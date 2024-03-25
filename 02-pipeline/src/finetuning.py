import os
import random
import gc

import pickle
import torchmetrics
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.dummy import DummyClassifier
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score, confusion_matrix, multilabel_confusion_matrix
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.utils.class_weight import compute_class_weight
from transformers import get_linear_schedule_with_warmup, logging as tf_logging
from torch.utils.data import TensorDataset, random_split
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pdb
import glob

import numpy as np
import torch
import wandb

import src.models as models

# ********************************************************************************************************************************


def set_seeds(seed):
	torch.manual_seed(seed)
	random.seed(seed)
	np.random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	torch.cuda.manual_seed(seed)
	torch.use_deterministic_algorithms(True)
	torch.backends.cudnn.deterministic = True


def get_tgseed(seed):
	g = torch.Generator()
	g.manual_seed(seed)
	return g


def read_x_from_csv(filename):
	all_rows = []
	with open(filename, 'r') as f:
		lines = f.readlines()
		for line in lines:
			all_rows.append(line.replace("\n", ""))
	return np.array(all_rows)


def encode_sentence(roberta, text):
	text = text.strip()
	tokens = roberta.encode(text).tolist()
	tokens = list(map(str, tokens))
	sample_encoded = " ".join(tokens)
	return sample_encoded


def loss_func(y_pred_logits, y_pred_embeddings, y_true, imbalance_strategy, class_weights, dvc):
		
	loss_weight = torch.tensor(class_weights, dtype=torch.float32).to(dvc) if imbalance_strategy == 'loss_weight' else None

	ce_loss = torch.nn.CrossEntropyLoss(weight=loss_weight, reduction='none')(y_pred_logits, y_true)
	# ce_loss = torch.nn.CrossEntropyLoss(weight=loss_weight, reduction='mean')(y_pred_logits, y_true)
	
	ce_loss = torch.mean(ce_loss)

	return ce_loss


def fn_init_wandb(run_id, rand_seed, wandb_config):
	wand_exp = wandb.init(project=wandb_config.get("project"), entity=wandb_config.get("entity"))
	wand_exp.name = run_id + "-seed-" + str(rand_seed)


def log_metric(metric, val, epoch, log_with_wandb):
	if log_with_wandb:
		wandb.log({metric: val, "epoch": epoch})


def empty_cuda_cache():
	gc.collect()
	torch.cuda.empty_cache()


def check_label_distr(y_all, ds, is_debug):
	if is_debug: print(f"\n[{ds}] label distribution:")

	max_cnt = -np.inf
	min_cnt = np.inf

	for label in list(np.unique(y_all)):
		cnt = np.sum(y_all == label)

		max_cnt = np.max((cnt, max_cnt))
		min_cnt = np.min((cnt, min_cnt))

		if is_debug: print(f"{label} : {cnt} samples")

	max_cnt = int(max_cnt)
	min_cnt = int(min_cnt)

	if is_debug: print("max cnt: ", max_cnt)
	if is_debug: print("min cnt: ", min_cnt)
	if is_debug: print("")

	return max_cnt, min_cnt


def init_misc(rand_seed, run_id, is_debug, remove_log_files=True):
	set_seeds(rand_seed)
	os.makedirs("./ckpt/" + run_id, exist_ok=True)

	if os.path.exists("./output"):
		if remove_log_files:
			files = [ f for f in glob.glob('./output/*.csv') if f"{rand_seed}" in f ]
			for f in files:
				os.remove(f)
			preds_file = "./output/predictions-x-unlabeled.csv"
			if os.path.exists(preds_file):
				os.remove(preds_file)
	else:
		os.makedirs("./output/", exist_ok=True)

	dvc = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	torch.cuda.empty_cache()

	# for DEV
	#tf_logging.set_verbosity_warning()

	# for PRODUCTION
	tf_logging.set_verbosity_error()

	return dvc


def print_failed_classifier_majority(y_all, rand_seed):

	clf = DummyClassifier(strategy='most_frequent')
	clf = clf.fit(np.random.rand(y_all.shape[0]), y_all)
	y_pred = clf.predict(y_all)

	print("failed classifier (simply voting on most frequent class)")
	compute_classification_metrics(y_all, y_pred, None, "majority-voting", rand_seed, None, True)
	print("")



def resample_dataset(x, y, imbalance_strategy, ds, is_debug):
	x_resampled = np.array([])
	y_resampled = np.array([])

	max_cnt, min_cnt = check_label_distr(y, ds, is_debug)
	
	for label in list(np.unique(y)):
		mask_label = y == label

		y_label = y[mask_label]
		x_label = x[mask_label]

		if imbalance_strategy == "upsampling":
			if is_debug: print(f"[{ds}] label {label}: upsampling...")
			idx_resampled = np.random.choice(np.arange(len(y_label)), max_cnt)
		elif imbalance_strategy == "undersampling":
			if is_debug: print(f"[{ds}] label {label}: undersampling...")
			idx_resampled = np.random.choice(np.arange(len(y_label)), min_cnt)
		else:
			if is_debug: print(f"[{ds}] label {label}: no resampling...")
			idx_resampled = np.arange(len(y_label))

		y_resampled = np.concatenate((y_resampled, y_label[idx_resampled]))
		x_resampled = np.concatenate((x_resampled, x_label[idx_resampled]))

	check_label_distr(y_resampled, ds, is_debug)

	return x_resampled, y_resampled


def resample_datasets(x_train, x_val, y_train, y_val, imbalance_strategy, run_id, rand_seed, do_validation_set, is_debug):

	if y_val is not None:
		y_all = np.concatenate((y_train, y_val))
	else:
		y_all = y_train

	# check for class imbalance
	n_classes = len(list(np.unique(y_all)))
	if is_debug: print("num of classes: ", n_classes)

	# failed classifier: simply assign majority class
	if is_debug: print_failed_classifier_majority(y_all, rand_seed)

	class_weights_train = compute_class_weight(class_weight='balanced', classes=np.unique(y_all), y=y_train)
	if is_debug: print("class_weights: ", class_weights_train)

	x_train_resampled, y_train_resampled = resample_dataset(x_train, y_train, imbalance_strategy, "train", is_debug)

	if do_validation_set:
		x_val_resampled, y_val_resampled = resample_dataset(x_val, y_val, imbalance_strategy, "val", is_debug)
		return x_train_resampled, y_train_resampled, x_val_resampled, y_val_resampled, n_classes, class_weights_train
	
	return x_train_resampled, y_train_resampled, x_val, y_val, n_classes, class_weights_train


def make_onehot_encoding_for_y(y_train, y_val, do_validation_set, run_id):

	y_onehot_encoder = OneHotEncoder(sparse=False)
	
	if do_validation_set:
		y_onehot_encoder = y_onehot_encoder.fit(np.expand_dims(np.concatenate((y_train, y_val)), axis=1))
		y_val_encoded = y_onehot_encoder.transform(np.expand_dims([int(i) for i in y_val], 1))
	else:
		y_onehot_encoder = y_onehot_encoder.fit(np.expand_dims(y_train, axis=1))
		y_val_encoded = None

	y_train_encoded = y_onehot_encoder.transform(np.expand_dims([int(i) for i in y_train], 1))

	with open(f"./ckpt/{run_id}/y_label_transformer", "wb") as f:
		pickle.dump(y_onehot_encoder, f)

	return y_train_encoded, y_val_encoded, y_onehot_encoder


def get_embeddings_from_sentences(x_all, max_seq_length, language_model, language_for_model, custom_model_name, is_debug):

	if is_debug: print("run tokenizer...")
	tokenizer = models.get_tokenizer(language_model, language_for_model, custom_model_name)

	if is_debug: print("check how many samples get truncated...")
	num_truncated_samples = 0
	for x in tqdm(x_all):
		encoded_dict = tokenizer.encode_plus(
			x,
			add_special_tokens=True,
			max_length=None,
			padding=False,
			truncation=False,
			return_attention_mask=True,
			return_tensors='pt',
		)
		if encoded_dict['input_ids'].shape[1] >= max_seq_length:
			num_truncated_samples += 1

	if is_debug: print(f"# of samples which will get truncated: {num_truncated_samples}")
	if is_debug: print("encode samples...")

	input_ids = []
	attention_masks = []
	for x in tqdm(x_all):
		encoded_dict = tokenizer.encode_plus(
			x,
			add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
			max_length=max_seq_length,  # Pad & truncate all sentences.
			padding='max_length',
			truncation=True,
			return_attention_mask=True,  # Construct attn. masks.
			return_tensors='pt',  # Return pytorch tensors.
		)
		input_ids.append(encoded_dict['input_ids'])
		attention_masks.append(encoded_dict['attention_mask'])

	x_input_ids = torch.cat(input_ids, dim=0)
	x_attention_masks = torch.cat(attention_masks, dim=0)

	return x_input_ids, x_attention_masks


def init_dataloaders(x_train, y_train, x_val, y_val, do_validation_set, batch_size, max_seq_length, language_model, language_for_model, custom_model_name, rand_seed, is_debug):

	x_train_input_ids, x_train_attention_masks = get_embeddings_from_sentences(x_train, max_seq_length, language_model, language_for_model, custom_model_name, is_debug)
	train_dataset = TensorDataset(x_train_input_ids, x_train_attention_masks, torch.tensor(y_train))

	do_pin_memory = True
	num_workers = 0

	train_dataloader = torch.utils.data.DataLoader(
		train_dataset,
		batch_size=batch_size,
		generator=get_tgseed(rand_seed),
		shuffle=True,
		pin_memory=do_pin_memory,
		num_workers=num_workers,
	)

	if do_validation_set:

		x_val_input_ids, x_val_attention_masks = get_embeddings_from_sentences(x_val, max_seq_length, language_model, language_for_model, custom_model_name, is_debug)
		val_dataset = TensorDataset(x_val_input_ids, x_val_attention_masks, torch.tensor(y_val))

		val_dataloader = torch.utils.data.DataLoader(
		val_dataset,
		batch_size=batch_size,
		generator=get_tgseed(rand_seed),
		shuffle=True,
		pin_memory=do_pin_memory,
		num_workers=num_workers,
	)

	print("")
	print("# of train samples: ", train_dataset.__len__())
	if do_validation_set: 
		print("# of val samples: ", val_dataset.__len__())
		print("")
		return train_dataloader, train_dataset.__len__(), val_dataloader, val_dataset.__len__()
	else:
		print("")
		return train_dataloader, train_dataset.__len__(), None, None


def init_model(language_model, language_for_model, custom_model_name, learning_rate, dropout_rate, n_classes, dvc, n_epochs, gradient_accumulation_steps, len_train_dataloader):

	model = models.get_ptlm_for_classification(language_model, language_for_model, custom_model_name, dropout_rate, n_classes, dvc)

	# faster training
	# model = torch.compile(model)

	if len_train_dataloader is not None:
		
		optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

		total_steps = (len_train_dataloader / gradient_accumulation_steps) * n_epochs

		#scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
		#scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=total_steps*0.1, num_training_steps=total_steps)
		scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=total_steps*0.06, num_training_steps=total_steps)

		return model, optimizer, scheduler

	return model, None, None


def get_metrics(y_true, y_pred, dataset, is_debug):
	
	acc_score = compute_accuracy_score(y_true, y_pred, dataset, is_debug)
	prec_score = compute_precision_score(y_true, y_pred, dataset, is_debug)
	rec_score = compute_recall_score(y_true, y_pred, dataset, is_debug)
	f1_score_macro = compute_f1_score_macro(y_true, y_pred, dataset, is_debug)
	f1_score_weighted = compute_f1_score_weighted(y_true, y_pred, dataset, is_debug)

	return prec_score, f1_score_weighted, f1_score_macro, acc_score, rec_score


def compute_classification_metrics(y_true, y_pred, epoch, dataset, rand_seed, log_with_wandb, is_debug):

	prec_score, f1_score_weighted, f1_score_macro, acc_score, rec_score = get_metrics(y_true, y_pred, dataset, is_debug)

	file_name = f"./output/results-{dataset}-{rand_seed}.csv"
	if not os.path.exists(file_name):
		with open(file_name, "w") as file:
			file.write(f"epoch,precision,f1 score (weighted),f1 score (macro),accuracy,recall\n")	
	
	with open(file_name, "a") as file:
		file.write(f"{epoch},{prec_score},{f1_score_weighted},{f1_score_macro},{acc_score},{rec_score}\n")

	compute_confusion_matrix(y_true, y_pred, dataset, is_debug)
	
	log_metric(f"{dataset} precision", prec_score, epoch, log_with_wandb)
	log_metric(f"{dataset} f1 weighted", f1_score_weighted, epoch, log_with_wandb)
	log_metric(f"{dataset} f1 macro", f1_score_macro, epoch, log_with_wandb)
	log_metric(f"{dataset} accuracy", acc_score, epoch, log_with_wandb)
	log_metric(f"{dataset} recall", rec_score, epoch, log_with_wandb)

	return prec_score, f1_score_weighted, f1_score_macro, acc_score, rec_score


def compute_metrics_from_onehot_y(y_pred_enc, y_true_enc, loss_all, dataset, epoch, log_with_wandb, y_onehot_encoder, validation_set_metric, rand_seed, is_debug):

	y_pred_dec = y_onehot_encoder.inverse_transform(y_pred_enc)
	y_true_dec = y_onehot_encoder.inverse_transform(y_true_enc)

	if is_debug: print("")

	loss_mean = np.mean(loss_all)
	log_metric(f"{dataset} loss", loss_mean, epoch, log_with_wandb)
	if is_debug: print(f"[{dataset} loss]: ", loss_mean)

	prec_score, f1_score_weighted, f1_score_macro, acc_score, rec_score = compute_classification_metrics(y_true_dec, y_pred_dec, epoch, dataset, rand_seed, log_with_wandb, is_debug)

	if validation_set_metric == "f1_score_macro":
		val_metric = f1_score_macro
	elif validation_set_metric == "f1_score_weighted":
		val_metric = f1_score_weighted
	elif validation_set_metric == "accuracy":
		val_metric = acc_score
	elif validation_set_metric == "recall":
		val_metric = rec_score
	else:
		val_metric = prec_score

	return val_metric, y_pred_dec


def compute_and_print_metrics_for_dataset_b(y, y_pred, epoch, rand_seed, log_with_wandb, is_debug):

	if(type(y_pred) is not list):
		compute_classification_metrics(y, y_pred, epoch, "test", rand_seed, log_with_wandb, is_debug)
	else:
		y_preds = y_pred

		prec_scores = []
		f1_scores_weighted = []
		f1_scores_macro = []
		acc_scores = []
		rec_scores = []
		for y_pred in y_preds:
			prec_score, f1_score_weighted, f1_score_macro, acc_score, rec_score = get_metrics(y, y_pred, None, False)
			prec_scores.append(prec_score)
			f1_scores_weighted.append(f1_score_weighted)
			f1_scores_macro.append(f1_score_macro)
			acc_scores.append(acc_score)
			rec_scores.append(rec_score)

		acc_fmt = f"accuracy: {round(np.mean(acc_scores), 2)} +/- {round(np.std(acc_scores), 2)}"
		prec_fmt = f"precision: {round(np.mean(prec_scores), 2)} +/- {round(np.std(prec_scores), 2)}"
		rec_fmt = f"recall: {round(np.mean(rec_scores), 2)} +/- {round(np.std(rec_scores), 2)}"
		f1_macro_fmt = f"f1 score macro: {round(np.mean(f1_scores_macro), 2)} +/- {round(np.std(f1_scores_macro), 2)}"
		f1_weighted_fmt = f"f1 score weighted: {round(np.mean(f1_scores_weighted), 2)} +/- {round(np.std(f1_scores_weighted), 2)}"

		with open("stats.txt", "w") as f:
			f.write(acc_fmt + "\n" + prec_fmt + "\n" + rec_fmt + "\n" + f1_macro_fmt + "\n" + f1_weighted_fmt)

		print("")
		print(acc_fmt)
		print(prec_fmt)
		print(rec_fmt)
		print(f1_macro_fmt)
		print(f1_weighted_fmt)
		print("")


def compute_mse_score(y_true, y_pred, dataset, is_debug):
	mse_score = mean_squared_error(y_true, y_pred)
	if is_debug: print(f"{dataset} MSE: {mse_score}")
	return mse_score


def compute_rmse_score(y_true, y_pred, dataset, is_debug):
	rmse_score = mean_squared_error(y_true, y_pred, squared=False)
	if is_debug: print(f"{dataset} RMSE: {rmse_score}")
	return rmse_score


def compute_mae_score(y_true, y_pred, dataset, is_debug):
	mae_score = mean_absolute_error(y_true, y_pred)
	if is_debug: print(f"{dataset} MAE: {mae_score}")
	return mae_score


def compute_r2_score(y_true, y_pred, dataset, is_debug):
	r2_sc = r2_score(y_true, y_pred)
	if is_debug: print(f"{dataset} R2 Score: {r2_sc}")
	return r2_sc


def compute_confusion_matrix(y_true, y_pred, dataset, is_debug):
	if is_debug: print(f"{dataset} confusion matrix:")
	if is_debug: print(multilabel_confusion_matrix(y_true, y_pred))


def compute_accuracy_score(y_true, y_pred, dataset, is_debug):
	accuracy = accuracy_score(y_true, y_pred)
	if is_debug: print(f"{dataset} accuracy: {accuracy}")

	return accuracy


def compute_precision_score(y_true, y_pred, dataset, is_debug):
	prec_score = precision_score(y_true, y_pred, average='weighted')
	if is_debug: print(f"{dataset} precision: {prec_score}")

	return prec_score


def compute_recall_score(y_true, y_pred, dataset, is_debug):
	recall = recall_score(y_true, y_pred, average='weighted')
	if is_debug: print(f"{dataset} recall: {recall}")

	return recall


def compute_f1_score_weighted(y_true, y_pred, dataset, is_debug):
	f1_score_weighted = f1_score(y_true, y_pred, average='weighted')
	if is_debug: print(f"{dataset} f1 weighted: {f1_score_weighted}")

	return f1_score_weighted


def compute_f1_score_macro(y_true, y_pred, dataset, is_debug):
	f1_score_macro = f1_score(y_true, y_pred, average='macro')
	if is_debug: print(f"{dataset} f1 macro: {f1_score_macro}")

	return f1_score_macro


def run_forward_pass_with_loss(batch, model, imbalance_strategy, class_weights, dvc):

	x_ids, x_masks, y_true = batch
	x_ids, x_masks, y_true = x_ids.to(dvc), x_masks.to(dvc), y_true.to(dvc)

	_, y_ind = torch.max(y_true, dim=1)
	output = model(x_ids, attention_mask=x_masks, labels=y_ind)
	
	y_pred = torch.nn.Softmax(dim=1)(output.logits.float())
	loss_mean = loss_func(output.logits.float(), output.hidden_states, y_ind, imbalance_strategy, class_weights, dvc)

	return loss_mean, y_pred, y_true


def run_forward_pass_only(model, test_dataloader, test_n_total, n_classes, y_label_transformer, dvc):

	with torch.no_grad():

		model.eval()

		y_pred_all = np.zeros((test_n_total, n_classes))

		idx = 0
		for _, batch in enumerate(tqdm(test_dataloader)):

			x_ids, x_masks = batch
			x_ids, x_masks = x_ids.to(dvc), x_masks.to(dvc)

			output = model(x_ids, attention_mask=x_masks)

			y_pred = torch.nn.Softmax(dim=1)(output.logits.float())

			n_batch = y_pred.shape[0]

			y_pred_all[idx:(idx + n_batch), :] = y_pred.detach().cpu().numpy()
			
			idx += n_batch

		y_pred_dec = y_label_transformer.inverse_transform(y_pred_all).squeeze()

		empty_cuda_cache()

		return y_pred_dec


def run_train_epoch(epoch, rand_seed, model, optimizer, gradient_accumulation_steps, imbalance_strategy, class_weights, scheduler, train_dataloader, train_n_total, n_classes, y_label_transformer, validation_set_metric, log_with_wandb, dvc, is_debug):

	model.train()
	optimizer.zero_grad()

	loss_batchwise = np.zeros(len(train_dataloader))
	y_pred_all = np.zeros((train_n_total, n_classes))
	y_true_all = np.zeros((train_n_total, n_classes))

	idx = 0
	for idx_batch, batch in enumerate(tqdm(train_dataloader)):
		loss_batch_mean, y_pred, y_true = run_forward_pass_with_loss(batch, model, imbalance_strategy, class_weights, dvc)

		# gradient accumulation: scale loss by steps
		loss_batch_mean = loss_batch_mean / gradient_accumulation_steps

		n_batch = y_pred.shape[0]
		loss_batchwise[idx_batch] = loss_batch_mean.detach().item()
		y_pred_all[idx:(idx + n_batch), :] = y_pred.detach().cpu().numpy()
		y_true_all[idx:(idx + n_batch), :] = y_true.detach().cpu().numpy()
		idx += n_batch

		# do optimization step
		loss_batch_mean.backward()

		if ((idx_batch + 1) % gradient_accumulation_steps == 0) or (idx_batch + 1 == len(train_dataloader)):		
			torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # TODO: set to 2.0 ??
			optimizer.step()
			scheduler.step()
			optimizer.zero_grad()

	compute_metrics_from_onehot_y(y_pred_all, y_true_all, loss_batchwise, "train", epoch, log_with_wandb, y_label_transformer, validation_set_metric, rand_seed, is_debug)

	empty_cuda_cache()


def run_val_epoch(epoch, rand_seed, model, imbalance_strategy, class_weights, val_dataloader, val_n_total, n_classes, y_label_transformer, validation_set_metric, log_with_wandb, dvc, is_debug):

	with torch.no_grad():

		model.eval()

		loss_batchwise = np.zeros(len(val_dataloader))
		y_pred_all = np.zeros((val_n_total, n_classes))
		y_true_all = np.zeros((val_n_total, n_classes))

		idx = 0
		for idx_batch, batch in enumerate(tqdm(val_dataloader)):
			loss_batch_mean, y_pred, y_true = run_forward_pass_with_loss(batch, model, imbalance_strategy, class_weights, dvc)

			n_batch = y_pred.shape[0]
			loss_batchwise[idx_batch] = loss_batch_mean.detach().item()
			y_pred_all[idx:(idx + n_batch), :] = y_pred.detach().cpu().numpy()
			y_true_all[idx:(idx + n_batch), :] = y_true.detach().cpu().numpy()
			idx += n_batch

		val_metric, y_pred_dec = compute_metrics_from_onehot_y(y_pred_all, y_true_all, loss_batchwise, "val", epoch, log_with_wandb, y_label_transformer, validation_set_metric, rand_seed, is_debug)

		empty_cuda_cache()

		return val_metric


def run_training(n_epochs, run_id, rand_seed, model, optimizer, imbalance_strategy, class_weights, scheduler, train_dataloader, train_n_total, val_dataloader, val_n_total, n_classes, y_label_transformer, do_validation_set, validation_set_metric, log_with_wandb, dvc, language_model, language_for_model, custom_model_name, max_seq_length, dataset_B_unlabelled_x, batch_size, gradient_accumulation_steps, is_debug):

	best_val_score = -np.inf

	for epoch in range(n_epochs):

		print(f"\nepoch: {epoch}...")

		empty_cuda_cache()

		run_train_epoch(epoch, rand_seed, model, optimizer, gradient_accumulation_steps, imbalance_strategy, class_weights, scheduler, train_dataloader, train_n_total, n_classes, y_label_transformer, validation_set_metric, log_with_wandb, dvc, is_debug)

		if do_validation_set:
			val_score = run_val_epoch(epoch, rand_seed, model, imbalance_strategy, class_weights, val_dataloader, val_n_total, n_classes, y_label_transformer, validation_set_metric, log_with_wandb, dvc, is_debug)

			is_best = val_score >= best_val_score
			if is_best:
				best_val_score = val_score
		else:
			best_val_score = None
			is_best = False

		models.save_model(
			{
				'epoch': epoch,
				'best_val_loss': best_val_score,
				'state_dict': model.state_dict(),
				'optimizer': optimizer.state_dict()
			}, run_id, is_best, rand_seed)


def predict_y_from_trained_model(run_id, language_model, language_for_model, custom_model_name, dataset_B_unlabelled_x, model, batch_size, rand_seed, max_seq_length, dvc, is_debug):

	#ckpt_path = f"./ckpt/{run_id}/ckpt_best_seed_1234.pth.tar"
	ckpt_path = f"./ckpt/{run_id}/ckpt_last.pth.tar"

	if is_debug: print(f"resume model from {ckpt_path}")
	ckpt = torch.load(ckpt_path, map_location=dvc)
	model.load_state_dict(ckpt['state_dict'])

	with open(f"./ckpt/{run_id}/y_label_transformer", "rb") as f:
		y_label_transformer = pickle.load(f)

	n_classes = len(y_label_transformer.categories_[0])

	input_ids, attention_masks = get_embeddings_from_sentences(dataset_B_unlabelled_x, max_seq_length, language_model, language_for_model, custom_model_name, is_debug)

	test_dataset = TensorDataset(input_ids, attention_masks)
	test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

	return run_forward_pass_only(model, test_dataloader, test_dataset.__len__(), n_classes, y_label_transformer, dvc)


def do_train_val_split(dataset_A_labelled_x, dataset_A_labelled_y, rand_seed, do_validation_set):
	
	if do_validation_set:
		train_size = int(0.9 * len(dataset_A_labelled_x)) if do_validation_set else int(len(dataset_A_labelled_x))
		val_size = (len(dataset_A_labelled_x) - train_size)
		x_train, x_val, y_train, y_val = train_test_split(dataset_A_labelled_x, dataset_A_labelled_y, test_size=val_size, random_state=rand_seed)
		return x_train, x_val, y_train, y_val
	else:
		return dataset_A_labelled_x, None, dataset_A_labelled_y, None


def train_and_predict_test(dataset_A_labelled_x,
							dataset_A_labelled_y,
							run_id,
							n_epochs,
							imbalance_strategy,
							dataset_B_unlabelled_x=None,
							learning_rate=2e-5,
							dropout_rate=0.1,
							batch_size=32,
							gradient_accumulation_steps=1,
							rand_seed=1234,
							max_seq_length=512,
							language_model="BERT",
							language_for_model="en",
							custom_model_name=None,
							do_validation_set=True,
							validation_set_metric="f1_macro",
							is_debug=False,
							log_with_wandb=False,
							wandb_config=None):

	print("start training...")
	print(f"run id: {run_id}")
	print(f"random seed: {rand_seed}")
	print(f"batch size: {batch_size}")
	print(f"learning rate: {learning_rate}")
	print(f"dropout rate: {dropout_rate}")
	print(f"imbalance strategy: {imbalance_strategy}")
	print(f"is CUDA available: {torch.cuda.is_available()}")
	print("")

	if log_with_wandb: fn_init_wandb(run_id, rand_seed, wandb_config)

	if is_debug: print(f"start training for random seed: {rand_seed}\n")

	dvc = init_misc(rand_seed, run_id, is_debug)

	x_train, x_val, y_train, y_val = do_train_val_split(dataset_A_labelled_x, dataset_A_labelled_y, rand_seed, do_validation_set)

	x_train, y_train, x_val, y_val, n_classes, class_weights = resample_datasets(x_train, x_val, y_train, y_val, imbalance_strategy, run_id, rand_seed, do_validation_set, is_debug)
	
	y_train, y_val, y_label_transformer = make_onehot_encoding_for_y(y_train, y_val, do_validation_set, run_id)

	train_dataloader, train_n_total, val_dataloader, val_n_total = init_dataloaders(x_train, y_train, x_val, y_val, do_validation_set, batch_size, max_seq_length, language_model, language_for_model, custom_model_name, rand_seed, is_debug)

	model, optimizer, scheduler = init_model(language_model, language_for_model, custom_model_name, learning_rate, dropout_rate, n_classes, dvc, n_epochs, gradient_accumulation_steps, len(train_dataloader))

	run_training(n_epochs, run_id, rand_seed, model, optimizer, imbalance_strategy, class_weights, scheduler, train_dataloader, train_n_total, val_dataloader, val_n_total, n_classes, y_label_transformer, do_validation_set, validation_set_metric, log_with_wandb, dvc, language_model, language_for_model, custom_model_name, max_seq_length, dataset_B_unlabelled_x, batch_size, gradient_accumulation_steps, is_debug)

	y_pred = None
	if dataset_B_unlabelled_x is not None:
		y_pred = predict_y_from_trained_model(run_id, language_model, language_for_model, custom_model_name, dataset_B_unlabelled_x, model, batch_size, rand_seed, max_seq_length, dvc, is_debug)

	if log_with_wandb: wandb.finish()
	
	empty_cuda_cache()
	
	print("\nfinished with training!\n")

	return y_pred

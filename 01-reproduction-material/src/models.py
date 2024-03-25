import torch
from transformers import BertForSequenceClassification, BertConfig, RobertaForSequenceClassification, RobertaTokenizer, BertTokenizer, DistilBertForSequenceClassification, DistilBertTokenizer, AlbertTokenizer, AlbertForSequenceClassification, RobertaConfig, DistilBertConfig, XLMRobertaTokenizer
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig


def save_model(state, run_id, is_best, seed):	
	if is_best:
		torch.save(state, './ckpt/' + run_id + f'/ckpt_best.pth.tar')
		torch.save(state, './ckpt/' + run_id + f'/ckpt_last.pth.tar')
	else:
		torch.save(state, './ckpt/' + run_id + f'/ckpt_last.pth.tar')


def get_model_name_from_lang_model_and_lang(language_model, language_for_model):
	
	if language_model == "ROB-BASE":
		if language_for_model == "de":
			return "benjamin/roberta-base-wechsel-german"
		else:
			return "roberta-base"
	elif language_model == "ROB-LRG":
		if language_for_model == "de":
			return "xlm-roberta-large-finetuned-conll03-german"
		else:
			return "roberta-large"
	elif language_model == "ELE-LRG":
		if language_for_model == "de":
			raise Exception("No german model for ELE-LRG")
		else:
			return "google/electra-large-discriminator"
	elif language_model == "ELE-BS-GER":
		if language_for_model == "de":
			return "german-nlp-group/electra-base-german-uncased"
		else:
			raise Exception("No model for ELE-BS-GER in any other language than German available")
	elif language_model == "DEB-V3":
		if language_for_model == "de":
			raise Exception("No german model for DEB-V3")
		else:
			return "microsoft/deberta-v3-large"
	elif language_model == "XLNET-LRG":
		if language_for_model == "de":
			raise Exception("No german model for XLNET-LRG")
		else:
			return "xlnet-large-cased"
	else:			
		raise Exception("Could not find appropriate language model. Please try another model name")


def get_tokenizer(language_model, language_for_model, custom_model_name=None):

	if custom_model_name is None:
		custom_model_name = get_model_name_from_lang_model_and_lang(language_model, language_for_model)

	return AutoTokenizer.from_pretrained(custom_model_name)


def get_ptlm_for_classification(language_model, language_for_model, custom_model_name, dropout_rate, n_classes, dvc):

	if custom_model_name is None:
		custom_model_name = get_model_name_from_lang_model_and_lang(language_model, language_for_model)

	print(f"using language model: {custom_model_name}")

	if language_model == "bert":
		config = BertConfig().from_pretrained(custom_model_name)
		config.num_labels = n_classes
		config.output_attentions = False
		config.output_hidden_states = True
		config.classifier_dropout = dropout_rate

		return BertForSequenceClassification.from_pretrained(
			custom_model_name,
			config=config
		).to(dvc)

	elif language_model == "roberta":
		config = RobertaConfig().from_pretrained(custom_model_name)
		config.num_labels = n_classes
		config.output_attentions = False
		config.output_hidden_states = True
		config.classifier_dropout = dropout_rate

		return RobertaForSequenceClassification.from_pretrained(
			custom_model_name,
			config=config
		).to(dvc)

	elif language_model == "distilbert":
		config = DistilBertConfig().from_pretrained(custom_model_name)
		config.num_labels = n_classes
		config.output_attentions = False
		config.output_hidden_states = True
		config.seq_classif_dropout = dropout_rate
		return DistilBertForSequenceClassification.from_pretrained(
			custom_model_name,
			config=config,
		).to(dvc)

	elif language_model == "albert":
		config = AlbertConfig().from_pretrained(custom_model_name)
		config.num_labels = n_classes
		config.output_attentions = False
		config.output_hidden_states = True
		config.classifier_dropout_prob = dropout_rate

		return AlbertForSequenceClassification.from_pretrained(
			custom_model_name,
			config=config,
		).to(dvc)

	else:
		config = AutoConfig.from_pretrained(custom_model_name)
		config.num_labels = n_classes
		config.output_attentions = False
		config.output_hidden_states = True

		# TODO: fix dropout for this one or ignore?

		return AutoModelForSequenceClassification.from_pretrained(
			custom_model_name,
			config=config,
		).to(dvc)

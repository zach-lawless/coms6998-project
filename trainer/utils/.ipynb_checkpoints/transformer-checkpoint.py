import torch
import transformers
from transformers import AdapterType
from transformers import BertTokenizerFast, BertForSequenceClassification


def get_tokenizer(model_name):
    if model_name == 'bert-base-uncased':
        tokenizer = BertTokenizerFast.from_pretrained(model_name)
    else:
        raise NotImplementedError

    return tokenizer


def get_transformer(model_name, num_labels, adapter, dataset):
    if model_name == 'bert-base-uncased':
        model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        if adapter:
            model.add_adapter(dataset, AdapterType.text_task)
            model.train_adapter(dataset)
    else:
        raise NotImplementedError

    return model


def get_criterion(num_labels):
    if num_labels == 2:
        criterion = torch.nn.CrossEntropyLoss()
    else:
        raise NotImplementedError

    return criterion

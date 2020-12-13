import datasets
import torch
from torch.utils.data import DataLoader, TensorDataset


def get_dataset(dataset):
    ds = datasets.load_dataset('glue', dataset)
    num_classes = ds['train'].features['label'].num_classes
    return ds, num_classes


def create_dataset_from_text_dataset(ds, tokenizer):
    encoding = tokenizer(ds['sentence'], return_tensors='pt', padding=True, truncation=True)
    input_ids = encoding['input_ids']
    attn_masks = encoding['attention_mask']
    labels = torch.tensor(ds['label'])
    return TensorDataset(input_ids, attn_masks, labels)


def get_tensor_datasets(dataset_dict, splits, tokenizer):
    split_datasets = {}
    for s in splits:
        split_datasets[s] = create_dataset_from_text_dataset(dataset_dict[s], tokenizer)
    return split_datasets


def get_data_loaders(split_datasets, batch_size):
    train_loader = DataLoader(split_datasets['train'], batch_size, shuffle=True)
    val_loader = DataLoader(split_datasets['validation'], batch_size, shuffle=False)
    test_loader = DataLoader(split_datasets['test'], batch_size, shuffle=False)
    return train_loader, val_loader, test_loader

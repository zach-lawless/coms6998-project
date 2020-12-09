import argparse

from trainer.trainer import Trainer
from trainer.utils.dataset import get_dataset, get_tensor_datasets, get_data_loaders
from trainer.utils.learning_scheme import get_learning_scheme
from trainer.utils.transformer import get_tokenizer, get_transformer, get_criterion

SUPPORTED_TRANSFORMERS = ['bert-base-uncased']
SUPPORTED_DATASETS = ['sst2']
SUPPORTED_LEARNING_SCHEMES = ['differential']


def main(args_dict):

    # Load dataset
    dataset = args_dict['dataset']
    print(f'Loading {dataset} dataset...')
    dataset_dict, num_classes = get_dataset(dataset)

    # Load tokenizer
    model_name = args_dict['transformer']
    print(f'Loading tokenizer for {model_name}...')
    tokenizer = get_tokenizer(model_name)

    # Create data loader for each split
    splits = list(dataset_dict.keys())
    print(f'Creating data loader for {splits} splits...')
    split_datasets = get_tensor_datasets(dataset_dict, splits, tokenizer)
    train_loader, val_loader = get_data_loaders(split_datasets)

    # Load model
    adapter = args_dict['adapter']
    print(f'Loading {model_name} with adapters={adapter}...')
    model = get_transformer(model_name,
                            num_labels=num_classes,
                            adapter=adapter,
                            dataset=dataset)
    criterion = get_criterion(num_labels=num_classes)

    # Get learning scheme
    learning_scheme = args_dict['learning_scheme']
    print(f'Configuring {learning_scheme} learning scheme...')
    optimizer = get_learning_scheme(learning_scheme, model)

    # Initialize Trainer
    print(f'Initializing Trainer object')
    trainer = Trainer(model=model,
                      n_epochs=2,
                      optimizer=optimizer,
                      scheduler=None,
                      criterion=criterion)

    # Perform training
    trainer.train_loop(train_loader, val_loader)

    # TODO: Save results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run a finetuning trial')
    parser.add_argument('-t', '--transformer',
                        choices=SUPPORTED_TRANSFORMERS,
                        type=str,
                        help='the specific transformer to finetune')
    parser.add_argument('-d', '--dataset',
                        choices=SUPPORTED_DATASETS,
                        type=str,
                        help='the dataset you are finetuning on')
    parser.add_argument('-a', '--adapter',
                        type=bool,
                        help='whether to add adapters to the transformer or not')
    parser.add_argument('-l', '--learning-scheme',
                        choices=SUPPORTED_LEARNING_SCHEMES,
                        type=str,
                        help='the learning scheme to fine tune with')
    args = parser.parse_args()
    args_dict = vars(args)
    main(args_dict)

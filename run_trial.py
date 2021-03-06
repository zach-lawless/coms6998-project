import argparse

from trainer.trainer import Trainer
from trainer.utils.dataset import get_dataset, get_tensor_datasets, get_data_loaders
from trainer.utils.learning_scheme import get_learning_scheme, get_scheduler
from trainer.utils.transformer import get_tokenizer, get_transformer, get_criterion

SUPPORTED_TRANSFORMERS = ['bert-base-uncased']
SUPPORTED_DATASETS = ['sst2', 'cola']
SUPPORTED_LEARNING_SCHEMES = ['differential', 'fixed', 'nesterov', 'gradual-unfreeze']
SUPPORTED_SCHEDULERS = ['cyclic-triangular']


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
    train_loader, val_loader, test_loader = get_data_loaders(split_datasets, args_dict['batch_size'])

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
    learning_rate = args_dict['learning_rate']
    print(f'Configuring {learning_scheme} learning scheme...')
    optimizer = get_learning_scheme(learning_scheme, model, learning_rate, adapter, epoch=0)

    # Get scheduler
    print('Setting up scheduler if any...')
    scheduler = get_scheduler(args_dict['scheduler'], optimizer, learning_rate, args_dict['max_learning_rate'])

    # Initialize Trainer
    print(f'Initializing Trainer object...')
    trainer = Trainer(model=model,
                      n_epochs=args_dict['epochs'],
                      optimizer=optimizer,
                      scheduler=scheduler,
                      criterion=criterion,
                      learning_scheme=learning_scheme,
                      learning_rate=learning_rate,
                      adapter=adapter)

    # Perform training
    print(f'Beginning finetuning...')
    epoch_history, batch_history = trainer.train_loop(train_loader, val_loader, args_dict['batch_logging'])
    
    # Print holdout test metrics
    # NOTE: Commenting out as all labels in the test set are -1 and an error is thrown
    # print(f'Evaluating on test dataset...')
    # test_loss, test_acc = trainer.measure_performance(test_loader)
    # print(f'Test loss: {test_loss.item()}, Test accuracy: {test_acc.item()}')

    # Save results
    trial_name = args_dict['name']
    epoch_history['name'] = trial_name
    batch_history['name'] = trial_name
    epoch_history.to_csv(trial_name+'_epoch_history.csv', index=False)
    batch_history.to_csv(trial_name+'_batch_history.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run a finetuning trial')
    parser.add_argument('--name',
                        type=str,
                        help='name to give trial (for output saving purposes)')
    parser.add_argument('--transformer',
                        choices=SUPPORTED_TRANSFORMERS,
                        type=str,
                        help='the specific transformer to finetune')
    parser.add_argument('--dataset',
                        choices=SUPPORTED_DATASETS,
                        type=str,
                        help='the dataset you are finetuning on')
    parser.add_argument('--batch-size',
                        type=int,
                        default=128,
                        help='how large of batches to feed to the transformer')
    parser.add_argument('--batch-logging',
                        type=int,
                        default=10,
                        help='how frequently to log and print finetuning batch info')
    parser.add_argument('--adapter',
                        type=bool,
                        help='whether to add adapters to the transformer or not')
    parser.add_argument('--learning-scheme',
                        choices=SUPPORTED_LEARNING_SCHEMES,
                        type=str,
                        help='the learning scheme to fine tune with')
    parser.add_argument('--learning-rate',
                        type=float,
                        help='the learning rate to use for finetuning')
    parser.add_argument('--max-learning-rate',
                        type=float,
                        default=0.1,
                        help='the max learning rate if using a scheduler')
    parser.add_argument('--epochs',
                        type=int,
                        default=2,
                        help='how many epochs to finetune for')
    parser.add_argument('--scheduler',
                        type=str,
                        choices=SUPPORTED_SCHEDULERS,
                        default=None,
                        help='learning rate scheduler to use, if any')
    args = parser.parse_args()
    main(vars(args))

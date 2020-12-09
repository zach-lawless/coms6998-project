import argparse

SUPPORTED_TRANSFORMERS = ['bert-base-uncased']
SUPPORTED_DATASETS = ['sst2']
SUPPORTED_ADAPTERS = []
SUPPORTED_LEARNING_SCHEMES = ['differential']

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
                        choices=SUPPORTED_ADAPTERS,
                        type=str,
                        help='the adapter to add for finetunig')
    parser.add_argument('-l', '--learning-scheme',
                        choices=SUPPORTED_LEARNING_SCHEMES,
                        type=str,
                        help='the learning scheme to fine tune with')
    args = parser.parse_args()
    print(args)

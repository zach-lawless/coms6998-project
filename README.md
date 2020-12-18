# coms6998-project

This is the repo for our Survey of Finetuning Techniques final project in Columbia's Fall 2020 COMS 6998 Deep Learning
System Performance course.

Team Members:
* Shawn Pachgade (snp2128@columbia.edu)
* Zach Lawless (ztl2103@columbia.edu)

The general purpose of this repository is for prototyping and module development of a benchmarking framework that
enables running experiments for a variety of finetuning techniques.

The corresponding report for this assignment is in the repo as [report.pdf](report.pdf).

# Repository Overview

The general structure of this repository is as follows

```

final_report.pdf: our final report for this project
notebooks: where we stored our own prototyping work
outputs: where we persisted our experiment training history
trainer: the trainer module we developed for running experiments
    utils: helper functions necessary for intermediate steps in the overall experiment process
        dataset.py: helper functions for loading and preprocessing data
        learning_scheme.py: helper functions for setting up the various learning schemes
        transformer.py: helper functions for loading the transformer to use
    trainer.py: where the Trainer class is instantiated
requirements.txt: contains all of the necessary packages for running this module
run_trial.py: the main python script to call for performing experiments

```

The notebooks in this repository should be considered as our scratch work and nothing more. The actual executable 
experimenting modules are in the `trainer` directory and called via the `run_trial.py` script.

In order to install all of the necessary packages, you can run:

```

pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html

```

# Running an Experiment

You can view the command line arguments that the `run_trial.py` expects by running:

```
python run_trial.py --help
```

The output of this command will look like:

```

usage: run_trial.py [-h] [--name NAME] [--transformer {bert-base-uncased}] [--dataset {sst2,cola}] [--batch-size BATCH_SIZE] [--batch-logging BATCH_LOGGING]
                    [--adapter ADAPTER] [--learning-scheme {differential,fixed,nesterov,gradual-unfreeze}] [--learning-rate LEARNING_RATE]
                    [--max-learning-rate MAX_LEARNING_RATE] [--epochs EPOCHS] [--scheduler {cyclic-triangular}]

Run a finetuning trial

optional arguments:
  -h, --help            show this help message and exit
  --name NAME           name to give trial (for output saving purposes)
  --transformer {bert-base-uncased}
                        the specific transformer to finetune
  --dataset {sst2,cola}
                        the dataset you are finetuning on
  --batch-size BATCH_SIZE
                        how large of batches to feed to the transformer
  --batch-logging BATCH_LOGGING
                        how frequently to log and print finetuning batch info
  --adapter ADAPTER     whether to add adapters to the transformer or not
  --learning-scheme {differential,fixed,nesterov,gradual-unfreeze}
                        the learning scheme to fine tune with
  --learning-rate LEARNING_RATE
                        the learning rate to use for finetuning
  --max-learning-rate MAX_LEARNING_RATE
                        the max learning rate if using a scheduler
  --epochs EPOCHS       how many epochs to finetune for
  --scheduler {cyclic-triangular}
                        learning rate scheduler to use, if any

```

The command line documentation should be descriptive enough to help run any form of trial that is supported
(defaults and valid options are shown in the documentation).

An example of running an experiment using the `bert-base-uncased` Transformer on the CoLA dataset for 5 epochs with a
minibatch size of 32 and an Adapter with Differential Learning rates starting at 0.01 is below:

```

python run_trial.py \
    --name my_experiment \
    --transformer bert-base-uncased \
    --dataset cola \
    --epochs 5 \
    --batch-size 32 \
    --adapter True \
    --learning-scheme differential
    --learning-rate 0.01

```

# Results Summary

The following table contains the performance metrics associated with the experiments that we performed on the SST2 dataset.

| Dataset | Epochs | Batch Size | Learning Scheme | Learning Rate | Adapter | Scheduler | Validation Loss | Validation Accuracy |
|---------|--------|------------|-----------------|---------------|---------|-----------|-----------------|---------------------|
| SST2    | 5      | 32         | Fixed           | 0.01          |         |           | 0.235           | 0.907               |
| SST2    | 5      | 32         | Fixed           | 0.01          | Yes     |           | 0.251           | 0.900               |
| SST2    | 5      | 32         | Fixed           | 0.01          |         | Triangle  | 0.238           | 0.914               |
| SST2    | 5      | 32         | Fixed           | 0.01          | Yes     | Triangle  | 0.229           | 0.925               |
| SST2    | 5      | 32         | Nesterov        | 0.01          |         |           | 0.227           | 0.911               |
| SST2    | 5      | 32         | Nesterov        | 0.01          | Yes     |           | 0.234           | 0.907               |
| SST2    | 5      | 32         | Differential    | 0.01          |         |           | 0.346           | 0.842               |
| SST2    | 5      | 32         | Differential    | 0.01          | Yes     |           | 0.345           | 0.858               |
| SST2    | 5      | 32         | Gradual Unfreeze| 0.01          |         |           | 0.389           | 0.835               |
| SST2    | 5      | 32         | Gradual Unfreeze| 0.01          | Yes     |           | 0.424           | 0.806               |

The following table contains the performance metrics associated with the experiments that we performed on the CoLA dataset.

| Dataset | Epochs | Batch Size | Learning Scheme | Learning Rate | Adapter | Scheduler | Validation Loss | Validation Accuracy |
|---------|--------|------------|-----------------|---------------|---------|-----------|-----------------|---------------------|
| CoLA    | 10     | 32         | Fixed           | 0.01          |         |           | 0.483           | 0.772               |
| CoLA    | 10     | 32         | Fixed           | 0.01          | Yes     |           | 0.458           | 0.776               |
| CoLA    | 10     | 32         | Fixed           | 0.01          |         | Triangle  | 0.433           | 0.822               |
| CoLA    | 10     | 32         | Fixed           | 0.01          | Yes     | Triangle  | 0.407           | 0.831               |
| CoLA    | 10     | 32         | Nesterov        | 0.01          |         |           | 0.449           | 0.819               |
| CoLA    | 10     | 32         | Nesterov        | 0.01          | Yes     |           | 0.417           | 0.814               |
| CoLA    | 10     | 32         | Differential    | 0.01          |         |           | 0.548           | 0.742               |
| CoLA    | 10     | 32         | Differential    | 0.01          | Yes     |           | 0.560           | 0.739               |
| CoLA    | 10     | 32         | Gradual Unfreeze| 0.01          |         |           | 0.554           | 0.737               |
| CoLA    | 10     | 32         | Gradual Unfreeze| 0.01          | Yes     |           | 0.574           | 0.737               |

# Next Steps
Feel free to use this code in however you see fit. We do not plan to maintain or make updates at this time, but that
could always change based on interest or requests for additional features.

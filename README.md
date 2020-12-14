# coms6998-project
Project repo for Columbia's Fall 2020 COMS 6998 Deep Learning System Performance course


# Repository Overview


# Running an Experiment


# Results Summary

The following table contains the performance metrics associated with the experiments that we performed.

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
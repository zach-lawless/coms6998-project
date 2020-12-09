import torch


def get_learning_scheme(learning_scheme, model):
    if learning_scheme == 'differential':
        optimizer_grouped_parameters = differential_learning_scheme(model)
        optimizer = torch.optim.SGD(optimizer_grouped_parameters)
    else:
        raise NotImplementedError

    return optimizer


def differential_learning_scheme(model, learning_rate=0.1, divisor=2.6):
    param_prefixes = {}
    for n, p in model.named_parameters():
        base = n.partition('.weight')[0].partition('.bias')[0]
        if base not in param_prefixes:
            param_prefixes[base] = 0

    param_prefix_divisors = list(reversed([divisor * i for i in range(1, len(param_prefixes))])) + [1]
    param_learning_rates = [learning_rate / ld for ld in param_prefix_divisors]

    param_prefix_lr_lookup = dict(zip(param_prefixes.keys(), param_learning_rates))

    optimizer_grouped_parameters = [
        {'params': p, 'lr': param_prefix_lr_lookup[n.partition('.weight')[0].partition('.bias')[0]]}
        for n, p in model.named_parameters()
    ]

    return optimizer_grouped_parameters

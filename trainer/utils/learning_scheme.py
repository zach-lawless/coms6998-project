import torch


def get_learning_scheme(learning_scheme, model, learning_rate):
    if learning_scheme == 'differential':
        optimizer_grouped_parameters = differential_learning_scheme(model, learning_rate)
        optimizer = torch.optim.SGD(optimizer_grouped_parameters)
    elif learning_scheme == 'fixed':
        optimizer = torch.optim.SGD(learning_rate)
    elif learning_scheme == 'nesterov':
        optimizer = torch.optim.SGD(learning_rate, momentum=0.9, nesterov=True)
    else:
        raise NotImplementedError

    return optimizer


def differential_learning_scheme(model, learning_rate=0.1, divisor=2.6):
    param_prefixes = {}
    for n, p in model.named_parameters():
        if p.requires_grad:
            base = n.partition('.weight')[0].partition('.bias')[0]
            if base not in param_prefixes:
                param_prefixes[base] = 0

    param_prefix_divisors = list(reversed([divisor * i for i in range(1, len(param_prefixes))])) + [1]
    param_learning_rates = [learning_rate / ld for ld in param_prefix_divisors]

    param_prefix_lr_lookup = dict(zip(param_prefixes.keys(), param_learning_rates))

    optimizer_grouped_parameters = [
        {'params': p, 'lr': param_prefix_lr_lookup[n.partition('.weight')[0].partition('.bias')[0]]}
        for n, p in model.named_parameters() if p.requires_grad
    ]

    return optimizer_grouped_parameters


def get_scheduler(scheduler, optimizer, learning_rate, max_lr):
    if scheduler:
        if scheduler == 'cyclic-triangular':
            scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer,
                                                          base_lr=learning_rate,
                                                          max_lr=max_lr,
                                                          mode='triangular')
        else:
            raise NotImplementedError

    return scheduler

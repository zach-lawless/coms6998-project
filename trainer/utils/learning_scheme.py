import torch


def get_learning_scheme(learning_scheme, model, learning_rate, adapter, epoch):
    if learning_scheme == 'differential':
        optimizer_grouped_parameters = differential_learning_scheme(model, learning_rate)
        optimizer = torch.optim.SGD(optimizer_grouped_parameters)
    elif learning_scheme == 'fixed':
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    elif learning_scheme == 'nesterov':
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, nesterov=True)
    elif learning_scheme == 'gradual-unfreeze':
        optimizer_grouped_parameters = gradual_unfreezing_learning_scheme(model, learning_rate, adapter, epoch)
        optimizer = torch.optim.SGD(optimizer_grouped_parameters)
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


def gradual_unfreezing_learning_scheme(model, learning_rate, adapter, epoch=1):
    trainable_layers = []
    for n, p in model.named_parameters():
        if p.requires_grad:
            base = n.partition('.weight')[0].partition('.bias')[0]
            if adapter:
                if base not in trainable_layers and 'adapter' or 'classifier' in base:
                    trainable_layers.append(base)
            else:
                if base not in trainable_layers:
                    trainable_layers.append(base)

    optimizer_grouped_parameters = [
        {'params': p, 'lr': learning_rate}
        for n, p in model.named_parameters() if p.requires_grad and n.partition('.weight')[0].partition('.bias')[0] in trainable_layers[-epoch:]
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

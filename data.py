from math import ceil
import torch

DIVISION_MODULO_OPERATIONS = {
    "xy/y": lambda x, y, p: ((x * y) % p, y, x),
    "x/y": lambda x, y, p: (x, y, (x / y).to(torch.int64) % p),
}

ALL_MODULO_OPERATIONS = {
    "x+y": lambda x, y, p: (x, y, (x + y) % p),
    "x-y": lambda x, y, p: (x, y, (x - y) % p),
    "x*y": lambda x, y, p: (x, y, (x * y) % p),
    **DIVISION_MODULO_OPERATIONS,
}

ALL_OPERATIONS = {
    **ALL_MODULO_OPERATIONS,
}

def operation_mod_p_data(operation: str, p: int, eq_token: int, op_token: int):
    """
    x◦y (mod p) for 0 <= x < p, 1 <= y < p if operation in DIVISION_MODULO_OPERATIONS
    x◦y (mod p) for 0 <= x, y < p otherwise
    """
    x = torch.arange(0, p)
    y = torch.arange(0 if not operation in DIVISION_MODULO_OPERATIONS else 1, p)
    
    # For division modulo (p = 97),
    # x = [0, 0, ..., 0 (96 times), 1, 1,..., 1, ..., 96, ..., 96]
    # y = [1, 2, ..., 96, 1, 2, ..., 96, ..., 1, ..., 96]    
    x, y = torch.cartesian_prod(x, y).T

    eq = torch.ones_like(x) * eq_token
    op = torch.ones_like(x) * op_token

    x, y, labels = ALL_OPERATIONS[operation](x, y, p)

    inputs = torch.stack([x, op, y, eq], dim=1)
    if operation in DIVISION_MODULO_OPERATIONS:
        assert list(inputs.size()) == [p * (p - 1), 4]
        assert list(labels.size()) == [p * (p - 1)]
    else: 
        assert list(inputs.size()) == [p * p, 4]
        assert list(labels.size()) == [p * p]

    # It is okay to set the token value same as original number
    # Even we set to a permutation, it will be used as an index to the embedding
    # The whole embedding is initialized 
    # So, it does not matter

    return inputs, labels

def get_data(operation: str, prime: int, training_fraction: float, batch_size: int):
    inputs, labels = operation_mod_p_data(operation, prime, prime, prime+1)
    dataset = torch.utils.data.TensorDataset(inputs, labels)

    train_size = int(training_fraction * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    batch_size = min(batch_size, ceil(len(dataset) / 2))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, val_loader

def get_vocab_size(prime):
    return prime + 2

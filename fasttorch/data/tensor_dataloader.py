import torch as T


class TensorDataLoader:
    """
    Warning:
        `TensorDataLoader` doesn't support distributed training now.
    """
    def __init__(self, *tensors, batch_size, shuffle=False, pin_memory=False):
        assert all(t.shape[0] == tensors[0].shape[0] for t in tensors)
        self.dataset_len = tensors[0].shape[0]
        self.pin_memory = pin_memory and T.cuda.is_available()
        self.tensors = [T.as_tensor(t) for t in tensors]
        if self.pin_memory:
            self.tensors = [t.pin_memory() for t in self.tensors]
        self.batch_size = batch_size
        self.shuffle = shuffle
        # Calculate # batches
        n_batches, remainder = divmod(self.dataset_len, self.batch_size)
        if remainder > 0:
            n_batches += 1
        self.n_batches = n_batches

    def __iter__(self):
        if self.shuffle:
            r = T.randperm(self.dataset_len)
            self.tensors = [t[r] for t in self.tensors]
            del r
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.dataset_len:
            raise StopIteration
        batch = tuple(t[self.i: self.i+self.batch_size] for t in self.tensors)
        self.i += self.batch_size
        return batch

    def __len__(self):
        return self.n_batches

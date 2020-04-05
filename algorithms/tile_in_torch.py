import torch
import numpy as np
device='cpu'

def tile(a, dim, n_tile):
    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])).to(device)
    return torch.index_select(a, dim, order_index)

tensor=torch.tensor([[1,2,3], [4,5,6]])

print(tile(tensor, 0, 4))

print(tile(tensor, 1, 4))
import torch.nn.functional as F
from torch import Tensor
# normalize(input, p=2, dim=1, eps=1e-12)
def max_singular_value(W, u=None, Ip=1):
    """
    Apply power iteration for the weight parameter
    """
    if not Ip >= 1:
        raise ValueError("The number of power iterations should be positive integer")
    # todo
    if u is None:
        u = Tensor(1, W.shape[0]).normal_().type_as(W)
    _u = u.clone()
    for _ in range(Ip):
        # todo
        _v = F.normalize(Tensor.mm(_u, W.data), p=2)
        _u = F.normalize(Tensor.mm(_v, W.data.t()), p=2)

    sigma = Tensor.mm(Tensor.mm(_u, W.data), _v.t())
    return sigma, _u, _v

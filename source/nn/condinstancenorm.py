import torch
from torch.nn import functional as F
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter

class CondInstanceNorm(Module):

    def __init__(self, num_features, num_labels, eps=1e-5, momentum=0.1):
        super(CondInstanceNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.weight = Parameter(torch.Tensor(num_labels, num_features))
        self.bias = Parameter(torch.Tensor(num_labels, num_features))

        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.reset_parameters()

    def reset_parameters(self):
        self.running_mean.zero_()
        self.running_var.fill_(1)
        self.weight.data.uniform_()
        self.bias.data.zero_()

    def _check_input_dim(self, input):
        return NotImplemented

    def forward(self, input, label):
        self._check_input_dim(input)
        b, c = input.size(0), input.size(1)

        # Repeat stored stats and affine transform params
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)

        # Apply instance norm
        input_reshaped = input.contiguous().view(1, b * c, *input.size()[2:])

        weight_per_sample = F.embedding(label.long(), self.weight).reshape(b * c)
        bias_per_sample = F.embedding(label.long(), self.bias).reshape(b * c)

        out = F.batch_norm(
            input_reshaped, running_mean, running_var, weight_per_sample, bias_per_sample,
            True, self.momentum, self.eps)

        # Reshape back
        self.running_mean.copy_(running_mean.view(b, c).mean(0, keepdim=False))
        self.running_var.copy_(running_var.view(b, c).mean(0, keepdim=False))

        return out.view(b, c, *input.size()[2:])

    def eval(self):
        return self

    def __repr__(self):
        return ('{name}({num_features}, eps={eps}, momentum={momentum},'
                .format(name=self.__class__.__name__, **self.__dict__))

class CondInstanceNorm1d(CondInstanceNorm):
    def _check_input_dim(self, input):
        if input.dim() != 3:
            raise ValueError('expected 3D input (got {}D input)'
                             .format(input.dim()))

class CondInstanceNorm2d(CondInstanceNorm):
    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))

class CondInstanceNorm3d(CondInstanceNorm):
    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError('expected 5D input (got {}D input)'
                             .format(input.dim()))
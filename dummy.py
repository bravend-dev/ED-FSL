import torch
from torch.distributions import distribution

def print_shape(tensor, name='-'):
    if isinstance(tensor, torch.Tensor):
        print(f'| {name} {tensor.names}', tuple(tensor.shape))



if __name__ == '__main__':

    torch.random.manual_seed(1)


    a = torch.rand((5,11), names=['B','D'])
    print_shape(a, 'a')
    mu, sigma, gamma = a.chunk(chunks=3, dim=1)

    print_shape(mu, 'mu')
    print_shape(sigma, 'sigma')
    print_shape(gamma, 'gamma')
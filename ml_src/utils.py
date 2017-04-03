import torch


def is_gpu_available():
    return torch.cuda.is_available()
import numpy as np
import torch
import torch.nn as nn

print("hello_world")
array = np.array([1, 2, 3, 4, 5])
print(array)
print(np.sum(array))

dat = np.loadtxt("iris.dat")
print(dat)

lin = nn.Linear(5, 3)
data = torch.randn(2, 5)
print(lin(data))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

x = torch.rand(5, 3)
if torch.cuda.is_available():
    device = torch.device("cuda")          # a CUDA device object
    y = torch.ones_like(x, device=device)  # directly create a tensor on GPU
    x = x.to(device)                       # or just use strings ``.to("cuda")``
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))       # ``.to`` can also change dtype together!

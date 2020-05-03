import torch
from model import am_loss_function as amlf

if __name__ == "__main__":
    """
    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([2.0, 3.0, 4.0])
    adam = torch.optim.Adam([a, b], lr=0.1)
    a.requires_grad = True
    b.requires_grad = True
    c = (a - b) ** 2
    c = c.sum()
    c.backward()
    adam.step()
    print(a)
    print(b)
    print(c)
    """
    loss = amlf.AmLossFunction()
    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    b = torch.tensor([[0.0, 1.0], [1.0, 1.0]])
    a.requires_grad = True
    b.requires_grad = True
    res = loss.forward(a, b)
    res.backward()
    print(res)

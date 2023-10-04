#!/usr/bin/python
import numpy as np
import torch.nn as nn
import torch


# IIC loss function
def IIC(z, zt, C=3,EPS=0.001):
    z1 = z.unsqueeze(2)
    zt1 = zt.unsqueeze(1)
    P1 = (z1*zt1)
    P2 = P1.sum(dim=0)
    P2 = ((P2 + P2.t()) / 2) / P2.sum()
    P2[(P2 < EPS).data] = EPS
    Pi = P2.sum(dim=1).view(C, 1).expand(C, C)
    Pj = P2.sum(dim=0).view(1, C).expand(C, C)
    return (P2*(torch.log(Pi) + torch.log(Pj) - torch.log(P2))).sum()




z = np.array([[1,0,0],[0,1,0],[1,0,0], [1,0,0], [1,0,0]])
zt = np.array([[1,0,0],[0,1,0],[1,0,0], [1,0,0], [1,0,0]])

z = torch.from_numpy(z)
zt = torch.from_numpy(zt)

P = IIC(z, zt)

print(P)
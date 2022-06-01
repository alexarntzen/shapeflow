"""These curves are not the same mod Diff+, now we transormd c_2 , r=SRVT(q) by x**3"""
import torch
from math import pi, sqrt
from neural_reparam.plotting import plot_curve


# parametrization
def ksi(t):
    return (t - torch.sin(2 * pi * t) / (2 * pi)) ** (1 / 3)


def d_ksi_dt(t):
    return 1 / 3 * t ** (3 / 2) * (1 - torch.cos(2 * pi * t))


def q(t):
    q_x = torch.cos(pi * t)
    q_y = torch.sin(pi * t)
    return torch.cat((q_x, q_y), dim=-1)


def r(t):
    # r = Q(c_2)

    r_x = torch.zeros_like(t)
    r_y = sqrt(3) * t
    return torch.cat((r_x, r_y), dim=-1)


DIST_R_Q = 2 - sqrt(2)

# run this to show the curves
if __name__ == "__main__":
    # Data frame with dat
    plot_curve(q)
    plot_curve(r)

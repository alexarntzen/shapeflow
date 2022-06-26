import matplotlib.pyplot as plt

import shapeflow.reparam as reparam


def plot_reparams(N=64, max_step=8):
    states = reparam.generate_reaparm(N, max_step=max_step)
    plt.plot(states[:, 0], states[:, 1])
    plt.show()


if __name__ == "__main__":
    plot_reparams()

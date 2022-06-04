import numpy as np
import scipy.interpolate as spint


def generate_reparam(N: int, max_step: int = 5):
    states = [np.zeros(2)]
    while np.all(states[-1] < N):
        states.append(states[-1] + np.random.randint(0, max_step, size=2))
    states[-1] = np.ones(2) * N
    return np.stack(states) / N


def reparam_curve_data(
    curve_data: np.ndarray,
    reparam_data: np.ndarray = None,
    N: int = 1,
    times: np.ndarray = None,
):
    if times is None:
        times = np.linspace(0, 1, N)
    else:
        N = len(times)
        # times = times.flatten()

    if reparam_data is None:
        # random reparam
        reparam_data = generate_reparam(N)

    times = times.flatten()

    new_times = spint.interp1d(
        x=reparam_data[:, 0], y=reparam_data[:, 1], assume_sorted=True
    )(times)
    new_curve = spint.interp1d(x=times, y=curve_data, assume_sorted=True, axis=0)(
        new_times
    )
    return new_curve


def reparam_curve(
    curve: callable,
    reparam_data: np.ndarray = None,
    N: int = 1,
    times: np.ndarray = None,
    max_step: int = 5,
):
    if times is None:

        times = np.linspace(0, 1, N)
    else:
        N = len(times)
        # times = times.flatten()

    if reparam_data is None:
        # random reparam
        reparam_data = generate_reparam(N, max_step=max_step)

    times = times.flatten()

    new_times = spint.interp1d(
        x=reparam_data[:, 0], y=reparam_data[:, 1], assume_sorted=True
    )(times)
    return curve(new_times)

from typing import List, Type, Union

import deepthermal.validation
from tqdm import tqdm
import numpy as np
import torch
import torch.distributions as dist
import nflows.transforms
import nflows.flows
import flowtorch as ft
import flowtorch.bijectors
import flowtorch.distributions
import shapeflow as sf


# from shapeflow.utilsimport WrapInverseModel, WrapModel


# def get_composed_flow(
#     get_transform: callable,
#     base_dist: dist.Distribution,
#     num_layers: int,
#     inverse_model: bool = True,
#     **transform_kwargs,
# ) -> ft.distributions.Flow:
#     wrapp = sf.WrapInverseModel if inverse_model else sf.WrapModel
#     # create a bijector that wraps the original model. See test for example
#     bijector = wrapp(
#         params_fn=sf.LazyModule(get_transform=get_transform, **transform_kwargs)
#     )
#     model = ft.distributions.Flow(base_dist=base_dist, bijector=bijector)
#     return model
#


def animation_to_eulers(animations: List, reduce_shape=True, **kwargs) -> np.array:
    # assumes that all the skeletions in animations have the same keys

    # init array
    if reduce_shape:
        total_frames = sum(
            map(lambda animation: len(animation.get_frames()), animations)
        )
        total_angles = len(frame_to_euler(animations[0].get_frames()[0], **kwargs))
        angle_array = np.zeros((total_frames, total_angles))

        # put angles in array
        frame_iter = 0
        for animation in tqdm(animations):

            for frame in animation.get_frames():
                angle_array[frame_iter] = frame_to_euler(frame, **kwargs)
                frame_iter += 1

        return angle_array
    else:
        num_animations = len(animations)
        min_frames = min([len(animation.get_frames()) for animation in animations])
        total_angles = len(frame_to_euler(animations[0].get_frames()[0], **kwargs))

        angle_array = np.zeros((num_animations, min_frames, total_angles))
        for a, animation in enumerate(tqdm(animations)):
            for f, frame in enumerate(animation.get_frames()):
                if f == min_frames:
                    break
                angle_array[a, f] = frame_to_euler(frame, **kwargs)
        return angle_array


def frame_to_euler(frame, deg2rad=True, remove_root=False) -> np.array:
    # assumes the first bone is (x,y,z, phi, theta,..)
    # we do not include (x,y,z)
    root_and_angles = np.concatenate([frame[bone] for bone in frame.keys()])

    if deg2rad:
        root_and_angles = np.deg2rad(root_and_angles)

    if remove_root:
        root_and_angles = root_and_angles[3:]

    return root_and_angles


def data_to_motion_array(data: torch.Tensor, transposed: bool = True) -> np.ndarray:
    """Converts to degrees and pads zero position"""
    pads = (0, 0, 3, 0)
    _data = data
    if len(_data.shape) == 4:
        _data = _data[0][0]
    if len(_data.shape) == 3:
        _data = _data[0]
    assert len(_data.shape) == 2, "wrong shape for motion capture array "
    if transposed:
        _data = _data.T
    return (
        torch.nn.functional.pad(torch.rad2deg(_data), pads, "constant", 0)
        .detach()
        .numpy()
    )


def motion_array_to_data(
    motion_array: np.ndarray, transposed: bool = True
) -> torch.Tensor:
    """Converts to degrees and pads zero position"""
    if transposed:
        motion_array = motion_array.T
    return torch.tensor(np.deg2rad(motion_array[:, 3:])).float()

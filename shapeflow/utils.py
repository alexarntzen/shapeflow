from typing import List, Type
from tqdm import tqdm
import numpy as np
import torch
import torch.distributions as dist
import nflows.transforms
import nflows.flows
import flowtorch as ft
import flowtorch.distributions
from shapeflow import WrapInverseModel, WrapModel

# from shapeflow.utilsimport WrapInverseModel, WrapModel


def get_transform_nflow(
    Transform: Type, shape: torch.Size, **kwargs
) -> nflows.transforms.Transform:
    assert len(shape) == 1
    transform = Transform(features=shape[0], **kwargs)
    if isinstance(transform, nflows.flows.Flow):
        transform = transform._transform

    if isinstance(transform, nflows.transforms.Transform):
        return transform
    else:
        RuntimeError("Failed to create Transform")


def get_flow(
    get_transform: callable,
    base_dist: dist.Distribution,
    inverse_model: bool = True,
    **transform_kwargs
) -> ft.distributions.Flow:
    wrapp = WrapInverseModel if inverse_model else WrapModel

    # create a bijector that wraps the original model. See test for example
    bijector = wrapp(get_transform=get_transform, **transform_kwargs)
    model = ft.distributions.Flow(base_dist=base_dist, bijector=bijector)
    return model


def animation_to_eulers(animations: List, reduce_shape=True, **kwargs) -> np.array:
    # asssumes that all the skeletions in animations have the same keys

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
    # we do not inclue (x,y,z
    root_and_angles = np.concatenate([frame[bone] for bone in frame.keys()])

    if deg2rad:
        root_and_angles = np.deg2rad(root_and_angles)

    if remove_root:
        root_and_angles = root_and_angles[3:]

    return root_and_angles

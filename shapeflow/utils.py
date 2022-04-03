from typing import List
import numpy as np
from tqdm import tqdm


def animation_to_eulers(animations: List):
    # asssumes that all the skeletions in animations have the same keys

    # init array
    total_frames = sum(map(lambda animation: len(animation.get_frames()), animations))
    total_angles = len(frame_to_euler(animations[0].get_frames()[0]))

    angle_array = np.zeros((total_frames, total_angles))

    # put angles in array
    frame_iter = 0
    for animation in tqdm(animations):

        for frame in animation.get_frames():
            angle_array[frame_iter] = frame_to_euler(frame)
            frame_iter += 1

    return angle_array


def frame_to_euler(frame, deg2rad=True):
    # assumes the first bone is (x,y,z, phi, theta,..)
    # we do not inclue (x,y,z
    root_and_angles = np.concatenate([frame[bone] for bone in frame.keys()])
    if deg2rad:
        return np.deg2rad(root_and_angles[3:])
    else:
        return root_and_angles[3:]

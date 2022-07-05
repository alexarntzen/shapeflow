[![Python testing](https://github.com/alexarntzen/shapeflow/workflows/Python%20testing/badge.svg)](https://github.com/alexarntzen/shapeflow/actions/workflows/python_test.yml)
[![Python linting](https://github.com/alexarntzen/shapeflow/workflows/Python%20linting/badge.svg)](https://github.com/alexarntzen/shapeflow/actions/workflows/python_lint.yml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# Shapeflow

Normalizing flows for shape analysis.

Part of a master thesis Norwegian University of Science and Technology.

The experiments are placed in the `experimets` directory. Each of the notebooks corresponds to one of the six experiments in the thesis. The experiments are divided into two categories **clustering** and **interpolation**. Each of the categories is then performed on three datasets; test datasets such as `moons`, frames of human motions, and human motions.

To run the experiments in the notebooks, clone the repository, and install dependencies in `requirements.txt`. For instance with

    pip3 install -r requirements.txt 

from the project root. Then most of the notebooks can be run. 

To run notebooks with motion capture data the human motion database needs to be installed on your computer. Set up the motion capture database as explained in [alexarntzen/signatureshape](https://github.com/alexarntzen/signatureshape). This is a copy of [paalel/Signatures-in-Shape-Analysis](https://github.com/paalel/Signatures-in-Shape-Analysis) that works with `python3`.


The videos of motion and frame interpolation are located in the `videos` directory.
### Some plots from test datasets



#### Interpolation:

<img src="https://user-images.githubusercontent.com/48533802/177403355-4966270f-17c7-49ef-80fc-66b94f13cf8d.png" height="200"><img src="https://user-images.githubusercontent.com/48533802/177406299-200e43d2-52fb-43bb-9100-31891720a96e.png" height="200">

#### Clustering:

<img src="https://user-images.githubusercontent.com/48533802/177405389-0b98c4cd-db26-4c33-89cc-fef88842bc57.png" height="200"><<img src="https://user-images.githubusercontent.com/48533802/177405811-c636b0d3-8592-45f4-b8bd-aaf855eddefb.png" height="200">


### Videos of interpolations:

![Latent space interpolation of human frames (20 interpolation frames)](https://raw.githubusercontent.com/alexarntzen/shapeflow/main/videos/feature_interpolation_frame.mp4)

[Latent space interpolation of human motion (20 interpolating motions of 1 second)]( https://raw.githubusercontent.com/alexarntzen/shapeflow/main/videos/feature_interpolation_motion.mp4)






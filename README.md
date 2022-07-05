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

[Latent space interpolation of human frames](https://raw.githubusercontent.com/alexarntzen/shapeflow/main/videos/feature_interpolation_frame.mp4)

[Latent space interpolation of human motion](https://raw.githubusercontent.com/alexarntzen/shapeflow/main/videos/feature_interpolation_motion.mp4)





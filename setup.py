from setuptools import setup

setup(
    name="shapeflow",
    version="0.1",
    packages=["shapeflow"],
    url="https://github.com/alexarntzen/shapeflow",
    license="MIT",
    author="Alexander Johan Arntzen",
    author_email="hello@alexarntzen.com",
    description="For master thesis on normalizing flows in shape analysis",
    install_requires=[
        "torch>=1.8.1",
        "numpy>=1.18.2",
        "matplotlib>=3.2.0",
        "flowtorch>=0.8",
        "torchdyn~=1.0.1",
        "tqdm~=4.64.0",
        "extratorch @ git+https://github.com/alexarntzen/extratorch.git",
        "residual-flows @ git+https://github.com/VincentStimper/residual-flows.git",
        "normflow @ git+https://github.com/VincentStimper/normalizing-flows.git",
    ],
)

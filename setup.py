import setuptools
from pathlib import Path

with open("README.md", "r") as fh:
    long_description = fh.read()

def get_install_requires():
    """Returns requirements.txt parsed to a list"""
    fname = Path(__file__).parent / 'requirements.txt'
    targets = []
    if fname.exists():
        with open(fname, 'r') as f:
            targets = f.read().splitlines()
    return targets

setuptools.setup(
    name="distrans",
    version="0.0.1",
    author="Qiauil",
    author_email="qiangliu.7@outlook.com",
    description="A PyTorch library for Distribution Transformation in Generative Modeling",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/qiauil/distrans",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=get_install_requires(),
)

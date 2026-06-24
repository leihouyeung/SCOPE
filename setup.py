"""Minimal setup.py so the repo can be installed with `pip install -e .`."""
from pathlib import Path
from setuptools import find_packages, setup

setup(
    name="scope",
    version="1.0.0",
    description="Single-Cell multimOdal sPatial intEgration",
    long_description=(Path(__file__).parent / "README.md").read_text(),
    long_description_content_type="text/markdown",
    license="MIT",
    python_requires=">=3.10",
    packages=find_packages(include=["scope", "scope.*", "evaluation", "evaluation.*"]),
    install_requires=[
        "torch>=2.1",
        "torch-geometric>=2.5",
        "numpy",
        "scipy",
        "scikit-learn>=1.3",
        "pandas",
        "h5py",
        "tqdm",
        "anndata>=0.10",
        "scanpy>=1.10",
        "leidenalg",
        "lifelines>=0.27",
    ],
)

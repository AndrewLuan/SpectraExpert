from setuptools import setup, find_packages

setup(
    name="spectraexpert",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24",
        "pandas>=2.0",
        "scipy>=1.10",
        "pyarrow>=12.0",
        "matplotlib>=3.7",
        "rdkit>=2023.9",
        "scikit-learn>=1.3",
        "torch>=2.0",
        "transformers>=4.35",
    ],
)

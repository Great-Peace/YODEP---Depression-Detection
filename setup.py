"""Package setup for YODEP research repository."""

from setuptools import setup, find_packages

setup(
    name="yodep",
    version="1.0.0",
    description=(
        "YODEP: Yoruba-English Acted Depression Speech Corpus "
        "and F0 Transferability Study"
    ),
    author="Carnegie Mellon University Africa",
    python_requires=">=3.10",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.26",
        "scipy>=1.12",
        "pandas>=2.2",
        "scikit-learn>=1.4",
        "librosa>=0.10",
        "soundfile>=0.12",
        "praat-parselmouth>=0.4",
        "opensmile>=2.5",
        "torch>=2.2",
        "torchaudio>=2.2",
        "transformers>=4.39",
        "pyyaml>=6.0",
        "tqdm>=4.66",
        "matplotlib>=3.8",
        "seaborn>=0.13",
        "click>=8.1",
    ],
    extras_require={
        "dev": ["pytest>=8.1", "pytest-cov>=5.0"],
        "notebook": ["jupyter>=1.0", "ipykernel>=6.29"],
    },
)

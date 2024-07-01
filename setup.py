import codecs
from os import path

from setuptools import find_packages, setup

from wnb import __version__

setup(
    name="wnb",
    version=__version__,
    description="Python library for the implementations of general and weighted naive Bayes (WNB) classifiers.",
    keywords=["python", "bayes", "naive bayes", "classifier", "probabilistic"],
    author="Mehdi Samsami",
    author_email="mehdisamsami@live.com",
    license="BSD License",
    url="https://github.com/msamsami/weighted-naive-bayes",
    long_description=codecs.open(
        path.join(path.abspath(path.dirname(__file__)), "README.md"), encoding="utf-8"
    ).read(),
    long_description_content_type="text/markdown",
    include_package_data=True,
    packages=find_packages(exclude=["tests*"]),
    classifiers=[
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: BSD License",
    ],
    python_requires=">=3.8,<3.13",
    install_requires=[
        "pandas>=1.4.1",
        "numpy<2.0.0",
        "scipy>=1.8.0",
        "scikit-learn>=1.0.2",
        "typing-extensions>=4.8.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.3.1",
            "black>=24.4.2",
            "tqdm>=4.65.0",
            "pre-commit>=3.7.1",
            "isort==5.13.2",
        ]
    },
)

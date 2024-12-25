import codecs
from os import path

from setuptools import find_packages, setup

from wnb import __version__

setup(
    name="wnb",
    version=__version__,
    description="Python library for the implementations of general and weighted naive Bayes (WNB) classifiers.",
    keywords=["python", "machine learning", "bayes", "naive bayes", "classifier"],
    author="Mehdi Samsami",
    author_email="mehdisamsami@live.com",
    license="BSD License",
    url="https://github.com/msamsami/wnb",
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
        "Programming Language :: Python :: 3.13",
        "License :: OSI Approved :: BSD License",
    ],
    python_requires=">=3.8,<3.14",
    install_requires=[
        "pandas>=1.4.1",
        "scipy>=1.8.0",
        "scikit-learn>=1.0.2",
        "typing-extensions>=4.8.0; python_full_version < '3.11'",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=24.8.0",
            "tqdm",
            "pre-commit>=3.5.0",
            "isort",
        ]
    },
)

import codecs
from os import path
from setuptools import setup, find_packages


setup(
    name="wnb",
    version="0.1.18",
    description="Python library for the implementations of general and weighted naive Bayes (WNB) classifiers.",
    keywords=["python", "bayes", "naivebayes", "classifier", "probabilistic"],
    author="Mehdi Samsami",
    author_email="mehdisamsami@live.com",
    license="BSD License",
    url="https://github.com/msamsami/weighted-naive-bayes",
    long_description=codecs.open(
        path.join(path.abspath(path.dirname(__file__)), "README.md"), encoding="utf-8"
    ).read(),
    long_description_content_type="text/markdown",
    packages=find_packages(exclude=["tests*"]),
    include_package_data=True,
    classifiers=[
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: BSD License",
    ],
    python_requires=">=3.7",
    install_requires=["pandas==1.4.1", "scikit-learn>=1.0.2"],
    extras_require={"dev": ["pytest==7.3.1", "black>=23.9.0"]},
)

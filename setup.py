import codecs
from os import path
from setuptools import setup


setup(
    name='wnb',
    version='0.1.7',
    description='Python library for the implementations of general and weighted naive Bayes (WNB) classifiers.',
    keywords=['python', 'bayes', 'naivebayes', 'classifier', 'probabilistic'],
    author='Mehdi Samsami',
    author_email='mehdisamsami@live.com',
    url='https://github.com/msamsami/weighted-naive-bayes',
    long_description=codecs.open(path.join(path.abspath(path.dirname(__file__)), 'README.md'), encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    packages=['wnb'],
    classifiers=[
        'Topic :: Machine Learning',
        'Topic :: Pattern Recognition',
        'Topic :: Probabilistic Classification',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'License :: OSI Approved :: MIT License',
    ],
    python_requires=">=3.7",
    install_requires=['pandas==1.4.1', 'scikit-learn>=1.0.2'],
    extras_require={},
)

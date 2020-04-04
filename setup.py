from setuptools import setup, find_packages
from codecs import open
import os

here = os.path.abspath(os.path.dirname(__file__))

# Get the long description from the README file
with open(os.path.join(here, 'README.md'), 'r') as f:
    long_description = f.read()

setup(
    name='seq2attn',
    version='1.0',
    description='Implementation of the Seq2Attn architecture for sequence-to-sequence task in PyTorch',
    long_description=long_description,
    url='https://github.com/i-machine-think/seq2attn',
    license='Apache License 2.0',

    classifiers=[
        'Intended Audience :: Research',
        'Topic :: Software Development',
        'Programming Language :: Python :: 3.6'
    ],
    keywords='seq2seq py-torch development',
    packages=find_packages(),
    install_requires=[
        'matplotlib',
    ],
    python_requires='>3.6',
)

from setuptools import setup, find_packages

setup(
    name='resnet34lib',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'torch',
        'torchvision'
    ],
    author='Your Name',
    author_email='your.email@example.com',
    description='A simple library that provides ResNet34 model.',
    url='https://your-repo-url.com',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License'
    ],
)

from setuptools import setup, find_packages

# read the contents of your README file
from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Needed for dependencies
INSTALL_REQUIRES = [
      'wheel',
      'gym',
]

setup(
    name = 'robocrop',
    packages = find_packages(),
    version = '0.1.0',
    description = 'A gym-like environment to simulate robot planting and harvesting crop and to test different RL algorithms.',
    long_description_content_type = 'text/markdown',
    long_description = long_description,
    author='MattiaCinelli',
    classifiers=[
        'License :: OSI Approved :: MIT License',
        ],
    install_requires = INSTALL_REQUIRES,
    python_requires = '>=3.7',
    test_suite='tests',
    zip_safe = False
)
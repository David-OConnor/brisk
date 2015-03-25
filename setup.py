from setuptools import setup, find_packages

with open('README.rst') as f:
    readme = f.read()

setup(
    name = "quick",
    version = "0.1",
    packages = find_packages(),

    install_requires = ['numba>=0.17.0',
                        'numpy>=1.9.1',],

    author = "David O'Connor",
    author_email = "david.alan.oconnor@gmail.com",
    description = "Fast implementation of numerical functions using Numba",
    long_description = readme,
    license = "LGPL",
    keywords = "fast, numba, numerical, optimized",

)

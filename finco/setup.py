# -*- coding: utf-8 -*-
"""
Script for building Python extension performing propagation

Should run using:
> python setup.py build_ext --inplace
"""

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

setup(
    name='doprop',
    ext_modules=cythonize(Extension(
            "doprop",
            sources=["_doprop.pyx"],
            include_dirs=[np.get_include()]
            ), annotate=True, language_level='3'),
    zip_safe=False,
)
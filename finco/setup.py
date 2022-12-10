# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 18:27:58 2022

@author: Owner
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
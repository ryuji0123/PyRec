from setuptools import setup, find_packages, Extension
from codecs import open
from os import path

"""
Release instruction:
"""

__version__ = "0.0.2"

try:
    import numpy as np
except ImportError:
    exit("Please install numpy")


try:
    from Cython.Build import cythonize
    from Cython.Distutils import build_ext
except ImportError:
    USE_CYTHON = False
else:
    USE_CYTHON = True

BASE_PATH = path.abspath(path.dirname(__file__))

with open(path.join(BASE_PATH, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

with open(path.join(BASE_PATH, "requirements_dev.txt"), encoding="utf-8") as f:
    all_reqs = f.read().split("\n")

ext = ".pyx" if USE_CYTHON else ".c"

extensions = [
    Extension(
        "pyrec.prediction_algorithms.matrix_factorization",
        ["pyrec/prediction_algorithms/matrix_factorization" + ext],
        include_dirs=[np.get_include()]
    ),
]

cmd_class = {}

if USE_CYTHON:
    ext_modules = cythonize(extensions)
    cmd_class.update({"build_ext": build_ext})
else:
    ext_modules = extensions

setup(
    name="pyrec",
    author="Ryo Okuda",
    author_email="contact2ryo.okuda@gmail.com",

    long_description=long_description,
    long_description_content_type="text/markdown",

    version=__version__,

    keywords="recommender system",

    ext_modules=ext_modules,
    cmd_class=cmd_class,

    entry_points={"console_scripts": ["pyrec = pyrec.__main__:main"]},
)
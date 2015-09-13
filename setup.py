import numpy as np
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

cmdclass = {}
ext_modules = [
    Extension(
        "utils.cython_nms",
        ["utils/nms.pyx"],
        extra_compile_args=["-Wno-cpp", "-Wno-unused-function"],
        include_dirs=[np.get_include()]
    )
]
cmdclass.update({'build_ext': build_ext})

setup(
    name='vdetlib',
    cmdclass=cmdclass,
    ext_modules=ext_modules
)

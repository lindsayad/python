from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

setup(
    extensions = [Extension('cython_file', ["cython_file.pyx"])],
    ext_modules=cythonize(Extension("cython_file", ["cython_file.pyx"]),
                          gdb_debug=True)
)

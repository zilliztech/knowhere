from setuptools import setup
from setuptools.extension import Extension
from setuptools.command.build_py import build_py
import os
import shutil

NAME = "pyknowhere"

class get_numpy_include(object):
    def __str__(self):
        import numpy as np
        return np.get_include()


class CustomBuildPy(build_py):
    """Run build_ext before build_py to compile swig code."""

    def run(self):
        # Verify that libknowhere exists but don't copy it
        so_src = os.path.join("..", "build", "Release", "libknowhere.so")
        dylib_src = os.path.join("..", "build", "Release", "libknowhere.dylib")
        if os.path.exists(so_src):
            shutil.copyfile(so_src, os.path.join("knowhere", "libknowhere.so"))
        elif os.path.exists(dylib_src):
            shutil.copyfile(dylib_src, os.path.join("knowhere", "libknowhere.dylib"))
        else:
            raise FileNotFoundError("libknowhere.so or libknowhere.dylib not found")
        self.run_command("build_ext")
        return build_py.run(self)


def get_thirdparty_prefix(lib_name):
    prefix = ""
    with open(os.path.join("..", "build", "Release", "generators", lib_name + ".pc")) as f:
        for line in f.readlines():
            if line.startswith("prefix="):
                prefix = line.strip().split("=")[1]
                break
    return prefix

def get_readme():
    with open(os.path.join("..", "README.md"), "r") as f:
        return f.read()


DEFINE_MACROS = [
    ("FINTEGER", "int"),
    ("SWIGWORDSIZE64", "1"),
    ("GLOG_USE_GLOG_EXPORT", None),
]

INCLUDE_DIRS = [
    get_numpy_include(),
    os.path.join("..", "include"),
    os.path.join("..", "thirdparty"),
    os.path.join("..", "build", "Release", "milvus-common-src", "include"),
    get_thirdparty_prefix("boost-headers") + "/include",
    get_thirdparty_prefix("nlohmann_json") + "/include",
    get_thirdparty_prefix("libglog") + "/include",
    get_thirdparty_prefix("gflags") + "/include"
]

BUILD_DIR = os.path.abspath(os.path.join("..", "build", "Release"))
MILVUS_COMMON_LIB_DIR = os.path.join(BUILD_DIR, "milvus-common-build")

LIBRARY_DIRS = [
    BUILD_DIR,
    MILVUS_COMMON_LIB_DIR
]
EXTRA_COMPILE_ARGS = ["-fPIC", "-std=gnu++17"]
EXTRA_LINK_ARGS = [
    "-lknowhere",
    "-lmilvus-common",
    f"-Wl,-rpath,{BUILD_DIR}",
    f"-Wl,-rpath,{MILVUS_COMMON_LIB_DIR}",
]

SWIG_OPTS = [
    "-c++",
    "-I" + os.path.join("..", "include")
]

EXTRA_OBJECTS = []

_swigknowhere = Extension(
    "knowhere._swigknowhere",
    sources=[
        os.path.join("knowhere", "knowhere.i"),
    ],
    language="c++",
    define_macros=DEFINE_MACROS,
    include_dirs=INCLUDE_DIRS,
    library_dirs=LIBRARY_DIRS,
    extra_compile_args=EXTRA_COMPILE_ARGS,
    extra_link_args=EXTRA_LINK_ARGS,
    swig_opts=SWIG_OPTS,
    extra_objects=EXTRA_OBJECTS
)

setup(
    name=NAME,
    description=(
        "A library for efficient similarity search and clustering of vectors."
    ),
    url="https://github.com/zilliztech/knowhere",
    author="Milvus Team",
    author_email="milvus-team@zilliz.com",
    license='Apache License 2.0',
    keywords="search nearest neighbors",
    setup_requires=["numpy", "setuptools_scm"],
    #use_scm_version={'root': '..', 'local_scheme': 'no-local-version', 'version_scheme': 'release-branch-semver'},
    long_description=get_readme(),
    long_description_content_type="text/markdown",
    packages=["knowhere"],
    include_package_data=True,
    package_data={"knowhere": ["libknowhere.so"]},
    python_requires=">=3.8",
    ext_modules=[_swigknowhere],
    cmdclass={"build_py": CustomBuildPy},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: POSIX",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)

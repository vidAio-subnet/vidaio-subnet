import re
import os
import codecs
from os import path
from io import open
from setuptools import setup, find_packages

def read_requirements(path):
    with open(path, "r") as f:
        requirements = f.read().splitlines()
        processed_requirements = []

        for req in requirements:
            if req.startswith("git+") or "@" in req:
                pkg_name = re.search(r"(#egg=)([\w\-_]+)", req)
                if pkg_name:
                    processed_requirements.append(pkg_name.group(2))
                else:
                    continue
            else:
                processed_requirements.append(req)
        return processed_requirements

requirements = read_requirements("requirements.txt")
here = path.abspath(path.dirname(__file__))

with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

# loading version from setup.py
with codecs.open(
    os.path.join(here, "vidaio_subnet_core/__init__.py"), encoding="utf-8"
) as init_file:
    version_match = re.search(
        r"^__version__ = ['\"]([^'\"]*)['\"]", init_file.read(), re.M
    )
    version_string = version_match.group(1)

setup(
    name="vidaio_subnet",
    version=version_string,
    description="video_processing_subnet",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/vidaio-subnet/vidaio-subnet",
    author="https://vidaio.io",  
    packages=find_packages(),
    include_package_data=True,
    author_email="",
    license="MIT",
    python_requires=">=3.10",
    install_requires=requirements,
    classifiers=[
        "Development Status :: Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Build Tools",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Vidao Processing",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)
#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import datetime
import os
import subprocess
from pathlib import Path
import setuptools

this_dir = os.path.dirname(os.path.abspath(__file__))


def get_local_version_suffix() -> str:
    date_suffix = datetime.datetime.now().strftime("%Y%m%d")
    git_hash = subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"], cwd=Path(__file__).parent
    ).decode("ascii")[:-1]
    return f"+{git_hash}.d{date_suffix}"


def write_version_file():
    version_path = os.path.join(this_dir, "version.py")
    with open(version_path, "w") as f:
        f.write("# noqa: C801\n")
        f.write(f'__version__ = "{version}"\n')
        tag = os.getenv("GIT_TAG")
        if tag is not None:
            f.write(f'git_tag = "{tag}"\n')

def get_install_requires():
    with open(Path(this_dir) / 'requirements.txt', 'r') as f:
        return f.readlines()

if __name__ == "__main__":
    version_txt = os.path.join(this_dir, "version.txt")
    with open(version_txt) as f:
        version = f.readline().strip()
    version += get_local_version_suffix()
    write_version_file()

    setuptools.setup(
        name="olla",
        description="Optimizing the Lifetime and Location of Arrays",
        version=version,
        license='MIT',
        packages=setuptools.find_packages(exclude=("tests", "tests.*")),
        install_requires=get_install_requires(),
        python_requires=">=3.6",
        author="Facebook AI Research",
        classifiers=[
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
            "Operating System :: OS Independent",
        ],
        project_urls={"Source": "https://github.com/facebookresearch/OLLA"},
    )

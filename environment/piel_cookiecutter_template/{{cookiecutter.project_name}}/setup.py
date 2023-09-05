#!/usr/bin/env python
from distutils.core import setup

setup(
    name="{{cookiecutter.project_name}}",
    version="0.0.1",
    description="{{cookiecutter.project_name}} project with a piel template.",
    author="{{cookiecutter.author_name}}",
    author_email="{{cookiecutter.author_email}}",
    url="https://github.com/daquintero/piel",
    packages=["{{cookiecutter.project_name}}"],
)

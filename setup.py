#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open("README.md") as readme_file:
    readme = readme_file.read()

requirements = [
    # "amaranth",  # Until they sort out their packaging issue, this dependency has to be installed separately. .
    "amaranth-yosys==0.25.0.0.post75",
    "cython==3.0.2",
    "jupytext==1.15.0",
    "Click>=7.0",
    "cocotb==1.8.0",
    "femwell==0.1.8",
    "hdl21==4.0.0",
    "jax==0.4.14",
    "jaxlib==0.4.14",
    "gdsfactory==7.3.0",  # Pinned for pydantic <v2 compatibility.
    "networkx==3.1",
    "numpy==1.24.4",
    "openlane==2.0.0b8",
    "pandas==1.5.3",
    "qutip==4.7.3",
    "sax==0.8.8",  # Pinned for pydantic <v2 compatibility.
    "thewalrus==0.20.0",
]

test_requirements = [
    "pytest>=3",
]

setup(
    author="Dario Quintero",
    author_email="darioaquintero@gmail.com",
    python_requires=">=3.10",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    description="Photonic Integrated Electronics: microservices to codesign photonics, electronics, communications, quantum, and more.",
    entry_points={
        "console_scripts": [
            "piel=piel.cli:main",
        ],
    },
    dependency_links=[
        "git+https://github.com/fact-project/smart_fact_crawler.git@master#egg=smart_fact_crawler-0"
    ],
    extras_require={
        "develop": [
            "sphinx",
            "sphinx_autodoc_typehints",
            "sphinx-pydantic",
            "sphinx-autoapi",
            "sphinx-autobuild",
            "sphinx_rtd_theme",
            "sphinx-gallery",
            "nbsphinx",
            "myst_parser",
            "pandoc",
            "flake8",
        ]
    },
    install_requires=requirements,
    license="MIT license",
    long_description_content_type="text/markdown",
    long_description=readme,
    include_package_data=True,
    keywords="piel",
    name="piel",
    packages=find_packages(include=["piel", "piel.*"]),
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/daquintero/piel",
    version="0.0.51",
    zip_safe=False,
)

from setuptools import setup, find_packages

d = {}
exec(open("probeinterface/version.py").read(), None, d)
version = d['version']
long_description = open("README.md").read()

pkg_name = "probeinterface"

setup(
    name=pkg_name,
    version=version,
    author="Samuel Garcia",
    author_email="sam.garcia.die@gmail.com",
    description="Python package to handle probe layout, geometry and wiring to device.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/SpikeInterface/probeinterface",
    packages=find_packages(),
    package_data={},
    install_requires=[
        'numpy',
    ],
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    )
)

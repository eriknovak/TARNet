from setuptools import setup, find_packages

with open("README.md", mode="r") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as fh:
    requirements = fh.readlines()

setup(
    name="TARNet",
    version="0.0.0",
    author="Ranak Roy Chowdhury",
    author_email="",
    description="Setup the experiment environment",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=[req for req in requirements if req[:2] != "# "],
    setup_requires=["flake8"],
)
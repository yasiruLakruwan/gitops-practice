from setuptools import setup,find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name="GitOps project practicing",
    version="0.0.1",
    author="Yapa",
    install_requires=requirements, 
    packages=find_packages()
)
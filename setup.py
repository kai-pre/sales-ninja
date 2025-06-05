from setuptools import setup
from setuptools import find_packages

# list dependencies from file
with open('requirements.txt') as f:
    content = f.readlines()
requirements = [x.strip() for x in content]

setup(name='salesninja',
      description="Python package for the Sales Ninja backend",
      version='0.0.2',
      packages=find_packages(),
      install_requires=requirements)

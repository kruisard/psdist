from setuptools import find_packages
from setuptools import setup


with open('README.md') as f:
    readme = f.read()

with open('LICENSE.txt') as f:
    license = f.read()

setup(
    name='psdist',
    version='0.0.0',
    description='Analysis/visualization of phase space distributions',
    long_description=readme,
    author='Austin Hoover',
    author_email='ahoover1218@gmail.com',
    url='https://github.com/austin-hoover/psdist',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)
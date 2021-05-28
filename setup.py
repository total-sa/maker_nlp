from setuptools import setup, find_packages
from os import path

with open('README.md', 'r') as fh:
    long_description = fh.read()

required_setup = ['pytest-runner',
                  'wheel',
                  'flake8']
required_tests = ['pytest',
                  'pytest-cov',
                  'treon']

here = path.abspath(path.dirname(__file__))
with open(path.join(here, 'requirements.txt'), encoding='utf-8') as f:
    all_reqs = f.read().split('\n')

required_install = [x.strip() for x in all_reqs]

__version__ = 'init'
exec(open('maker_nlp/version.py').read())

setup(
    name='maker_nlp',
    packages=find_packages(),
    version=__version__,
    description="Introduction to Natural language processing",
    author='TOTAL/TDF/DATASTUDIO',
    long_description=long_description,
    long_description_content_type="text/markdown",
    setup_requires=required_setup,
    tests_require=required_tests,
    install_requires=required_install,
    license='TOTAL - TDF',
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.5',
)

from setuptools import setup, find_packages

setup(
    name='gradflow',
    version='0.1.0',
    description='Reverse mode automatic differentiation engine',
    author='Raj Sekhar Dey',
    author_email='heysekhar.box@gmail.com',
    url='https://github.com/banished-panda/gradflow',
    packages=find_packages(),
    install_requires=[
        'numpy',
    ],
)
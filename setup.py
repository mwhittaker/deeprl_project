from setuptools import setup, find_packages

setup(
    name='deeprl',
    version='0.0',
    description='Deep RL Final Project',
    packages=find_packages(),
    author='Vladimir Feinberg, Samvit Jain, Michael Whittaker',
    install_requires=[
        'scipy',
        'numpy>=1.13',
        'mujoco_py==0.5.7',
        'gym[all]',
        'tensorflow-gpu', # or tensorflow if cpu only
        'universe'
    ],
)

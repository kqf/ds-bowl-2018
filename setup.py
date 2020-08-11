from setuptools import setup, find_packages

setup(
    name="nuclei-segmentation",
    version="0.0.1",
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'merge-masks=model.data:main',
            'nuclei=model.main:main',
        ],
    },
)

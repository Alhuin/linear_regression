from setuptools import setup

setup(
    name='linear_regression',
    version='1.0',
    description='42 AI Project',
    author='jjanin-r',
    author_email='jjanin-r@student.42lyon.fr',
    packages=['linear_regression'],
    extras_require={
        'dev': [
            'pytest',
            'flake8',
            'coverage'
        ],
    },
    install_requires=[
        'wheel',
        'matplotlib',
        'numpy',
    ],
)

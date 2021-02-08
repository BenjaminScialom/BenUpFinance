from setuptools import find_packages, setup
setup(
    name='BenUpFin',
    packages=find_packages(include=['BenUpFin']),
    version='0.1.0',
    description='Pyhton library related to finance: risk , portfolio management etc..',
    author='Benjamin Scialom',
    license='MIT',
    install_requires=['pandas'],
    setup_requires=['pytest-runner'],
    tests_require=['pytest==4.4.1'],
    test_suite='tests',
)
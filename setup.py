import pip
import logging
import pkg_resources

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


def _parse_requirements(filename):
    """ load requirements from a pip requirements file """
    lineiter = (line.strip() for line in open(filename))
    return [line for line in lineiter if line and not line.startswith("#")]


# parse_requirements() returns generator of pip.req.InstallRequirement objects
try:
    install_reqs = _parse_requirements("requirements.txt")
except Exception:
    logging.warning("Fail load requirements file, so using default ones.")
    install_reqs = []

setup(
    name="custom_train_loop",
    packages=["custom_train_loop"],
    version="0.1",
    license="MIT",
    description="TESTING FIRST PYPI PACKAGE",
    author="Kyi Thar Hein",
    author_email="kyitharhein18@gmail.com",
    url="https://github.com/1chimaruGin/custom_train_loop",
    download_url="https://github.com/1chimaruGin/custom_train_loop/archive/v_01.tar.gz",
    keywords=["CUSTOM", "TRAINING", "LOOP"],
    install_requires=install_reqs,
    classifiers=[
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Developers',
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
  ],
)

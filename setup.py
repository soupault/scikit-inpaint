#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup
from pip.req import parse_requirements


with open('README.md') as readme_file:
    readme = readme_file.read()

install_reqs = parse_requirements('requirements.txt',
                                  session='hack')
requirements = [str(ir.req) for ir in install_reqs]

test_requirements = [
    'numpy',
    'pytest'
]

setup(
    name='skinpaint',
    version='0.1.0',
    description="Collection of image inpainting algorithms implemented in Python",
    long_description=readme,
    author="Egor Panfilov",
    author_email='egor.v.panfilov@gmail.com',
    url='https://github.com/soupault/skinpaint',
    packages=[
        'skinpaint',
    ],
    package_dir={'skinpaint':
                 'skinpaint'},
    include_package_data=True,
    install_requires=requirements,
    zip_safe=False,
    license="MIT license",
    keywords='skinpaint',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Topic :: Software Development',
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        "Programming Language :: Python :: 2",
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
    test_suite='tests',
    tests_require=test_requirements
)

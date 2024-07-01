from codecs import open
from os import path
from typing import Any, Dict

from setuptools import find_packages, setup

here = path.abspath(path.dirname(__file__))

about: Dict[Any, Any] = {}
with open('./futureexpert/__about__.py') as fp:
    exec(fp.read(), about)

setup(
    name='futureexpert',

    # Versions should comply with PEP440.  For a discussion on single-sourcing
    # the version across setup.py and the project code, see
    # https://packaging.python.org/en/latest/single_source_version.html
    version=about['__version__'],

    description='TODO',
    long_description='TODO',

    # The project's main homepage.
    url='https://git.prognostica.de/prognostica/future/futureapp/expert-api',

    # Author details
    author='prognostica GmbH',
    author_email='info@prognostica.de',

    license='proprietary and confidential',

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Data scientists and client applications',
        'Topic :: Data-based forecasting',
        'License :: Other/Proprietary License',
        'Programming Language :: Python :: 3.9',
    ],

    keywords='time-series, forecast, ml',
    packages=find_packages(
        exclude=['contrib', 'docs', 'tests', 'tests.*', 'exploration']),

    # List run-time dependencies here.  These will be installed by pip when
    # your project is installed. For an analysis of "install_requires" vs pip's
    # requirements files see:
    # https://packaging.python.org/en/latest/requirements.html
    install_requires=['python-keycloak == 3.12.0',  # MIT License
                      'tenacity >= 8.2.0',          # Apache Software License
                      'requests >= 2.31.0',         # Apache Software License
                      'matplotlib >= 3.8.0',        # Python Software Foundation License
                      'pydantic >= 2.5.0,< 3.0.0',  # MIT License
                      'numpy >= 1.22.4,< 2.0.0',    # BSD-3-Clause License
                      'pandas >= 1.4.3',            # BSD-3-Clause License
                      'python-dotenv'],             # BSD-3-Clause License

    # List additional groups of dependencies here (e.g. development
    # dependencies). You can install these using the following syntax,
    # for example:
    # $ pip install -e .[dev,test]
    extras_require={
        "dev": ['pytest',
                'ipytest',
                'isort',
                'mypy == 1.0.*',
                'autopep8',
                'cloudpickle',
                'pdoc3',
                'types-pkg_resources',
                'types-python-dateutil',
                'types-requests',
                'types-toml',
                'jupyter == 1.0.0',
                'nbconvert == 7.2.6',
                'nbqa == 1.5.3',
                'python-dotenv']
    },

    # If there are data files included in your packages that need to be
    # installed, specify them here.  If using Python 2.6 or less, then these
    # have to be included in MANIFEST.in as well.
    package_data={'futureexpert': ['py.typed']},

    # Although 'package_data' is the preferred approach, in some case you may
    # need to place data files outside of your packages. See:
    # http://docs.python.org/3.4/distutils/setupscript.html#installing-additional-files # noqa
    # In this case, 'data_file' will be installed into '<sys.prefix>/my_data'
    data_files=[],
)

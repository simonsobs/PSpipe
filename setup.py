#- 
# setup.py
#-

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

config = {
    'description': 'PS PY',
    'author':  "The SO collaboration",
    'url': 'https://github.com/simonsobs/PSpipe',
    'download_url': 'https://github.com/simonsobs/PSpipe',
    'version': '0.0.1',
    'install_requires': [
        'numpy',
        'pixell'
        ],
    'packages': [
        'pspy',
        'pspy.wigner3j',
        'pspy.mcm_fortran'
        ],
    'scripts': [],
    'name': 'pspy',
    'include_package_data':True,
    'data_files': [('data/Planck_Parchment_RGB.txt')]

}

setup(**config)


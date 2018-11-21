#- 
# setup.py
#-

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

config = {
    'description': 'PS PY',
    'author':  "Steve Choi, Thibaut Louis, Dongwon 'DW' HAN",
    'url': 'https://github.com/simonsobs/ps_py',
    'download_url': 'https://github.com/simonsobs/ps_py',
    'version': '0.0.1',
    'install_requires': [
        'numpy',
        'pixell'
        ],
    'packages': [
        'pspy',
        ],
    'scripts': [],
    'name': 'pspy',
    'include_package_data':True,
    'data_files': [('data/Planck_Parchment_RGB.txt')]

}

setup(**config)


from setuptools import find_packages, setup

import versioneer

with open("README.rst") as readme_file:
    readme = readme_file.read()

setup(name="pspipe",
      version=versioneer.get_version(),
      cmdclass=versioneer.get_cmdclass(),
      author="Simons Observatory Collaboration Power Spectrum Task Force",
      url="https://github.com/simonsobs/PSpipe",
      description="Pipeline analysis for the SO LAT experiment",
      long_description=readme,
      license="BSD license",
      python_requires=">=3.5",
      zip_safe=True,
      packages=find_packages(),
      install_requires=[
          "scipy",
          "camb",
          "pspy @ git+https://github.com/simonsobs/pspy@master#egg=pspy",
          "pymaster @ git+https://github.com/LSSTDESC/NaMaster@easier_libsharp#egg=pymaster",
          "mflike @ git+https://github.com/simonsobs/LAT_MFLike@master#egg=mflike",
      ])

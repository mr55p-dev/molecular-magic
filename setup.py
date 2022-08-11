import setuptools

setuptools.setup(
  name="molecular_magic",
  version="1.1.0",
  packages=["magic"],
  install_requires=[],
  entry_points={
      'console_scripts': ['molmagic=magic.cli:main'],
  }
)

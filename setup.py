import setuptools

setuptools.setup(
  name="molecular_magic",
  version="0.0.1",
  packages=["magic"],
  install_requires=[],
  entry_points={
      'console_scripts': ['molmagic=magic.cli:main'],
  }
)

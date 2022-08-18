import setuptools

setuptools.setup(
  name="molecular_magic",
  version="2.1.0",
  packages=["molmagic"],
  install_requires=[],
  entry_points={
      'console_scripts': ['magic=molmagic.cli:main'],
  }
)

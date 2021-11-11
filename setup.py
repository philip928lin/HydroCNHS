from setuptools import setup
setup(name='HydroCNHS',
      version='1.0.0',
      description='HydroCNHS is a pure python CNHS simulation model.',
      url='',
      author='Chung-Yi Lin',
      author_email='philip928lin@gmail.com',
      license='NotOpenYet',
      packages=['HydroCNHS'],
      install_requires = ["ruamel.yaml", "tqdm", "numpy", "pandas", "joblib", "scipy", "matplotlib", "sklearn", "adjustText", "deap"],
      zip_safe=False,
      include_package_data = True,        # Enable MANIFEST.in  https://python-packaging.readthedocs.io/en/latest/non-code-files.html
      python_requires='>=3.7')




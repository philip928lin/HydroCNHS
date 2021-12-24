from setuptools import setup
setup(name='HydroCNHS',
      version='1.0.0',
      description='A Python Package of Hydrological Model for Coupled Natural Human Systems.',
      url='',
      author='Chung-Yi Lin',
      author_email='philip928lin@gmail.com',
      license='GPL-3.0 License',
      packages=['HydroCNHS'],
      install_requires = ["ruamel.yaml", "tqdm", "numpy", "pandas", "joblib",
                          "scipy", "matplotlib", "sklearn", "deap"],
      zip_safe=False,
      # Enable MANIFEST.in 
      # https://python-packaging.readthedocs.io/en/latest/non-code-files.html
      include_package_data = True,
      python_requires='>=3.9')




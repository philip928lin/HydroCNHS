import pathlib
from setuptools import setup

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
#README = (HERE / "README.md").read_text()
with open("README.md", "r", encoding='utf8') as fh:
  README = fh.read()
  
    
setup(name='hydrocnhs',
      version='0.0.3',
      description='A Python Package of Hydrological Model for Coupled Natural–Human Systems.',
      long_description=README,
      long_description_content_type="text/markdown",
      url='https://github.com/philip928lin/HydroCNHS',
      author='Chung-Yi Lin',
      author_email='philip928lin@gmail.com',
      license='GPL-3.0 License',
      packages=['HydroCNHS'],
      install_requires = ["ruamel.yaml", "tqdm", "numpy", "pandas", "joblib",
                          "scipy", "matplotlib", "scikit-learn","pyyaml", "deap"],
      classifiers=[
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Framework :: IPython",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.8",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: Unix",
        "Topic :: Education",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Hydrology",
    ],
      zip_safe=False,
      # Enable MANIFEST.in 
      # https://python-packaging.readthedocs.io/en/latest/non-code-files.html
      include_package_data = True,
      python_requires='>=3.8')




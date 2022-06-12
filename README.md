[![Documentation Status](https://readthedocs.org/projects/hydrocnhs/badge/?version=latest)](https://hydrocnhs.readthedocs.io)


# HydroCNHS
A Python Package of Hydrological Model for Coupled Natural–Human Systems

<img src="https://github.com/philip928lin/HydroCNHS/blob/main/docs/figs/fig3_hydrocnhs.png?raw=true" alt="Complex Adaptive Water System" width="500"/>

Modeling Coupled Natural–Human Systems (CNHS) to inform comprehensive water resources management policies or describe hydrological cycles in the Anthropocene has become popular in recent years. To fulfill this need, we developed a semi-distributed Hydrological model for Coupled Natural–Human Systems, HydroCNHS. The HydroCNHS is an open-source Python package supporting four Application Programming Interfaces (APIs) that enable users to integrate their human decision models, which can be programmed with the agent-based modeling concept, into the HydroCNHS. Specifically, we design Dam API, RiverDiv API, Conveying API, and InSitu API to integrate, respectively, customized man-made infrastructures such as reservoirs, off-stream diversions, trans-basin aqueducts, and drainage systems that abstract human behaviors (e.g., operator and farmers’ water use decisions). Each of the HydroCNHS APIs has a unique plug-in structure that respects within-subbasin and inter-subbasin (i.e., river) routing logic for maintaining the water balance. In addition, the HydroCNHS uses a single model configuration file to organize input features for the hydrological model and case-specific human systems models. Also, HydroCNHS enables the model calibration using parallel computing power. We demonstrate the functionalities of the HydroCNHS package through a case study in the Northwest United States. Given the integrity of the modeling framework, HydroCNHS can benefit water resources planning and management in various aspects, including the uncertainty analysis in CNHS modeling and more complex agent design. 


## Install
Install HydroCNHS by *pip*.
```
pip install hydrocnhs
```
To install the latest version (beta version) of  HydroCNHS, users can (1) install HydroCNHS by *git*.
```
pip install git+https://github.com/philip928lin/HydroCNHS.git
```
Or, (2) download the HydroCNHS package directly from the HydroCNHS GitHub repository. Then, install HydroCNHS from the *setup.py*.
```
# Need to move to the folder containing setup.py first.
python setup.py install
```
If you fail to install HydroCNHS due to the DEAP package, first downgrade setuptools to 57 and try to install HydroCNHS again.
```
pip install setuptools==57
```


## User Manual & Example
Click [![Documentation Status](https://readthedocs.org/projects/hydrocnhs/badge/?version=latest)](https://hydrocnhs.readthedocs.io)

## How to cite?
Lin, C. Y., Yang, Y. C. E., & Wi S. (2022, under review). HydroCNHS: A Python Package of Hydrological Model for Coupled Natural Human Systems, Journal of Water Resources Planning and Management.

## Related studies
Lin, C. Y., & Yang, Y. E. The effects of model complexity on model output uncertainty in co‐evolved coupled natural–human systems. Earth's Future, e2021EF002403.

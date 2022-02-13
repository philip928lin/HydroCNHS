[![Documentation Status](https://readthedocs.org/projects/hydrocnhs/badge/?version=latest)](https://hydrocnhs.readthedocs.io/en/latest/?badge=latest)


# HydroCNHS
A Python Package of Hydrological Model for Coupled Natural–Human Systems

<img src="https://github.com/philip928lin/HydroCNHS/blob/main/docs/figs/CAWS.png?raw=true" alt="Complex Adaptive Water System" width="500"/>

Modeling Coupled Natural–Human Systems (CNHS) to inform comprehensive water resources management policies or describe hydrological cycles in the Anthropocene has become popular in recent years. To fulfill this need, we developed a semi-distributed Hydrological model for Coupled Natural–Human Systems, HydroCNHS. The HydroCNHS is an open-source Python package supporting four Application Programming Interfaces (APIs) that enable users to integrate their human decision models, which can be programmed with the agent-based modeling concept, into the HydroCNHS. Specifically, we design Dam API, RiverDiv API, Conveying API, and InSitu API to integrate, respectively, customized man-made infrastructures (i.e., defined as agent behaviors) such as reservoirs, off-stream diversions, trans-basin aqueducts, and drainage systems. Each of the HydroCNHS APIs has a unique plug-in structure that respects within-subbasin and inter-subbasin (i.e., river) routing logic for maintaining the water balance. In addition, the HydroCNHS requires a single model configuration file to organize input features for the hydrological model and case-specific human systems models. Also, HydroCNHS enable the model calibration using parallel computing power. We demonstrate the functionalities of the HydroCNHS package through a case study in the Northwest United States. Given the integrity of the modeling framework, HydroCNHS can potentially benefit the uncertainty analysis in CNHS modeling and more complex agent design.


## Install
```
pip install hydrocnhs
```

## User Manual & Example
Click [here](https://hydrocnhs.readthedocs.io/en/latest/?badge=latest)!

## How to cite?
Lin, C. Y., Yang, Y. C. E., & Wi S. (2022). HydroCNHS: A Python Package of Hydrological Model for Coupled Natural Human Systems, Journal of Water Resources Planning and Management. (Pending)


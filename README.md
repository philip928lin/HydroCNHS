[![Documentation Status](https://readthedocs.org/projects/hydrocnhs/badge/?version=latest)](https://hydrocnhs.readthedocs.io)


# HydroCNHS
A Python Package of Hydrological Model for Coupled Natural–Human Systems

<img src="https://github.com/philip928lin/HydroCNHS/blob/main/docs/source/figs/fig3_hydrocnhs.png" alt="Complex Adaptive Water System" width="500"/>

Modeling Coupled Natural–Human Systems (CNHS) to inform comprehensive water resources management policies or describe hydrological cycles in the Anthropocene has become popular in recent years. To fulfill this need, we developed a semi-distributed Hydrological model for Coupled Natural–Human Systems, HydroCNHS. The HydroCNHS is an open-source Python package supporting four Application Programming Interfaces (APIs) that enable users to integrate their human decision models, which can be programmed with the agent-based modeling concept, into the HydroCNHS. Specifically, we design Dam API, RiverDiv API, Conveying API, and InSitu API to integrate, respectively, customized man-made infrastructures such as reservoirs, off-stream diversions, trans-basin aqueducts, and drainage systems that abstract human behaviors (e.g., operator and farmers’ water use decisions). Each of the HydroCNHS APIs has a unique plug-in structure that respects within-subbasin and inter-subbasin (i.e., river) routing logic for maintaining the water balance. In addition, the HydroCNHS uses a single model configuration file to organize input features for the hydrological model and case-specific human systems models. Also, HydroCNHS enables the model calibration using parallel computing power. We demonstrate the functionalities of the HydroCNHS package through a case study in the Northwest United States. Given the integrity of the modeling framework, HydroCNHS can benefit water resources planning and management in various aspects, including the uncertainty analysis in CNHS modeling and more complex agent design.


## Install
Install HydroCNHS by *pip*.
```
pip install hydrocnhs
```
To install the latest version (recommend) of  HydroCNHS, users can (1) install HydroCNHS by *git*.
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

## When should you use HydroCNHS?
1.	Want to build a hydrological model with auto-parameter-tuning (calibration) features.
2.	Want to add human components into a hydrological model under a unified framework without worrying about routing logic.
3.	Want to calibrate the entire integrated model (hydrological + ABM modules) with customized agents' parameters.
4.	Want to design human behaviors with a high degree of freedom, including coupling with external software.
5.	Want to conduct extensive numerical experiments to test various human behavioral designs (i.e., integrating and testing many ABM modules).

## Feature highlights
- Built-in genetic algorithm calibration module that can utilize parallel computing power.
- A unified framework allowing calibration of the entire integrated model (hydrological + ABM models), where parameters in the ABM module (i.e., human model) are customizable.
- Automatic integration and simulation through four APIs.
- Built-in data collector module that can collect data from the hydrological and user-defined ABM modules.
- Built-in model builder module to assist users in creating the HydroCNHS model and generating ABM module (*.py*) templates.
- Built-in indicator calculation and visualization tools for simple result analysis.

## Supporting APIs for incorporating human/agent components
### Dam API
Link to human actions that completely redefine downstream flow, e.g., instream objects like reservoirs. The downstream flow is governed by a reservoir's operational rules and operators' decisions.

### RiverDiv API
Link to human actions that divert water from the river (with return water to the downstream subbasins), e.g., off-stream irrigation water diversions governed by farmers or regional district managers' diversion behaviors.

### Conveying API
Link to human actions that convey water beyond basin boundaries and gravitational limitations, e.g., trans-basin aqueducts that convey water from one basin to another and pump stations that convey water from downstream to upstream.

### InSitu API
Link to human actions that locally alter the runoff in a subbasin, e.g., drainage systems, urbanization, and water diversion from wells or local streams (within-subbasin).


Those human/agent actions can be modeled with various complexities according to users' ABM design. Actions can simply be an inputted daily time series or governed by complex endogenous behavior rules (i.e., decision-making rules). Institutional decisions (a group of agents making decisions together with institutional rules) are allowed in HydroCNHS. Users require to have some level of object-oriented programing concept to design an ABM module (*.py*).

## How to cite?
Lin, C. Y., Yang, Y. C. E., & Wi S. (2022). [HydroCNHS: A Python Package of Hydrological Model for Coupled Natural Human Systems](https://doi.org/10.1061/(ASCE)WR.1943-5452.0001630), Journal of Water Resources Planning and Management, 148(12), 6022005.

## Related studies
Lin, C. Y., & Yang, Y. E. (2022). [The effects of model complexity on model output uncertainty in co‐evolved coupled natural–human systems](https://doi.org/10.1029/2021EF002403). Earth's Future, e2021EF002403.

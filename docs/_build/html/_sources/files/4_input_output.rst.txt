Input/Output
============

Inputs
------

HydroCNHS has the following three main inputs:

(1) Daily climate data including precipitation, temperature, and (optional) potential evapotranspiration (python *dictionary object*), 

(2) A model configuration file (*.yaml*; settings for the HydroCNHS and ABM models), and 

(3) ABM modules (*.py*; customized human models).

Outputs
-------

The outputs of HydroCNHS are stored in a data collector object, an attribute of HydroCNHS (e.g., *model.dc*). The main output is the daily streamflow at routing outlets. However, this data collector object will also contain other user-specified agent outputs as long as users use this data collector object in their ABM modules. See :ref:`Integrate an ABM` and :ref:`Data collector` for more information.

Daily climate data
------------------

Temperature (temp; degC) and precipitation (prec; cm) are two required weather inputs for a simulation. Potential evapotranspiration (pet; cm) can be automatically calculated by HydroCNHS using the Hamon method. However, we encourage users to pre-calculate pet and provide it to HydroCNHS as an input to avoid repeated calculation, especially when running calibration. The daily climate inputs are in a python dictionary format, in which the subbasin outlet name is the key, and a daily time series (*list*) is the value, as shown below.

.. code-block:: python

  temp = {'subbasin1': [7.7, 7.0, 6.6, 6.3, .......],
          'subbasin2': [8.5, 4.0, 5.3, 6.2, .......]}

Model configuration file
------------------------

A model configuration file is a YAML file that contains settings for the entire water system. The main sections in a model configuration file include Path, WaterSystem, LSM, Routing, and ABM. Please see the following description for each setting. Parameter and inputs definitions for rainfall-runoff and routing models are summarized in :numref:`table2` and :numref:`table3`, respectively. HydroCNHS has an embedded model builder. Therefore, users do not need to manually create the model configuration file (see the example in the following section).

.. code-block:: yaml

  Path:
    WD:       <working directory>
    Modules:  <directory to the folder containing ABM modules>

  WaterSystem:
    StartDate:    <simulation start date, e.g., 1981/1/1>
    EndDate:      <simulation end date, e.g., 2013/12/31> 
    DataLength:   <total simulation length/days, e.g., 12053>
    NumSubbasins: <number of subbasins>
    Outlets:      <a list of subbasins'/outlets' names, e.g., ["outlet1", "outlet2"]>
    NodeGroups:   <a list of node group lists, e.g., [["outlet1", "outlet2"], []]>
    LSM:          <selected rainfall-runoff models, e.g., 'GWLF' or 'ABCD' or 'Other'>
    Routing:      <selected routing model, e.g., 'Lohmann'>
    ABM:
      Modules:    <a list of ABM modules, e.g., ['TRB_ABM_Instit.py']>
      InstitDMClasses:   
          <a dict of {InstitDMClass: a list of institutional decision-making objects}>
      DMClasses:      <a list of decision-making classes, e.g., ['ReleaseDM', 'TransferDM']>
      DamAPI:         <a list of agent type classes using DamAPI, e.g., ['Reservoir_AgtType']>
      RiverDivAPI:    <a list of agent type classes using RiverDivAPI, e.g., ['Diversion_AgtType']>
      InsituAPI:      <a list of agent type classes using InsituAPI, e.g., ['Drain_AgtType']>
      ConveyingAPI:   <a list of agent type classes using ConveyingAPI, e.g., ['Pipe_AgtType']>
      Institutions:
          <a dict of {an institutional decision-making agent: a list of agent members}>

  RainfallRunoff:
    <an outlet name, e.g., 'outlet1'>:
      Inputs: <input dict for selected rainfall-runoff model at outlet1>
      Pars:   <parameter dict for selected rainfall-runoff model at outlet1>
    <an outlet name, e.g., 'outlet2'>:
      Inputs: <input dict for selected rainfall-runoff model at outlet2>
      Pars:   <parameter dict for selected rainfall-runoff model at outlet2>

  Routing:
    <a routing outlet name, e.g., 'outlet1'>:
      <an upstream outlet name, e.g., 'outlet1'>:
        Inputs:   <input dict of Lohmann routing model for link between outlet1 and the routing outlet>
        Pars:     <parameter dict of Lohmann routing model for link between outlet1 and the routing outlet>
      <an upstream outlet name, e.g., 'outlet2'>:
        Inputs:   <input dict of Lohmann routing model for link between outlet2 and the routing outlet>
        Pars:     <parameter dict of Lohmann routing model for link between outlet2 and the routing outlet>

  ABM:
    <an agent type class name, e.g., 'Reservoir_AgtType'>:
      <an agent name belongs to this class, e.g., 'ResAgt'>:
        Attributes: "agent's attributes dict, e.g., {}"
        Inputs:
          Priority:   <exercution piority is conflict occurs, e.g., 0>
          Links:      <linkage dict, e.g., divert from 'outlet1' and return to 'outlet2,' {'outlet1': -1, 'outlet2': 1}>
          DMClass:    <assigned decision-making class or institution or none, e.g., 'ReleaseDM'>
        Pars:     <parameter dict of the agent, e.g., {}>


.. _table2:
.. table:: Hydrological model parameters and suggested bounds.
  :align: center
  :width: 100%

  +----------+------------------------------------------+--------------+-------------------+---------------------+
  |Module    |Parameter name                            |Unit          |Parameter          |Bound                |
  +==========+==========================================+==============+===================+=====================+
  |GWLF      |Curve number                              |--            |:math:`CN2`        |[25, 100]            |
  |          +------------------------------------------+--------------+-------------------+---------------------+
  |          |Interception coefficient                  |--            |:math:`IS`         |[0, 0.5]             |
  |          +------------------------------------------+--------------+-------------------+---------------------+
  |          |Recession coefficient                     |--            |:math:`Res`        |[10\ :sup:`-3`\, 0.5]|
  |          +------------------------------------------+--------------+-------------------+---------------------+
  |          |Deep seepage coefficient                  |--            |:math:`Sep`        |[0, 0.5]             |
  |          +------------------------------------------+--------------+-------------------+---------------------+
  |          |Baseflow coefficient                      |--            |:math:`\alpha`     |[0, 1]               |
  |          +------------------------------------------+--------------+-------------------+---------------------+
  |          |Percolation coefficient                   |--            |:math:`\beta`      |[0, 1]               |
  |          +------------------------------------------+--------------+-------------------+---------------------+
  |          |Available/soil water capacity             |cm            |:math:`U_r`        |[1, 15]              |
  |          +------------------------------------------+--------------+-------------------+---------------------+
  |          |Degree-day coefficient for snowmelt       |cm/°C         |:math:`D_f`        |[0, 1]               |
  |          +------------------------------------------+--------------+-------------------+---------------------+
  |          |Land cover coefficient                    |--            |:math:`K_c`        |[0.5, 1.5]           |
  +----------+------------------------------------------+--------------+-------------------+---------------------+
  |ABCD      |Controls the amount of runoff and recharge|--            |:math:`a`          |[0, 1]               |
  |          |during unsaturated soil                   |              |                   |                     |
  |          +------------------------------------------+--------------+-------------------+---------------------+
  |          |Control of the saturation level of the    |--            |:math:`b`          |[0, 400]             |
  |          |soils                                     |              |                   |                     |
  |          +------------------------------------------+--------------+-------------------+---------------------+
  |          |Ratio of groundwater recharge to runoff   |--            |:math:`c`          |[0, 1]               |
  |          +------------------------------------------+--------------+-------------------+---------------------+
  |          |Control of groundwater discharge rate     |--            |:math:`d`          |[0, 1]               |
  |          +------------------------------------------+--------------+-------------------+---------------------+
  |          |Degree-day coefficient for snowmelt       |cm/°C         |:math:`D_f`        |[0, 1]               |
  +----------+------------------------------------------+--------------+-------------------+---------------------+
  | | Lohmann|Subbasin unit hydrograph shape parameter  |--            |:math:`G_{shape}`  |[1, 100]             |
  | | routing+------------------------------------------+--------------+-------------------+---------------------+
  |          |Subbasin unit hydrograph rate parameter   |--            |:math:`G_{scale}`  |[10\ :sup:`-2`\, 150]|
  |          +------------------------------------------+--------------+-------------------+---------------------+
  |          |Wave velocity in the linearized Saint-    |m/s           |:math:`Velo`       |[0.5, 100]           |
  |          |Venant equation                           |              |                   |                     |
  |          +------------------------------------------+--------------+-------------------+---------------------+
  |          |Diffusivity in the linearized Saint-      |m\ :sup:`2`\/s|:math:`Diff`       |[200, 5000]          |
  |          |Venant equation                           |              |                   |                     |
  +----------+------------------------------------------+--------------+-------------------+---------------------+


.. _table3:
.. table:: Hydrological model inputs and default values.
  :align: center
  :width: 100%

  +----------+---------------------------------------------+--------------+------------------------+---------------------+
  |Module    |Parameter name                               |Unit          |Parameter               |Default              |
  +==========+=============================================+==============+========================+=====================+
  |GWLF      |Subbasin's drainage area                     |ha            |:math:`Area`            |--                   |
  |          +---------------------------------------------+--------------+------------------------+---------------------+
  |          |Latitude                                     |deg           |:math:`Latitude`        |--                   |
  |          +---------------------------------------------+--------------+------------------------+---------------------+
  |          |Initial shallow saturated soil water content |cm            |:math:`S0`              |2                    |
  |          +---------------------------------------------+--------------+------------------------+---------------------+
  |          |Initial unsaturated soil water content       |cm            |:math:`U0`              |10                   |
  |          +---------------------------------------------+--------------+------------------------+---------------------+
  |          |Initial snow storage                         |cm            |:math:`SnowS`           |5                    |
  +----------+---------------------------------------------+--------------+------------------------+---------------------+
  |ABCD      |Subbasin's drainage area                     |--            |:math:`Area`            |--                   |
  |          +---------------------------------------------+--------------+------------------------+---------------------+
  |          |Latitude                                     |deg           |:math:`Latitude`        |--                   |
  |          +---------------------------------------------+--------------+------------------------+---------------------+
  |          |Initial saturated soil water content         |cm            |:math:`XL`              |2                    |
  |          +---------------------------------------------+--------------+------------------------+---------------------+
  |          |Initial snow storage                         |cm            |:math:`SnowS`           |5                    |
  +----------+---------------------------------------------+--------------+------------------------+---------------------+
  | | Lohmann|Flow length between two outlets              |m             |:math:`FlowLength`      |--                   |
  | | routing+---------------------------------------------+--------------+------------------------+---------------------+
  |          |An instream control object, e.g., a reservoir|--            |:math:`InstreamControl` |False                |
  +----------+---------------------------------------------+--------------+------------------------+---------------------+

ABM modules
-----------

ABM modules are customized python scripts in which human components are designed through programming agent type classes and decision-making classes. HydroCNHS will load those user-specified classes and use them to initialize agents. More details are provided in the :ref:`Integrate an ABM` section.

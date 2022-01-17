Quick start!
============
Let's start with running a simulation using the Tualatin River Basin (TRB) example!

The TRB tutorial codes are located at **./HydroCNHS/tutorials/Tualatin River Basin/** 

.. _TRB:
.. figure:: ./figs/TRB.png
  :align: center
  :width: 500
  :alt: The Tualatin River Basin system diagram. 

  The Tualatin River Basin system diagram (Lin et al., 2022). TRTR, 
  Hagg\ :sub:`In`\, DLLO, TRGC, DAIRY, RCTV, and WSLO are seven subbasins. 
  PipeAgt, ResAgt, and DivAgt are trans-basin aqueduct, Hagg reservoir, and 
  irrigation diversion agents, respectively. DrainAgt1 and DrainAgt2 are two 
  drainage system agents for the runoff-changing scenario.
  
  
The TRB, consisting of 1844.07 km2 in northwest Oregon, US, is covered by 
densely populated area (20%), agricultural area (30%), and forest (50%) 
(Tualatin River Watershed Council, 2021). Its agriculture heavily relies on 
irrigation because seasonal rainfall concentrates in winter (November - 
February). The Spring Hill Pumping Plant is the largest diversion facility in 
the TRB for irrigating Tualatin Valley Irrigation District (TVID; DivAgt), 
where the Hagg reservoir (ResAgt) is the primary water source. During the 
summer period, water is transferred from the Barney reservoir (outside of the 
TRB) through a trans-basin aqueduct (PipeAgt) to augment the low flow for 
ecological purposes.

The full background of the TRB can be found in Lin et al., (2022).

Run a simulation
-----------------
Assuming we have already calibrated the TRB model, let's take one minute to 
browse through the following simulation code. Then, you should be able run the 
script directly.


.. code-block:: python

	import os
	import pickle
	import HydroCNHS

	##### Setup Working Directory
	# Get this .py file's directory.
	prj_path, this_filename = os.path.split(__file__)
	wd = prj_path

	##### Load Daily Weather Time Series.
	# We store all required time series data in the pickle format.
	with open(os.path.join(prj_path, "Inputs", "TRB_inputs.pickle"), "rb") as file:
		(temp, prec, pet, obv_D, obv_M, obv_Y) = pickle.load(file)
		
	##### Load Model.yaml.
	best_gwlf_abm_path = os.path.join(prj_path, "Calibrated_model", "Best_gwlf_abm_KGE.yaml")
	model_dict_gwlf = HydroCNHS.load_model(best_gwlf_abm_path)
	# Change the path according to this .py file's directory.
	model_dict_gwlf["Path"]["WD"] = wd
	model_dict_gwlf["Path"]["Modules"] = os.path.join(prj_path, "ABM_modules")

	##### Create HydroCNHS Model Object for Simulation.
	model_gwlf = HydroCNHS.Model(model_dict_gwlf)

	##### Run a simulation
	Q_gwlf = model_gwlf.run(temp, prec, pet) # pet is optional.


To run a coupled natural human model simulation, three items are required.

1. Daily weather time series including temperture (temp) and precipitation (prec). Potential evapotranspiration (pet) is optional.
2. Model file (e.g., Best_gwlf_abm_KGE.yaml).
3. User-provided ABM module(s) (e.g., TRB_ABM_dm.py). This is not required for pure hydrological model, non-coupled model.

Now, we are going to introduce these three items in a more detail.


Daily weather time series
-------------------------
Temperture and precipitation are two required weather inputs for the 
simulation. If potential evapotranspiration is not provided, HydroCNHS will 
automatically calculate it by Hamon method. The weather inputs are in a 
dictionary format shown below.

.. code-block:: python

	# 'DAIRY', 'DLLO', 'RCTV', 'HaggIn', 'TRGC', 'TRTR', and 'WSLO' are subbasins' names.
	temp = {'DAIRY': [7.7, 7.0, 6.6, 6.3, .......],
		'DLLO': [7.9, 7.5, 7.1, 6.1, .......],
		'RCTV': [8.0, 7.8, 7.8, 7.5, .......],
		'HaggIn': [8.1, 7.4, 7.0, 6.2, .......],
		'TRGC': [5.7, 5.5, 5.1, 4.0, .......],
		'TRTR': [7.9, 6.9, 6.5, 6.1, .......],
		'WSLO': [7.8, 7.4, 7.3, 7.3, .......]}
	# Similar to prep and pet.
		
The dictionary will contain weather time series (i.e., a list) for each 
subbasin. The length of each time series has to be identical.


Model.yaml
-------------------------
The model file (.yaml) contains settings for hydrological model (e.g., 
rainfall-runoff and routing) and ABM model (e.g., how to coupled).
The model file has six sections:

1. Path
^^^^^^^^^^^^^^^^^^^
.. code-block:: yaml
	
	# Path for working directory (outputing log file) and user-provided ABM
	# modules (optional).
	Path: {
	  WD: 'wd path',
	  Modules: 'ABM module path'}


2. WaterSystem
^^^^^^^^^^^^^^^^^^^
.. code-block:: yaml

	WaterSystem:
		StartDate: 1981/1/1
		EndDate: 2013/12/31
		NumSubbasins: 7
		NumGauges: 2
		NumAgents: 3
		Outlets: [TRTR, HaggIn, DLLO, TRGC, DAIRY, RCTV, WSLO]
		GaugedOutlets: [DLLO, WSLO]		# Optional
		DataLength: 12053

DataLength can be automatically calculated if EndDate is provided, vice versa.

3. LSM
^^^^^^^^^^^^^^^^^^^
HydroCNHS provides user two rainfall-runoff simulation options, General
Water Loading Function (GWLF; 9 parameters) and ABCD (5 parameters). Their
settings are shown below. 

The detailed documentation for GWLF and ABCD can be found at the supplementary 
material of (Lin et al., 2022).

**GWLF:**

.. code-block:: yaml

	LSM:
		Model: GWLF
		TRTR:
			Inputs: {Area: 329.80, Latitude: 45.45, S0: 2.0, U0: 10.0, SnowS: 5.0}
			Pars: {CN2: 97.148, IS: 0.404, Res: 0.178, Sep: 0.482, Alpha: 0.868, Beta: 0.708,
			Ur: 14.341, Df: 0.124, Kc: 0.988}
		HaggIn:
			Inputs: {Area: 10034.24, Latitude: 45.46, S0: 2.0, U0: 10.0, SnowS: 5.0}
			Pars: {CN2: 36.984, IS: 0.178, Res: 0.122, Sep: 0.178, Alpha: 0.235, Beta: 0.112,
			Ur: 13.932, Df: 0.588, Kc: 0.765}
		DLLO:
			Inputs: {Area: 22238.43, Latitude: 45.47, S0: 2.0, U0: 10.0, SnowS: 5.0}
			Pars: {CN2: 98.249, IS: 0.369, Res: 0.281, Sep: 0.117, Alpha: 0.909, Beta: 0.381,
			Ur: 9.0, Df: 0.947, Kc: 1.057}
		TRGC:
			Inputs: {Area: 24044.63, Latitude: 45.50, S0: 2.0, U0: 10.0, SnowS: 5.0}
			Pars: {CN2: 89.963, IS: 0.098, Res: 0.047, Sep: 0.418, Alpha: 0.535, Beta: 0.115,
			Ur: 9.107, Df: 0.727, Kc: 1.013}
		DAIRY:
			Inputs: {Area: 59822.75, Latitude: 45.52, S0: 2.0, U0: 10.0, SnowS: 5.0}
			Pars: {CN2: 95.962, IS: 0.026, Res: 0.241, Sep: 0.199, Alpha: 0.432, Beta: 0.299,
			Ur: 4.049, Df: 0.034, Kc: 1.012}
		RCTV:
			Inputs: {Area: 19682.60, Latitude: 45.50, S0: 2.0, U0: 10.0, SnowS: 5.0}
			Pars: {CN2: 93.049, IS: 0.341, Res: 0.319, Sep: 0.05, Alpha: 0.316, Beta: 0.7,
			Ur: 11.288, Df: 0.741, Kc: 0.746}
		WSLO:
			Inputs: {Area: 47646.84, Latitude: 45.35, S0: 2.0, U0: 10.0, SnowS: 5.0}
			Pars: {CN2: 63.818, IS: 0.039, Res: 0.305, Sep: 0.045, Alpha: 0.207, Beta: 0.496,
			Ur: 13.919, Df: 0.175, Kc: 1.451}

**ABCD:** 

.. code-block:: yaml

	LSM:
		Model: ABCD
		TRTR:
			Inputs: {Area: 329.80, Latitude: 45.45, XL: 2.0, SnowS: 5.0}
			Pars: {a: 0.998, b: 53.403, c: 0.345, d: 0.559, Df: 0.636}
		HaggIn:
			Inputs: {Area: 10034.24, Latitude: 45.46, XL: 2.0, SnowS: 5.0}
			Pars: {a: 0.196, b: 2.316, c: 0.78, d: 0.248, Df: 0.697}
		DLLO:
			Inputs: {Area: 22238.43, Latitude: 45.47, XL: 2.0, SnowS: 5.0}
			Pars: {a: 0.237, b: 16.746, c: 0.693, d: 0.834, Df: 0.16}
		TRGC:
			Inputs: {Area: 24044.63, Latitude: 45.50, XL: 2.0, SnowS: 5.0}
			Pars: {a: 0.306, b: 91.475, c: 0.695, d: 0.263, Df: 0.727}
		DAIRY:
			Inputs: {Area: 59822.75, Latitude: 45.52, XL: 2.0, SnowS: 5.0}
			Pars: {a: 0.836, b: 5.536, c: 0.795, d: 0.491, Df: 0.515}
		RCTV:
			Inputs: {Area: 19682.60, Latitude: 45.50, XL: 2.0, SnowS: 5.0}
			Pars: {a: 0.516, b: 6.632, c: 0.971, d: 0.148, Df: 0.681}
		WSLO:
			Inputs: {Area: 47646.84, Latitude: 45.35, XL: 2.0, SnowS: 5.0}
			Pars: {a: 0.777, b: 307.23, c: 0.453, d: 0.638, Df: 0.159}

4. Routing
^^^^^^^^^^^^^^^^^^^
HydroCNHS adopts Lohmann routing model to simulate within-subbasin routing and 
inter-subbasin routing process. We adpot a nested struture to setup the routing 
setting for each routing outlets (:numref:`TRB`), as shown below.

.. code-block:: yaml

	Routing:
		Model: Lohmann
		# WSLO, TRGC, DLLO, and HaggIn are four routing outlets.
		WSLO:
			# TRGC is a routing outlet. No within-subbasin routing at here (at
			# its own routing setting below). Namely, TRGC will be routed
			# first. Therefore, GShape and GScale are null.
			TRGC:
				Inputs: {FlowLength: 80064.86, InstreamControl: false}
				Pars: {GShape: null, GScale: null, Velo: 29.84, Diff: 2840.985}
			DAIRY:
				Inputs: {FlowLength: 70988.16, InstreamControl: false}
				Pars: {GShape: 38.652, GScale: 393.641, Velo: 15.281, Diff: 1753.171}
			RCTV:
				Inputs: {FlowLength: 60398.68, InstreamControl: false}
				Pars: {GShape: 55.996, GScale: 751.942, Velo: 2.973, Diff: 2573.258}
			# WSLO is the routing outlet itself. No river routing is needed
			# since the FlowLength is 0. Therefore, Velo and Diff are null.
			WSLO:
				Inputs: {FlowLength: 0, InstreamControl: false}
      			Pars: {GShape: 1.026, GScale: 119.961, Velo: null, Diff: null}
		TRGC:
			# DLLO is a routing outlet. No within-subbasin routing at here (at
			# its own routing setting below). Namely, DLLO will be routed
			# first. Therefore, GShape and GScale are null.
			DLLO:
				Inputs: {FlowLength: 11748.21, InstreamControl: false}
				Pars: {GShape: null, GScale: null, Velo: 15.178, Diff: 2382.476}
			# TRGC is the routing outlet itself. No river routing is needed
			# since the FlowLength is 0. Therefore, Velo and Diff are null.
			TRGC:
				Inputs: {FlowLength: 0, InstreamControl: false}
				Pars: {GShape: 3.823, GScale: 694.564, Velo: null, Diff: null}
		DLLO:
			# ResAgt is a reservoir agent. There is no within-subbasin routing. 
			# Its release flow is the streamflow at this spot. Therefore,
			# GShape and GScale are null.
			ResAgt:
				Inputs: {FlowLength: 9656.06, InstreamControl: true}
				Pars: {GShape: null, GScale: null, Velo: 40.546, Diff: 2250.142}
			TRTR:
				Inputs: {FlowLength: 30899.40, InstreamControl: false}
				Pars: {GShape: 50.454, GScale: 444.699, Velo: 27.168, Diff: 3625.861}
			# DLLO is the routing outlet itself. No river routing is needed
			# since the FlowLength is 0. Therefore, Velo and Diff are null.
			DLLO:
				Inputs: {FlowLength: 0, InstreamControl: false}
				Pars: {GShape: 2.505, GScale: 85.004, Velo: null, Diff: null}
		HaggIn:
			# HaggIn is the routing outlet itself. No river routing is needed
			# since the FlowLength is 0. Therefore, Velo and Diff are null
			HaggIn:
				Inputs: {FlowLength: 0, InstreamControl: false}
				Pars: {GShape: 49.412, GScale: 1.774, Velo: null, Diff: null}

5. ABM (optional)
^^^^^^^^^^^^^^^^^^^
ABM section is a highly customized setting section. Users will assign each 
active agent type class to corresponding APIs. After that, users will define 
agents created by each agent type class in a nested structure.

.. code-block:: yaml

	ABM:
		Inputs:
			# Assign user-defined agent classes to corresponding APIs. Here, we
			# defined three agent classes in TRB_ABM_dm.py: ResDam_AgType 
			# (reservoir), IrrDiv_AgType (diversion), and Pipe_AgType 
			# (trans-basin conveying water).
			DamAgentTypes: [ResDam_AgType]		# Dam API
			RiverDivAgentTypes: [IrrDiv_AgType]	# RiverDiv API
			InsituAgentTypes: [Drain_AgType] 	# InSitu API
			ConveyAgentTypes: [Pipe_AgType]		# Conveying API
			# Activate user-defined decision-making classes in TRB_ABM_dm.py.
			DMClasses: [ResDM, DivDM, PipeDM]
			# User-defined ABM module, TRB_ABM_dm.py.
			Modules: [TRB_ABM_dm.py]
			# Agent group is for agents make decisions and act together.
			AgGroup: null
		Pipe_AgType:	# Agent type class's name.
			PipeAgt:	# Agent created by the above agent type class.
			Attributes: {}
			Inputs:
				Piority: 0
				Links: {TRTR: 1}
				DMClass: PipeDM
			Pars: null
		ResDam_AgType:
			ResAgt:
			Attributes: {}
			Inputs:
				Piority: 0
				Links: {HaggIn: -1, ResAgt: 1}
				DMClass: ResDM
			Pars: null
		IrrDiv_AgType:
			DivAgt:
			Attributes: {}
			Inputs:
				Piority: 1
				Links:
				TRGC: -1
				WSLO: [ReturnFactor, 0, Plus]
				DMClass: DivDM
			Pars:
				ReturnFactor: [0.12]
				a: 0.52
				b: -0.04
		Drain_AgType:
			DrainAgt1:
			Attributes: {}
			Inputs:
				Piority: 1
				Links:
				RCTV: 1
				DMClass: null
			Pars: null
			DrainAgt2:
			Attributes: {}
			Inputs:
				Piority: 1
				Links:
				WSLO: 1
				DMClass: null
			Pars: null


For the "Inputs" setting of ABM section, first, we assign the user-defined
agent classes (defined in ABM modules) to corresponding coupling APIs. Then,
we ativate the decision-making classes (defined in ABM modules) if any. Next,
we assign ABM module(s), where agent classes and decision-making classes are
defined. Finally, agent group is for agents make decisions and act together.
For example, two diversion agents make diversion requests together and share 
the water deficiency together based on their water rights. Namely, their 
diversion behaviors are not piority-based. The agent group will be defined as 
a single function in a ABM module, which users can define a more detailed
interactions among agents in an agent group. See 
:ref:`How to build a ABM module?<How to build a ABM module?>` for more details.

Following the "Inputs" setting, we will define agent objects created by certain
agent classes (defined in ABM modules). For example, we create PipeAgt agent 
using Pipe_AgType class. Under each agent object (e.g., PipeAgt), it has three
sub-sections: "Attributes", "Inputs", and "Pars." 

**a) Inputs** are required information including "Piority", "Links", and "DMClass." 

* Piority: 
  
The lower value has higher piority when conflicts happen. For example, two
diverion agents divert at the same routing outlet. If users want a 
non-priority-based behaviors. "AgGroup" should be applied. See 
:ref:`How to build a ABM module?<How to build a ABM module?>` for more details.

Note that agents coupling with Dam API has to have Piority = 0.

* Links:

"Links" is a dictionary containing information of which outlets for agent to
take/add water from/to (e.g., {<outlet>: factor}). The **positive** factor 
means add the water to that outlet. The **negative** number factor take water
from that outlet. The "factor" is a number in a range of [-1,1] defining the 
portion of the agent's decision to be implemented at this specific outlet. For 
example, an irrigation diversion agent divert from *a* outlet but reture to *b* 
and *c* outlets (i.e., subbasins) with the ratios, 0.3 and 0.7. Then, we have

.. code-block:: yaml

	Links: {a: -1, b: 0.3, c: 0.7}

Assuming the diversion request is 10, the actual diversion is also 10 (i.e., no
deficiency), and return flow coefficent is 0.5, we have

.. math::

	 Flow_{a,new} = Flow_{a,org} + factor_a \times 10 = Flow_{a,org} -1 \times 10

.. math::
	
	Flow_{return} = 0.5 \times 10 = 5

.. math::
	
	Flow_{b,new} = Flow_{b,org} + factor_b \times 5 = Flow_{b,org} + 0.3 \times 5

.. math::
	
	Flow_{c,new} = Flow_{c,org} + factor_c \times 5 = Flow_{c,org} + 0.7 \times 5


If the "factor" is a calibrated parameter (i.e., an unknown). We can link it to 
a parameter by *[parameter name, its index, Plus/Minus]*. The parameter has to  
be a list format. For example, if *Links = {WSLO: [ReturnFactor, 0, Plus]}*, 
the factor will be extracted from ReturnFactor parameter (a list) at index 0.  
"Plus" will tell the program that we add water to WSLO ("Minus" means divert 
from the given outlet).

* DMClass:

This is optional. If there is no specific decision-making class to be assigned,
put "null" instead. Namely, users can code everything in "agent type class." See 
:ref:`How to build a ABM module?<How to build a ABM module?>` for more details.

**b) Pars** collects agents' parameters for calibration. We 
offer two types of parameter formats: 

(1) A constant (e.g., 9), or 

(2) A list of constants (e.g., [9, 4.5]).

**c) Attributes** is a space for users to store any other information for their 
agents' calculation that is not belong to "Pars" or "Inputs."

6. SystemParsedData (auto-generated)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This section will be automatically generated by HydroCNHS. The model file don't
need to include this section.

.. note::
   :ref:`ModelBuilder<Model builder>` can help you to create an initial model 
   template! Check it out!


ABM module(s)
-------------------------
Agent-based model (ABM) is an user-provided human model. HydroCNHS support 
multiple ABM modules to be used at a single simulation. In the ABM module, 
users have 100% of freedom to design agent class (e.g., irrigation diversion 
agent class, reservoir agent class, etc.); however, some input and output 
protocals have to be followed. Please visit 
:ref:`How to build a ABM module?<How to build a ABM module?>` for more detailed 
instructions. 

.. note::
   If you only need a hydrological model and do not require any human 
   components, then you can skip this ABM part!
 


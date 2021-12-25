Quick start!
============
Let's start with running a simulation!
Here we will use the Tualatin River Basin (TRB) in the Northwest US as an example. The full code is located at **./HydroCNHS/tutorials/Tualatin River Basin/**.

Take one minute to browse through the following simulation code.

.. code-block:: python

	import os
	import pickle
	import HydroCNHS

	##### Setup Working Directory
	# Get this .py file's directory.
	prj_path, this_filename = os.path.split(__file__)
	wd = prj_path

	##### Load Daily Weather Time Series.
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

	##### Run simulation
	Q_gwlf = model_gwlf.run(temp, prec, pet) # pet is optional.



To run a coupled natural human model simulation, three items are required.

1. Daily weather time series including temperture (temp) and precipitation (prec). Potential evapotranspiration (pet) is optional.
2. Model file (e.g., Best_gwlf_abm_KGE.yaml).
3. User-provided ABM module(s) (e.g., TRB_ABM_dm.py). This is not required for pure hydrological model, non-coupled model.

Now, we are going to introduce these three items in a more detail.


Daily weather time series
-------------------------
Temperture and precipitation are two required weather inputs for the simulation. If potential evapotranspiration is not provided, HydroCNHS will automatically calculate it by Hamon method. The weather inputs are in dictionary form shown below.

.. code-block:: python

	# 'DAIRY', 'DLLO', 'RCTV', 'SCOO', 'TRGC', 'TRTR', and 'WSLO' are subbasins' names.
	temp = {'DAIRY': [7.7, 7.0, 6.6, 6.3, .......],
		'DLLO': [7.9, 7.5, 7.1, 6.1, .......],
		'RCTV': [8.0, 7.8, 7.8, 7.5, .......],
		'SCOO': [8.1, 7.4, 7.0, 6.2, .......],
		'TRGC': [5.7, 5.5, 5.1, 4.0, .......],
		'TRTR': [7.9, 6.9, 6.5, 6.1, .......],
		'WSLO': [7.8, 7.4, 7.3, 7.3, .......]}
	# Similar to prep and pet.
		
The dictionary will contain a time series (i.e., a list) for each subbasin. The length of each time series has to be identical.


Model.yaml
-------------------------
The model file (.yaml) contains settings for hydrological model (e.g., rainfall-runoff and routing) and ABM model (e.g., how to coupled).
The model file has six sections:

1. Path

.. code-block:: yaml
	
	# Path for working directory (outputing log file) and user-provided ABM
	# modules (optional).
	Path: {
	  WD: 'wd path',
	  Modules: 'ABM module path'}


2. WaterSystem

.. code-block:: yaml

	WaterSystem:
		StartDate: 1981/1/1
		EndDate: 2013/12/31
		NumSubbasins: 7
		NumGauges: 2
		NumAgents: 3
		Outlets: [TRTR, SCOO, DLLO, TRGC, DAIRY, RCTV, WSLO]
		GaugedOutlets: [DLLO, WSLO]		# Optional
		DataLength: 12053

DataLength can be automatically calculated if EndDate is provided, vice versa.

3. LSM

HydroCNHS provides user two rainfall-runoff simulation options, the General
Water Loading Function (GWLF; 9 parameters) and ABCD (5 parameters). Their
settings are shown below. 

The detailed documentation for GWLF and ABCD can be found at: SM.

**GWLF**

.. code-block:: yaml

	LSM:
		Model: GWLF
		TRTR:
			Inputs: {Area: 329.80, Latitude: 45.45, S0: 2.0, U0: 10.0, SnowS: 5.0}
			Pars: {CN2: 85.450, IS: 0.415, Res: 0.054, Sep: 0.311, Alpha: 0.862,
				Beta: 0.348, Ur: 13.215, Df: 0.920, Kc: 0.838}
		SCOO:
			Inputs: {Area: 10034.24, Latitude: 45.46, S0: 2.0, U0: 10.0, SnowS: 5.0}
			Pars: {CN2: 46.490, IS: 0.268, Res: 0.289, Sep: 0.078, Alpha: 0.174,
				Beta: 0.477, Ur: 12.266, Df: 0.899, Kc: 0.651}
		DLLO:
			Inputs: {Area: 22238.43, Latitude: 45.47, S0: 2.0, U0: 10.0, SnowS: 5.0}
			Pars: {CN2: 39.047, IS: 0.224, Res: 0.425, Sep: 0.284, Alpha: 0.101,
				Beta: 0.398, Ur: 6.386, Df: 0.753, Kc: 0.918}
		TRGC:
			Inputs: {Area: 24044.63, Latitude: 45.50, S0: 2.0, U0: 10.0, SnowS: 5.0}
			Pars: {CN2: 59.783, IS: 0.450, Res: 0.407, Sep: 0.135, Alpha: 0.939,
				Beta: 0.441, Ur: 2.579, Df: 0.516, Kc: 0.733}
		DAIRY:
			Inputs: {Area: 59822.75, Latitude: 45.52, S0: 2.0, U0: 10.0, SnowS: 5.0}
			Pars: {CN2: 99.729, IS: 0.107, Res: 0.198, Sep: 0.332, Alpha: 0.043,
				Beta: 0.101, Ur: 8.570, Df: 0.914, Kc: 1.468}
		RCTV:
			Inputs: {Area: 19682.60, Latitude: 45.50, S0: 2.0, U0: 10.0, SnowS: 5.0}
			Pars: {CN2: 57.345, IS: 0.251, Res: 0.094, Sep: 0.416, Alpha: 0.772,
				Beta: 0.034, Ur: 5.6732, Df: 0.334, Kc: 0.576}
		WSLO:
			Inputs: {Area: 47646.84, Latitude: 45.35, S0: 2.0, U0: 10.0, SnowS: 5.0}
			Pars: {CN2: 60.151, IS: 0.498, Res: 0.095, Sep: 0.038, Alpha: 0.484,
				Beta: 0.371, Ur: 14.347, Df: 0.811, Kc: 0.720}

**ABCD** 

.. code-block:: yaml

	LSM:
		Model: ABCD
		TRTR:
			Inputs: {Area: 329.80, Latitude: 45.45, XL: 2.0, SnowS: 5.0}
			Pars: {a: 0.241, b: 281.131, c: 0.915, d: 0.510, Df: 0.492}
		SCOO:
			Inputs: {Area: 10034.24, Latitude: 45.46, XL: 2.0, SnowS: 5.0}
			Pars: {a: 0.438, b: 13.751, c: 0.990, d: 0.330, Df: 0.576}
		DLLO:
			Inputs: {Area: 22238.43, Latitude: 45.47, XL: 2.0, SnowS: 5.0}
			Pars: {a: 0.438, b: 317.570, c: 0.765, d: 0.400, Df: 0.834}
		TRGC:
			Inputs: {Area: 24044.63, Latitude: 45.50, XL: 2.0, SnowS: 5.0}
			Pars: {a: 0.197, b: 157.836, c: 0.785, d: 0.584, Df: 0.503}
		DAIRY:
			Inputs: {Area: 59822.75, Latitude: 45.52, XL: 2.0, SnowS: 5.0}
			Pars: {a: 0.294, b: 102.755, c: 0.466, d: 0.529, Df: 0.503}
		RCTV:
			Inputs: {Area: 19682.60, Latitude: 45.50, XL: 2.0, SnowS: 5.0}
			Pars: {a: 0.195, b: 52.505, c: 0.226, d: 0.492, Df: 0.865}
		WSLO:
			Inputs: {Area: 47646.84, Latitude: 45.35, XL: 2.0, SnowS: 5.0}
			Pars: {a: 0.781, b: 2.738, c: 0.961, d: 0.785, Df: 0.055}

4. Routing

HydroCNHS adopts Lohmann routing model to simulate within-subbasin routing and inter subbasin routing process. Its setting is shown below.

.. code-block:: yaml

	Routing:
		Model: Lohmann
		# WSLO, TRGC, DLLO, and SCOO are four routing outlets.
		WSLO:
			# TRGC is a routing outlet. No within-subbasin routing at here (at
			# its own routing setting below). Namely, TRGC will be routed
			# first. Therefore, GShape and GScale are null.
			TRGC:
				Inputs: {FlowLength: 80064.86, InstreamControl: false}
				Pars: {GShape: null, GScale: null, Velo: 53.28, Diff: 1991.52}
			DAIRY:
				Inputs: {FlowLength: 70988.16, InstreamControl: false}
				Pars: {GShape: 68.40, GScale: 545.55, Velo: 45.32, Diff: 935.13}
			RCTV:
				Inputs: {FlowLength: 60398.68, InstreamControl: false}
				Pars: {GShape: 53.37, GScale: 462.47, Velo: 53.57, Diff: 3339.43}
			# WSLO is the routing outlet itself. No river routing is needed
			# since the FlowLength is 0. Therefore, Velo and Diff are null.
			WSLO:
				Inputs: {FlowLength: 0, InstreamControl: false}
				Pars: {GShape: 6.86, GScale: 0.67, Velo: null, Diff: null}
		TRGC:
			# DLLO is a routing outlet. No within-subbasin routing at here (at
			# its own routing setting below). Namely, DLLO will be routed
			# first. Therefore, GShape and GScale are null.
			DLLO:
				Inputs: {FlowLength: 11748.21, InstreamControl: false}
				Pars: {GShape: null, GScale: null, Velo: 5.97, Diff: 1864.99}
			# TRGC is the routing outlet itself. No river routing is needed
			# since the FlowLength is 0. Therefore, Velo and Diff are null.
			TRGC:
				Inputs: {FlowLength: 0, InstreamControl: false}
				Pars: {GShape: 27.22, GScale: 0.29, Velo: null, Diff: null}
		DLLO:
			# R1 is the reservoir agent. There is no within-subbasin routing. 
			# Its release flow is the streamflow at this spot. Therefore,
			# GShape and GScale are null.
			R1:
				Inputs: {FlowLength: 9656.06, InstreamControl: true}
				Pars: {GShape: null, GScale: null, Velo: 53.95, Diff: 852.67}
			TRTR:
				Inputs: {FlowLength: 30899.40, InstreamControl: false}
				Pars: {GShape: 83.52, GScale: 755.91, Velo: 18.73, Diff: 2388.09}
			# DLLO is the routing outlet itself. No river routing is needed
			# since the FlowLength is 0. Therefore, Velo and Diff are null.
			DLLO:
				Inputs: {FlowLength: 0, InstreamControl: false}
				Pars: {GShape: 75.30, GScale: 1.62, Velo: null, Diff: null}
		SCOO:
			# SCOO is the routing outlet itself. No river routing is needed
			# since the FlowLength is 0. Therefore, Velo and Diff are null
			SCOO:
				Inputs: {FlowLength: 0, InstreamControl: false}
				Pars: {GShape: 27.22, GScale: 0.29, Velo: null, Diff: null}

Put the system diagram here.

5. ABM

For the "Inputs" setting of ABM section, first, we assign the user-defined
agent classes (defined in ABM modules) to corresponding coupling APIs. Then,
we ativate the decision-making classes (defined in ABM modules) if any. Next,
we assign ABM module(s), where agent classes and decision-making classes are
defined. Finally, agent group is for agents make decisions and act together.
For example, two diversion agents make diversion requests together and share 
the water deficiency together based on their water rights. Namely, their 
diversion behaviors are not piority-based. The agent group will be defined as 
a single function in a ABM module, which users can define a more detailed
interactions amond agents in an agent group. See Build ABM for more details.

Following the "Inputs" setting, we will define agent objects created by certain
agent classes (defined in ABM modules). For example, we create Barney agent 
using Pipe_AgType class. Under each agent object (e.g., Barney), it has three
sub-sections: "Attributes", "Inputs", and "Pars." 

**"Inputs"** is required information including "Piority", "Links", and "DMClass." 

* Piority: 
  
The lower value has higher piority when conflicts happen. For example, two
diverion agents divert at the same routing outlet. If users want a 
non-priority-based behaviors. "AgGroup" should be applied. See Build ABM for
more details.

Note that agents coupling with Dam API has to have Piority = 0.

* Links:

"Links" is a dictionary containing information which outlets for agent to
take/add water from/to. The positive number means add the water to that outlet.
Negative number means take water from that outlet. The number is a "factor" 
in a range of [-1,1] defining the portion of the agent's decision to be
implemented at this specific outlet. For example, an irrigation diversion agent
divert from a point but reture to b and c points with the ratios, 0.3
and 0.7. Then, we have

.. code-block:: yaml

	Links: {a: -1, b: 0.3, c: 0.7}

Assuming the diversion request is 10, the actual diversion is also 10 (i.e., no
deficiency), and return flow coefficent is 0.5.

.. math::

	 Flow_{a,new} = Flow_{a,org} -1 \times 10

.. math::
	
	reture_flow = 0.5 \times 10 = 5

.. math::
	
	Flow_{b,new} = Flow_{b,org} + 0.3 \times 5

.. math::
	
	Flow_{c,new} = Flow_{c,org} + 0.7 \times 5


If the "factor" is a calibrated parameter. We can link it to a parameter by
[parameter name, its index, Plus/Minus]. The parameter has to be a list
format. For example, if Links = {WSLO: [ReturnFactor, 0, Plus]}, the factor
will be extracted from ReturnFactor parameter (a list) at index 0. "Plus" 
will tell the program that we add water to WSLO (for forming simulation
purpose).

* DMClass:

This is optional. If there is no specific decision-making class to be assigned,
put "null" instead. See Build ABM for more details.

**"Pars"** is a section for collecting agents' parameters for calibration. We offer
two types of parameter formats: a single constant (e.g., 9), (2) a list of 
constants (e.g., [9, 4.5]).

**"Attributes"** is a space for users to store any other information for their 
agents' calculation that is not belong to "Pars" or "Inputs."

.. code-block:: yaml

	ABM:
		Inputs:
			# Assign user-defined agent classes to corresponding APIs. Here, we
			# define three agent classes in TRB_ABM_dm.py: ResDam_AgType 
			# (reservoir), IrrDiv_AgType (diversion), and Pipe_AgType 
			# (trans-basin conveying water).
			DamAgentTypes: [ResDam_AgType]		# Dam API
			RiverDivAgentTypes: [IrrDiv_AgType]	# RiverDiv API
			InsituAgentTypes: []				# InSitu API
			ConveyAgentTypes: [Pipe_AgType]		# Conveying API
			# Activate user-defined decision-making classes in TRB_ABM_dm.py.
			DMClasses: [ResDM, DivDM, PipeDM]
			# User-defined ABM module, TRB_ABM_dm.py.
			Modules: [TRB_ABM_dm.py]
			# Agent group is for agents make decisions and act together.
			AgGroup: null
		Pipe_AgType:
			# Create agent objects using Pipe_AgType class. Here, we only have 
			# one Pipe_AgType agent, Barney.
			Barney:
				Attributes: {} 	# According to users' needs, optional.
				# Inputs are required information.
				Inputs:
					Piority: 0 	
					Links: {TRTR: 1}
					DMClass: PipeDM
				Pars:			# According to users' needs, optional.
					ReturnFactor: []
					a: null
					b: null
		ResDam_AgType:
			# Create agent objects using ResDam_AgType class. Here, we only 
			# have one ResDam_AgType agent, R1.
			R1:
				Attributes: {}	# According to users' needs, optional.
				# Inputs are required information.
				Inputs:
					Piority: 0
					Links: {SCOO: -1, R1: 1}
					DMClass: ResDM
				Pars:			# According to users' needs, optional.
					ReturnFactor: []
					a: null
					b: null
		IrrDiv_AgType:
			# Create agent objects using IrrDiv_AgType class. Here, we only 
			# have one IrrDiv_AgType agent, SHPP.
			SHPP:
				Attributes: {}	# According to users' needs, optional.
				# Inputs are required information.
				Inputs:
					Piority: 1
					Links:
					TRGC: -1
					WSLO: [ReturnFactor, 0, Plus]
					DMClass: DivDM
				Pars:			# According to users' needs, optional.
					ReturnFactor: [0.30086264868779805]
					a: -0.92169837578325
					b: 0.09731044387555121

6. SystemParsedData
This section will be automatically generated by HydroCNHS. The model file don't
need to include this section.

.. note::
   ModelBuilder can help you to create an initial model template! Check it out!


ABM module(s)
-------------------------
Agent-based model (ABM) is an user-provided human model. HydroCNHS support multiple ABM modules to be used at a single simulation. In the ABM module, users have 100% of freedom to design agent class (e.g., irrigation diversion agent class, reservoir agent class, etc.); however, some input and output protocals have to be followed.
Please visit Build ABM module for more detailed instruction. 

.. note::
   If you only need a hydrological model and don't require any human components, then you can skip this ABM part!
 


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
	  GaugedOutlets: [DLLO, WSLO]
	  DataLength: 12053

DataLength can be automatically calculated if EndDate is provided, vice versa.

3. LSM

HydroCNHS provides user two rainfall-runoff simulation options, GWLF (9 parameters) and ABCD (5 parameters). Their settings are shown below. 
The detailed documentation for GWLF and ABCD can be found at: SM.

**GWLF**

.. code-block:: yaml

	LSM:
	  Model: ABCD

**ABCD** 

.. code-block:: yaml

	LSM:
	  Model: ABCD

4. Routing
HydroCNHS adopt Lohmann routing model to simulate within-subbasin routing and inter subbasin routing process. Its setting is shown below.

.. code-block:: yaml

	Routing:
	  Model: Lohmann
	  WSLO:
		TRGC:
		  Inputs: {FlowLength: 80064.864, InstreamControl: false}
		  Pars: {GShape: null, GScale: null, Velo: 27.96625624979407, Diff: 2863.4082828311643}
		DAIRY:
		  Inputs: {FlowLength: 70988.16384, InstreamControl: false}
		  Pars: {GShape: 81.12165115626, GScale: 6.33, Velo: 23.556139717018198,
			Diff: 2555.76308439851}
		RCTV:
		  Inputs: {FlowLength: 60398.68032, InstreamControl: false}
		  Pars: {GShape: 93.65203474797343, GScale: 9.35, Velo: 53.6240069222664,
			Diff: 416.6625889040875}
		WSLO:
		  Inputs: {FlowLength: 0, InstreamControl: false}
		  Pars: {GShape: 84.11705456562375, GScale: 3.45, Velo: null, Diff: null}
	  TRGC:
		DLLO:
		  Inputs: {FlowLength: 11748.2112, InstreamControl: false}
		  Pars: {GShape: null, GScale: null, Velo: 7.814731413762071, Diff: 844.2277689361871}

Put the system diagram here.

5. ABM

.. code-block:: yaml

	ABM:
	  Inputs:
		DamAgentTypes: [ResDam_AgType]
		RiverDivAgentTypes: [IrrDiv_AgType]
		InsituAgentTypes: []
		ConveyAgentTypes: [Pipe_AgType]
		DMClasses: [ResDM, DivDM, PipeDM]
		Modules: [TRB_ABM_dm.py]
		AgGroup: null
	  Pipe_AgType:
		Barney:
		  Attributes: {}
		  Inputs:
			Piority: 0
			Links: {TRTR: 1}
			DMClass: PipeDM
		  Pars:
			ReturnFactor: []
			a: null
			b: null
	  ResDam_AgType:
		R1:
		  Attributes: {}
		  Inputs:
			Piority: 0
			Links: {SCOO: -1, R1: 1}
			DMClass: ResDM
		  Pars:
			ReturnFactor: []
			a: null
			b: null
	  IrrDiv_AgType:
		SHPP:
		  Attributes: {}
		  Inputs:
			Piority: 1
			Links:
			  TRGC: -1
			  WSLO: [ReturnFactor, 0, Plus]
			DMClass: DivDM
		  Pars:
			ReturnFactor: [0.23561383574933084]
			a: 0.4152467680396592
			b: -0.012094413038078455

6. SystemParsedData
This section will be automatically generated by HydroCNHS. The model file don't need to include this section.

.. note::
   ModelBuilder can help you to create an initial model template! Check it out!


ABM module(s)
-------------------------
Agent-based model (ABM) is an user-provided human model. HydroCNHS support multiple ABM modules to be used at a single simulation. In the ABM module, users have 100% of freedom to design agent class (e.g., irrigation diversion agent class, reservoir agent class, etc.); however, some input and output protocals have to be followed.
Please visit Build ABM module for more detailed instruction. 

.. note::
   If you only need a hydrological model and don't require any human components, then you can skip this ABM part!
 


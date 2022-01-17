Model builder
===================

The ModelBuilder() of model builder module can quickly help users to establish
the outline of a model (.yaml). Namely, ModelBuilder can create a template 
model for :ref:`calibration<Calibration>`. First, we initialize a ModelBuilder
object as shown in the following.

.. code-block:: python

    import HydroCNHS

    wd = "working directory"
    mb = HydroCNHS.ModelBuilder(wd)

    r'''
    # Print out message.
    Follow the following steps to create model template:
        Step 1: set_water_system()
        Step 2: set_lsm()
        Step 3: set_routing_outlet(), one at a time.
        Step 4: set_ABM() if you want to build a coupled model.
        Step 5: write_model_to_yaml()
    After creating model.yaml template, open it and further edit it.
    Use .help to re-print the above instruction.
    '''


Then, we follow the print out instructions to create a model template.

Here, we demonstrate how to create a coupled model template for the Tualatin 
River Basin (TRB).

.. figure:: ./figs/TRB.png
  :align: center
  :width: 500
  :alt: The Tualatin River Basin system diagram. 

  The Tualatin River Basin system diagram (Lin et al., 2022). TRTR, 
  Hagg\ :sub:`In`\, DLLO, TRGC, DAIRY, RCTV, and WSLO are seven subbasins. 
  PipeAgt, ResAgt, and DivAgt are trans-basin aqueduct, Hagg reservoir, and 
  irrigation diversion agents, respectively. DrainAgt1 and DrainAgt2 are two 
  drainage-system agents for the runoff-changing scenario.


.. code-block:: python

    ### Setup a water system simulation information
    mb.set_water_system(start_date="1981/1/1", end_date="2013/12/31")

    ### Setup land surface model (rainfall-runoff model)
    # Here we have seven subbasins and we select GWLF as the rainfall-runoff model.
    outlet_list = ["TRTR", "HaggIn", "DLLO", "TRGC", "DAIRY", "RCTV", "WSLO"]
    mb.set_lsm(outlet_list=outlet_list, lsm_model="GWLF")   # or lsm_model="ABCD"

    ### Setup routing 
    # We have four routing outlets, which we will add them into the model one by 
    # one.
    mb.set_routing_outlet(routing_outlet="WSLO", 
                        upstream_outlet_list=["TRGC", "DAIRY", "RCTV", "WSLO"])
    mb.set_routing_outlet(routing_outlet="TRGC", 
                        upstream_outlet_list=["DLLO", "TRGC"])
    mb.set_routing_outlet(routing_outlet="DLLO", 
                        upstream_outlet_list=["ResAgt", "TRTR", "DLLO"], 
                        instream_outlets=["ResAgt"]) 
    # Note: ResAgt (Hagg Lake) is the reservoir agent, whcih is considerred as 
    # an instream control object.

    mb.set_routing_outlet(routing_outlet="HaggIn", 
                        upstream_outlet_list=["HaggIn"])

    ### Setup ABM
    abm_module_path="abm_module path"
    mb.set_ABM(abm_module_path=abm_module_path)

    ### Save to a .yaml model file for further editting.
    filename = "output directory/model.yaml"
    mb.write_model_to_yaml(filename)


After "write_model_to_yaml()", open the create model.yaml (see below) to 
further edit it. Users need to manually add necessary "inputs" information
for all sections. Namely, users need to fill all **null** *except those nested 
under "Pars" subsections*. For the ABM section, users are asked to manually add 
"agents" to the model. 

Note that the **-99** values under "Pars" subsections are the parameters 
required :ref:`calibration<Calibration>`. 
  
.. code-block:: yaml

    Path: {WD: working directory, Modules: abm_module path}
    WaterSystem:
    StartDate: 1981/1/1
    EndDate: 2013/12/31
    NumSubbasins: 7
    NumGauges: null
    NumAgents: null
    Outlets: [TRTR, HaggIn, DLLO, TRGC, DAIRY, RCTV, WSLO]
    GaugedOutlets: []
    DataLength: 12053
    LSM:
    Model: GWLF
    TRTR:
        Inputs: {Area: null, Latitude: null, S0: null, U0: null, SnowS: null}
        Pars: {CN2: -99, IS: -99, Res: -99, Sep: -99, Alpha: -99, Beta: -99, Ur: -99,
        Df: -99, Kc: -99}
    HaggIn:
        Inputs: {Area: null, Latitude: null, S0: null, U0: null, SnowS: null}
        Pars: {CN2: -99, IS: -99, Res: -99, Sep: -99, Alpha: -99, Beta: -99, Ur: -99,
        Df: -99, Kc: -99}
    DLLO:
        Inputs: {Area: null, Latitude: null, S0: null, U0: null, SnowS: null}
        Pars: {CN2: -99, IS: -99, Res: -99, Sep: -99, Alpha: -99, Beta: -99, Ur: -99,
        Df: -99, Kc: -99}
    TRGC:
        Inputs: {Area: null, Latitude: null, S0: null, U0: null, SnowS: null}
        Pars: {CN2: -99, IS: -99, Res: -99, Sep: -99, Alpha: -99, Beta: -99, Ur: -99,
        Df: -99, Kc: -99}
    DAIRY:
        Inputs: {Area: null, Latitude: null, S0: null, U0: null, SnowS: null}
        Pars: {CN2: -99, IS: -99, Res: -99, Sep: -99, Alpha: -99, Beta: -99, Ur: -99,
        Df: -99, Kc: -99}
    RCTV:
        Inputs: {Area: null, Latitude: null, S0: null, U0: null, SnowS: null}
        Pars: {CN2: -99, IS: -99, Res: -99, Sep: -99, Alpha: -99, Beta: -99, Ur: -99,
        Df: -99, Kc: -99}
    WSLO:
        Inputs: {Area: null, Latitude: null, S0: null, U0: null, SnowS: null}
        Pars: {CN2: -99, IS: -99, Res: -99, Sep: -99, Alpha: -99, Beta: -99, Ur: -99,
        Df: -99, Kc: -99}
    Routing:
    Model: Lohmann
    WSLO:
        TRGC:
            Inputs: {FlowLength: null, InstreamControl: false}
            Pars: {GShape: null, GScale: null, Velo: -99, Diff: -99}
        DAIRY:
            Inputs: {FlowLength: null, InstreamControl: false}
            Pars: {GShape: -99, GScale: -99, Velo: -99, Diff: -99}
        RCTV:
            Inputs: {FlowLength: null, InstreamControl: false}
            Pars: {GShape: -99, GScale: -99, Velo: -99, Diff: -99}
        WSLO:
            Inputs: {FlowLength: null, InstreamControl: false}
            Pars: {GShape: -99, GScale: -99, Velo: null, Diff: null}
    TRGC:
        DLLO:
            Inputs: {FlowLength: null, InstreamControl: false}
            Pars: {GShape: null, GScale: null, Velo: -99, Diff: -99}
        TRGC:
            Inputs: {FlowLength: null, InstreamControl: false}
            Pars: {GShape: -99, GScale: -99, Velo: null, Diff: null}
    DLLO:
        ResAgt:
            Inputs: {FlowLength: null, InstreamControl: false}
            Pars: {GShape: null, GScale: null, Velo: -99, Diff: -99}
        TRTR:
            Inputs: {FlowLength: null, InstreamControl: false}
            Pars: {GShape: -99, GScale: -99, Velo: -99, Diff: -99}
        DLLO:
            Inputs: {FlowLength: null, InstreamControl: false}
            Pars: {GShape: -99, GScale: -99, Velo: null, Diff: null}
    HaggIn:
        HaggIn:
            Inputs: {FlowLength: null, InstreamControl: false}
            Pars: {GShape: -99, GScale: -99, Velo: null, Diff: null}
    ABM:
    Inputs:
        DamAgentTypes: []
        RiverDivAgentTypes: []
        InsituAgentTypes: []
        ConveyAgentTypes: []
        DMClasses: []
        Modules: []
        AgGroup: null

After filling in the necessary information (e.g., Inputs and ABM settings, we 
will obtain a model template (see below) ready to be calibrated (i.e., those 
-99 values).

.. code-block:: yaml

    Path: {WD: working directory, Modules: abm_module path}
    WaterSystem:
    StartDate: 1981/1/1
    EndDate: 2013/12/31
    NumSubbasins: 7
    NumGauges: 2
    NumAgents: 3
    Outlets: [TRTR, HaggIn, DLLO, TRGC, DAIRY, RCTV, WSLO]
    GaugedOutlets: [DLLO, WSLO]
    DataLength: 12053
    LSM:
    Model: GWLF
    TRTR:
        Inputs: {Area: 329.8013, Latitude: 45.458136, XL: 2.0, SnowS: 5.0}
        Pars: {CN2: -99, IS: -99, Res: -99, Sep: -99, Alpha: -99, Beta: -99, Ur: -99,
        Df: -99, Kc: -99}
    HaggIn:
        Inputs: {Area: 10034.2408, Latitude: 45.469444, XL: 2.0, SnowS: 5.0}
        Pars: {CN2: -99, IS: -99, Res: -99, Sep: -99, Alpha: -99, Beta: -99, Ur: -99,
        Df: -99, Kc: -99}
    DLLO:
        Inputs: {Area: 22238.4391, Latitude: 45.475, XL: 2.0, SnowS: 5.0}
        Pars: {CN2: -99, IS: -99, Res: -99, Sep: -99, Alpha: -99, Beta: -99, Ur: -99,
        Df: -99, Kc: -99}
    TRGC:
        Inputs: {Area: 24044.6363, Latitude: 45.502222, XL: 2.0, SnowS: 5.0}
        Pars: {CN2: -99, IS: -99, Res: -99, Sep: -99, Alpha: -99, Beta: -99, Ur: -99,
        Df: -99, Kc: -99}
    DAIRY:
        Inputs: {Area: 59822.7546, Latitude: 45.52, XL: 2.0, SnowS: 5.0}
        Pars: {CN2: -99, IS: -99, Res: -99, Sep: -99, Alpha: -99, Beta: -99, Ur: -99,
        Df: -99, Kc: -99}
    RCTV:
        Inputs: {Area: 19682.6046, Latitude: 45.502222, XL: 2.0, SnowS: 5.0}
        Pars: {CN2: -99, IS: -99, Res: -99, Sep: -99, Alpha: -99, Beta: -99, Ur: -99,
        Df: -99, Kc: -99}
    WSLO:
        Inputs: {Area: 47646.8477, Latitude: 45.350833, XL: 2.0, SnowS: 5.0}
        Pars: {CN2: -99, IS: -99, Res: -99, Sep: -99, Alpha: -99, Beta: -99, Ur: -99,
        Df: -99, Kc: -99}
    Routing:
    Model: Lohmann
    WSLO:
        TRGC:
            Inputs: {FlowLength: 80064.864, InstreamControl: false}
            Pars: {GShape: null, GScale: null, Velo: -99, Diff: -99}
        DAIRY:
            Inputs: {FlowLength: 70988.16384, InstreamControl: false}
            Pars: {GShape: -99, GScale: -99, Velo: -99, Diff: -99}
        RCTV:
            Inputs: {FlowLength: 60398.68032, InstreamControl: false}
            Pars: {GShape: -99, GScale: -99, Velo: -99, Diff: -99}
        WSLO:
            Inputs: {FlowLength: 0, InstreamControl: false}
            Pars: {GShape: -99, GScale: -99, Velo: null, Diff: null}
    TRGC:
        DLLO:
            Inputs: {FlowLength: 11748.2112, InstreamControl: false}
            Pars: {GShape: null, GScale: null, Velo: -99, Diff: -99}
        TRGC:
            Inputs: {FlowLength: 0, InstreamControl: false}
            Pars: {GShape: -99, GScale: -99, Velo: null, Diff: null}
    DLLO:
        ResAgt:
            Inputs: {FlowLength: 9656.064, InstreamControl: false}
            Pars: {GShape: null, GScale: null, Velo: -99, Diff: -99}
        TRTR:
            Inputs: {FlowLength: 30899.4048, InstreamControl: false}
            Pars: {GShape: -99, GScale: -99, Velo: -99, Diff: -99}
        DLLO:
            Inputs: {FlowLength: 0, InstreamControl: false}
            Pars: {GShape: -99, GScale: -99, Velo: null, Diff: null}
    HaggIn:
        HaggIn:
            Inputs: {FlowLength: 0, InstreamControl: false}
            Pars: {GShape: -99, GScale: -99, Velo: null, Diff: null}
    ABM:
    Inputs:
        DamAgentTypes: [ResDam_AgType]
        RiverDivAgentTypes: [IrrDiv_AgType]
        InsituAgentTypes: []
        ConveyAgentTypes: [Pipe_AgType]
        DMClasses: [ResDM, DivDM, PipeDM]
        Modules: [TRB_ABM_dm.py]        # user-provided ABM module.
        AgGroup: null
    Pipe_AgType:
        PipeAgt:
        Attributes: {}
        Inputs:
            Piority: 0
            Links: {TRTR: 1}
            DMClass: PipeDM
        Pars: {}    # No parameter
    ResDam_AgType:
        ResAgt:
        Attributes: {}
        Inputs:
            Piority: 0
            Links: {SCOO: -1, R1: 1}
            DMClass: ResDM
        Pars: {}    # No parameter
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
            ReturnFactor: [-99]
            a: -99
            b: -99

import logging
from HydroCNHS.abm import Base, read_factor
logger = logging.getLogger("ABM")
# logger can be used to log message (e.g., logger.info(msg)).

### Global variable and data
# Users can load data or define variables that are commonly available to all 
# agent type classes and decision-making classes here. 



"""
This is the auto-generated script template for a ABM module.
Make sure to add ABM module (this filename) to the model file
(.yaml). 

Note:
Below is the list of inherited attributes for agent type 
classes (AgtType) and (institutional) decision-making classes 
(DMClass). These attributes will be assigned to each
initialized object in a HydroCNHS simulation. Namely, they 
can be used for model design.

Agent type class (AgtType):
self.name         = agent's name.
self.config       = agent's configuration dictionary,
                  {'Attributes': ..., 'Inputs': ..., 'Pars': ...}.
self.start_date   = start date (datetime object).
self.current_date = current date (datetime object).
self.data_length  = data/simulation length.
self.t            = current timestep.
self.dc           = data collector object containing data. Routed streamflow  
                    (Q_routed) is also collected at here.
self.rn_gen       = NumPy random number generator.
self.agents       = a dictionary of all initialized agents, 
                    {agt_name: agt object}.
self.dm           = (institutional) decision-making object if
                    DMClass or institution is assigned to the
                    agent, else None.

(Institutional) decision-making classes (DMClass):
self.name         = name of the agent or institute.
self.dc           = data collector object containing data. Routed streamflow  
                    (Q_routed) is also collected at here.
self.rn_gen       = NumPy random number generator.

Q_routed is the routed streamflow.

Please visit HydroCNHS manual for more examples.
https://hydrocnhs.readthedocs.io
"""


# AgtType
class Reservoir_AgtType(Base):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # The AgtType inherited attributes are applied. 
        # See the note at top.

    def act(self, outlet):
        # Read corresponding factor of the given outlet
        factor = read_factor(self.config, outlet)
        
        # Common usage:
        # Get streamflow of outlet at timestep t
        Q = self.dc.Q_routed[outlet][self.t]
        
        # Make decision from (Institutional) decision-making
        # object if self.dm is not None.
        #decision = self.dm.make_dm(your_arguments)
        
        if factor <= 0:     # Divert from the outlet
            action = 0
        elif factor > 0:    # Add to the outlet
            action = 0
        
        return action


# AgtType
class Diversion_AgtType(Base):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # The AgtType inherited attributes are applied. 
        # See the note at top.

    def act(self, outlet):
        # Read corresponding factor of the given outlet
        factor = read_factor(self.config, outlet)
        
        # Common usage:
        # Get streamflow of outlet at timestep t
        Q = self.dc.Q_routed[outlet][self.t]
        
        # Make decision from (Institutional) decision-making
        # object if self.dm is not None.
        #decision = self.dm.make_dm(your_arguments)
        
        if factor <= 0:     # Divert from the outlet
            action = 0
        elif factor > 0:    # Add to the outlet
            action = 0
        
        return action


# AgtType
class Pipe_AgtType(Base):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # The AgtType inherited attributes are applied. 
        # See the note at top.

    def act(self, outlet):
        # Read corresponding factor of the given outlet
        factor = read_factor(self.config, outlet)
        
        # Common usage:
        # Get streamflow of outlet at timestep t
        Q = self.dc.Q_routed[outlet][self.t]
        
        # Make decision from (Institutional) decision-making
        # object if self.dm is not None.
        #decision = self.dm.make_dm(your_arguments)
        
        if factor <= 0:     # Divert from the outlet
            action = 0
        elif factor > 0:    # Add to the outlet
            action = 0
        
        return action


# DMClass
class ReleaseDM(Base):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # The (Institutional) DMClass inherited attributes are applied. 
        # See the note at top.

    def make_dm(self, your_arguments):
        # Decision-making calculation.
        decision = None
        return decision


# DMClass
class DivertDM(Base):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # The (Institutional) DMClass inherited attributes are applied. 
        # See the note at top.

    def make_dm(self, your_arguments):
        # Decision-making calculation.
        decision = None
        return decision


# DMClass
class TransferDM(Base):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # The (Institutional) DMClass inherited attributes are applied. 
        # See the note at top.

    def make_dm(self, your_arguments):
        # Decision-making calculation.
        decision = None
        return decision

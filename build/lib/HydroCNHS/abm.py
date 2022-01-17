# ABM helper function and class.
# by Chung-Yi Lin @ Lehigh University (philip928lin@gmail.com) 
# Last update at 2022/01/14.

class Base():
    """
    Agent_type class's available items:
    * name: agent's name.
    * config: agent's configuration dictionary the model file (.yaml).
    * start_date: datetime object.
    * data_length: length of the simulation.
    * data_collector: a container to store simulated data.
    * rn_gen: random number generator to ensure reproducibility (e.g., 
    * self.rn_gen.random()). Note that do NOT set a global random seed in 
    * this module! All type of random number should be created by "rn_gen."
    * dm: decision making object if assigned in the model file (.yaml).
    
     Decision-making class's available items:
    * start_date: datetime object.
    * data_length: length of the simulation.
    * abm: the ABM configuration dictionary from the model file (.yaml).
    * data_collector: a container to store simulated data.
    * rn_gen: random number generator to ensure reproducibility (e.g., 
    * self.rn_gen.random()). Note that do NOT set a global random seed in 
    * this module! All type of random number should be created by "rn_gen.
    """
    def __init__(self, **kwargs):
        for key in kwargs:  # Load back all the previous class attributions.
            setattr(self, key, kwargs[key])

def read_factor(ag_config, outlet):
    """Read factor from agent's ag_config at a given outlet.

    Parameters
    ----------
    ag_config : dict
        Agent's configuration.
    outlet : str
        Outlet name.
    """
    factor = ag_config["Inputs"]["Links"][outlet]
    # For parameterized factor.
    if isinstance(factor, list):
        factor = ag_config["Pars"][factor[0]][factor[1]]
    return factor
Advanced ABM coding tips
========================

In this section, we provide some coding tips for ABM module designs.

Enhancing computational speed
-----------------------------

Avoid extensive calls to DataFrame
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Reading data to a DataFrame (e.g., *df.loc[,]*) tends to be slow. We suggest users use NumPy array, list, or dictionary for calculations or data storage.

Avoid repeated loading of external data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
If there is common data among multiple agent-type classes, we suggest loading the data once to a global variable at the top of the ABM module and using the variable across classes. This might save some time from repeated loading of external data inside each class (e.g., at *def __init__(self)*).

Avoid extensive deepcopy
^^^^^^^^^^^^^^^^^^^^^^^^
deepcopy is a function to create a copy with a different storage address (not just copy a pointer that points to the same storage address). Therefore, it will take a longer time to complete the task. We suggest using deepcopy only when it is necessary.

Avoid storing redundant data in a data collector
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
A data collector is a data container object to store model outputs. We encourage users to utilize it to store agents' results; however, users will need to consider the storage capacity of their computing devices, especially with a considerable number of agents. Overusing the computer storage might also slow down the computational speed.

Logging
-------

Logging is a python package to organize model output messages. We encourage users to adopt it in their ABM module design. This will help you to integrate your agent output messages into HydroCNHS. 

.. code-block:: python

    import logging
    logger = logging.getLogger("ABM")
    logger.info(<general information message>)
    logger.error(<error message>)

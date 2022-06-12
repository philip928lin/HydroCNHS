Data collector
==============

A data collector is a container object created by HydroCNHS in each simulation that can be passed around HydroCNHS and user-defined ABM modules. A data collector object can store dictionaries and lists. Each of the items is associated with a unique field name. Each field has properties including data, data type (e.g., a dictionary or a list), description, and unit. We provide some usage examples below.

First, we manually create a data collector object for demonstration.

.. code-block:: python

    import HydroCNHS

    ### Create a data collector object
    dc = HydroCNHS.Data_collector()

Then, we add two fields, "field_1" and "field 2", with corresponding field information to the collector. Spaces are not allowed here, and the code will convert "field 2" to "field_2". 

.. code-block:: python
    ### Add fields
    dc.add_field("field_1", data_type={}, desc="Demo dc ex1.", unit="no unit")
    dc.add_field("field 2", data_type=[1,2,3], unit="cm")
    ### Print out existed fields
    dc.list_fields()
    # field_1
    #   type
    #     <class 'dict'>
    #   desc
    #     Demo dc ex1.
    #   unit
    #     no unit
    # field_2
    #   type
    #     <class 'list'>
    #   desc
    #     None
    #   unit
    #     cm

To read the data in a data collector, e.g., reading field_2, we may do the following:

.. code-block:: python

    ### Read data
    dc.field_2
    # Out[0]: [1, 2, 3]

We can also create a shortcut for accessing a field by the following command, in which any modifications on the shortcut will be passed into the data collector object.

.. code-block:: python

    ### Get a field shortcut
    shortcut = dc.get_field("field_1", copy=False)  
    shortcut["new_key"] = "new value"
    dc.field_1
    # Out[0]: {'new_key': 'new value'}

If we want to get a copy of a field (not a shortcut), we must assign "True" to the "copy" argument.

.. code-block:: python

    ### Get a copy of a field
    copied = dc.get_field("field_1", copy=True) 
    copied["new_key2"] = "new value2"
    dc.field_1
    # Out[0]: {'new_key': 'new value'}
    print(copied)
    # {'new_key': 'new value', 'new_key2': 'new value2'}

We can also delete a field using the following commands.

.. code-block:: python

    ### Delete a field
    dc.del_field("field_1")
    dc.list_fields()
    # field_2
    #   type
    #     <class 'list'>
    #   desc
    #     None
    #   unit
    #     cm

Finally, users can export the entire data collector to a dictionary.

.. code-block:: python

    ### Export the entire data collector to a dictionary
    dictionary = dc.get_dict(copy=True)
    print(dictionary)
    # {'field_info': {'field_2': {'type': <class 'list'>, 'desc': None, 'unit': 'cm'}},
    #  'field_2': [1, 2, 3]}

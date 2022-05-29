import HydroCNHS

### Create a data collector object
dc = HydroCNHS.Data_collector()

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

### Read data
dc.field_2
# Out[0]: [1, 2, 3]

### Get a field shortcut
shortcut = dc.get_field("field_1", copy=False)  
shortcut["new_key"] = "new value"
dc.field_1
# Out[0]: {'new_key': 'new value'}

### Get a copy of a field
copied = dc.get_field("field_1", copy=True) 
copied["new_key2"] = "new value2"
dc.field_1
# Out[0]: {'new_key': 'new value'}
print(copied)
# {'new_key': 'new value', 'new_key2': 'new value2'}

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

### Export the entire data collector to a dictionary
dictionary = dc.get_dict(copy=True)
print(dictionary)
# {'field_info': {'field_2': {'type': <class 'list'>, 'desc': None, 'unit': 'cm'}},
#  'field_2': [1, 2, 3]}


# Data collector module
# Here we create a basic Data_collector aiming to collect agents' outputs.
# by Chung-Yi Lin @ Lehigh University (philip928lin@gmail.com) 
# Last update at 2021/12/23.

# Future plan:
# Turn field into a class, which can contain unit, attributes, etc.

import logging
logger = logging.getLogger("HydroCNHS.dc")   

class Data_collector(object):
    def __init__(self):
        """A general data container, where each added field is a sub-container.
        """
        self.field_list = []
        pass
    def add_field(self, field, data_type={}, check_exist=True):
        """Add a field to the data collector.

        Parameters
        ----------
        field : str
            Field name.
        data_type : dict, optional
            Data type of the field (e.g., {} and [])., by default {}
        check_exist : bool, optional
            If Ture, check the given field name is not existed before adding,
            by default True
        """
        if check_exist:
            if field in self.field_list:
                logger.error("Field {} already exist in the ".format(field)+
                             "data collector. {} is not added.".format(field))
            else:
                setattr(self, field, data_type)
                self.field_list.append(field)
                logger.info("Add field {}.".format(field))
        else:
            setattr(self, field, data_type)
            self.field_list.append(field)
            logger.info("Add field {}.".format(field))
            
    def get_field(self, field):
        """Get field.

        Parameters
        ----------
        field : str
            Field name.

        Returns
        -------
        Assigned field type.
        """
        try:
            return getattr(self, field)
        except Exception as e:
            print(e)
            logger.error(e)
    def list_fields(self):
        """Print available fields in the data collector.
        """
        print(self.field_list)
    def get_dict(self):
        """Get data collector in dictionary format.

        Returns
        -------
        dict
            A dictionary contains all fields.
        """
        return self.__dict__    
    
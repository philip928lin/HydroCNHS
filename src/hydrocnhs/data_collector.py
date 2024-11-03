# Data collector module
# Here we create a basic Data_collector aiming to collect agents' outputs.
# by Chung-Yi Lin @ Lehigh University (philip928lin@gmail.com)
# Last update at 2021/12/23.

# Future plan:
# Turn field into a class, which can contain unit, attributes, etc.

import logging

logger = logging.getLogger("HydroCNHS.dc")
from copy import deepcopy
from .util import dict_to_string


class Data_collector(object):
    def __init__(self):
        """A general data container, where each added field is a sub-container.

        A field can be a dictionary or a list. It can be called as an attribute
        of a Data_collector object.
        """
        self.field_info = {}
        pass

    def add_field(self, field, data_type={}, desc=None, unit=None, check_exist=True):
        """Add a field to the data collector.

        A field can be a dictionary or a list. A Data_collector object
        cannot have duplicated field name.

        Parameters
        ----------
        field : str
            Field name. Cannot have space in a field name.
        data_type : dict, optional
            Data type of the field (e.g., {} and []), by default {}. User can
            also populate the field by directly assigning data here.
        desc : str
            Field description.
        unit :
            Unit of the field.
        check_exist : bool, optional
            If Ture, check the given field name is not existed before adding,
            by default True.
        """
        field = field.replace(" ", "_")
        if check_exist:
            if field in self.field_info.keys():
                logger.error(
                    "Field {} already exist in the ".format(field)
                    + "data collector. {} is not added.".format(field)
                )
            else:
                setattr(self, field, data_type)
                self.field_info[field] = {
                    "type": type(data_type),
                    "desc": desc,
                    "unit": unit,
                }
                logger.info("Add field {} ({}).".format(field, type(data_type)))
        else:
            setattr(self, field, data_type)
            self.field_info[field] = {
                "type": type(data_type),
                "desc": desc,
                "unit": unit,
            }
            logger.info("Add field {} ({}).".format(field, type(data_type)))

    def del_field(self, field):
        """Delete a field from the data collector.

        Parameters
        ----------
        field : str
            Field name.
        """
        if field in self.field_info.keys():
            delattr(self, field)
            self.field_info.pop(field, None)
            logger.info("Delete field {}.".format(field))
        else:
            logger.info("Field {} is not exist.".format(field))

    def get_field(self, field, copy=False):
        """Get a field.

        This function create a shortcut to access a field. Namely, changes of
        a local variable of get_field() will be accumulated back to the
        original data_collector. copy=Ture if a copy of a field is needed.

        Parameters
        ----------
        field : str
            Field name.
        copy : bool
            If true, create a copy of a field, which has seperate storage
        pointer than the original data_collector. Otherwise, return a shortcut
        of a field to the original data_collector.
        Returns
        -------
        Assigned field type.
        """
        try:
            if copy:
                return deepcopy(getattr(self, field))
            else:
                return getattr(self, field)
        except Exception as e:
            print(e)
            logger.error(e)

    def list_fields(self):
        """Print available fields in the data collector."""
        print(dict_to_string(self.field_info))

    def get_dict(self, copy=False):
        """Get data collector in dictionary format.

        Note that if copy=False, any modification on a variable assigned with
        the returned dictionary will also affect the data stored in the data
        collector object.

        Parameters
        ----------
        copy : str
            If true, a copy of dictionary will be returned, else a pointer will
            be returned. Default False.

        Returns
        -------
        dict
            A dictionary contains all fields.
        """
        if copy:
            return deepcopy(self.__dict__)
        else:
            return self.__dict__

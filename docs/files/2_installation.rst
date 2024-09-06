Installation
============
Install HydroCNHS by *pip*.

.. code-block:: python

   pip install hydrocnhs

To install the latest version (beta version) of  `HydroCNHS <https://github.com/philip928lin/HydroCNHS>`_, users can (1) install HydroCNHS by *git*.

.. code-block:: python

   pip install git+https://github.com/philip928lin/HydroCNHS.git

Or, (2) download the HydroCNHS package directly from the HydroCNHS GitHub repository. Then, install HydroCNHS from the *setup.py*.

.. code-block:: python

   # Need to move to the folder containing setup.py first.
   python setup.py install


If you fail to install HydroCNHS due to the DEAP package, first downgrade setuptools to 57 and try to install HydroCNHS again.

.. code-block:: python

   pip install setuptools==57

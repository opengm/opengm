Construct a Graphical Model
---------------------------    
A graphical model (gm) is always constructed from a sequence containing the
number of labels for all the variables.
The number of variables is given by the length of the sequence.
The type of the sequence can be a list or an 1d-numpy.ndarray

A gm can be constructed from a list in the following way::

    numberOfLabels=[2,2,2,2,2]  
    gm=opengm.graphicalModel(numberOfLabels)
    
The result is a gm with 5 variables. All of them have two labels / states.
The operator of the gm is an Adder (+).
The operator can also be specified.
The following lines will construct two graphical models, one with an Adder(+) as operator, and one with a Multiplier(*) as operator.
This time a numpy.ndarray is used as number of labels sequence::
        
    numberOfLabels=numpy.array([4,4,4] ,dtype=numpy.uint64) 
    gm1=opengm.graphicalModel(numberOfLabels,operator='adder')
    gm2=opengm.graphicalModel(numberOfLabels,operator='multiplier') 
    
The result will be two graphical models, each with 3 variables where each variable has four states.
The operator of gm1 is an Adder(+), the operator of gm2 is an Multiplier (*)


Add Factors and Functions to a Graphical Model
++++++++++++++++++++++++++++++++++++++++++++++ 

.. note::

   TODO: describe why function and factors are two different things!
.. note::

   Variable Indices must always be sorted!

.. literalinclude:: ../../examples/add_functions.py

Add Multiple Factors and Functions to a Graphical Model at once
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

.. literalinclude:: ../../examples/add_multiple_unaries.py

Save and Load a Graphical Model
-------------------------------


Save a gm::
   
   opengm.hdf5.saveGraphicalModel(gm,'path','dataset')
   
Load a gm::
   
   opengm.hdf5.loadGraphicalModel(gm,'path','dataset')
   
.. note:: 
   Currently only graphical models with which use only the explicit function can be loaded.
   If the gm has been saved from Python only the explicit function  is used and this is not
   a limitation.
   If the gm has been saved from C++ this can be a limitation.
   Within the next release of OpenGM Python this will be changed so that all default function types can
   be loaded.

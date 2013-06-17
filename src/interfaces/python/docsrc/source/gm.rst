Construct a Graphical Model
---------------------------    
A graphical model (gm) is always constructed from a sequence containing the
number of labels for all the variables.
The number of variables is given by the length of the sequence.

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

   Variable Indices must always be sorted!

+-----------------------------------------------------------+------------------------------------------------+
| Code                                                      |    Factor Graph                                |
+-----------------------------------------------------------+------------------------------------------------+
| .. literalinclude:: ../../examples/visu/chain.py          | .. image:: img/chain_non_shared.png            |  
|                                                           |    :scale: 20%                                 |     
|                                                           | .. image:: img/chain_shared.png                |  
|                                                           |    :scale: 20%                                 |     
+-----------------------------------------------------------+------------------------------------------------+
| .. literalinclude:: ../../examples/visu/grid.py           | .. image:: img/grid.png                        |  
|                                                           |    :scale: 20%                                 |        
+-----------------------------------------------------------+------------------------------------------------+
| .. literalinclude:: ../../examples/visu/triangle.py       | .. image:: img/triangle.png                    |  
|                                                           |    :scale: 20%                                 |       
+-----------------------------------------------------------+------------------------------------------------+
| .. literalinclude:: ../../examples/visu/full.py           | .. image:: img/full_non_shared.png             |  
|                                                           |    :scale: 20%                                 |     
|                                                           | .. image:: img/full_shared.png                 |  
|                                                           |    :scale: 20%                                 |     
+-----------------------------------------------------------+------------------------------------------------+


Add Functions to a Graphical Model
++++++++++++++++++++++++++++++++++++++++++++++ 

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
   

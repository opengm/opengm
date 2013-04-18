.. opengm documentation master file, created by thoesten beier
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. contents::
.. toctree::
   :maxdepth: 2
Introduction
======================
`OpenGM <http://hci.iwr.uni-heidelberg.de/opengm2/>`_  is a C++ template library for discrete factor graph models and distributive operations on these models. 
OpenGM Python exports the functionality of the C++ library OpenGM to Python. 
It includes state-of-the-art optimization and inference algorithms beyond message passing. 
OpenGM handles large models efficiently, since  functions that occur repeatedly need to be stored only once .
No restrictions are imposed on the factor graph or the operations of the model. 
The graphical model data structure, inference algorithms and different encodings of functions interoperate through well-defined interfaces. 
The binary OpenGM file format is based on the HDF5 standard and incorporates user extensions automatically. 


The mathematical foundation can be found in the `OpenGM Manual <http://hci.iwr.uni-heidelberg.de/opengm2/download/opengm-2.0.2-beta-manual.pdf>`_ .

OpenGM Python Tutorial
======================

.. toctree::
  import.rst    
  gm.rst    
  inference.rst


Examples
=======================
.. toctree::
    examples.rst   

OpenGM Python Reference Documentation
======================= 
.. toctree::
  refdocu.rst
  opengm.rst

Contact
======================
opengm (AT) hci.iwr.uni-heidelberg (DOT) de 

Authors
======================
OpenGM Authors
---------------------------    
- Bjoern Andres
    - bjoern (AT) andres (DOT) sc
    - www.andres.sc
    - Bjoern Andres,SEAS, Harvard University
- Thorsten Beier
    - thorsten.beier (AT) iwr (DOT) uni-heidelberg (DOT) de
    - Thorsten Beier, Multidimensional Image Processing Group Heidelberg Collaboratory for Image Processing (HCI), University of Heidelberg
- Joerg H. Kappes
    - kappes (AT) math (DOT) uni-heidelberg (DOT) de
    - ipa.iwr.uni-heidelberg.de/jkappes
    - Joerg Hendrik Kappes, Image & Pattern Analysis Group Heidelberg Collaboratory for Image Processing (HCI), University of Heidelberg

OpenGM Python Author(s)
---------------------------   
- Thorsten Beier
    - thorsten.beier (AT) iwr (DOT) uni-heidelberg (DOT) de
    - Thorsten Beier, Multidimensional Image Processing Group Heidelberg Collaboratory for Image Processing (HCI), University of Heidelberg
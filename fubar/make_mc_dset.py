import numpy
import opengm
from opengm import learning
import vigra
from progressbar import *
from functools import partial


getPbar


def make_mmwc_dataset(nSemanticClasses, modelSizes, edges, nodeFeatures, edgeFeatures, allowCutsWithin):

    assert len(modelSize)==len(edges)
    assert len(edges) == len(nodeFeatures)
    assert len(edges) == len(nodeFeatures)
    
    for modelSize,edge


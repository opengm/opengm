import cPickle as pickle
import numpy as np
import opengm
from opengm import learning
import h5py

# download the Pascal VOC datasets from
# 
# http://www.ais.uni-bonn.de/deep_learning/downloads.html
# 

# converter from pystruct EdgeFeatureGraphCRF to opengm learnable
fns = ['./data_train.pickle', './data_val.pickle']
ds_suffixes = ['_train']#, '_val']
ogm_dss = [None, None]
ww = [None, None]
num_samples = None
out_dir = './'
out_prefix = 'pascal_voc'

num_labels = 21

# get label weights from training set:
# loss should be weighted inversely by the number of 
# occurrences of each class in the training set    
Y = pickle.load(open(fns[0], 'r'))['Y'][:num_samples]
Y = np.hstack(Y)
Y[Y==-1]=0 # FIXME: introduce a void label, so long: make the void label background 
label_weights = 1. / np.bincount(Y)
label_weights= np.ones(27,dtype=opengm.value_type)
label_weights[np.isinf(label_weights)] = 0
label_weights *= 1. / np.sum(label_weights)

for ii, (fn, ds_suffix) in enumerate(zip(fns, ds_suffixes)):
    ds = pickle.load(open(fn, 'r'))

    # X is a list of instances of a dataset where (for EdgeFeatureGraphCRF)
    # each instance is a tuple of (unary_feats, edges, edge_feats)
    X = ds['X'][:num_samples]

    # the ground truth labels
    Y = ds['Y'][:num_samples]

    # superpixels (for reference)
    #superpixels_train = ds['superpixels'][:num_samples]

    # filenames (for reference)
    #fns_train = ds['file_names'][:num_samples]

    num_edge_feats = X[0][2].shape[1]
    num_unary_feats = num_labels * X[0][0].shape[1]
    num_weights = num_unary_feats + num_edge_feats
    # create and initialize weights
    print 'num_weights =', num_weights
    print 'num_instances =', len(X)

    ogm_dss[ii] = learning.createDataset(num_weights, numInstances=len(X))
    #ogm_ds = ogm_dss[ii]
    ww[ii] = ogm_dss[ii].getWeights()

    for idx, (x, y) in enumerate(zip(X, Y)):
        print idx
        y[y==-1]=0  # FIXME: introduce a void label, so long: make the void label background 
        unary_feats, edges, edge_feats = x
        num_vars = unary_feats.shape[0]

        states = np.ones(num_vars, dtype=opengm.label_type) * num_labels
        
        gm = opengm.gm(states, operator='adder')

        lossParam =  learning.LossParameter(lossType='hamming', labelMult=label_weights)
        lossParam.setLabelLossMultiplier(label_weights)

        # add unary factors
        weight_ids = np.arange(0, num_labels * unary_feats.shape[1]).reshape((num_labels, -1))

            
            
        # the features are different for each function instance
        # but a single feature vector is shared between all
        # labels for one particular instance.
        # The weights the same for all function instances
        # but each label has a particular set of weights
        lUnaries = learning.lUnaryFunctions(weights = ww[ii],numberOfLabels = num_labels, 
                                            features=unary_feats,weightIds = weight_ids,
                                            featurePolicy= learning.FeaturePolicy.sharedBetweenLabels)
        gm.addFactors(gm.addFunctions(lUnaries), np.arange(num_vars)) 



        # add all pairwise factors at once
        weight_ids = np.arange(num_unary_feats, num_unary_feats+num_edge_feats)
        lp = learning.lPottsFunctions(weights=ww[ii], numberOfLabels=num_labels,
                                     features=edge_feats, weightIds=weight_ids)
        gm.addFactors(gm.addFunctions(lp), edges) 


        # add the model to the dataset
        ogm_dss[ii].setInstanceWithLossParam(idx, gm, y.astype(dtype=opengm.label_type), lossParam)


    ogm_dss[ii].save(out_dir, out_prefix + ds_suffix + '_')







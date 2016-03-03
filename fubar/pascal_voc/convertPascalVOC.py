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
ds_suffixes = ['_train', '_val']

num_samples = 5 # None for all
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
label_weights[np.isinf(label_weights)] = 0
label_weights *= 1. / np.sum(label_weights)

for fn, ds_suffix in zip(fns, ds_suffixes):
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
    ogm_ds = learning.createDataset(num_weights, numInstances=len(X), loss="generalized-hamming")
    weights = ogm_ds.getWeights()

    for idx, (x, y) in enumerate(zip(X, Y)):
        y[y==-1]=0  # FIXME: introduce a void label, so long: make the void label background 
        unary_feats, edges, edge_feats = x
        num_vars = unary_feats.shape[0]

        states = np.ones(num_vars, dtype=opengm.index_type) * num_labels
        
        gm = opengm.graphicalModel(states, operator='adder')

        lossParam = learning.GeneralizedHammingLossParameter()
        lossParam.setLabelLossMultiplier(np.array(label_weights))

        # add unary factors
        weight_ids = np.arange(0, num_labels * unary_feats.shape[1]).reshape((num_labels, -1))
        for feat_idx, unary_feat in enumerate(unary_feats):
            # make that each label sees all features, but use their own weights
            unary_feat_array = np.repeat(unary_feat.reshape((-1,1)), num_labels, axis=1)
            f = learning.lUnaryFunction(weights, num_labels, unary_feat_array, weight_ids)
            var_idxs = np.array([feat_idx], dtype=np.uint64)
            fid = gm.addFunction(f)
            gm.addFactor(fid, var_idxs)
        #var_idxs = np.arange(0, num_vars, dtype=np.uint64)
        #gm.addFactors(fids, var_idxs)

        # add pairwise factors
        for edge, edge_feat in zip(edges, edge_feats):
            var_idxs = edge.astype(opengm.index_type)
            weight_ids = np.arange(num_unary_feats, num_unary_feats+num_edge_feats, dtype=opengm.index_type)
            f = opengm.LPottsFunction(weights=weights, numberOfLabels=num_labels,
                                      weightIds=weight_ids, features=edge_feat)
            fid = gm.addFunction(f)
            gm.addFactor(fid, var_idxs)

        print idx, y.shape, lossParam
        ogm_ds.setInstanceWithLossParam(idx, gm, y.astype(dtype=opengm.label_type), lossParam)

    ogm_ds.save(out_dir, out_prefix + ds_suffix + '_')


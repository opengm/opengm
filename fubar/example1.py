import opengm
import opengm.learning



datasetA = opengm.learning.createDataset(loss='hamming')
datasetB = opengm.learning.createDataset(loss='generalized-hamming')

print datasetA, datasetB

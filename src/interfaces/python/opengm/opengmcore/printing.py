from shapewalker import shapeWalker

def prettyValueTable(valueTable,vis=None):
    try:
        from prettytable import PrettyTable
    except:
        raise ImportError("following imports failed :\nimport PrettyTable from prettytabble")

    shape=valueTable.shape
    order=len(shape)
    nRows = valueTable.size

    if vis is None:
        visName=["V_%s"%(str(v),) for v in xrange(order)]
    else:
        visName=["V_%s"%(str(v),)  for v in vis]
    visName.append("Value")

    x = PrettyTable(visName)
    for labeling in shapeWalker(shape):
        x.add_row(labeling + [valueTable[labeling]])

    return x
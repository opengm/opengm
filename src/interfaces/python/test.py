import numpy
import opengm
import os
import sys

class TestAllExampes:
    def test_run(self):
        for r, d, f in os.walk("examples"):
            for files in f:
                if files.endswith(".py"):
                    if(not str(files).endswith('gui.py')):
                        pass
                        # execfile(filePath)
                        # subprocess.call([filePath, arg1, arg2])
                        #execfile("examples/" + files)


def lenOfGen(gen):
    return len([i for i in gen])


def generate_grid(dimx, dimy, labels, beta1, beta2, operator="adder"):
    nos = numpy.ones(dimx * dimy, dtype=numpy.uint64) * labels
    gm = opengm.gm(nos, operator, 0)

    for vi in range(dimx * dimy):
        f1 = numpy.random.random((labels,)).astype(numpy.float32) * 0.6 + 0.2
        assert len(f1.shape) == 1
        assert f1.shape[0] == labels
        fid1 = gm.addFunction(f1)
        gm.addFactor(fid1, (vi,))
    f2 = numpy.ones([labels, labels], dtype=numpy.float32)
    for l in range(labels):
        f2[l, l] = beta1
    fid2 = gm.addFunction(f2)
    for y in range(dimy):
        for x in range(dimx):
            if x + 1 < dimx:
                vis = [x + y * dimx, x + 1 + y * dimx]
                assert vis.sort is not None
                vis.sort
                gm.addFactor(fid2, vis)
            if y + 1 < dimy:
                vis = [x + y * dimx, x + (y + 1) * dimx]
                vis.sort()
                gm.addFactor(fid2, vis)
    return gm


def makeGrid(dimx, dimy, labels, beta, acc="min"):
    nos = numpy.ones(dimx * dimy, dtype=numpy.uint64) * labels
    if acc == "min":
        gm = opengm.adder.GraphicalModel(nos)
    else:
        gm = opengm.multiplier.GraphicalModel(nos)
    for vi in range(dimx * dimy):
        f1 = numpy.random.random((labels,)).astype(numpy.float32)
        fid1 = gm.addFunction(f1)
        gm.addFactor(fid1, (vi,))
    f2 = numpy.ones(labels * labels, dtype=numpy.float32).reshape(
        labels, labels) * beta
    for l in range(labels):
        f2[l, l] = 0
    fid2 = gm.addFunction(f2)
    for y in range(dimy):
        for x in range(dimx):
            if x + 1 < dimx - 1:
                gm.addFactor(fid2, [x + y * dimx, x + 1 + y * dimx])
            if y + 1 < dimy - 1:
                gm.addFactor(fid2, [x + y * dimx, x + (y + 1) * dimx])
    return gm


def checkSolution(gm, argOpt, arg, acc="min", tolerance=None, check=True):
    valOpt = gm.evaluate(argOpt)
    val = gm.evaluate(arg)
    numtol = 0.00000000001
    if check:
        if acc == "min":
            if tolerance is None:
                tol = numtol
                assert(val - tol <= valOpt)
            else:
                tol = valOpt * tolerance
                assert(val - tol <= valOpt)
        if acc == "max":
            if tolerance is None:
                tol = numtol
                assert(val - tol >= valOpt)
            else:
                tol = valOpt * tolerance + numtol
                assert(val - tol >= valOpt)


def checkInference(gm, solver, argOpt, optimal=False, tolerance=None,
                   acc="min"):
    solver.infer()
    arg = solver.arg()
    checkSolution(gm, argOpt, arg, acc, tolerance, optimal)


class TestUtilities:
    def test_vector(self):
        assert(True)

    def test_enums(self):
        assert(True)

    def test_is_build_in_simple_parameter(self):
        class MyClass(object):
            def __init__(self):
                pass
        assert(not opengm._to_native_converter.is_build_in_simple_parameter(
            classType=MyClass))
        assert(not opengm._to_native_converter.is_build_in_simple_parameter(
            instanceType=MyClass()))
        assert(opengm._to_native_converter.is_build_in_simple_parameter(
            classType=bool))
        assert(opengm._to_native_converter.is_build_in_simple_parameter(
            instanceType=bool()))
        assert(opengm._to_native_converter.is_build_in_simple_parameter(
            instanceType=1))
        assert(opengm._to_native_converter.is_build_in_simple_parameter(
            instanceType=1.0))
        assert(opengm._to_native_converter.is_build_in_simple_parameter(
            instanceType='1.0'))
        simple_types = [int, long, float, bool, str]
        for st in simple_types:
            assert(opengm._to_native_converter.is_build_in_simple_parameter(
                classType=st))
            assert(opengm._to_native_converter.is_build_in_simple_parameter(
                instanceType=st()))

    def test_is_tribool(self):
        assert(opengm._to_native_converter.is_tribool(
            classType=opengm.Tribool))
        assert(opengm._to_native_converter.is_tribool(
            instanceType=opengm.Tribool(0)))
        assert(not opengm._to_native_converter.is_tribool(classType=bool))
        assert(not opengm._to_native_converter.is_tribool(
            instanceType=True))


class TestSparseFunction:
    def test_constructor(self):
        functions = []
        functions.append(opengm.SparseFunction([2, 3, 4], 1))
        functions.append(opengm.SparseFunction((2, 3, 4), 1))
        for f in functions:
            assert(f.defaultValue == 1)
            assert(f.dimension == 3)
            assert(f.shape[0] == 2)
            assert(f.shape[1] == 3)
            assert(f.shape[2] == 4)
            assert(len(f.shape) == 3)
            assert(f.size == 2 * 3 * 4)

    def test_key_to_coordinate(self):
        f = opengm.SparseFunction([2, 3, 4], 0)
        c = numpy.ones(3, dtype=numpy.uint64)
        for key, cTrue in enumerate(opengm.shapeWalker(f.shape)):
            f.keyToCoordinate(key, c)
            for ct, cOwn in zip(cTrue, c):
                assert ct == cOwn

    def test_dense_assignment(self):
        f = opengm.SparseFunction()
        fDense = numpy.zeros([3, 4])
        fDense[0, 1] = 1
        fDense[0, 2] = 2
        f.assignDense(fDense, 0)
        assert f.dimension == 2
        assert f.shape[0] == 3
        assert f.shape[1] == 4
        assert f[[0, 0]] == 0
        assert f[[0, 1]] == 1
        assert f[[0, 2]] == 2
        for c in opengm.shapeWalker(f.shape):
            assert f[c] == fDense[c[0], c[1]]
        assert len(f.container) == 2


class TestFunctions:
    def test_potts(self):
        nl1 = numpy.ones(10, dtype=numpy.uint64) * 2
        nl2 = numpy.ones(5, dtype=numpy.uint64) * 3
        veq = numpy.zeros(1, dtype=numpy.float32)
        vnew = numpy.arange(0, 10, dtype=numpy.float32)

        pottsFunctionVector = opengm.PottsFunctionVector(nl1, nl2, veq, vnew)

        assert len(pottsFunctionVector) == 10

        for i, f in enumerate(pottsFunctionVector):
            assert f.shape[0] == 2
            assert f.shape[1] == 3
            assert f[0, 0] == 0
            assert f[[1, 1]] == 0
            assert f[[0, 1]] == vnew[i]


class TestGm:
    def test_vectorized_factors(self):
        gm = generate_grid(
            dimx=2, dimy=2, labels=2, beta1=0.2, beta2=0.6, operator='adder')
        # res=gm.isSubmodular(range(gm.numberOfFactors))

        def foo(factor):
            return factor.isSubmodular()

        # f=functools.partial(gm.isSubmodular,self=gm)
        res = gm.vectorizedFactorFunction(
            gm.factorClass.isSubmodular, range(gm.numberOfFactors))
        assert len(res) == gm.numberOfFactors

        for f, r in zip(gm.factors(), res):
            assert r == f.isSubmodular()

        res = gm.vectorizedFactorFunction(gm.factorClass.isSubmodular)
        assert len(res) == gm.numberOfFactors

        for f, r in zip(gm.factors(), res):
            assert r == f.isSubmodular()

    def test_constructor_generic(self):
        def mygen():
            yield 2
            yield 3
            yield 4

        nos_list = [
            numpy.arange(2, 5, dtype=numpy.uint64),
            [2, 3, 4],
            (2, 3, 4),
            (x for x in xrange(2, 5)),
            mygen(),
            opengm.IndexVector(x for x in xrange(2, 5))
        ]
        for i, nos in enumerate(nos_list):
            if(type(nos) != type(mygen())):
                pass
                # assert(len(nos)==3)
            gm = opengm.gm(nos, operator='adder')
            assert(gm.numberOfVariables == 3)
            assert(gm.numberOfLabels(0) == 2)
            assert(gm.numberOfLabels(1) == 3)
            assert(gm.numberOfLabels(2) == 4)
            assert(gm.space().numberOfVariables == 3)
            assert(gm.space()[0] == 2)
            assert(gm.space()[1] == 3)
            assert(gm.space()[2] == 4)

        nos_list = [
            numpy.arange(2, 5, dtype=numpy.uint64),
            [2, 3, 4],
            (2, 3, 4),
            (x for x in xrange(2, 5)),
            mygen(),
            opengm.IndexVector(x for x in xrange(2, 5))
        ]
        for i, nos in enumerate(nos_list):
            if(type(nos) != type(mygen())):
                pass  # assert(len(nos)==3)
            gm = opengm.adder.GraphicalModel()
            gm.assign(nos)
            assert(gm.numberOfVariables == 3)
            assert(gm.numberOfLabels(0) == 2)
            assert(gm.numberOfLabels(1) == 3)
            assert(gm.numberOfLabels(2) == 4)
            assert(gm.space().numberOfVariables == 3)
            assert(gm.space()[0] == 2)
            assert(gm.space()[1] == 3)
            assert(gm.space()[2] == 4)

    def test_add_factors_generic(self):
        def mygen():
            yield 0
            yield 1
        gm = opengm.gm([2, 4])
        f = opengm.PottsFunction([2, 4], 0.0, 1.0)
        fid = gm.addFunction(f)
        vis_list = [
            [0, 1],
            (0, 1),
            (x for x in xrange(2)),
            mygen(),
            opengm.IndexVector(x for x in xrange(0, 2)),
            numpy.arange(0, 2, dtype=numpy.uint64)
        ]
        for i, vis in enumerate(vis_list):
            fIndex = gm.addFactor(fid, vis)
            assert(gm.numberOfFactors == i + 1)
            assert(fIndex == i)
            assert(gm[fIndex].numberOfVariables == 2)
            assert(gm[fIndex].shape[0] == 2)
            assert(gm[fIndex].shape[1] == 4)
            assert(gm[fIndex].variableIndices[0] == 0)
            assert(gm[fIndex].variableIndices[1] == 1)

    def test_add_function(self):
        numberOfStates = [2, 3, 4]
        gm = opengm.adder.GraphicalModel(numberOfStates)
        f1 = numpy.ones(6 * 4, numpy.float32)
        p = 1
        for i in range(2 * 3 * 4):
            f1[i] = i
            p *= i
        f1 = f1.reshape(2, 3, 4)
        idf = gm.addFunction(f1)
        gm.addFactor(idf, (0, 1, 2))

        assert(gm[0].min() == 0)
        assert(gm[0].max() == 2 * 3 * 4 - 1)
        assert(gm[0].sum() == sum(range(2 * 3 * 4)))
        assert(gm[0].product() == p)

        nf1 = gm[0].asNumpy()
        assert(len(f1.shape) == len(nf1.shape))
        for i in range(len(f1.shape)):
            assert(f1.shape[i] == nf1.shape[i])

        for k in range(f1.shape[2]):
            for j in range(f1.shape[1]):
                for i in range(f1.shape[0]):
                    assert(gm[0][numpy.array(
                        [i, j, k], dtype=numpy.uint64)] == f1[i, j, k])
                    assert(gm[0][(i, j, k)] == f1[i, j, k])
                    assert(gm[0][(i, j, k)] == nf1[i, j, k])

    def test_add_multiple_functions(self):
        nVar = 10
        nLabels = 2
        for nFunctions in [1, 10]:
            for order in [1, 2, 3, 4]:
                gm = opengm.gm([nLabels] * nVar)
                # add functionS
                fShape = [nFunctions] + [nLabels] * order
                f = numpy.ones(fShape, dtype=opengm.value_type).reshape(-1)
                f[:] = numpy.random.rand(f.size)[:]
                f = f.reshape(fShape)
                fids = gm.addFunctions(f)
                # assertions
            assert len(fids) == nFunctions

    def test_add_multiple_functions_with_map(self):

        gm = opengm.gm([2] * 10)

        def add_a_function(w):
            return gm.addFunction(opengm.differenceFunction(shape=[2, 2],
                                                            weight=w))

        weights = [0.2, 0.3, 0.4]
        fidList = map(add_a_function, weights)

        assert isinstance(fidList, list)
        assert len(fidList) == len(weights)

        gm.addFactors(fidList, [[0, 1], [1, 2], [3, 4]])

    def test_evaluate(self):
        numberOfStates = [2, 2, 2, 2]
        gm = opengm.adder.GraphicalModel(numberOfStates)
        f1 = numpy.ones(2, dtype=numpy.float32).reshape(2)
        f2 = numpy.ones(4, dtype=numpy.float32).reshape(2, 2)
        for i in range(3):
            gm.addFactor(gm.addFunction(f1), [i])
        for i in range(2):
            gm.addFactor(gm.addFunction(f2), [i, i + 1])
        sequenceList = [0, 1, 0, 1]
        valueList = gm.evaluate(sequenceList)
        assert(float(valueList) == float(gm.numberOfFactors))
        sequenceNumpy = numpy.array([0, 1, 0, 1], dtype=numpy.uint64)
        valueNumpy = gm.evaluate(sequenceNumpy)
        assert(float(valueNumpy) == float(gm.numberOfFactors))
        assert(float(valueNumpy) == float(valueList))

    def test_variables_generator(self):
        nos = [2, 3, 4, 5, 6]
        gm = opengm.adder.GraphicalModel(nos)
        truevis = [0, 1, 2, 3, 4]
        myvis = [vi for vi in gm.variables()]
        assert (len(truevis) == len(myvis))
        for a, b in zip(truevis, myvis):
            assert a == b
        truevis = [2]
        myvis = [vi for vi in gm.variables(labels=4)]
        assert (len(truevis) == len(myvis))
        for a, b in zip(truevis, myvis):
            assert a == b

        truevis = [1, 2, 3, 4]
        myvis = [vi for vi in gm.variables(minLabels=3)]
        assert (len(truevis) == len(myvis))
        for a, b in zip(truevis, myvis):
            assert a == b

        truevis = [0, 1, 2]
        myvis = [vi for vi in gm.variables(maxLabels=4)]
        assert (len(truevis) == len(myvis))
        for a, b in zip(truevis, myvis):
            assert a == b

        truevis = [1, 2]
        myvis = [vi for vi in gm.variables(minLabels=3, maxLabels=4)]
        assert (len(truevis) == len(myvis))
        for a, b in zip(truevis, myvis):
            assert a == b

    def test_factor_generators(self):
        numberOfStates = [2, 2, 2, 2, 2]
        gm = opengm.adder.GraphicalModel(numberOfStates)
        functions = [numpy.ones(2, dtype=numpy.float32).reshape(2),
                     numpy.ones(4, dtype=numpy.float32).reshape(2, 2),
                     numpy.ones(8, dtype=numpy.float32).reshape(2, 2, 2),
                     numpy.ones(16, dtype=numpy.float32).reshape(2, 2, 2, 2),
                     numpy.ones(32,
                                dtype=numpy.float32).reshape(2, 2, 2, 2, 2)]

        for f in functions:
            fid = gm.addFunction(f)
            vis = [i for i in xrange(len(f.shape))]
            gm.addFactor(fid, vis)

        assert gm.numberOfVariables == 5

        # test generators
        for i, factor in enumerate(gm.factors(), start=1):
            assert factor.numberOfVariables == i

        for i, fId in enumerate(gm.factorIds()):
            assert fId == i

        for i, (factor, fId) in enumerate(gm.factorsAndIds()):
            assert fId == i
            assert factor.numberOfVariables == i + 1

        # with order
        for order in xrange(1, 6):
            gens = []
            gens.append(gm.factors(order=order))
            gens.append(gm.factorIds(order=order))
            gens.append(gm.factorsAndIds(order=order))
            for gen in gens:
                assert lenOfGen(gen) == 1
            gens = []
            gens.append(gm.factors(order=order))
            gens.append(gm.factorIds(order=order))
            gens.append(gm.factorsAndIds(order=order))
            for factor in gens[0]:
                assert factor.numberOfVariables == order
            for fId in gens[1]:
                assert gm[fId].numberOfVariables == order
            for factor, fId in gens[2]:
                assert factor.numberOfVariables == order
                assert gm[fId].numberOfVariables == order

        # with order
        for order in xrange(1, 6):
            orderSets = [set(), set(), set()]
            gens = [gm.factors(minOrder=order), gm.factorIds(
                minOrder=order), gm.factorsAndIds(minOrder=order)]
            assert(len(gens) == 3)
            for gen in gens:
                print "len assert"
                assert lenOfGen(gen) == 6 - order
            gens = [gm.factors(minOrder=order), gm.factorIds(
                minOrder=order), gm.factorsAndIds(minOrder=order)]
            for factor in gens[0]:
                assert factor.numberOfVariables >= order
                orderSets[0].add(factor.numberOfVariables)
            for fId in gens[1]:
                assert gm[fId].numberOfVariables >= order
                orderSets[1].add(gm[fId].numberOfVariables)
            for factor, fId in gens[2]:
                assert factor.numberOfVariables >= order
                assert gm[fId].numberOfVariables >= order
                orderSets[2].add(factor.numberOfVariables)
            for oset in orderSets:
                assert len(oset) == 6 - order

        for order in xrange(2, 6):
            orderSets = [set(), set(), set()]
            gens = [gm.factors(maxOrder=order), gm.factorIds(
                maxOrder=order), gm.factorsAndIds(maxOrder=order)]
            assert(len(gens) == 3)
            for gen in gens:
                print "len assert"
                assert lenOfGen(gen) == order
            gens = [gm.factors(maxOrder=order), gm.factorIds(
                maxOrder=order), gm.factorsAndIds(maxOrder=order)]
            for factor in gens[0]:
                assert factor.numberOfVariables <= order
                orderSets[0].add(factor.numberOfVariables)
            for fId in gens[1]:
                assert gm[fId].numberOfVariables <= order
                orderSets[1].add(gm[fId].numberOfVariables)
            for factor, fId in gens[2]:
                assert factor.numberOfVariables <= order
                assert gm[fId].numberOfVariables <= order
                orderSets[2].add(factor.numberOfVariables)
            for oset in orderSets:
                assert len(oset) == order

        for order in xrange(1, 6):
            orderSets = [set(), set(), set()]
            gens = [gm.factors(minOrder=order, maxOrder=4),
                    gm.factorIds(minOrder=order, maxOrder=4),
                    gm.factorsAndIds(minOrder=order, maxOrder=4)]
            assert(len(gens) == 3)
            for gen in gens:
                print "len assert"
                assert lenOfGen(gen) == 6 - order - 1
            gens = [gm.factors(minOrder=order, maxOrder=4),
                    gm.factorIds(minOrder=order, maxOrder=4),
                    gm.factorsAndIds(minOrder=order, maxOrder=4)]
            for factor in gens[0]:
                assert (factor.numberOfVariables >= order
                        and factor.numberOfVariables <= 4)
                orderSets[0].add(factor.numberOfVariables)
            for fId in gens[1]:
                assert gm[fId].numberOfVariables >= order and gm[
                    fId].numberOfVariables <= 4
                orderSets[1].add(gm[fId].numberOfVariables)
            for factor, fId in gens[2]:
                assert(factor.numberOfVariables >= order
                       and factor.numberOfVariables <= 4)
                assert gm[fId].numberOfVariables >= order and gm[
                    fId].numberOfVariables <= 4
                orderSets[2].add(factor.numberOfVariables)
            for oset in orderSets:
                assert len(oset) == 6 - order - 1


class TestFactor:
    def test_factor_shape(self):
        numberOfStates = [2, 3, 4]
        gm = opengm.adder.GraphicalModel(numberOfStates)
        f1 = numpy.ones(6 * 4, numpy.float32).reshape(2, 3, 4)
        idf = gm.addFunction(f1)
        gm.addFactor(idf, (0, 1, 2))
        nf1 = gm[0].asNumpy()  # not used?
        for i in range(3):
            assert(gm[0].shape[i] == numberOfStates[i])
            assert(gm[0].shape.asNumpy()[i] == numberOfStates[i])
            assert(gm[0].shape.asList()[i] == numberOfStates[i])
            assert(gm[0].shape.asTuple()[i] == numberOfStates[i])

    def test_factor_vi(self):
        numberOfStates = [2, 3, 4]
        gm = opengm.adder.GraphicalModel(numberOfStates)
        f1 = numpy.ones(6 * 4, numpy.float32).reshape(2, 3, 4)
        idf = gm.addFunction(f1)
        gm.addFactor(idf, (0, 1, 2))
        nf1 = gm[0].asNumpy()  # not used?
        for i in range(3):
            assert(gm[0].variableIndices[i] == i)
            assert(gm[0].variableIndices.asNumpy()[i] == i)
            assert(gm[0].variableIndices.asList()[i] == i)
            assert(gm[0].variableIndices.asTuple()[i] == i)

    def test_factor_properties(self):
        numberOfStates = [2, 2, 2, 2]
        gm = opengm.adder.GraphicalModel(numberOfStates)
        assert(gm.space().numberOfVariables == 4)
        assert(gm.numberOfFactors == 0)
        f1 = numpy.array([2, 3], numpy.float32)
        f2 = numpy.array([1, 2, 3, 4], numpy.float32).reshape(2, 2)
        if1 = gm.addFunction(f1)
        if2 = gm.addFunction(f2)
        gm.addFactor(if1, (0,))
        gm.addFactor(if2, (0, 1))
        nf0 = gm[0].asNumpy()
        nf1 = gm[1].asNumpy()
        for i in range(f1.shape[0]):
            assert(nf0[i] == gm[0][(i,)])
            assert(nf0[i] == f1[i])
        for i in range(f2.shape[0]):
            for j in range(f2.shape[1]):
                assert(nf1[i, j] == gm[1][(i, j)])
                assert(nf1[i, j] == f2[i, j])
        assert(gm[0].min() == 2)
        assert(gm[0].max() == 3)
        assert(gm[0].sum() == 5)
        assert(gm[0].product() == 6)
        assert(gm[0][(0,)] == 2)
        assert(gm[0][(1,)] == 3)
        assert(gm[1].min() == 1)
        assert(gm[1].max() == 4)
        assert(gm[1].sum() == 1 + 2 + 3 + 4)
        assert(gm[1].product() == 1 * 2 * 3 * 4)


def genericSolverCheck(solverClass, params, gms, semiRings):

    for operator, accumulator in semiRings:
        for gmGen in gms:
            gm = gmGen[operator]
            for param in params:

                # start inference
                solver = solverClass(
                    gm=gm, accumulator=accumulator, parameter=param)
                solver.infer()
                arg = solver.arg()  # no used?


class Test_Inference():
    def __init__(self):
        self.gridGm = {
            'adder': generate_grid(dimx=2, dimy=2, labels=2, beta1=0.1,
                                   beta2=0.2, operator='adder'),
            'multiplier': generate_grid(dimx=2, dimy=2, labels=2, beta1=0.1,
                                        beta2=0.2, operator='multiplier'),
        }

        self.gridGm3 = {
            'adder': generate_grid(dimx=3, dimy=2, labels=3, beta1=0.1,
                                   beta2=0.2, operator='adder'),
            'multiplier': generate_grid(dimx=3, dimy=2, labels=3, beta1=0.1,
                                        beta2=0.2, operator='multiplier'),
        }
        self.chainGm = {
            'adder': generate_grid(dimx=4, dimy=1, labels=2, beta1=0.1,
                                   beta2=0.2, operator='adder'),
            'multiplier': generate_grid(dimx=4, dimy=1, labels=2, beta1=0.1,
                                        beta2=0.2, operator='multiplier')
        }
        self.chainGm3 = {
            'adder': generate_grid(dimx=4, dimy=1, labels=3, beta1=0.1,
                                   beta2=0.2, operator='adder'),
            'multiplier': generate_grid(dimx=4, dimy=1, labels=3, beta1=0.1,
                                        beta2=0.2, operator='multiplier')
        }

        self.all = [('adder', 'minimizer'), ('adder', 'maximizer'), (
            'multiplier', 'minimizer'), ('multiplier', 'maximizer')]
        self.minSum = [('adder', 'minimizer')]
        self.minSumMaxProd = [('adder', 'minimizer'), (
            'multiplier', 'maximizer')]

    def test_bruteforce(self):
        solverClass = opengm.inference.Bruteforce
        params = [None, opengm.InfParam()]
        genericSolverCheck(solverClass, params=params,
                           gms=[self.gridGm, self.chainGm, self.gridGm3,
                                self.chainGm3],
                           semiRings=self.all)

    def test_icm_fast(self):
        solverClass = opengm.inference.AStar
        params = [None, opengm.InfParam(heuristic='fast')]
        genericSolverCheck(solverClass, params=params,
                           gms=[self.gridGm, self.chainGm, self.gridGm3,
                                self.chainGm3],
                           semiRings=self.all)

    def test_icm(self):
        solverClass = opengm.inference.Icm
        params = [None, opengm.InfParam(moveType='variable'), opengm.InfParam(
            moveType='factor'), opengm.InfParam()]
        genericSolverCheck(solverClass, params=params,
                           gms=[self.gridGm, self.chainGm, self.gridGm3,
                                self.chainGm3],
                           semiRings=self.all)

    def test_lazyflipper(self):
        solverClass = opengm.inference.LazyFlipper
        params = [None, opengm.InfParam(
            maxSubgraphSize=2), opengm.InfParam()]
        genericSolverCheck(solverClass, params=params,
                           gms=[self.gridGm, self.chainGm, self.gridGm3,
                                self.chainGm3],
                           semiRings=self.all)

    def test_loc(self):
        solverClass = opengm.inference.Loc
        params = [None, opengm.InfParam(
            phi=0.5), opengm.InfParam(phi=0.5, maxRadius=10, steps=100)]
        genericSolverCheck(solverClass, params=params,
                           gms=[self.gridGm, self.chainGm, self.gridGm3,
                                self.chainGm3],
                           semiRings=self.all)

    def test_dualdecompostion_subgradient(self):
        solverClass = opengm.inference.DualDecompositionSubgradient
        params = [opengm.InfParam()]
        try:
            genericSolverCheck(solverClass, params=params,
                               gms=[self.gridGm, self.gridGm3, self.chainGm,
                                    self.chainGm3],
                               semiRings=self.minSum)
        except RuntimeError as detail:
            raise RuntimeError("Error In C++ Impl. of "
                               "DualDecompositionSubgradient:\n\nReason: %s"
                               % (str(detail),))

    def test_dualdecompostion_subgradient_dynamic_programming(self):
        solverClass = opengm.inference.DualDecompositionSubgradient
        params = [opengm.InfParam(
            subInference='dynamic-programming', subInfParam=opengm.InfParam()),
            opengm.InfParam(subInference='dynamic-programming',
                            decompositionId='tree',
                            subInfParam=opengm.InfParam())
        ]
        try:
            genericSolverCheck(solverClass, params=params,
                               gms=[self.gridGm, self.gridGm3, self.chainGm,
                                    self.chainGm3],
                               semiRings=self.minSum)
        except RuntimeError as detail:
            raise RuntimeError("Error In C++ Impl. of "
                               "DualDecompositionSubgradient:\n\nReason: %s"
                               % (str(detail),))

    def test_dualdecompostion_subgradient_graph_cut(self):
        solverClass = opengm.inference.DualDecompositionSubgradient
        params = [opengm.InfParam(subInference='graph-cut',
                                  decompositionId='blocks',
                                  subInfParam=opengm.InfParam())]
        try:
            genericSolverCheck(solverClass, params=params,
                               gms=[self.gridGm, self.chainGm],
                               semiRings=self.minSum)
        except RuntimeError as detail:
            raise RuntimeError("Error In C++ Impl. of "
                               "DualDecompositionSubgradient:\n\nReason: %s" %
                               (str(detail),))

    def test_gibbs(self):
        solverClass = opengm.inference.Gibbs
        params = [opengm.InfParam(steps=10000)]
        genericSolverCheck(solverClass, params=params,
                           gms=[self.gridGm, self.chainGm, self.gridGm3,
                                self.chainGm3],
                           semiRings=self.minSumMaxProd)

    def test_bp(self):
        solverClass = opengm.inference.BeliefPropagation
        params = [opengm.InfParam(steps=10)]
        genericSolverCheck(solverClass, params=params,
                           gms=[self.gridGm, self.chainGm, self.gridGm3,
                                self.chainGm3],
                           semiRings=self.minSumMaxProd)

    def test_trwbp(self):
        solverClass = opengm.inference.TreeReweightedBp
        params = [opengm.InfParam(steps=10)]
        genericSolverCheck(solverClass, params=params,
                           gms=[self.gridGm, self.chainGm, self.gridGm3,
                                self.chainGm3],
                           semiRings=self.minSumMaxProd)

    def test_trws_external(self):
        if opengm.configuration.withTrws:
            solverClass = opengm.inference.TrwsExternal
            params = [None, opengm.InfParam(),
                      opengm.InfParam(steps=100, energyType='view'),
                      opengm.InfParam(steps=1, energyType='tables')]
            genericSolverCheck(solverClass, params=params,
                               gms=[self.gridGm, self.chainGm, self.gridGm3,
                                    self.chainGm3],
                               semiRings=self.minSum)

    def test_graphcut(self):
        solverClass = opengm.inference.GraphCut
        params = [None, opengm.InfParam(),
                  opengm.InfParam(minStCut='boost-kolmogorov'),
                  opengm.InfParam(minStCut='push-relabel')]
        if opengm.configuration.withMaxflow:
            params.append(opengm.InfParam(minStCut='kolmogorov'))
        genericSolverCheck(solverClass, params=params,gms=[self.gridGm, self.chainGm], semiRings=self.minSum)

    def test_graphcut_maxflow_ibfs(self):
        if opengm.configuration.withMaxflowIbfs :
            solverClass = opengm.inference.GraphCut
            params=[ opengm.InfParam(minStCut='ibfs') ]
            genericSolverCheck(solverClass, params=params,gms=[self.gridGm, self.chainGm], semiRings=self.minSum)

    def test_qpbo_external(self):
        if opengm.configuration.withQpbo:
            solverClass = opengm.inference.QpboExternal
            params = [None, opengm.InfParam(),
                      opengm.InfParam(useProbeing=True),
                      opengm.InfParam(strongPersistency=True),
                      opengm.InfParam(useImproveing=True)]
            genericSolverCheck(solverClass, params=params,
                               gms=[self.gridGm, self.chainGm],
                               semiRings=self.minSum)

    def test_alpha_beta_swap(self):
        solverClass = opengm.inference.AlphaBetaSwap
        params = [None, opengm.InfParam(steps=10),
                  opengm.InfParam(minStCut='boost-kolmogorov', steps=10),
                  opengm.InfParam(minStCut='push-relabel', steps=10)]
        if opengm.configuration.withMaxflow:
            params.append(opengm.InfParam(minStCut='kolmogorov', steps=10))
        genericSolverCheck(solverClass, params=params, gms=[
                           self.gridGm3, self.chainGm3], semiRings=self.minSum)

    def test_alpha_expansion(self):
        solverClass = opengm.inference.AlphaExpansion
        params = [None, opengm.InfParam(steps=10),
                  opengm.InfParam(minStCut='boost-kolmogorov', steps=10),
                  opengm.InfParam(minStCut='push-relabel', steps=10)]
        if opengm.configuration.withMaxflow:
            params.append(opengm.InfParam(minStCut='kolmogorov', steps=10))
        genericSolverCheck(solverClass, params=params, gms=[
                           self.gridGm3, self.chainGm3], semiRings=self.minSum)

    def test_lpcplex(self):
        if opengm.configuration.withCplex:
            solverClass = opengm.inference.LpCplex
            params = [None, opengm.InfParam(),
                      opengm.InfParam(integerConstraint=True),
                      opengm.InfParam(integerConstraint=False)]
            genericSolverCheck(solverClass, params=params,
                               gms=[self.gridGm, self.chainGm, self.gridGm3,
                                    self.chainGm3],
                               semiRings=self.minSum)

    ################################
    # LIB DAI
    ################################
    def test_libdai_bp(self):
        if opengm.configuration.withLibdai:
            solverClass = opengm.inference.BeliefPropagationLibDai
            params = [None, opengm.InfParam(), opengm.InfParam(
                updateRule='parall'), opengm.InfParam(updateRule='seqrnd')]
            genericSolverCheck(solverClass, params=params,
                               gms=[self.gridGm, self.chainGm, self.gridGm3,
                                    self.chainGm3],
                               semiRings=self.minSum)

    def test_libdai_fractional_bp(self):
        if opengm.configuration.withLibdai:
            solverClass = opengm.inference.FractionalBpLibDai
            params = [None, opengm.InfParam(), opengm.InfParam(
                updateRule='parall'), opengm.InfParam(updateRule='seqrnd')]
            genericSolverCheck(solverClass, params=params,
                               gms=[self.gridGm, self.chainGm, self.gridGm3,
                                    self.chainGm3],
                               semiRings=self.minSum)

    def test_libdai_trw_bp(self):
        if opengm.configuration.withLibdai:
            solverClass = opengm.inference.TreeReweightedBpLibDai
            params = [None, opengm.InfParam(),
                      opengm.InfParam(updateRule='parall'),
                      opengm.InfParam(updateRule='seqrnd', ntrees=2)]
            genericSolverCheck(solverClass, params=params,
                               gms=[self.gridGm, self.chainGm, self.gridGm3,
                                    self.chainGm3],
                               semiRings=self.minSum)

    def test_libdai_gibbs(self):
        if opengm.configuration.withLibdai:
            solverClass = opengm.inference.GibbsLibDai
            params = [None, opengm.InfParam(),
                      opengm.InfParam(steps=100)]
            genericSolverCheck(solverClass, params=params,
                               gms=[self.gridGm, self.chainGm, self.gridGm3,
                                    self.chainGm3],
                               semiRings=self.minSum)

    def test_libdai_junction_tree(self):
        if opengm.configuration.withLibdai:
            solverClass = opengm.inference.JunctionTreeLibDai
            params = [None, opengm.InfParam()]
            genericSolverCheck(solverClass, params=params,
                               gms=[self.gridGm, self.chainGm, self.gridGm3,
                                    self.chainGm3],
                               semiRings=self.minSum)

    def test_libdai_decimation(self):
        if opengm.configuration.withLibdai:
            solverClass = opengm.inference.DecimationLibDai
            params = [None, opengm.InfParam()]
            genericSolverCheck(solverClass, params=params,
                               gms=[self.gridGm, self.chainGm, self.gridGm3,
                                    self.chainGm3],
                               semiRings=self.minSum)

    def test_libdai_decimation_bp(self):
        if opengm.configuration.withLibdai:
            solverClass = opengm.inference.DecimationLibDai
            params = [opengm.InfParam(subInference='bp')]
            genericSolverCheck(solverClass, params=params,
                               gms=[self.gridGm, self.chainGm, self.gridGm3,
                                    self.chainGm3],
                               semiRings=self.minSum)

    def test_libdai_decimation_trwbp(self):
        if opengm.configuration.withLibdai:
            solverClass = opengm.inference.DecimationLibDai
            params = [opengm.InfParam(subInference='trwBp')]
            genericSolverCheck(solverClass, params=params,
                               gms=[self.gridGm, self.chainGm, self.gridGm3,
                                    self.chainGm3],
                               semiRings=self.minSum)

    def test_libdai_decimation_fractional_bp(self):
        if opengm.configuration.withLibdai:
            solverClass = opengm.inference.DecimationLibDai
            params = [opengm.InfParam(subInference='fractionalBp')]
            genericSolverCheck(solverClass, params=params,
                               gms=[self.gridGm, self.chainGm, self.gridGm3,
                                    self.chainGm3],
                               semiRings=self.minSum)

    def test_libdai_decimation_gibbs(self):
        if opengm.configuration.withLibdai:
            solverClass = opengm.inference.DecimationLibDai
            params = [opengm.InfParam(subInference='gibbs')]
            genericSolverCheck(solverClass, params=params,
                               gms=[self.gridGm, self.chainGm, self.gridGm3,
                                    self.chainGm3],
                               semiRings=self.minSum)


if __name__ == "__main__":
    t = Test_Inference()
    t.test_bp()

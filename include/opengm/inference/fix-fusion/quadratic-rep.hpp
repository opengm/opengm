#ifndef _QUADRATIC_REP_HPP_
#define _QUADRATIC_REP_HPP_

/*
 * quadratic-rep.hpp
 *
 * Copyright 2012 Alexander Fix
 * See LICENSE.txt for license information
 *
 * Dummy quadratic representation:
 *
 * Doesn't implement any functionality, but any class to be used with
 * HigherOrderEnergy::ToQuadratic must implement this interface
 */
template <typename REAL>
class DummyQuadraticRep {
    public:
        typedef int NodeId;

        DummyQuadraticRep() : _nodeCounter(0) { }
        virtual ~DummyQuadraticRep() { }

        virtual NodeId AddNode(int n) 
        {
            int counter = _nodeCounter; _nodeCounter += n; return counter;
        }
        virtual void SetMaxEdgeNum(int n) { }

        virtual void AddConstantTerm(REAL c) { }
        virtual void AddUnaryTerm(NodeId n, REAL coeff) { }
        virtual void AddUnaryTerm(NodeId n, REAL E0, REAL E1) { }
        virtual void AddPairwiseTerm(NodeId n1, NodeId n2, REAL coeff) { }
        virtual void AddPairwiseTerm(NodeId n1, NodeId n2, REAL E00, REAL E01, REAL E10, REAL E11) { }

        virtual void MergeParallelEdges() { }
        virtual void Solve() { }

        virtual int GetNodeNum() { return _nodeCounter; }
        virtual int GetLabel(NodeId n) { return -1; }

    protected:
        int _nodeCounter;
};
#endif

#ifndef _HIGHER_ORDER_ENERGY_HPP_
#define _HIGHER_ORDER_ENERGY_HPP_

/*
 * higher-order-energy.hpp
 *
 * Copyright 2012 Alexander Fix
 * See LICENSE.txt for license information
 *
 * A representation for higher order pseudo-boolean energy functions
 *
 * Built up by calls to AddVar and AddTerm
 * Main operation is the reduction to quadratic form, ToQuadratic()
 * ToQuadratic() impements the reduction described in 
 *  A Graph Cut Algorithm for Higher Order Markov Random Fields
 *  Alexander Fix, Artinan Gruber, Endre Boros, Ramin Zabih, ICCV 2011
 *
 * Template parameters:
 *  Energy   The result type of the energy function
 *  D        The maximum degree of any monomial 
 *
 * Example usage:
 *
 * To represent the function
 *      f(x_0, ..., x_3) = 7x_0 + 9x_1 - 8x_2 + 13x_3 + 27x_1x_2x_3 
 *                       - 31x_1x_3x_4 + 18x_1x_2x_3x_4
 * we can do the following:
 *
 * HigherOrderEnergy<int, 4> f;
 * f.AddVars(4); 
 * f.AddUnaryTerm(0, 7);
 * f.AddUnaryTerm(1, 9);
 * f.AddUnaryTerm(2, -8);
 * f.AddUnaryTerm(3, 13);
 *
 * VarId term1[] = {1, 2, 3};
 * VarId term2[] = {1, 3, 4};
 * VarId term3[] = {1, 2, 3, 4};
 * f.AddTerm(27, 3, term1);
 * f.AddTerm(-31, 3, term2);
 * f.AddTerm(18, 4, term3);
 *
 *
 * Then, we can get an equivalent quadratic form (by adding additional 
 * variables and performing transformations) by calling ToQuadratic. We can
 * solve this using QPBO, or any other quadratic optimizer that has the same
 * interface.
 *
 * QPBO<int> qr;
 * f.ToQuadratic(qr);
 * qr.Solve();
 *
 *
 */

#include <vector>
#include <list>
#include <boost/foreach.hpp>

template <typename R, int D>
class HigherOrderEnergy {
    public:
        typedef int VarId;

        // Constructs empty HigherOrderEnergy with no variables or terms
        HigherOrderEnergy();

        // Adds variables to the HigherOrderEnergy. Variables must be added 
        // before any terms referencing them can be added
        VarId AddVar();
        VarId AddVars(int n);
        VarId NumVars() const { return _varCounter; }

        // Adds a monomial to the HigherOrderEnergy. degree must be <= D
        // vars is an array of length at least degree, with the indices of 
        // the corresponding variables or literals in the monomial
        void AddTerm(R coeff, int degree, const VarId vars[]);

        void AddUnaryTerm(VarId v, R coeff);
    
        // Reduces the HigherOrderEnergy to quadratic form
        // NOTE: THIS IS A DESTRUCTIVE OPERATION, so do not rely on the 
        // HigherOrderEnergy being in any useful state after this operation.
        //
        // This is a templated function, so it will work with any class 
        // implementing the necessary interface. See quadratic-rep.hpp for
        // the minimal requirements.
        template <typename QR>
        void ToQuadratic(QR &qr);

        void Clear();

    private:
        struct Term
        {
            R coeff;
            int degree;
            VarId vars[D];

            Term(R _coeff, int _degree, const VarId _vars[])
                : coeff(_coeff), degree(_degree)
            {
                for (int i = 0; i < degree; ++i)
                    vars[i] = _vars[i];
            }

            bool operator<(const Term& t) const;
            bool operator==(const Term& t) const;

            std::string ToString() const;

        };
        static int Compare(int d1, 
                const VarId vars1[], 
                int d2, 
                const VarId vars2[]);

        struct VarRecord {
            VarRecord(VarId id) 
                : _id(id), _positiveTerms(0), _higherOrderTerms(0), 
                _quadraticTerms(0), _sumDegrees(0), _terms(), _coeff(0) { }
            VarId _id;
            int _positiveTerms;
            int _higherOrderTerms;
            int _quadraticTerms;
            int _sumDegrees;
            std::list<Term> _terms;
            R _coeff;

            void PrintTerms() const;
        };

        R _constantTerm;

        void RemoveTerm(Term* tp);
        void _EliminatePositiveTerms();
        template <typename QR>
        void _ReduceNegativeTerms(QR& qr);
        void _ReportMultilinearStats();

        size_t NumTerms() const {
            size_t numTerms = 0;
            BOOST_FOREACH(const VarRecord& vr, _varRecords)
                numTerms += vr._terms.size();
            return numTerms;
        }

        VarId _varCounter;

        typedef std::vector<VarRecord> varRecordVec_t;
        varRecordVec_t _varRecords;
};

template <typename R, int D>
inline HigherOrderEnergy<R, D>::HigherOrderEnergy()
    : _constantTerm(0), _varCounter(0), _varRecords()
{ }


template <typename R, int D>
inline typename HigherOrderEnergy<R, D>::VarId 
HigherOrderEnergy<R, D>::AddVar() {
    VarRecord vr(_varCounter);
    _varRecords.push_back(vr);
    return _varCounter++;
}

template <typename R, int D>
inline typename HigherOrderEnergy<R, D>::VarId 
HigherOrderEnergy<R, D>::AddVars(int n) {
    VarId firstVar = _varCounter;
    for (int i = 0; i < n; ++i)
        this->AddVar();
    return firstVar;
}

template <typename R, int D>
inline void 
HigherOrderEnergy<R, D>::AddTerm(R coeff, int d, const VarId vars[]) {
    if(coeff == 0) {
        return;
    } else if (d == 0) {
        _constantTerm += coeff;
        return;
    } else if (d == 1) {
        _varRecords[vars[0]]._coeff += coeff;
        return;
    } else {
        VarRecord& smallestVarRec = _varRecords[vars[0]];
        typename std::list<Term>::iterator it = smallestVarRec._terms.begin();
        int compareVars = 1;
        while (it != smallestVarRec._terms.end()) {
            compareVars = Compare(d, vars, it->degree, it->vars); 
            if (compareVars == 0) {
                break;
            } else if (compareVars < 0) {
                break;
            } else {
                ++it;
            }
        }
        if (compareVars == 0) {
            it->coeff += coeff;
        } else {
            if (d > 2) {
                smallestVarRec._higherOrderTerms++;
                smallestVarRec._sumDegrees += d;
            } else {
                smallestVarRec._quadraticTerms++;
            }
            smallestVarRec._terms.insert(it, Term(coeff, d, vars));
        }
        if (coeff > 0)
            smallestVarRec._positiveTerms++;
        return;
    }
}

template <typename R, int D>
inline void HigherOrderEnergy<R, D>::AddUnaryTerm(VarId var, R coeff) {
    _varRecords[var]._coeff += coeff;
}

template <typename R, int D>
void HigherOrderEnergy<R, D>::_EliminatePositiveTerms() {
    size_t numVars = _varRecords.size();
    for (size_t varIndex = 0; varIndex < numVars; ++varIndex) {
        R positiveSum = 0;
        VarId newPosVar = AddVar();

        VarRecord& vr = _varRecords[varIndex];

        typename std::list<Term>::iterator termIt = vr._terms.begin();
        VarId newVars[D];
        while (termIt != vr._terms.end()) {
            Term& t = *termIt;
            //std::cout << "\t" << t.ToString() << std::endl;
            typename std::list<Term>::iterator currIt = termIt;
            ++termIt;

            if (t.coeff > 0) {
                positiveSum += t.coeff;
                for (int i = 0; i < t.degree - 1; ++i) {
                    newVars[i] = t.vars[i+1];
                    assert(newVars[i] >= vr._id);
                }
                AddTerm(t.coeff, t.degree - 1, newVars);
                newVars[t.degree - 1] = newPosVar;
                AddTerm(-t.coeff, t.degree, newVars);
                if (t.degree == 2) {
                    vr._quadraticTerms--;
                } else {
                    vr._higherOrderTerms--;
                    vr._sumDegrees -= t.degree;
                }
                vr._terms.erase(currIt);
            }
        }
        VarId quadratic[2];
        quadratic[0] = vr._id;
        quadratic[1] = newPosVar;
        AddTerm(positiveSum, 2, quadratic);
    }
}

template <typename R, int D>
template <typename QR>
void HigherOrderEnergy<R, D>::_ReduceNegativeTerms(QR& qr) {
    // Estimate expected size of quadratic problem. Only nodes/edges are
    // created below, so we can count them ahead of time
    int expectedVars = _varCounter;
    int expectedEdges = 0;
    BOOST_FOREACH(const VarRecord& vr, _varRecords) {
        expectedVars += vr._higherOrderTerms;
        expectedEdges += vr._quadraticTerms;
        expectedEdges += vr._sumDegrees;
    }

    //std::cout << "\tExpected Vars: " << expectedVars << "\tExpected Edges: " << expectedEdges << std::endl;

    qr.SetMaxEdgeNum(expectedEdges);
    qr.AddNode(_varCounter);

    // Term-by-term reduction from Friedman & Drineas
    BOOST_FOREACH(VarRecord& vr, _varRecords) {
        BOOST_FOREACH(Term& t, vr._terms) {
            if (t.degree == 2) {
                qr.AddPairwiseTerm(t.vars[0], t.vars[1], 0, 0, 0, t.coeff);
            } else {
                typename QR::NodeId w = qr.AddNode();
                assert(t.coeff <= 0);
                for (int i = 0; i < t.degree; ++i) {
                    qr.AddPairwiseTerm(t.vars[i], w, 0, 0, 0, t.coeff);
                }
                qr.AddUnaryTerm(w, 0, t.coeff*(1-t.degree));
            }
        }
    }
    BOOST_FOREACH(VarRecord& vr, _varRecords) {
        qr.AddUnaryTerm(vr._id, 0, vr._coeff);
    }
}

template <typename R, int D>
template <typename QR>
inline void HigherOrderEnergy<R, D>::ToQuadratic(QR& qr) {
    _EliminatePositiveTerms();
    _ReduceNegativeTerms(qr);
}

template <typename R, int D>
inline int HigherOrderEnergy<R, D>::Compare(int d1, const VarId vars1[], int d2, const VarId vars2[]) {
    if (d1 < d2)
        return -1;
    if (d1 > d2)
        return 1;
    for (int index = 0; index < d1; ++index) {
        if (vars1[index] != vars2[index])
            return (vars1[index] < vars2[index]) ? -1 : 1;
    }
    return 0;
}

template <typename R, int D>
inline void HigherOrderEnergy<R, D>::Clear() {
    _varRecords.clear();
};

#endif

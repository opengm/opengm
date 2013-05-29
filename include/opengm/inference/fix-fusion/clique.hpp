#ifndef _CLIQUE_HPP_
#define _CLIQUE_HPP_

/*
 * clique.hpp
 *
 * Copyright 2012 Alexander Fix
 * See LICENSE.txt for license information
 *
 * This file contains classes for defining an energy function as a sum of
 * "local clique energies". That is, we define an energy function
 *      f(x_1, x_2, ..., x_n)
 *  by writing f as a sum of local energies, f_C, defined on "cliques"
 *  or "patches" of the image. For instance, we could let the set of cliques
 *  be all 2x2 patches of the image (i.e., sliding windows of size 2x2).
 *
 *  The fusion move algorithm (in fusion-move.hpp) can handle energies which
 *  are the sum of the individual energy functions for each clique, i.e.,
 *          f(x_1, ..., x_n) = sum_C f_C(x_C)
 *  where f_C is any real-valued function on the boolean variables in C,
 *  and x_C is just the labeling x restricted to the subset C.
 *
 *
 *  To make the above concrete, we provide two classes: CliqueEnergy and
 *  CliqueSystem. A CliqueEnergy has two important parts: it keeps track of a 
 *  subset of the variables which it depends on, and it can call operator()
 *  on a labeling of these variables, returning the local energy for that 
 *  particular labeling.
 *
 *  A CliqueSystem is basically just a container for CliqueEnergy objects. 
 *  The total energy to be optimized is the sum over all cliques of the local
 *  energies.
 *
 *
 *  To make this as flexible as possible, CliqueEnergy is an abstract base
 *  class. To use the fusion move code, the user should inherit from 
 *  CliqueEnergy and implement their own operator(). This operator() should 
 *  take in an array of labels (corresponding to a labeling of just the
 *  variables in the clique) and return the local energy corresponding to that
 *  labeling.
 *
 *  The reason this isn't just a simple function pointer is that the user may
 *  want to have each local energy depend on some parameter (for instance, 
 *  the unary terms will typically depend on the observed data values). 
 *  In this case, the user is free to add data members to their derived class.
 *  CliqueSystem is smart and stores the CliqueEnergy polymorphically, so no
 *  slicing occurs.
 */

#include <vector>
#include <boost/shared_ptr.hpp>
#include <boost/foreach.hpp>


/*
 * Abstract base class for defining the local energy of a clique.
 *
 * Template parameters:
 *  R   "real" type that is the type of energy-values
 *  T   "image" type, the base domain of the image labelling
 *  D   Maximum degree of the clique
 *
 * To use: derive from this class and implement operator()
 *
 * In the fusion-move code, these clique energies are stored by pointer, so 
 * derived types can add additional data-members without worrying about 
 * slicing. This is useful if your clique energy function needs to be a 
 * closure over additional data than just the current labelling of the clique
 *
 * See foe-cliques.{hpp|cpp} for example of how to derive from this class.
 */
template<typename R, typename T, int D>
class CliqueEnergy {
    public:
        /*
         * Parameters
         *  size    the size of the clique. Must be <= D
         *  nbd     array of indices that defines the clique. Must be at least
         *          size 'size' long.
         */
        CliqueEnergy(int size, int nbd[]) 
            : _size(size)
        {
            assert(size <= D);
            for (int i = 0; i < size; ++i) 
                _neighborhood[i] = nbd[i];
        }

        /*
         * operator() computes the energy of the clique, given the current 
         * labelling of the clique.
         *
         *  buf     will be an array of size _size containing the current 
         *          labelling of the image, with buf[i] corresponding to
         *          _neighborhood[i]
         */
        virtual R operator()(const T buf[]) const = 0;

        int _size;
        int _neighborhood[D];
};

/*
 * Container for a set of cliques.
 *
 * Stores CliqueEnergy polymorphically, so multiple derived types can be part 
 * of the same CliqueSystem
 */
template<typename R, typename T, int D>
class CliqueSystem {
    public:
        typedef boost::shared_ptr<CliqueEnergy<R, T, D> > CliquePointer;

        CliqueSystem() { }

        /*
         * Add a clique to the CliqueSystem
         */
        void AddClique(const CliquePointer& cp);

        /*
         * Returns the total energy of the current labelling by summing
         * the results of the individual CliqueEnergy
         */
        R Energy(const T* im) const;

        /*
         * Returns a vector containing all the cliques in the CliqueSystem
         */
        const std::vector<CliquePointer>& GetCliques() const;

    private:
        std::vector<CliquePointer> _cliques;
};

/*
 * IMPLEMENTATION
 */

template<typename R, typename T, int D>
void CliqueSystem<R, T, D>::AddClique(const CliquePointer& cp) {
    _cliques.push_back(cp);
}

template <typename R, typename T, int D>
R CliqueSystem<R, T, D>::Energy(const T* im) const {
    R energy = 0;
    BOOST_FOREACH(const CliquePointer& cp, _cliques) {
        const CliqueEnergy<R, T, D>& c = *cp;
        T buf[c._size];
        for (int i = 0; i < c._size; ++i)
            buf[i] = im[c._neighborhood[i]];
        energy += c(buf);
    }
    return energy;
}

template <typename R, typename T, int D>
const std::vector<typename CliqueSystem<R, T, D>::CliquePointer>& 
CliqueSystem<R, T, D>::GetCliques() const {
    return _cliques;
}

#endif

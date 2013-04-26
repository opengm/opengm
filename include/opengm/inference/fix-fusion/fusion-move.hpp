#ifndef _FUSION_MOVE_HPP_
#define _FUSION_MOVE_HPP_

#include <iostream>
#include <sstream>
#include <boost/foreach.hpp>
#include "higher-order-energy.hpp"
#include "clique.hpp"

#define NO_QPBO
#ifndef NO_QPBO
#include "QPBO.h"
#endif

/*
 * fusion-move.hpp
 *
 * Copyright 2012 Alexander Fix
 * See LICENSE.txt for license information
 *
 * Computes a fusion move between the current and proposed image.
 *
 * A fusion move takes two images (current and proposed) and tries to perform
 * the optimal move where each pixel is allowed to either stay at its current
 * value, or switch to its label in the proposed image. This is a 
 * generalization of alpha-expansion, where in alpha-expansion each pixel is 
 * allowed to either stay the same, or change to a fixed value alpha. That is,
 * alpha expansion is a fusion move where the proposed image is just the flat
 * image with value alpha at all pixels.
 *
 * Template Parameters:
 *  RandomAccessIterator    Needs to support operator[] giving results of type 
 *                              Label, so that iter[i] gives the label of 
 *                              pixel i
 *  Energy                  The value type of the energy function
 *  Label                   The label space of the function
 *  D                       The maximum number of variables in any clique
 *  QuadraticRep            The type used for optimizing quadratic boolean 
 *                              functions
 *
 * Parameters:
 *  size            The number of variables (pixels) in the image
 *  current         The current image
 *  proposed        The proposed image to fuse with
 *  out             Result will be written here
 *  cliqueSystem    The cliques defining the energy
 *  qr              An empty QuadraticRep. The results of the higher-order 
 *                      reduction will be written to this variable, and then 
 *                      solved.
 *
 * All of current, proposed and out must be of size 'size'
 * cliqueSystem should refer to variables only in the range 0..(size-1)
 */
template <typename RandomAccessIterator, 
         typename Energy, 
         typename Label, 
         int D, 
         typename QuadraticRep>
void FusionMove(size_t size, 
        RandomAccessIterator current, 
        RandomAccessIterator proposed, 
        RandomAccessIterator out, 
        const CliqueSystem<Energy, Label, D>& cliqueSystem,
        QuadraticRep& qr);

#ifndef NO_QPBO
/*
 * Computes a fusion move with default quadratic representation
 *
 * Same as the above function, but with the QuadraticRep parameter set
 * to QPBO<Energy> as a default
 */
template <typename RandomAccessIterator, 
         typename Energy, 
         typename Label, 
         int D>
void FusionMove(size_t size, 
        RandomAccessIterator current, 
        RandomAccessIterator proposed, 
        RandomAccessIterator out, 
        const CliqueSystem<Energy, Label, D>& cliqueSystem);
#endif

/*
 * Set up the fusion move energy as a HigherOrderEnergy
 *
 * Template Parameters:
 *  Same as FusionMove
 *
 * Parameters:
 *  size, current, proposed, cliqueSystem same as FusionMove
 *  hoe     Output parameter. Result of setting up the fusion move as a 
 *              higher-order pseudo-boolean function is put here
 */
template <typename RandomAccessIterator, 
         typename Energy, 
         typename Label, 
         int D>
void SetupFusionEnergy(size_t size,
        RandomAccessIterator current,
        RandomAccessIterator proposed,
        const CliqueSystem<Energy, Label, D>& cliqueSystem,
        HigherOrderEnergy<Energy, D>& hoe);

/*
 * Given a labeling (according to the results of qr.GetLabel) fuse the images
 * current and proposed. 
 *
 * Labels of 0 and -1 (unlabeled) are assigned to their value in current
 * Labels of 1 are assigned to their value in proposed in the out image
 */
template <typename RandomAccessIterator, typename QuadraticRep>
void GetFusedImage(size_t size, 
        RandomAccessIterator current, 
        RandomAccessIterator proposed, 
        RandomAccessIterator out, 
        QuadraticRep& qr);

/*
 * Implementation
 */

template <typename RandomAccessIterator, 
    typename Energy, 
    typename Label, 
    int D, 
    typename QuadraticRep>
void FusionMove(size_t size, 
        RandomAccessIterator current, 
        RandomAccessIterator proposed, 
        RandomAccessIterator out, 
        const CliqueSystem<Energy, Label, D>& cliqueSystem,
        QuadraticRep& qr) 
{
    HigherOrderEnergy<Energy, D> hoe;
    SetupFusionEnergy(size, current, proposed, cliqueSystem, hoe);
    hoe.ToQuadratic(qr);
    qr.Solve();
    GetFusedImage(size, current, proposed, out, qr);
}


#ifndef NO_QPBO
template <typename RandomAccessIterator, 
    typename Energy, 
    typename Label, 
    int D>
void FusionMove(size_t size, 
        RandomAccessIterator current, 
        RandomAccessIterator proposed, 
        RandomAccessIterator out, 
        const CliqueSystem<Energy, Label, D>& cliqueSystem)
{
    QPBO<Energy> qr(size, 0);
    FusionMove(size, current, proposed, out, cliqueSystem, qr);
}
#endif

template <typename RandomAccessIterator, 
    typename Energy, 
    typename Label, 
    int D>
void SetupFusionEnergy(size_t size,
        RandomAccessIterator current,
        RandomAccessIterator proposed,
        const CliqueSystem<Energy, Label, D>& cliqueSystem,
        HigherOrderEnergy<Energy, D>& hoe)
{
    hoe.AddVars(size);
    typedef typename CliqueSystem<Energy, Label, D>::CliquePointer 
        CliquePointer;
    BOOST_FOREACH(const CliquePointer& cp, cliqueSystem.GetCliques()) {
        const CliqueEnergy<Energy, Label, D>& c = *cp;
        unsigned int size = c._size;

        if (size == 0) {
            continue;
        } else if (size == 1) {
            Energy e0 = c(&current[c._neighborhood[0]]);
            Energy e1 = c(&proposed[c._neighborhood[0]]);
            hoe.AddUnaryTerm(c._neighborhood[0], e1 - e0);
        } else {
            unsigned int numAssignments = 1 << size;
            Energy coeffs[numAssignments];
            for (unsigned int subset = 1; subset < numAssignments; ++subset) {
                coeffs[subset] = 0;
            }
            // For each boolean assignment, get the clique energy at the 
            // corresponding labeling
            Label cliqueLabels[size];
            for (unsigned int assignment = 0; 
                    assignment < numAssignments; 
                    ++assignment) 
            {
                for (unsigned int i = 0; i < size; ++i) {
                    if (assignment & (1 << i)) { 
                        cliqueLabels[i] = proposed[c._neighborhood[i]];
                    } else {
                        cliqueLabels[i] = current[c._neighborhood[i]];
                    }
                }
                Energy energy = c(cliqueLabels);
                for (unsigned int subset = 1; 
                        subset < numAssignments; 
                        ++subset) 
                {
                    if (assignment & ~subset) {
                        continue;
                    } else {
                        int parity = 0;
                        for (unsigned int b = 0; b < size; ++b) {
                            parity ^= 
                                (((assignment ^ subset) & (1 << b)) != 0);
                        }
                        coeffs[subset] += parity ? -energy : energy;
                    }
                }
            }
            typename HigherOrderEnergy<Energy, D>::VarId vars[D];
            for (unsigned int subset = 1; subset < numAssignments; ++subset) {
                int degree = 0;
                for (unsigned int b = 0; b < size; ++b) {
                    if (subset & (1 << b)) {
                        vars[degree++] = c._neighborhood[b];
                    }
                }
                std::sort(vars, vars+degree);
                hoe.AddTerm(coeffs[subset], degree, vars);
            }
        }
    }
}

template <typename RandomAccessIterator, typename QuadraticRep>
void GetFusedImage(size_t size, 
        RandomAccessIterator current, 
        RandomAccessIterator proposed, 
        RandomAccessIterator out, 
        QuadraticRep& qr) 
{
    for (size_t i = 0; i < size; ++i) {
        int label = qr.GetLabel(i);
        if (label == 1) {
            out[i] = proposed[i];
        } else if (label == 0) {
            out[i] = current[i];
        } else {
            out[i] = current[i];
        }
    }
}
#endif

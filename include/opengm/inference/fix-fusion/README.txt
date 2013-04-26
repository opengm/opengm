This library provides an implementation of the higher-order reduction
strategy described in the paper:
   A Graph Cut Algorithm for Higher Order Markov Random Fields
   Alexander Fix, Artinan Gruber, Endre Boros, Ramin Zabih, ICCV 2011

The representation of higher-order energy functions, and the reduction to
quadratic form are implemented in the file higher-order-energy.hpp

Since the Fusion Move algorithm is a common use case for higher-order
functions, the files fusion-move.hpp and clique.hpp are also provided to make
it easy to set up a higher-order fusion move.

Note that this is a HEADER ONLY LIBRARY, as all the classes are templates
according to the type used for representing energy values, as well as the
maximum size of a clique. 

An example of this library is provided in the example directory. This is a
simple denoising algorithm, using a Field of Experts prior with
blur-and-random proposals. This is one of the experiments described in the
paper linked at the top of this readme.

This example code requires the following prerequisites:
-- A reasonably recent version of boost (tested with 1.48, but will likely
    work with much earlier versions).

-- A copy of the QPBO code, available from Vladimir Kolmogorov's website at
    http://pub.ist.ac.at/~vnk/software.html

    To compile the examples, you will need to download the file
    QPBO-v1.3.src.tar.gz and extract the contents into a directory named qpbo
    in the same directory as this README.

    The overall directory structure should look like

    ./
        README.txt
        include/
            clique.hpp
            fusion-move.hpp
            ...
        example/
            higher-order-example.cpp
            ...
        qpbo/
            block.h
            QPBO.h
            ...

To compile and run this example, the following should work
    cd example
    make
    make test

Copyright and license information are included in LICENSE.txt. This code is
provided under the MIT license, but if for whatever reason this doesn't suit
your needs, feel free to contact the author at afix@cs.cornell.edu.

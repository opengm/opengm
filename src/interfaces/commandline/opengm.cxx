/**
 * @file
 * @brief Main file for the commandline interface.
 * @details This file contains the main() for the commandline interface and
 *          further support functions used by the main().
 */

/*
#include <iostream> 
#include <iomanip>
#include <vector>

#include <opengm/opengm.hxx> 
#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/graphicalmodel/graphicalmodel_hdf5.hxx>

//configuration file
#include "../common/helper/interfacesTypedefs.hxx"

//helper functions
#include "../common/helper/helper.hxx"

#include "../common/argument/argument.hxx"
#include "../common/argument/argument_executer.hxx"

#include "io_cmd.hxx"
*/

#include "cmd_interface.hxx"
#include "../common/helper/interfacesTypedefs.hxx"

using namespace opengm::interface;

/***************************
 * forward declarations... *
 ***************************/

/*********************
 * implementation... *
 *********************/

/**
 * @brief Main entry point for the commandline interface
 * @details This is the main function of the commandline interface. It
 * initializes all necessary variables, checks for help request and calls
 * corresponding proceeding functions.
 * @param[in] argc Number of arguments provided by the user.
 * @param[in] argv List of the aruments provided by the user.
 */
int main(int argc, char** argv) {
   if(argc < 2) {
      std::cerr << "At least one input argument required" << std::endl;
      std::cerr << "try \"-h\" for help" << std::endl;
      return 1;
   }

   //CMDInterface<ValueTypeList, OperatorTypeList, functionTypeList, AccumulatorTypeList, InferenceTypeList> interface(argc, argv);
   //interface.parse();

   return 0;
}

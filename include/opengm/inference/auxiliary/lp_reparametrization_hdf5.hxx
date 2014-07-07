/*
 * lp_reparametrization_hdf5.hxx
 *
 *  Created on: Jul 3, 2014
 *      Author: bsavchyn
 */

#ifndef LP_REPARAMETRIZATION_HDF5_HXX_
#define LP_REPARAMETRIZATION_HDF5_HXX_

#include <opengm/graphicalmodel/graphicalmodel_hdf5.hxx>
#include <opengm/inference/auxiliary/lp_reparametrization.hxx>

namespace opengm {
namespace hdf5 {

template<class GM>
void save(const LPReparametrisationStorage<GM>& repa,const std::string& filename,const std::string& modelname)
{
		hid_t file = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
	    OPENGM_ASSERT(file >= 0);
		marray::Vector<typename GM::ValueType> marr;
		repa.serialize(&marr);
		marray::hdf5::save(file,modelname.c_str(),marr);
	    H5Fclose(file);
}

template<class GM>
void load(LPReparametrisationStorage<GM>* prepa, const std::string& filename, const std::string& modelname)
{
	hid_t file = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
	OPENGM_ASSERT(file>=0);
	marray::Vector<typename GM::ValueType> marr;
	marray::hdf5::load(file,modelname.c_str(),marr);
	prepa->deserialize(marr);
	H5Fclose(file);
};
}
}

#endif /* LP_REPARAMETRIZATION_HDF5_HXX_ */

/*
 * test_primal_lpbound.openGM.cpp
 *
 *  Created on: Feb 5, 2013
 *      Author: bsavchyn
 */

#include <fstream>
#include <vector>
#include <opengm/inference/auxiliary/primal_lpbound.hxx>
namespace TST{
using std::string;

template <class FileStream>
void openfile_throw(const string& file_name,FileStream& file,const string& prefix="")
{
 file.open(file_name.c_str());
 if (file.fail())
  throw std::runtime_error(prefix+string("Can not open file: ")+file_name);
};

bool areEqualTXTfiles(const string& name1,const string& name2)
{
	std::ifstream f1,f2;
	openfile_throw(name1,f1);
    openfile_throw(name2,f2);

	string s1,s2;
	while (!f1.eof() && !f2.eof())
	{
		std::getline(f1,s1);
		std::getline(f2,s2);
		if (s1!=s2)
			return false;
	}
	if (!(f1.eof() && f2.eof()))
		return false;

	return true;
};

double RandomDouble(double max)
{
	return double(max*rand()/RAND_MAX);
};

};//TST

	template<class GM>
	void Create3x1Label3Model(GM& gm_return)
	{
		size_t numberOfNodes=3, numberOflabels=3;
		typename GM::SpaceType space(numberOfNodes,numberOflabels);
	    GM gm(space);

	    // single site factors
        opengm::ExplicitFunction<double> f1(&numberOflabels,&numberOflabels+1, 0.0);
        typename GM::IndexType j;
	    { j=0;
          f1(0) = 1.0;
	      f1(1) = 2.0;
	      f1(2) = 3.0;
	      typename GM::FunctionIdentifier fId = gm.addFunction(f1);
	      gm.addFactor(fId,&j, &j+1);}

	    { j=1;
          f1(0) = 4.0;
	      f1(1) = 5.0;
	      f1(2) = 6.0;
	      typename GM::FunctionIdentifier fId = gm.addFunction(f1);
	      gm.addFactor(fId,&j, &j+1);}

	    { j=2;
          f1(0) = 7.0;
	      f1(1) = 8.0;
	      f1(2) = 9.0;
	      typename GM::FunctionIdentifier fId = gm.addFunction(f1);
	      gm.addFactor(fId,&j, &j+1);}

	    //pw factors
        size_t shape[] = {numberOflabels,numberOflabels};
        opengm::ExplicitFunction<double> f2(shape,shape+2, 0.0);

	    { size_t vi[] = {0,1};
	      f2(0,0) = 5.0;
	      f2(0,1) = 0.0;
	      f2(0,2) = 1.0;

	      f2(1,0) = 5.0;
	      f2(1,1) = 0.0;
	      f2(1,2) = 1.0;

	      f2(2,0) = 1.0;
	      f2(2,1) = 0.0;
	      f2(2,2) = 10.0;
	      typename GM::FunctionIdentifier fId = gm.addFunction(f2);
	      gm.addFactor(fId,vi,vi+2);}

	    { size_t vi[] = {1,2};
	      f2(0,0) = 5.0;
	      f2(0,1) = 5.0;
	      f2(0,2) = 1.0;

	      f2(1,0) = 0.0;
	      f2(1,1) = 0.0;
	      f2(1,2) = 0.0;

	      f2(2,0) = 1.0;
	      f2(2,1) = 1.0;
	      f2(2,2) = 10.0;
	      typename GM::FunctionIdentifier fId = gm.addFunction(f2);
	      gm.addFactor(fId,vi,vi+2);}

	    gm_return=gm;
	};

	typedef opengm::GraphicalModel<double, opengm::Adder, opengm::ExplicitFunction<double>, opengm::DiscreteSpace<> > GraphicalModel;
    typedef opengm::PrimalLPBound<GraphicalModel,opengm::Maximizer> LPBounder;

	void test_PrimalLPBound()
	{
		GraphicalModel gm;
		Create3x1Label3Model(gm);

		LPBounder lpbound_max(gm);
		std::vector<double> val(gm.numberOfLabels(0),0.0);
		val[0]=1.0;
		lpbound_max.setVariable(0,val.begin());
		std::vector<double> val1(gm.numberOfLabels(0),-100.0);
		lpbound_max.getVariable(0,val1.begin());
		OPENGM_ASSERT(std::equal(val.begin(),val.end(),val1.begin()));

		val[0]=0.6; val[1]=0.0; val[2]=0.4;
		lpbound_max.setVariable(1,val.begin());
		OPENGM_ASSERT(!lpbound_max.IsValueBuffered(1));
		double varval=lpbound_max.getVariableValue(1);
		OPENGM_ASSERT(fabs(varval-4.8)<1e-8);
		OPENGM_ASSERT(!lpbound_max.IsValueBuffered(3));
		varval=lpbound_max.getFactorValue(3);
		OPENGM_ASSERT(fabs(varval-3.4)<1e-8);

		val[0]=0.3; val[1]=0.3; val[2]=0.4;
		lpbound_max.setVariable(2,val.begin());
		OPENGM_ASSERT(!lpbound_max.IsValueBuffered(4));
		double totalval=lpbound_max.getTotalValue();
		OPENGM_ASSERT(fabs(totalval-24.3)<1e-5);
		for (size_t i=0;i<gm.numberOfFactors();++i)
			OPENGM_ASSERT(lpbound_max.IsValueBuffered(i));

		TransportSolver::MatrixWrapper<double> matrix(3,3);
		OPENGM_ASSERT(lpbound_max.IsValueBuffered(4));
		OPENGM_ASSERT(lpbound_max.IsFactorVariableBuffered(4));
		lpbound_max.getFactorVariable(4,matrix);

		OPENGM_ASSERT(fabs(matrix(0,0)-0.3)<1e-8);
		OPENGM_ASSERT(fabs(matrix(0,1)-0.3)<1e-8);
		OPENGM_ASSERT(fabs(matrix(0,2)-0)<1e-8);
		OPENGM_ASSERT(fabs(matrix(1,0)-0)<1e-8);
		OPENGM_ASSERT(fabs(matrix(1,1)-0)<1e-8);
		OPENGM_ASSERT(fabs(matrix(1,2)-0)<1e-8);
		OPENGM_ASSERT(fabs(matrix(2,0)-0)<1e-8);
		OPENGM_ASSERT(fabs(matrix(2,1)-0)<1e-8);
		OPENGM_ASSERT(fabs(matrix(2,2)-0.4)<1e-8);

		OPENGM_ASSERT(lpbound_max.IsValueBuffered(4));
		varval=lpbound_max.getFactorValue(4);
		OPENGM_ASSERT(fabs(varval-7)<1e-8);

		LPBounder lpbound_max04(gm,LPBounder::Parameter(0.036,1000));
		val[0]=1.0; val[1]=0.0; val[2]=0.0;
		lpbound_max04.setVariable(0,val.begin());
		val[0]=0.6; val[1]=0.0; val[2]=0.4;
		lpbound_max04.setVariable(1,val.begin());
		val[0]=0.3; val[1]=0.3; val[2]=0.4;
		lpbound_max04.setVariable(2,val.begin());

		totalval=lpbound_max04.getTotalValue();
		for (size_t i=0;i<gm.numberOfFactors();++i)
			OPENGM_ASSERT(lpbound_max04.IsValueBuffered(i));

		OPENGM_ASSERT(fabs(totalval-21.9)<1e-8);
	};

int main()
{
	test_PrimalLPBound();
	return 0;
}

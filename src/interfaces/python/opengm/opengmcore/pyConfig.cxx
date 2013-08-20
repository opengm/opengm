#include <boost/python.hpp>
#include <stdexcept>
#include <string>
#include <sstream>
#include <stddef.h>
#include <opengm/graphicalmodel/graphicalmodel.hxx>

using namespace boost::python;

class PyOpengmConfig{
public:
   PyOpengmConfig(){}
   inline bool withConicbundle()const{
      #ifdef WITH_CONICBUNDLE
      return true;
      #else
      return false;
      #endif
   }
   inline bool withMaxflow()const{
      #ifdef WITH_MAXFLOW
      return true;
      #else
      return false;
      #endif
   }
   inline bool withMaxflowIbfs()const{
      #ifdef WITH_MAXFLOW_IBFS
      return true;
      #else
      return false;
      #endif
   }
   inline bool withMrf()const{
      #ifdef WITH_MRF
      return true;
      #else
      return false;
      #endif
   }
   inline bool withTrws()const{
      #ifdef WITH_TRWS
      return true;
      #else
      return false;
      #endif
   }
   inline bool withQpbo()const{
      #ifdef WITH_QPBO
      return true;
      #else
      return false;
      #endif
   }
   inline bool withCplex()const{
      #ifdef WITH_CPLEX
      return true;
      #else
      return false;
      #endif
   }
   inline bool withGurobi()const{
      #ifdef WITH_GUROBI
      return true;
      #else
      return false;
      #endif
   }
   inline bool withHdf5()const{
      #ifdef WITH_HDF5
      return true;
      #else
      return false;
      #endif
   }
   inline bool withLibdai()const{
      #ifdef WITH_LIBDAI
      return true;
      #else
      return false;
      #endif
   }

   inline bool withFastPd()const{
      #ifdef WITH_FASTPD
      return true;
      #else
      return false;
      #endif
   }

   inline bool withAd3()const{
      #ifdef WITH_AD3
      return true;
      #else
      return false;
      #endif
   }
   
   inline std::string opengmVersion()const{
      return "2.1.0";
   }
   inline std::string opengmPythonWrapperVersion()const{
      return "beta-0.9.5";
   }
   
   inline std::string asString() const {
      std::stringstream ss;
      ss<<"OpenGm Python Wrapper Version="<<opengmPythonWrapperVersion()<<"\n";
      ss<<"OpenGm Version="<<opengmVersion()<<"\n";
      ss<<"with Cplex="<<withCplex()<<"\n";
      ss<<"with Gurobi="<<withGurobi()<<"\n";
      ss<<"with ConicBundle="<<withConicbundle()<<"\n";
      ss<<"with Maxflow="<<withMaxflow()<<"\n";
      ss<<"with Maxflow Ibfs="<<withMaxflowIbfs()<<"\n";
      ss<<"with Mrf="<<withMrf()<<"\n";
      ss<<"with Qpbo="<<withQpbo()<<"\n";
      ss<<"with Trws="<<withTrws()<<"\n";
      ss<<"with Fastpd="<<withFastPd()<<"\n";
      ss<<"with Ad3="<<withAd3()<<"\n";
      ss<<"with Libdai="<<withLibdai()<<"\n";
      ss<<"with hdf5="<<withHdf5()<<"\n";
      
      return ss.str();
   }
};


void export_config() {
   class_<PyOpengmConfig > ("OpengmConfiguration", init< >())
   .def("__str__",&PyOpengmConfig::asString)
   .add_property("opengmPythonWrapperVersion", &PyOpengmConfig::opengmPythonWrapperVersion)
   .add_property("opengmVersion", &PyOpengmConfig::opengmVersion)
   .add_property("withConicbundle", &PyOpengmConfig::withConicbundle)
   .add_property("withMaxflow", &PyOpengmConfig::withMaxflow)
   .add_property("withMaxflowIbfs", &PyOpengmConfig::withMaxflowIbfs)
   .add_property("withMrf", &PyOpengmConfig::withMrf)
   .add_property("withQpbo", &PyOpengmConfig::withQpbo)
   .add_property("withTrws", &PyOpengmConfig::withTrws)
   .add_property("withCplex", &PyOpengmConfig::withCplex)
   .add_property("withGurobi", &PyOpengmConfig::withGurobi)
   .add_property("withFastPd", &PyOpengmConfig::withFastPd)
   .add_property("withAd3", &PyOpengmConfig::withAd3)
   .add_property("withLibdai", &PyOpengmConfig::withLibdai)
   .add_property("withHdf5", &PyOpengmConfig::withHdf5)
   ;
}



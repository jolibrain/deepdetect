#ifndef NATIVE_FACTORY_H
#define NATIVE_FACTORY_H

#include "native_net.h"
#include "./templates/nbeats.h"
#include "../torchinputconns.h"
#include "apidata.h"

namespace dd
{
  class NativeFactory
  {
  public:
    template <class TInputConnectorStrategy>
      static native_variant_type from_template(std::string tdef,
					       APIData template_params,
					       TInputConnectorStrategy &inputc)
   {
     (void)tdef;
     (void)inputc;
     (void)template_params;
     throw std::exception();
   }

    static bool valid_template_def(std::string tdef)
    {
      if (tdef.find("nbeats") != std::string::npos)
        return true;
      return false;
    }

    static bool is_timeserie(std::string tdef)
    {
      if (tdef.find("nbeats") != std::string::npos)
        return true;
      return false;
    }
  };

  template<>
    native_variant_type NativeFactory::from_template<CSVTSTorchInputFileConn>(
  //native_variant_type NativeFactory::from_template(
						   std::string tdef, APIData template_params,
						   CSVTSTorchInputFileConn &inputc)
  {
    if (tdef.find("nbeats") != std::string::npos)
      {
        std::vector<std::string> p = template_params.get("template_params")
	  .get<std::vector<std::string>>();
        return std::make_shared<NBeats>(inputc, p);
      }
      else
	{
	  // beware
	  //return torch::nn::Module();//NativeModule();
	  throw std::exception();
	  //return nullptr;
	}
    }
  
  
}
#endif

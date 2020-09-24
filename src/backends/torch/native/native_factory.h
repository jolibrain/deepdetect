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
    static NativeModule *from_template(const std::string tdef,
                                       const APIData template_params,
                                       const TInputConnectorStrategy &inputc)
    {
      (void)(tdef);
      (void)(template_params);
      (void)(inputc);
      return nullptr;
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

  template <>
  NativeModule *NativeFactory::from_template<CSVTSTorchInputFileConn>(
      const std::string tdef, const APIData template_params,
      const CSVTSTorchInputFileConn &inputc)
  {
    if (tdef.find("nbeats") != std::string::npos)
      {
        std::vector<std::string> p = template_params.get("template_params")
                                         .get<std::vector<std::string>>();
        return new NBeats(inputc, p);
      }
    else
      return nullptr;
  }
}
#endif

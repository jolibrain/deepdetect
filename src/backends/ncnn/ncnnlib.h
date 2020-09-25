/**
 * DeepDetect
 * Copyright (c) 2018 Jolibrain
 * Author: Corentin Barreau <corentin.barreau@epitech.eu>
 *
 * This file is part of deepdetect.
 *
 * deepdetect is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * deepdetect is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with deepdetect.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef NCNNLIB_H
#define NCNNLIB_H

#include <staticjson/staticjson.hpp>
#include "apidata.h"
#include "utils/utils.hpp"

// NCNN
#include "net.h"
#include "ncnnmodel.h"

namespace dd
{
  /* autodoc: nccn init parameters */
  struct NCNNLibInitParameters
  {

    /* number of output classes */
    int nclasses = 0;

    /* number of threads to use, 0 means all cpu cores */
    int threads = 0;

    /* light mode */
    bool lightmode = false;

    /* network input blob name */
    std::string inputBlob = "data";

    /* network output blob name */
    std::string outputBlob;

    void post_init()
    {
      if (threads <= 0)
        threads = dd_utils::my_hardware_concurrency();
    }
    void staticjson_init(staticjson::ObjectHandler *h)
    {
      h->add_property("nclasses", &nclasses, staticjson::Flags::Optional);
      h->add_property("threads", &threads, staticjson::Flags::Optional);
      h->add_property("lightmode", &lightmode, staticjson::Flags::Optional);
      h->add_property("inputblob", &inputBlob, staticjson::Flags::Optional);
      h->add_property("outputblob", &outputBlob, staticjson::Flags::Optional);
      h->set_flags(staticjson::Flags::DisallowUnknownKey);
    }
  };

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy,
            class TMLModel = NCNNModel>
  class NCNNLib : public MLLib<TInputConnectorStrategy,
                               TOutputConnectorStrategy, TMLModel>
  {
  public:
    NCNNLib(const NCNNModel &tmodel);
    NCNNLib(NCNNLib &&tl) noexcept;
    ~NCNNLib();

    /*- from mllib -*/
    void init_mllib(const APIData &ad);

    void clear_mllib(const APIData &ad);

    int train(const APIData &ad, APIData &out);

    int predict(const APIData &ad, APIData &out);

    void model_type(const std::string &param_file, std::string &mltype);

  public:
    ncnn::Net *_net = nullptr;
    bool _timeserie = false;

  private:
    NCNNLibInitParameters _mllib_params;
    static ncnn::UnlockedPoolAllocator _blob_pool_allocator;
    static ncnn::PoolAllocator _workspace_pool_allocator;

  protected:
    int _old_height = -1;
  };

}

#endif

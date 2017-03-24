/**
 * DeepDetect
 * Copyright (c) 2017 Emmanuel Benazera
 * Author: Emmanuel Benazera <beniz@droidnik.fr>
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

#ifndef TSNELIB_H
#define TSNELIB_H

#include "mllibstrategy.h"
#include "tsnemodel.h"
#include "tsne.h"

namespace dd
{
  /**
   * Multicore TSNE wrapper
   */
    template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel=TSNEModel>
    class TSNELib : public MLLib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>
    {
    public:
    TSNELib(const TSNEModel &tmodel);
    TSNELib(TSNELib &&tl) noexcept;
    ~TSNELib();

    /*- from mllib -*/
    void init_mllib(const APIData &ad);

    void clear_mllib(const APIData &d);

    int train(const APIData &ad, APIData &out);

    // N/A
    int predict(const APIData &ad, APIData &out)
    {
      (void)ad;
      (void)out;
      return 0;
    }

    public:
    TSNE _tsne;
    int _iterations = 5000;
    int _perplexity = 30;
    const int _no_dims = 2; /**< target dimensionality, backend lib only supports 2D */
    double _theta = 0.5; /**< angle */
    };

}

#endif

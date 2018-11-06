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

#include "ncnnmodel.h"

namespace dd
{
    int NCNNModel::read_from_repository(const std::shared_ptr<spdlog::logger> &logger)
    {
        static std::string params = ".param.bin";
        static std::string weights = ".bin";
        static std::string corresp = "corresp";
        std::unordered_set<std::string> lfiles;
        int e = fileops::list_directory(_repo,true,false,false,lfiles);
        if (e != 0) {
            logger->error("error reading or listing NCNN models in repository {}",_repo);
            return 1;
        }
        std::string paramsf,weightsf,correspf;
        int weight_t = -1;
        int params_t = -1;
        auto hit = lfiles.begin();
        while (hit != lfiles.end()) {
            std::cerr << *hit << std::endl;
            if ((*hit).find(params) != std::string::npos) {
                // stat file to pick the latest one
                long int pm = fileops::file_last_modif((*hit));
                if (pm > params_t) {
                    paramsf = (*hit);
                    params_t = pm;
                }
            } else if ((*hit).find(weights) != std::string::npos) {
                // stat file to pick the latest one
                long int wt = fileops::file_last_modif((*hit));
                if (wt > weight_t) {
        	        weightsf = (*hit);
        	        weight_t = wt;
                }
            } else if ((*hit).find(corresp) != std::string::npos) {
                correspf = (*hit);
            }
            ++hit;
        }
        _params = paramsf;
        _weights = weightsf;
        _corresp = correspf;
        return 0;
    }
}
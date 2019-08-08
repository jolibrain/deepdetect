/**
 * DeepDetect
 * Copyright (c) 2019 Jolibrain
 * Author: Louis Jean <ljean@etud.insa-toulouse.fr>
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

#include "torchmodel.h"

namespace dd
{
    int TorchModel::read_from_repository(
        const std::shared_ptr<spdlog::logger> &logger) {

        const std::string traced = "trace.pt";
        const std::string weights = ".pt";
        const std::string corresp = "corresp";

        std::unordered_set<std::string> files;
        int err = fileops::list_directory(_repo, true, false, false, files);

        if (err != 0) {
            logger->error("Listing pytorch models failed");
            return 1;
        }

        std::string tracedf,weightsf,correspf;
        int traced_t = -1, weights_t = -1, corresp_t = -1;

        for (const auto &file : files) {
            long int lm = fileops::file_last_modif(file);
            if (file.find(traced) != std::string::npos) {
                if (traced_t < lm) {
                    tracedf = file;
                    traced_t = lm;
                }
            }
            else if (file.find(weights) != std::string::npos) {
                if (weights_t < lm) {
                    weightsf = file;
                    weights_t = lm;
                }
            }
            else if (file.find(corresp) != std::string::npos) {
                if (corresp_t < lm) {
                    correspf = file;
                    corresp_t = lm;
                }
            }
        }

        _traced = tracedf;
        _weights = weightsf;
        _corresp = correspf;

        return 0;
    }
}
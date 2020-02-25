/**
 * DeepDetect
 * Copyright (c) 2019 Pixel Forensics, Inc.
 * Author: Cheni Chadowitz <cchadowitz@pixelforensics.com>
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
#ifndef DEEPDETECT_DLIB_ACTIONS_H
#define DEEPDETECT_DLIB_ACTIONS_H

#include "chain_actions.h"

namespace dd {

    class DlibAlignCropAction : public ChainAction
    {
    public:
        DlibAlignCropAction(const APIData &adc,
                            const std::string &action_id,
                            const std::string &action_type,
                            const std::shared_ptr<spdlog::logger> chain_logger)
          :ChainAction(adc,action_id,action_type,chain_logger) {}

        ~DlibAlignCropAction() {}

        void apply(APIData &model_out,
                   ChainData &cdata);
    };
}

#endif //DEEPDETECT_DLIB_ACTIONS_H

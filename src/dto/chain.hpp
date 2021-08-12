/**
 * DeepDetect
 * Copyright (c) 2021 Jolibrain SASU
 * Author: Louis Jean <louis.jean@jolibrain.com>
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

#ifndef DTO_CHAIN_HPP
#define DTO_CHAIN_HPP

#include "oatpp/core/Types.hpp"
#include "oatpp/core/macro/codegen.hpp"

#include "common.hpp"

namespace dd
{
  namespace DTO
  {
#include OATPP_CODEGEN_BEGIN(DTO) ///< Begin DTO codegen section

    class ChainHead : public oatpp::DTO
    {
      DTO_INIT(ChainHead, DTO)
    };

    class ChainBody : public oatpp::DTO
    {
      DTO_INIT(ChainBody, DTO)

      DTO_FIELD(Vector<UnorderedFields<Any>>, predictions)
          = Vector<UnorderedFields<Any>>::createShared();
      DTO_FIELD(Float64, time);
    };

    class ChainResponse : public oatpp::DTO
    {
      DTO_INIT(ChainResponse, DTO)

      DTO_FIELD(String, dd_msg);
      DTO_FIELD(Object<Status>, status);
      DTO_FIELD(Object<ChainHead>, head);
      DTO_FIELD(Object<ChainBody>, body);
    };

#include OATPP_CODEGEN_END(DTO) ///< End DTO codegen section
  }
}

#endif // DTO_CHAIN_HPP

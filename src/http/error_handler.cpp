/**
 * DeepDetect
 * Copyright (c) 2020 Jolibrain SASU
 * Author: Mehdi Abaakouk <mehdi.abaakouk@jolibrain.com>
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

#include "dd_types.h"
#include "http/error_handler.hpp"

#include <rapidjson/allocators.h>
#include <rapidjson/document.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/reader.h>
#include <rapidjson/writer.h>

#include <gflags/gflags.h>

DECLARE_string(allow_origin);

ErrorHandler::ErrorHandler(
    const std::shared_ptr<oatpp::data::mapping::ObjectMapper> &objectMapper)
    : m_objectMapper(objectMapper)
{
}

std::shared_ptr<ErrorHandler::OutgoingResponse>
ErrorHandler::handleError(const Status &status, const oatpp::String &message,
                          const Headers &headers)
{
  std::string status_message = status.description;
  status_message.append(": ");
  status_message.append(message.get()->std_str());

  JDoc jst;
  jst.SetObject();

  JVal jsv(rapidjson::kObjectType);
  jsv.AddMember("code", JVal(status.code).Move(), jst.GetAllocator());
  jsv.AddMember("msg",
                JVal().SetString(status_message.c_str(), jst.GetAllocator()),
                jst.GetAllocator());
  jst.AddMember("status", jsv, jst.GetAllocator());

  rapidjson::StringBuffer buffer;
  rapidjson::Writer<rapidjson::StringBuffer, rapidjson::UTF8<>,
                    rapidjson::UTF8<>, rapidjson::CrtAllocator,
                    rapidjson::kWriteNanAndInfFlag>
      writer(buffer);
  jst.Accept(writer);

  auto response
      = oatpp::web::protocol::http::outgoing::ResponseFactory::createResponse(
          status, buffer.GetString());
  for (const auto &pair : headers.getAll())
    {
      response->putHeader(pair.first.toString(), pair.second.toString());
    }

  response->putHeader(oatpp::web::protocol::http::Header::CONTENT_TYPE,
                      "application/json");

  if (!FLAGS_allow_origin.empty())
    response->putHeader(
        "Access-Control-Allow-Origin",
        oatpp::base::StrBuffer::createFromCString(FLAGS_allow_origin.c_str()));

  return response;
}

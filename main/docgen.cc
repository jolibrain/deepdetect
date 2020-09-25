/**
 * DeepDetect
 * Copyright (c) 2020 Jolibrain
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

#include <staticjson/staticjson.hpp>
#include <rapidjson/document.h>
#include <rapidjson/prettywriter.h>
#include <rapidjson/writer.h>
#include <rapidjson/filewritestream.h>
#include "backends/ncnn/ncnnlib.h"

int main()
{
  dd::NCNNLibInitParameters p;
  rapidjson::Document schema = staticjson::export_json_schema(&p);

  rapidjson::StringBuffer buffer;
  rapidjson::Writer<rapidjson::StringBuffer, rapidjson::UTF8<>,
                    rapidjson::UTF8<>, rapidjson::CrtAllocator,
                    rapidjson::kWriteNanAndInfFlag>
      writer(buffer);
  schema.Accept(writer);
  std::string out = buffer.GetString();
  std::cout << out << std::endl;
  return 0;
}

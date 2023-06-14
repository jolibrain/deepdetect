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

#ifndef DD_DTO_TYPES_HPP
#define DD_DTO_TYPES_HPP

#include <opencv2/opencv.hpp>

#include "oatpp/core/Types.hpp"
#include "oatpp/parser/json/mapping/ObjectMapper.hpp"
#include "apidata.h"
#include "utils/cv_utils.hpp"

namespace dd
{
  namespace DTO
  {
    struct VGpuIds
    {
      std::vector<int> _ids;

      VGpuIds() : _ids({ 0 })
      {
      }
      VGpuIds(const oatpp::Vector<oatpp::Int32> &vec)
          : _ids(vec->begin(), vec->end())
      {
      }
      VGpuIds(oatpp::Int32 i) : _ids({ i })
      {
      }
    };

    struct VImage
    {
      cv::Mat _img;
#ifdef USE_CUDA_CV
      cv::cuda::GpuMat _cuda_img;
#endif
      std::string _ext = ".png";

      VImage(const cv::Mat &img, const std::string &ext = ".png")
          : _img(img), _ext(ext)
      {
      }
#ifdef USE_CUDA_CV
      VImage(const cv::cuda::GpuMat &cuda_img, const std::string &ext = ".png")
          : _cuda_img(cuda_img), _ext(ext)
      {
      }
#endif
      bool is_cuda() const
      {
#ifdef USE_CUDA_CV
        return !_cuda_img.empty();
#else
        return false;
#endif
      }

      /** get image on CPU whether it's on GPU or not */
      const cv::Mat &get_img()
      {
#ifdef USE_CUDA_CV
        if (is_cuda())
          {
            _cuda_img.download(_img);
          }
#endif
        return _img;
      }
    };

    namespace __class
    {
      class APIDataClass;
      class GpuIdsClass;
      class ImageClass;
      template <typename T> class DTOVectorClass;
    }

    typedef oatpp::data::mapping::type::Primitive<APIData,
                                                  __class::APIDataClass>
        DTOApiData;
    typedef oatpp::data::mapping::type::Primitive<VGpuIds,
                                                  __class::GpuIdsClass>
        GpuIds;
    typedef oatpp::data::mapping::type::Primitive<VImage, __class::ImageClass>
        DTOImage;
    template <typename T>
    using DTOVector
        = oatpp::data::mapping::type::Primitive<std::vector<T>,
                                                __class::DTOVectorClass<T>>;

    namespace __class
    {
      class APIDataClass
      {
      public:
        static const oatpp::ClassId CLASS_ID;
        static oatpp::Type *getType()
        {
          static oatpp::Type type(CLASS_ID);
          return &type;
        }
      };

      class GpuIdsClass
      {
      public:
        static const oatpp::ClassId CLASS_ID;

        static oatpp::Type *getType()
        {
          static oatpp::Type type(CLASS_ID);
          return &type;
        }
      };

      class ImageClass
      {
      public:
        static const oatpp::ClassId CLASS_ID;

        static oatpp::Type *getType()
        {
          static oatpp::Type type(CLASS_ID);
          return &type;
        }
      };

      template <typename T> class DTOVectorClass
      {
      public:
        static const oatpp::ClassId CLASS_ID;

        static oatpp::Type *getType()
        {
          static oatpp::Type type(CLASS_ID);
          return &type;
        }
      };

      template class DTOVectorClass<double>;
      template class DTOVectorClass<bool>;
    }

    // ==== Serialization methods

    static inline oatpp::Void apiDataDeserialize(
        oatpp::parser::json::mapping::Deserializer *deserializer,
        oatpp::parser::Caret &caret, const oatpp::Type *const type)
    {
      (void)type;
      (void)deserializer;
      // XXX: this has a failure case if the stream contains excaped "{" or "}"
      // Since this is a temporary workaround until we use DTO everywhere, it
      // might not be required to be fixed
      if (caret.isAtChar('{'))
        {
          auto start = caret.getCurrData();
          int nest_lvl = 1;
          int length = 1;

          while (nest_lvl != 0)
            {
              if (!caret.canContinue())
                {
                  caret.setError("[apiDataDeserialize] missing '}'",
                                 oatpp::parser::json::mapping::Deserializer::
                                     ERROR_CODE_OBJECT_SCOPE_CLOSE);
                  return nullptr;
                }
              caret.inc();
              length++;
              if (caret.isAtChar('}'))
                nest_lvl--;
              else if (caret.isAtChar('{'))
                nest_lvl++;
            }

          // read to APIData
          DTOApiData dto_ad = APIData();
          JDoc d;
          d.Parse<rapidjson::kParseNanAndInfFlag>(start, length);
          dto_ad->fromRapidJson(d);
          return dto_ad;
        }
      else
        {
          caret.setError("[apiDataDeserialize] missing '{'",
                         oatpp::parser::json::mapping::Deserializer::
                             ERROR_CODE_OBJECT_SCOPE_OPEN);
          return nullptr;
        }
    }

    static inline void
    apiDataSerialize(oatpp::parser::json::mapping::Serializer *serializer,
                     oatpp::data::stream::ConsistentOutputStream *stream,
                     const oatpp::Void &obj)
    {
      (void)serializer;
      auto dto_ad = obj.cast<DTOApiData>();

      // APIData to string
      JDoc d;
      d.SetObject();
      dto_ad->toJDoc(d);

      rapidjson::StringBuffer buffer;
      rapidjson::Writer<rapidjson::StringBuffer, rapidjson::UTF8<>,
                        rapidjson::UTF8<>, rapidjson::CrtAllocator,
                        rapidjson::kWriteNanAndInfFlag>
          writer(buffer);
      bool done = d.Accept(writer);
      if (!done)
        throw DataConversionException("JSON rendering failed");

      stream->writeSimple(buffer.GetString());
    }

    static inline oatpp::Void
    gpuIdsDeserialize(oatpp::parser::json::mapping::Deserializer *deserializer,
                      oatpp::parser::Caret &caret,
                      const oatpp::Type *const type)
    {
      (void)type;
      if (caret.isAtChar('['))
        {
          return GpuIds(VGpuIds{
              deserializer
                  ->deserialize(caret,
                                oatpp::Vector<oatpp::Int32>::Class::getType())
                  .cast<oatpp::Vector<oatpp::Int32>>() });
        }
      else
        {
          return GpuIds(VGpuIds{
              deserializer->deserialize(caret, oatpp::Int32::Class::getType())
                  .cast<oatpp::Int32>() });
        }
    }

    static inline void
    gpuIdsSerialize(oatpp::parser::json::mapping::Serializer *serializer,
                    oatpp::data::stream::ConsistentOutputStream *stream,
                    const oatpp::Void &obj)
    {
      auto gpuid = obj.cast<GpuIds>();
      if (gpuid->_ids.size() == 1)
        {
          oatpp::Int32 id = gpuid->_ids[0];
          serializer->serializeToStream(stream, id);
        }
      else
        {
          oatpp::Vector<oatpp::Int32> ids;
          for (auto i : gpuid->_ids)
            {
              ids->push_back(i);
            }
          serializer->serializeToStream(stream, ids);
        }
    }

    static inline oatpp::Void
    imageDeserialize(oatpp::parser::json::mapping::Deserializer *deserializer,
                     oatpp::parser::Caret &caret,
                     const oatpp::Type *const type)
    {
      (void)type;
      auto str_base64
          = deserializer->deserialize(caret, oatpp::String::Class::getType())
                .cast<oatpp::String>();
      return DTOImage(VImage{ cv_utils::base64_to_image(*str_base64) });
    }

    static inline void
    imageSerialize(oatpp::parser::json::mapping::Serializer *serializer,
                   oatpp::data::stream::ConsistentOutputStream *stream,
                   const oatpp::Void &obj)
    {
      (void)serializer;
      auto img_dto = obj.cast<DTOImage>();
      std::string encoded
          = cv_utils::image_to_base64(img_dto->get_img(), img_dto->_ext);
      stream->writeSimple(encoded);
    }

    // Inspired by oatpp json deserializer
    template <typename T> inline T readVecElement(oatpp::parser::Caret &caret);

    template <>
    inline double readVecElement<double>(oatpp::parser::Caret &caret)
    {
      return caret.parseFloat64();
    }

    template <>
    inline uint8_t readVecElement<uint8_t>(oatpp::parser::Caret &caret)
    {
      return static_cast<uint8_t>(caret.parseUnsignedInt());
    }

    template <> inline bool readVecElement<bool>(oatpp::parser::Caret &caret)
    {
      if (caret.isAtText("true"))
        {
          return true;
        }
      else if (caret.isAtText("false"))
        {
          return false;
        }
      else
        {
          caret.setError("[readVecElement<bool>] expected 'true' or 'false'",
                         oatpp::parser::json::mapping::Deserializer::
                             ERROR_CODE_VALUE_BOOLEAN);
          return false;
        }
    }

    template <typename T>
    static inline oatpp::Void
    vectorDeserialize(oatpp::parser::json::mapping::Deserializer *deserializer,
                      oatpp::parser::Caret &caret,
                      const oatpp::Type *const type)
    {
      (void)deserializer;
      (void)type;
      if (caret.isAtText("null", true))
        {
          return oatpp::Void(type);
        }

      if (caret.canContinueAtChar('[', 1))
        {
          std::vector<T> vec;

          while (!caret.isAtChar(']') && caret.canContinue())
            {
              caret.skipBlankChars();
              vec.push_back(caret.parseFloat64());
              if (caret.hasError())
                {
                  return nullptr;
                }
              caret.skipBlankChars();
              caret.canContinueAtChar(',', 1);
            }

          if (!caret.canContinueAtChar(']', 1))
            {
              if (!caret.hasError())
                {
                  caret.setError(
                      "[oatpp::parser::json::mapping::Deserializer::"
                      "deserializeList()]: Error. ']' - expected",
                      oatpp::parser::json::mapping::Deserializer::
                          ERROR_CODE_ARRAY_SCOPE_CLOSE);
                }
              return nullptr;
            };

          return DTOVector<T>(std::move(vec));
        }
      else
        {
          caret.setError("[oatpp::parser::json::mapping::Deserializer::"
                         "deserializeList()]: Error. '[' - expected",
                         oatpp::parser::json::mapping::Deserializer::
                             ERROR_CODE_ARRAY_SCOPE_OPEN);
          return nullptr;
        }
    }

    template <typename T>
    static inline void
    vectorSerialize(oatpp::parser::json::mapping::Serializer *serializer,
                    oatpp::data::stream::ConsistentOutputStream *stream,
                    const oatpp::Void &obj)
    {
      (void)serializer;
      auto vec = obj.cast<DTOVector<T>>();
      stream->writeCharSimple('[');
      bool first = true;

      for (auto val : *vec)
        {
          if (first)
            first = false;
          else
            stream->writeCharSimple(',');
          stream->writeAsString(val);
        }

      stream->writeCharSimple(']');
    }
  }
}

#endif // DD_DTO_TYPES_HPP

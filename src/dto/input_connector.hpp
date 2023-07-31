/**
 * DeepDetect
 * Copyright (c) 2021 Jolibrain SASU
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

#ifndef DTO_INPUT_CONNECTOR_HPP
#define DTO_INPUT_CONNECTOR_HPP

#include "dd_config.h"
#include "oatpp/core/Types.hpp"
#include "oatpp/core/macro/codegen.hpp"

namespace dd
{
  namespace DTO
  {
#include OATPP_CODEGEN_BEGIN(DTO)

    class InputConnector : public oatpp::DTO
    {
      DTO_INIT(InputConnector, DTO /* extends */)
      // Connector type
      DTO_FIELD_INFO(connector)
      {
        info->description
            = "Type of the input connector, eg image, csv, text...";
      }
      DTO_FIELD(String, connector);

      DTO_FIELD_INFO(timeout)
      {
        info->description = "timeout on input data retrieval: -1 means using "
                            "default (600sec)";
      }
      DTO_FIELD(Int32, timeout) = -1;

      DTO_FIELD(Boolean, shuffle);
      DTO_FIELD(Int32, seed);
      DTO_FIELD(Float64, test_split);

      // bool for csv/csvts, float for img
      DTO_FIELD_INFO(scale)
      {
        info->description
            = "[csv] bool, whether input should be scaled"
              "\n\n"
              "[image] float, factor to scale pixel values. For exemple, "
              "1/256 means pixel values will land between 0 and 1.";
      }
      DTO_FIELD(Any, scale);

      // IMG Input Connector
      DTO_FIELD_INFO(width)
      {
        info->description
            = "Resize image to specified width. -1 for no resizing";
      }
      DTO_FIELD(Int32, width);

      DTO_FIELD_INFO(height)
      {
        info->description
            = "Resize image to specified height. -1 for no resizing";
      }
      DTO_FIELD(Int32, height);
      DTO_FIELD(Int32, crop_width);
      DTO_FIELD(Int32, crop_height);
      DTO_FIELD(Boolean, bw);
      DTO_FIELD(Boolean, rgb);
      DTO_FIELD(Boolean, histogram_equalization);
      DTO_FIELD(Boolean, unchanged_data);
      DTO_FIELD(Vector<Float32>, mean);
      DTO_FIELD(Vector<Float32>, std);
      DTO_FIELD(Boolean, scaled);
      DTO_FIELD(Int32, scale_min);
      DTO_FIELD(Int32, scale_max);
      DTO_FIELD(Boolean, keep_orig);
      DTO_FIELD(String, interp);

      DTO_FIELD_INFO(bbox)
      {
        info->description = "[training] true if data contains a bbox dataset";
      }
      DTO_FIELD(Boolean, bbox);

      // Text input connector
      DTO_FIELD_INFO(count)
      {
        info->description = "whether to add up word counters";
      }
      DTO_FIELD(Boolean, count);

      DTO_FIELD_INFO(tfidf)
      {
        info->description = "whether to use TF/IDF";
      }
      DTO_FIELD(Boolean, tfidf);

      DTO_FIELD_INFO(min_count)
      {
        info->description = "min word occurence";
      }
      DTO_FIELD(Int32, min_count);

      DTO_FIELD_INFO(min_word_length)
      {
        info->description = "min word length";
      }
      DTO_FIELD(Int32, min_word_length);

      DTO_FIELD_INFO(sentences)
      {
        info->description
            = "whether to consider every sentence (\\n separated) "
              "as a document";
      }
      DTO_FIELD(Boolean, sentences);

      DTO_FIELD_INFO(characters)
      {
        info->description = "whether to use character-level input features";
      }
      DTO_FIELD(Boolean, characters);

      DTO_FIELD_INFO(ordered_words)
      {
        info->description
            = "whether to consider the position of each words in "
              "the sentence";
      }
      DTO_FIELD(Boolean, ordered_words);

      DTO_FIELD_INFO(lower_case)
      {
        info->description = "whether the input should be lower cased before "
                            "processing";
      }
      DTO_FIELD(Boolean, lower_case);

      DTO_FIELD_INFO(wordpiece_tokens)
      {
        info->description = "whether to try to match word pieces from the "
                            "vocabulary";
      }
      DTO_FIELD(Boolean, wordpiece_tokens);

      DTO_FIELD_INFO(word_start)
      {
        info->description = "in most gpt2 vocabularies, start of word has "
                            "generally to be set to \"Ġ\".";
      }
      DTO_FIELD(String, word_start);

      DTO_FIELD_INFO(suffix_start)
      {
        info->description
            = "in most bert-like vocabularies, suffixes are prefixed by `##`.";
      }
      DTO_FIELD(String, suffix_start);

      DTO_FIELD_INFO(punctuation_tokens)
      {
        info->description = "accept punctuation tokens";
      }
      DTO_FIELD(Boolean, punctuation_tokens);

      DTO_FIELD_INFO(alphabet)
      {
        info->description = "character-level alphabet";
      }
      DTO_FIELD(String, alphabet);

      DTO_FIELD_INFO(sequence)
      {
        info->description
            = "sequence size when using character-level features";
      }
      DTO_FIELD(Int32, sequence);

      DTO_FIELD_INFO(read_forward)
      {
        info->description
            = "whether to read character-based sequences forward";
      }
      DTO_FIELD(Boolean, read_forward);

      // CSV Input Connector
      DTO_FIELD_INFO(id)
      {
        info->description = "[csv]";
      }
      DTO_FIELD(String, id);

      DTO_FIELD_INFO(separator)
      {
        info->description = "[csv]";
      }
      DTO_FIELD(String, separator);

      DTO_FIELD_INFO(quote)
      {
        info->description = "[csv]";
      }
      DTO_FIELD(String, quote);

      DTO_FIELD_INFO(ignore)
      {
        info->description = "set of ignored columns";
      }
      DTO_FIELD(Vector<String>, ignore);

      DTO_FIELD_INFO(label)
      {
        info->description = "label column, either a string or list of strings";
      }
      DTO_FIELD(Any, label);

      DTO_FIELD_INFO(label_offset)
      {
        info->description = "label offset, either an int or list of ints";
      }
      DTO_FIELD(Any, label_offset);

      DTO_FIELD_INFO(categoricals)
      {
        info->description = "auto-converted categorical variables";
      }
      DTO_FIELD(Vector<String>, categoricals);

      DTO_FIELD_INFO(categoricals_mapping)
      {
        info->description = "[csv]";
      }
      DTO_FIELD(DTOApiData, categoricals_mapping);

      // Scale vals
      DTO_FIELD_INFO(scale_type)
      {
        info->description = "type of scaling";
      }
      DTO_FIELD(String, scale_type);

      DTO_FIELD_INFO(min_vals)
      {
        info->description = "min values for scaling";
      }
      DTO_FIELD(Vector<Float32>, min_vals);

      DTO_FIELD_INFO(max_vals)
      {
        info->description = "max values for scaling";
      }
      DTO_FIELD(Vector<Float32>, max_vals);

      DTO_FIELD_INFO(mean_vals)
      {
        info->description = "mean values for scaling";
      }
      DTO_FIELD(Vector<Float32>, mean_vals);

      DTO_FIELD_INFO(variance_vals)
      {
        info->description = "variance values for scaling";
      }
      DTO_FIELD(Vector<Float32>, variance_vals);

      // CSVTS Input Connector
      DTO_FIELD_INFO(continuation)
      {
        info->description
            = "true if this call is the continuation of the previous one";
      }
      DTO_FIELD(Boolean, continuation) = false;

      // image resizing on GPU
#ifdef USE_CUDA_CV
      DTO_FIELD(Boolean, cuda);
#endif
    };

#include OATPP_CODEGEN_END(DTO)
  }
}

#endif

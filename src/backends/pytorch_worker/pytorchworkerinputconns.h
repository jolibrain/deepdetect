/**
 * DeepDetect
 * Copyright (c) 2026 Jolibrain
 *
 * This file is part of deepdetect.
 */

#ifndef PYTORCHWORKERINPUTCONNS_H
#define PYTORCHWORKERINPUTCONNS_H

#include "imginputfileconn.h"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <map>
#include <random>
#include <sstream>
#include <unistd.h>

namespace dd
{
  class ImgPytorchInputFileConn : public ImgInputFileConn
  {
  public:
    ImgPytorchInputFileConn() : ImgInputFileConn()
    {
    }

    ~ImgPytorchInputFileConn()
    {
      cleanup_inline_detection_pull_session();
    }

    ImgPytorchInputFileConn(const ImgPytorchInputFileConn &other)
        : ImgInputFileConn(other)
    {
    }

    int width() const
    {
      return _width;
    }

    int height() const
    {
      return _height;
    }

    APIData inline_detection_tensor_batches(const APIData &ad)
    {
      APIData input_params;
      if (ad.has("parameters") && ad.getobj("parameters").has("input"))
        input_params = ad.getobj("parameters").getobj("input");
      fillup_parameters(input_params);
      if (input_params.has("bbox") && !input_params.get("bbox").get<bool>())
        throw InputConnectorBadParamException(
            "connector_tensor_inline requires input bbox=true");

      if (!ad.has("data"))
        throw InputConnectorBadParamException(
            "connector_tensor_inline requires train data");
      const std::vector<std::string> data
          = ad.get("data").get<std::vector<std::string>>();
      if (data.empty())
        throw InputConnectorBadParamException(
            "connector_tensor_inline requires a train list path");
      const int max_samples = inline_tensor_max_samples(ad);

      APIData tensor_batches;
      tensor_batches.add("train",
                         inline_detection_batches(data[0], max_samples));
      std::vector<APIData> tests;
      for (size_t index = 1; index < data.size(); ++index)
        {
          APIData test_set;
          test_set.add("batches",
                       inline_detection_batches(data[index], max_samples));
          tests.push_back(test_set);
        }
      tensor_batches.add("tests", tests);
      return tensor_batches;
    }

    void start_inline_detection_pull_session(const APIData &ad)
    {
      cleanup_inline_detection_pull_session();
      APIData input_params;
      if (ad.has("parameters") && ad.getobj("parameters").has("input"))
        input_params = ad.getobj("parameters").getobj("input");
      fillup_parameters(input_params);
      configure_pull_transport(ad);
      if (input_params.has("bbox") && !input_params.get("bbox").get<bool>())
        throw InputConnectorBadParamException(
            "connector_tensor_pull requires input bbox=true");

      if (!ad.has("data"))
        throw InputConnectorBadParamException(
            "connector_tensor_pull requires train data");
      const std::vector<std::string> data
          = ad.get("data").get<std::vector<std::string>>();
      if (data.empty())
        throw InputConnectorBadParamException(
            "connector_tensor_pull requires a train list path");

      _pull_train = read_detection_pairs(data[0]);
      _pull_tests.clear();
      for (size_t index = 1; index < data.size(); ++index)
        _pull_tests.push_back(read_detection_pairs(data[index]));
      _pull_train_pos = 0;
      _pull_test_pos.assign(_pull_tests.size(), 0);
      _pull_epoch = 0;
      _pull_next_batch_id = 0;
      shuffle_pull_train();
      _pull_active = true;
    }

    APIData connector_batch_done(const APIData &params)
    {
      if (params.has("batch_id"))
        cleanup_pull_batch(params.get("batch_id").get<std::string>());
      APIData result;
      result.add("status", std::string("ok"));
      return result;
    }

    void cleanup_inline_detection_pull_session()
    {
      for (const auto &item : _pull_batch_files)
        for (const auto &path : item.second)
          {
            std::error_code ec;
            std::filesystem::remove(path, ec);
          }
      _pull_batch_files.clear();
      if (!_pull_shm_dir.empty())
        {
          std::error_code ec;
          std::filesystem::remove_all(_pull_shm_dir, ec);
          _pull_shm_dir.clear();
        }
      _pull_active = false;
    }

    APIData connector_dataset_info() const
    {
      if (!_pull_active)
        throw InputConnectorBadParamException(
            "connector_tensor_pull session is not active");
      std::vector<int> test_samples;
      test_samples.reserve(_pull_tests.size());
      for (const auto &test_set : _pull_tests)
        test_samples.push_back(static_cast<int>(test_set.size()));
      APIData info;
      info.add("task", std::string("detection"));
      info.add("boundary", std::string("connector-tensor-pull"));
      info.add("train_samples", static_cast<int>(_pull_train.size()));
      info.add("test_samples", test_samples);
      info.add("test_sets_total", static_cast<int>(_pull_tests.size()));
      return info;
    }

    APIData connector_batch_next(const APIData &params)
    {
      if (!_pull_active)
        throw InputConnectorBadParamException(
            "connector_tensor_pull session is not active");
      std::string split = "train";
      if (params.has("split"))
        split = params.get("split").get<std::string>();
      int batch_size = 1;
      if (params.has("batch_size"))
        batch_size = params.get("batch_size").get<int>();
      if (batch_size <= 0)
        throw InputConnectorBadParamException(
            "connector_tensor_pull batch_size must be positive");
      bool reset_epoch = false;
      if (params.has("reset_epoch"))
        reset_epoch = params.get("reset_epoch").get<bool>();

      if (split == "train")
        return connector_batch_next_from(_pull_train, _pull_train_pos,
                                         batch_size, reset_epoch, true);
      if (split != "test")
        throw InputConnectorBadParamException(
            "connector_tensor_pull split must be train or test");
      int test_index = 0;
      if (params.has("test_index"))
        test_index = params.get("test_index").get<int>();
      if (test_index < 0 || test_index >= static_cast<int>(_pull_tests.size()))
        throw InputConnectorBadParamException(
            "connector_tensor_pull test_index out of range");
      return connector_batch_next_from(
          _pull_tests[static_cast<size_t>(test_index)],
          _pull_test_pos[static_cast<size_t>(test_index)], batch_size,
          reset_epoch, false);
    }

  private:
    struct DetectionBBox
    {
      int label;
      double xmin;
      double ymin;
      double xmax;
      double ymax;
    };

    using DetectionPair
        = std::pair<std::filesystem::path, std::filesystem::path>;

    void configure_pull_transport(const APIData &ad)
    {
      _pull_transport = "shared_memory";
      std::filesystem::path base_dir("/dev/shm/deepdetect-pytorch");
      if (!std::filesystem::exists("/dev/shm"))
        base_dir
            = std::filesystem::temp_directory_path() / "deepdetect-pytorch";
      if (ad.has("parameters"))
        {
          APIData parameters = ad.getobj("parameters");
          if (parameters.has("mllib"))
            {
              APIData mllib = parameters.getobj("mllib");
              if (mllib.has("connector_tensor_transport"))
                _pull_transport = mllib.get("connector_tensor_transport")
                                      .get<std::string>();
              if (mllib.has("connector_shared_memory_dir"))
                base_dir = std::filesystem::path(
                    mllib.get("connector_shared_memory_dir")
                        .get<std::string>());
            }
        }
      if (_pull_transport != "shared_memory" && _pull_transport != "inline")
        throw InputConnectorBadParamException(
            "mllib.connector_tensor_transport must be shared_memory or "
            "inline");
      if (_pull_transport == "shared_memory")
        {
          const auto now
              = std::chrono::steady_clock::now().time_since_epoch().count();
          _pull_shm_dir
              = base_dir
                / ("session-"
                   + std::to_string(static_cast<long long>(getpid())) + "-"
                   + std::to_string(static_cast<long long>(now)));
          std::filesystem::create_directories(_pull_shm_dir);
          std::filesystem::permissions(_pull_shm_dir,
                                       std::filesystem::perms::owner_all,
                                       std::filesystem::perm_options::replace);
        }
    }

    std::vector<APIData> inline_detection_batches(const std::string &list_path,
                                                  int max_samples) const
    {
      std::vector<APIData> batches;
      const std::filesystem::path list_file
          = std::filesystem::absolute(std::filesystem::path(list_path));
      std::ifstream input(list_file);
      if (!input.is_open())
        throw InputConnectorBadParamException("Could not open image list: "
                                              + list_path);
      std::string line;
      int sample_index = 0;
      while (std::getline(input, line))
        {
          if (line.empty())
            continue;
          if (max_samples > 0 && sample_index >= max_samples)
            throw InputConnectorBadParamException(
                "connector_tensor_inline is limited to "
                + std::to_string(max_samples)
                + " samples per dataset list; use tiny smoke-test lists or "
                  "raise mllib.connector_tensor_inline_max_samples");
          std::istringstream row(line);
          std::string image_path;
          std::string bbox_path;
          row >> image_path >> bbox_path;
          if (image_path.empty() || bbox_path.empty())
            throw InputConnectorBadParamException(
                "connector_tensor_inline expects image and bbox path in "
                + list_path);
          batches.push_back(inline_detection_batch(
              resolve_dataset_path(list_file.parent_path(), image_path),
              resolve_dataset_path(list_file.parent_path(), bbox_path),
              sample_index));
          ++sample_index;
        }
      if (batches.empty())
        throw InputConnectorBadParamException(
            "image list contains no samples: " + list_path);
      return batches;
    }

    static int inline_tensor_max_samples(const APIData &ad)
    {
      constexpr int default_max_samples = 128;
      if (!ad.has("parameters"))
        return default_max_samples;
      APIData parameters = ad.getobj("parameters");
      if (!parameters.has("mllib"))
        return default_max_samples;
      APIData mllib = parameters.getobj("mllib");
      if (!mllib.has("connector_tensor_inline_max_samples"))
        return default_max_samples;
      int max_samples
          = mllib.get("connector_tensor_inline_max_samples").get<int>();
      if (max_samples <= 0)
        throw InputConnectorBadParamException(
            "mllib.connector_tensor_inline_max_samples must be positive");
      return max_samples;
    }

    std::vector<DetectionPair>
    read_detection_pairs(const std::string &list_path) const
    {
      std::vector<DetectionPair> pairs;
      const std::filesystem::path list_file
          = std::filesystem::absolute(std::filesystem::path(list_path));
      std::ifstream input(list_file);
      if (!input.is_open())
        throw InputConnectorBadParamException("Could not open image list: "
                                              + list_path);
      std::string line;
      while (std::getline(input, line))
        {
          if (line.empty())
            continue;
          std::istringstream row(line);
          std::string image_path;
          std::string bbox_path;
          row >> image_path >> bbox_path;
          if (image_path.empty() || bbox_path.empty())
            throw InputConnectorBadParamException(
                "connector_tensor_pull expects image and bbox path in "
                + list_path);
          pairs.emplace_back(
              resolve_dataset_path(list_file.parent_path(), image_path),
              resolve_dataset_path(list_file.parent_path(), bbox_path));
        }
      if (pairs.empty())
        throw InputConnectorBadParamException(
            "image list contains no samples: " + list_path);
      return pairs;
    }

    void shuffle_pull_train()
    {
      if (!_shuffle || _pull_train.empty())
        return;
      std::mt19937 rng(static_cast<unsigned int>(_seed + _pull_epoch));
      std::shuffle(_pull_train.begin(), _pull_train.end(), rng);
      ++_pull_epoch;
    }

    APIData connector_batch_next_from(const std::vector<DetectionPair> &pairs,
                                      size_t &cursor, int batch_size,
                                      bool reset_epoch, bool shuffle_on_reset)
    {
      if (reset_epoch)
        {
          cursor = 0;
          if (shuffle_on_reset)
            shuffle_pull_train();
        }
      APIData response;
      response.add("status", std::string("ok"));
      if (cursor >= pairs.size())
        {
          response.add("end", true);
          return response;
        }
      const size_t count
          = std::min(static_cast<size_t>(batch_size), pairs.size() - cursor);
      const std::string batch_id = next_pull_batch_id();
      response.add("end", false);
      response.add("batch_id", batch_id);
      response.add("batch",
                   inline_detection_batch(pairs, cursor, count, batch_id));
      cursor += count;
      return response;
    }

    std::string next_pull_batch_id()
    {
      ++_pull_next_batch_id;
      return std::to_string(_pull_next_batch_id);
    }

    APIData inline_detection_batch(const std::filesystem::path &image_path,
                                   const std::filesystem::path &bbox_path,
                                   int sample_index) const
    {
      DDImg dimg;
      copy_parameters_to(dimg);
      try
        {
          if (dimg.read_file(image_path.string(), -1))
            throw InputConnectorBadParamException("Could not read image: "
                                                  + image_path.string());
        }
      catch (const std::exception &error)
        {
          throw InputConnectorBadParamException("Could not read image: "
                                                + image_path.string() + ": "
                                                + error.what());
        }
      if (dimg._imgs.empty())
        throw InputConnectorBadParamException("Could not read image: "
                                              + image_path.string());
      const cv::Mat &image = dimg._imgs[0];
      const int orig_height
          = dimg._imgs_size.empty() ? image.rows : dimg._imgs_size[0].first;
      const int orig_width
          = dimg._imgs_size.empty() ? image.cols : dimg._imgs_size[0].second;
      const std::vector<DetectionBBox> bboxes
          = read_detection_bboxes(bbox_path, orig_width, orig_height);

      APIData batch;
      batch.add("kind", std::string("tensor_batch"));
      batch.add("inputs",
                std::vector<APIData>{ inline_image_tensor_ref(image) });
      batch.add("targets", detection_targets(bboxes));
      batch.add("meta", detection_meta(sample_index, image_path.string(),
                                       image.cols, image.rows));
      return batch;
    }

    APIData inline_detection_batch(const std::vector<DetectionPair> &pairs,
                                   size_t offset, size_t count,
                                   const std::string &batch_id)
    {
      std::vector<double> values;
      std::vector<std::vector<DetectionBBox>> targets;
      std::vector<int> sample_ids;
      std::vector<std::string> paths;
      std::vector<int> widths;
      std::vector<int> heights;
      int rows = 0;
      int cols = 0;
      int channels = 0;

      for (size_t index = 0; index < count; ++index)
        {
          const DetectionPair &pair = pairs[offset + index];
          DDImg dimg;
          copy_parameters_to(dimg);
          try
            {
              if (dimg.read_file(pair.first.string(), -1))
                throw InputConnectorBadParamException("Could not read image: "
                                                      + pair.first.string());
            }
          catch (const std::exception &error)
            {
              throw InputConnectorBadParamException("Could not read image: "
                                                    + pair.first.string()
                                                    + ": " + error.what());
            }
          if (dimg._imgs.empty())
            throw InputConnectorBadParamException("Could not read image: "
                                                  + pair.first.string());
          const cv::Mat &image = dimg._imgs[0];
          const int orig_height = dimg._imgs_size.empty()
                                      ? image.rows
                                      : dimg._imgs_size[0].first;
          const int orig_width = dimg._imgs_size.empty()
                                     ? image.cols
                                     : dimg._imgs_size[0].second;
          if (index == 0)
            {
              rows = image.rows;
              cols = image.cols;
              channels = image.channels();
            }
          else if (rows != image.rows || cols != image.cols
                   || channels != image.channels())
            {
              throw InputConnectorBadParamException(
                  "connector_tensor_pull batch images must have matching "
                  "preprocessed dimensions");
            }
          std::vector<double> image_values = image_values_chw(image);
          values.insert(values.end(), image_values.begin(),
                        image_values.end());
          targets.push_back(
              read_detection_bboxes(pair.second, orig_width, orig_height));
          sample_ids.push_back(static_cast<int>(offset + index));
          paths.push_back(pair.first.string());
          widths.push_back(image.cols);
          heights.push_back(image.rows);
        }

      APIData batch;
      batch.add("kind", std::string("tensor_batch"));
      batch.add("inputs", std::vector<APIData>{ pull_image_tensor_ref(
                              values, static_cast<int>(count), channels, rows,
                              cols, batch_id) });
      batch.add("targets", detection_targets(targets));
      batch.add("meta", detection_meta(sample_ids, paths, widths, heights));
      return batch;
    }

    APIData inline_image_tensor_ref(const cv::Mat &image) const
    {
      return inline_image_tensor_ref(image_values_chw(image), 1,
                                     image.channels(), image.rows, image.cols,
                                     "connector_tensor_inline");
    }

    APIData inline_image_tensor_ref(const std::vector<double> &values,
                                    int batch_size, int channels, int rows,
                                    int cols, const std::string &name) const
    {
      APIData storage;
      storage.add("type", std::string("inline_test_stub"));
      storage.add("name", name);
      storage.add("offset", 0);
      storage.add("nbytes", 0);
      storage.add("values", values);

      APIData lifetime;
      lifetime.add("owner", std::string("deepdetect"));
      lifetime.add("valid_until_ack", std::string("batch_done"));

      APIData cuda;

      APIData tensor;
      tensor.add("kind", std::string("tensor_ref"));
      tensor.add("device", std::string("cpu"));
      tensor.add("dtype", std::string("float32"));
      tensor.add("shape",
                 std::vector<int>{ batch_size, channels, rows, cols });
      tensor.add("layout", std::string("strided"));
      tensor.add("storage", storage);
      tensor.add("lifetime", lifetime);
      tensor.add("cuda", cuda);
      return tensor;
    }

    APIData pull_image_tensor_ref(const std::vector<double> &values,
                                  int batch_size, int channels, int rows,
                                  int cols, const std::string &batch_id)
    {
      if (_pull_transport == "inline")
        return inline_image_tensor_ref(values, batch_size, channels, rows,
                                       cols, "connector_tensor_pull");
      return shared_memory_image_tensor_ref(values, batch_size, channels, rows,
                                            cols, batch_id);
    }

    APIData shared_memory_image_tensor_ref(const std::vector<double> &values,
                                           int batch_size, int channels,
                                           int rows, int cols,
                                           const std::string &batch_id)
    {
      if (_pull_shm_dir.empty())
        throw InputConnectorBadParamException(
            "connector_tensor_pull shared memory session is not initialized");
      const std::filesystem::path path
          = _pull_shm_dir / ("batch-" + batch_id + "-input0.bin");
      std::ofstream out(path, std::ios::binary | std::ios::trunc);
      if (!out.is_open())
        throw InputConnectorBadParamException(
            "Could not create shared memory tensor file: " + path.string());
      for (double value : values)
        {
          const float stored = static_cast<float>(value);
          out.write(reinterpret_cast<const char *>(&stored), sizeof(stored));
        }
      out.close();
      if (!out.good())
        throw InputConnectorBadParamException(
            "Could not write shared memory tensor file: " + path.string());
      _pull_batch_files[batch_id].push_back(path);

      APIData storage;
      storage.add("type", std::string("shared_memory"));
      storage.add("name", path.string());
      storage.add("offset", 0);
      storage.add("nbytes", static_cast<int>(values.size() * sizeof(float)));

      APIData lifetime;
      lifetime.add("owner", std::string("deepdetect"));
      lifetime.add("valid_until_ack", std::string("batch_done"));
      lifetime.add("batch_id", batch_id);

      APIData cuda;

      APIData tensor;
      tensor.add("kind", std::string("tensor_ref"));
      tensor.add("device", std::string("cpu"));
      tensor.add("dtype", std::string("float32"));
      tensor.add("shape",
                 std::vector<int>{ batch_size, channels, rows, cols });
      tensor.add("layout", std::string("strided"));
      tensor.add("storage", storage);
      tensor.add("lifetime", lifetime);
      tensor.add("cuda", cuda);
      return tensor;
    }

    void cleanup_pull_batch(const std::string &batch_id)
    {
      auto it = _pull_batch_files.find(batch_id);
      if (it == _pull_batch_files.end())
        return;
      for (const auto &path : it->second)
        {
          std::error_code ec;
          std::filesystem::remove(path, ec);
        }
      _pull_batch_files.erase(it);
    }

    std::vector<double> image_values_chw(const cv::Mat &image) const
    {
      if (image.depth() != CV_8U)
        throw InputConnectorBadParamException(
            "connector_tensor_inline expects 8-bit preprocessed images");
      if (image.channels() != 3)
        throw InputConnectorBadParamException(
            "connector_tensor_inline expects 3-channel images");
      std::vector<double> values;
      values.reserve(static_cast<size_t>(image.channels()) * image.rows
                     * image.cols);
      for (int channel = 0; channel < image.channels(); ++channel)
        for (int row = 0; row < image.rows; ++row)
          for (int col = 0; col < image.cols; ++col)
            values.push_back(
                static_cast<double>(image.at<cv::Vec3b>(row, col)[channel])
                / 255.0);
      return values;
    }

    std::vector<DetectionBBox>
    read_detection_bboxes(const std::filesystem::path &bbox_path,
                          int orig_width, int orig_height) const
    {
      std::ifstream input(bbox_path);
      if (!input.is_open())
        throw InputConnectorBadParamException("Could not open bbox file: "
                                              + bbox_path.string());
      if (orig_width <= 0 || orig_height <= 0)
        throw InputConnectorBadParamException(
            "Could not determine original image size for bbox file: "
            + bbox_path.string());
      std::vector<DetectionBBox> bboxes;
      const double wfactor = _width > 0 ? static_cast<double>(_width)
                                              / static_cast<double>(orig_width)
                                        : 1.0;
      const double hfactor = _height > 0
                                 ? static_cast<double>(_height)
                                       / static_cast<double>(orig_height)
                                 : 1.0;
      std::string line;
      while (std::getline(input, line))
        {
          if (line.empty())
            continue;
          std::istringstream row(line);
          DetectionBBox bbox;
          row >> bbox.label >> bbox.xmin >> bbox.ymin >> bbox.xmax
              >> bbox.ymax;
          if (!row)
            throw InputConnectorBadParamException("Invalid bbox line in: "
                                                  + bbox_path.string());
          bbox.xmin *= wfactor;
          bbox.xmax *= wfactor;
          bbox.ymin *= hfactor;
          bbox.ymax *= hfactor;
          bboxes.push_back(bbox);
        }
      return bboxes;
    }

    APIData detection_targets(const std::vector<DetectionBBox> &bboxes) const
    {
      APIData sample;
      std::vector<APIData> boxes;
      std::vector<int> labels;
      for (const DetectionBBox &bbox : bboxes)
        {
          APIData box;
          box.add("xmin", bbox.xmin);
          box.add("ymin", bbox.ymin);
          box.add("xmax", bbox.xmax);
          box.add("ymax", bbox.ymax);
          boxes.push_back(box);
          labels.push_back(bbox.label);
        }
      sample.add("boxes", boxes);
      sample.add("labels", labels);

      APIData targets;
      targets.add("samples", std::vector<APIData>{ sample });
      return targets;
    }

    APIData detection_targets(
        const std::vector<std::vector<DetectionBBox>> &items) const
    {
      std::vector<APIData> samples;
      samples.reserve(items.size());
      for (const auto &bboxes : items)
        {
          APIData sample;
          std::vector<APIData> boxes;
          std::vector<int> labels;
          for (const DetectionBBox &bbox : bboxes)
            {
              APIData box;
              box.add("xmin", bbox.xmin);
              box.add("ymin", bbox.ymin);
              box.add("xmax", bbox.xmax);
              box.add("ymax", bbox.ymax);
              boxes.push_back(box);
              labels.push_back(bbox.label);
            }
          sample.add("boxes", boxes);
          sample.add("labels", labels);
          samples.push_back(sample);
        }
      APIData targets;
      targets.add("samples", samples);
      return targets;
    }

    APIData detection_meta(int sample_index, const std::string &path,
                           int width, int height) const
    {
      APIData meta;
      meta.add("sample_ids", std::vector<int>{ sample_index });
      meta.add("paths", std::vector<std::string>{ path });
      meta.add("widths", std::vector<int>{ width });
      meta.add("heights", std::vector<int>{ height });
      return meta;
    }

    APIData detection_meta(const std::vector<int> &sample_ids,
                           const std::vector<std::string> &paths,
                           const std::vector<int> &widths,
                           const std::vector<int> &heights) const
    {
      APIData meta;
      meta.add("sample_ids", sample_ids);
      meta.add("paths", paths);
      meta.add("widths", widths);
      meta.add("heights", heights);
      return meta;
    }

    static std::filesystem::path
    resolve_dataset_path(const std::filesystem::path &base,
                         const std::string &value)
    {
      std::filesystem::path path(value);
      if (path.is_relative())
        path = base / path;
      return std::filesystem::absolute(path);
    }

    std::vector<DetectionPair> _pull_train;
    std::vector<std::vector<DetectionPair>> _pull_tests;
    size_t _pull_train_pos = 0;
    std::vector<size_t> _pull_test_pos;
    std::string _pull_transport = "inline";
    std::filesystem::path _pull_shm_dir;
    std::map<std::string, std::vector<std::filesystem::path>>
        _pull_batch_files;
    int _pull_next_batch_id = 0;
    int _pull_epoch = 0;
    bool _pull_active = false;
  };
}

#endif

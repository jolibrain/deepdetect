/**
 * DeepDetect
 * Copyright (c) 2026 Jolibrain
 *
 * This file is part of deepdetect.
 */

#ifndef PYTORCHWORKERINPUTCONNS_H
#define PYTORCHWORKERINPUTCONNS_H

#include "imgdataaug.h"
#include "imginputfileconn.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
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

    APIData inline_tensor_batches(const APIData &ad, const APIData &mllib)
    {
      if (is_keypoint_task(tensor_task(mllib)))
        return inline_keypoint_tensor_batches(ad, keypoint_count(mllib),
                                              mllib);
      return inline_detection_tensor_batches(ad);
    }

    void start_tensor_pull_session(const APIData &ad, const APIData &mllib)
    {
      if (is_keypoint_task(tensor_task(mllib)))
        start_keypoint_pull_session(ad, keypoint_count(mllib), mllib);
      else
        start_inline_detection_pull_session(ad);
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
      configure_pull_augmentation(ad);
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
      _pull_task = "detection";
      _pull_nkeypoints = 0;
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
      info.add("task", _pull_task);
      if (is_keypoint_task(_pull_task))
        info.add("nkeypoints", _pull_nkeypoints);
      info.add("boundary", std::string("connector-tensor-pull"));
      info.add("train_samples", static_cast<int>(_pull_train.size()));
      info.add("test_samples", test_samples);
      info.add("test_sets_total", static_cast<int>(_pull_tests.size()));
      info.add("transport", _pull_transport);
      info.add("input_width", _width);
      info.add("input_height", _height);
      info.add("train_shuffle", _shuffle);
      info.add("augmentation_enabled", _pull_augmentation_enabled);
      info.add("augmentation_policy", _pull_augmentation_enabled
                                          ? std::string("opencv")
                                          : std::string("none"));
      info.add("augmentation_train_only", true);
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
                                         batch_size, reset_epoch, true, split,
                                         APINull());
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
          reset_epoch, false, split, test_index);
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

    struct Keypoint
    {
      double x = -1.0;
      double y = -1.0;
      bool valid = false;
    };

    using KeypointInstance = std::vector<Keypoint>;

    using DetectionPair
        = std::pair<std::filesystem::path, std::filesystem::path>;

    struct TensorWriteStats
    {
      long long int nbytes = 0;
      double shared_memory_write_ms = 0.0;
    };

    struct PullBatchBuildResult
    {
      APIData batch;
      TensorWriteStats tensor;
    };

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

    std::vector<APIData> inline_keypoint_batches(const std::string &list_path,
                                                 int max_samples,
                                                 int nkeypoints) const
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
          std::string keypoints_path;
          row >> image_path >> keypoints_path;
          if (image_path.empty() || keypoints_path.empty())
            throw InputConnectorBadParamException(
                "connector_tensor_inline expects image and keypoints path in "
                + list_path);
          batches.push_back(inline_keypoint_batch(
              resolve_dataset_path(list_file.parent_path(), image_path),
              resolve_dataset_path(list_file.parent_path(), keypoints_path),
              sample_index, nkeypoints));
          ++sample_index;
        }
      if (batches.empty())
        throw InputConnectorBadParamException(
            "image list contains no samples: " + list_path);
      return batches;
    }

    APIData inline_keypoint_tensor_batches(const APIData &ad, int nkeypoints,
                                           const APIData &mllib)
    {
      APIData input_params;
      if (ad.has("parameters") && ad.getobj("parameters").has("input"))
        input_params = ad.getobj("parameters").getobj("input");
      fillup_parameters(input_params);
      validate_keypoint_connector_config(input_params, mllib,
                                         "connector_tensor_inline");

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
      tensor_batches.add(
          "train", inline_keypoint_batches(data[0], max_samples, nkeypoints));
      std::vector<APIData> tests;
      for (size_t index = 1; index < data.size(); ++index)
        {
          APIData test_set;
          test_set.add("batches", inline_keypoint_batches(
                                      data[index], max_samples, nkeypoints));
          tests.push_back(test_set);
        }
      tensor_batches.add("tests", tests);
      return tensor_batches;
    }

    void start_keypoint_pull_session(const APIData &ad, int nkeypoints,
                                     const APIData &mllib)
    {
      cleanup_inline_detection_pull_session();
      APIData input_params;
      if (ad.has("parameters") && ad.getobj("parameters").has("input"))
        input_params = ad.getobj("parameters").getobj("input");
      fillup_parameters(input_params);
      validate_keypoint_connector_config(input_params, mllib,
                                         "connector_tensor_pull");
      configure_pull_transport(ad);

      if (!ad.has("data"))
        throw InputConnectorBadParamException(
            "connector_tensor_pull requires train data");
      const std::vector<std::string> data
          = ad.get("data").get<std::vector<std::string>>();
      if (data.empty())
        throw InputConnectorBadParamException(
            "connector_tensor_pull requires a train list path");

      _pull_train = read_keypoint_pairs(data[0]);
      _pull_tests.clear();
      for (size_t index = 1; index < data.size(); ++index)
        _pull_tests.push_back(read_keypoint_pairs(data[index]));
      _pull_train_pos = 0;
      _pull_test_pos.assign(_pull_tests.size(), 0);
      _pull_epoch = 0;
      _pull_next_batch_id = 0;
      _pull_task = "keypoint";
      _pull_nkeypoints = nkeypoints;
      _pull_augmentation_enabled = false;
      _pull_augmentation_policy = "none";
      shuffle_pull_train();
      _pull_active = true;
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

    std::vector<DetectionPair>
    read_keypoint_pairs(const std::string &list_path) const
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
          std::string keypoints_path;
          row >> image_path >> keypoints_path;
          if (image_path.empty() || keypoints_path.empty())
            throw InputConnectorBadParamException(
                "connector_tensor_pull expects image and keypoints path in "
                + list_path);
          pairs.emplace_back(
              resolve_dataset_path(list_file.parent_path(), image_path),
              resolve_dataset_path(list_file.parent_path(), keypoints_path));
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
                                      bool reset_epoch, bool shuffle_on_reset,
                                      const std::string &split,
                                      const ad_variant_type &test_index)
    {
      const auto total_start = std::chrono::steady_clock::now();
      if (reset_epoch)
        {
          cursor = 0;
          if (shuffle_on_reset)
            shuffle_pull_train();
        }
      APIData response;
      response.add("status", std::string("ok"));
      const size_t cursor_start = cursor;
      response.add("split", split);
      response.add("test_index", test_index);
      response.add("epoch", split == "train" ? _pull_epoch : 0);
      response.add("cursor_start", static_cast<int>(cursor_start));
      response.add("requested_batch_size", batch_size);
      response.add("transport", _pull_transport);
      if (cursor >= pairs.size())
        {
          response.add("end", true);
          response.add("cursor_end", static_cast<int>(cursor_start));
          response.add("sample_count", 0);
          response.add("tensor_nbytes", static_cast<long long int>(0));
          return response;
        }
      const size_t count
          = std::min(static_cast<size_t>(batch_size), pairs.size() - cursor);
      const std::string batch_id = next_pull_batch_id();
      const auto build_start = std::chrono::steady_clock::now();
      const bool apply_augmentation
          = split == "train" && _pull_augmentation_enabled;
      PullBatchBuildResult built
          = is_keypoint_task(_pull_task)
                ? inline_keypoint_batch(pairs, cursor, count, batch_id,
                                        _pull_nkeypoints)
                : inline_detection_batch(pairs, cursor, count, batch_id,
                                         apply_augmentation);
      const auto build_end = std::chrono::steady_clock::now();
      response.add("end", false);
      response.add("batch_id", batch_id);
      response.add("cursor_end", static_cast<int>(cursor + count));
      response.add("sample_count", static_cast<int>(count));
      response.add("tensor_nbytes", built.tensor.nbytes);
      response.add("batch", built.batch);
      cursor += count;
      const auto total_end = std::chrono::steady_clock::now();
      const double build_ms = elapsed_ms(build_start, build_end);
      const double shm_ms = built.tensor.shared_memory_write_ms;
      debug_log_connector_batch(batch_id, split, count,
                                std::max(0.0, build_ms - shm_ms), shm_ms,
                                built.tensor.nbytes, _pull_transport,
                                elapsed_ms(total_start, total_end));
      return response;
    }

    std::string next_pull_batch_id()
    {
      ++_pull_next_batch_id;
      return std::to_string(_pull_next_batch_id);
    }

    void configure_pull_augmentation(const APIData &ad)
    {
      _pull_augmentation_enabled = false;
      _pull_augmentation_policy = "none";
      _pull_img_rand_aug_cv = ImgRandAugCV();

      if (!ad.has("parameters"))
        return;
      APIData parameters = ad.getobj("parameters");
      if (!parameters.has("mllib"))
        return;
      APIData mllib = parameters.getobj("mllib");
      ImgRandAugCVConfig config
          = parse_img_rand_aug_cv_config(mllib, _width, _height, _bw, _rgb);
      if (!config.enabled)
        return;

      if (config.rotate_disabled_for_shape)
        _logger->warn(
            "rotate augment was not applied. To enable rotate, select "
            "img_width and img_height to be equal.");
      _pull_img_rand_aug_cv = make_img_rand_aug_cv(config, _seed);
      _pull_augmentation_enabled = true;
      _pull_augmentation_policy = "opencv";
    }

    void apply_detection_augmentation(cv::Mat &image,
                                      std::vector<DetectionBBox> &targets)
    {
      std::vector<std::vector<float>> bboxes;
      std::vector<int> classes;
      bboxes.reserve(targets.size());
      classes.reserve(targets.size());
      for (const DetectionBBox &target : targets)
        {
          bboxes.push_back({ static_cast<float>(target.xmin),
                             static_cast<float>(target.ymin),
                             static_cast<float>(target.xmax),
                             static_cast<float>(target.ymax) });
          classes.push_back(target.label);
        }

      _pull_img_rand_aug_cv.augment_with_bbox(image, bboxes, classes);
      targets.clear();
      const size_t count = std::min(bboxes.size(), classes.size());
      targets.reserve(count);
      for (size_t index = 0; index < count; ++index)
        {
          if (classes[index] <= 0 || bboxes[index].size() != 4)
            continue;
          DetectionBBox bbox;
          bbox.label = classes[index];
          bbox.xmin = std::max(0.0, static_cast<double>(bboxes[index][0]));
          bbox.ymin = std::max(0.0, static_cast<double>(bboxes[index][1]));
          bbox.xmax = std::min(static_cast<double>(image.cols),
                               static_cast<double>(bboxes[index][2]));
          bbox.ymax = std::min(static_cast<double>(image.rows),
                               static_cast<double>(bboxes[index][3]));
          if (bbox.xmax <= bbox.xmin || bbox.ymax <= bbox.ymin)
            continue;
          targets.push_back(bbox);
        }
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
                                       bbox_path.string(), orig_width,
                                       orig_height, image.cols, image.rows));
      return batch;
    }

    APIData inline_keypoint_batch(const std::filesystem::path &image_path,
                                  const std::filesystem::path &keypoints_path,
                                  int sample_index, int nkeypoints) const
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
      std::vector<KeypointInstance> instances
          = read_keypoints(keypoints_path, nkeypoints, orig_width, orig_height,
                           image.cols, image.rows);

      APIData batch;
      batch.add("kind", std::string("tensor_batch"));
      batch.add("inputs",
                std::vector<APIData>{ inline_image_tensor_ref(image) });
      batch.add("targets", keypoint_targets(instances));
      batch.add("meta",
                keypoint_meta(sample_index, image_path.string(),
                              keypoints_path.string(), orig_width, orig_height,
                              image.cols, image.rows, nkeypoints));
      return batch;
    }

    PullBatchBuildResult inline_detection_batch(
        const std::vector<DetectionPair> &pairs, size_t offset, size_t count,
        const std::string &batch_id, bool apply_augmentation = false)
    {
      std::vector<double> values;
      std::vector<std::vector<DetectionBBox>> targets;
      std::vector<int> sample_ids;
      std::vector<std::string> paths;
      std::vector<std::string> target_paths;
      std::vector<int> widths;
      std::vector<int> heights;
      std::vector<int> original_widths;
      std::vector<int> original_heights;
      std::vector<int> preprocessed_widths;
      std::vector<int> preprocessed_heights;
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
          cv::Mat image = dimg._imgs[0].clone();
          const int orig_height = dimg._imgs_size.empty()
                                      ? image.rows
                                      : dimg._imgs_size[0].first;
          const int orig_width = dimg._imgs_size.empty()
                                     ? image.cols
                                     : dimg._imgs_size[0].second;
          const int preprocessed_width = image.cols;
          const int preprocessed_height = image.rows;
          std::vector<DetectionBBox> sample_targets
              = read_detection_bboxes(pair.second, orig_width, orig_height);
          if (apply_augmentation)
            apply_detection_augmentation(image, sample_targets);
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
          targets.push_back(sample_targets);
          sample_ids.push_back(static_cast<int>(offset + index));
          paths.push_back(pair.first.string());
          target_paths.push_back(pair.second.string());
          widths.push_back(image.cols);
          heights.push_back(image.rows);
          original_widths.push_back(orig_width);
          original_heights.push_back(orig_height);
          preprocessed_widths.push_back(preprocessed_width);
          preprocessed_heights.push_back(preprocessed_height);
        }

      TensorWriteStats tensor_stats;
      APIData batch;
      batch.add("kind", std::string("tensor_batch"));
      batch.add("inputs", std::vector<APIData>{ pull_image_tensor_ref(
                              values, static_cast<int>(count), channels, rows,
                              cols, batch_id, tensor_stats) });
      batch.add("targets", detection_targets(targets));
      batch.add("meta",
                detection_meta(sample_ids, paths, target_paths,
                               original_widths, original_heights,
                               preprocessed_widths, preprocessed_heights,
                               widths, heights, apply_augmentation));
      return PullBatchBuildResult{ batch, tensor_stats };
    }

    PullBatchBuildResult
    inline_keypoint_batch(const std::vector<DetectionPair> &pairs,
                          size_t offset, size_t count,
                          const std::string &batch_id, int nkeypoints)
    {
      std::vector<double> values;
      std::vector<std::vector<KeypointInstance>> targets;
      std::vector<int> sample_ids;
      std::vector<std::string> paths;
      std::vector<std::string> target_paths;
      std::vector<int> widths;
      std::vector<int> heights;
      std::vector<int> original_widths;
      std::vector<int> original_heights;
      std::vector<int> preprocessed_widths;
      std::vector<int> preprocessed_heights;
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
          cv::Mat image = dimg._imgs[0].clone();
          const int orig_height = dimg._imgs_size.empty()
                                      ? image.rows
                                      : dimg._imgs_size[0].first;
          const int orig_width = dimg._imgs_size.empty()
                                     ? image.cols
                                     : dimg._imgs_size[0].second;
          const int preprocessed_width = image.cols;
          const int preprocessed_height = image.rows;
          std::vector<KeypointInstance> sample_targets
              = read_keypoints(pair.second, nkeypoints, orig_width,
                               orig_height, image.cols, image.rows);
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
          targets.push_back(sample_targets);
          sample_ids.push_back(static_cast<int>(offset + index));
          paths.push_back(pair.first.string());
          target_paths.push_back(pair.second.string());
          widths.push_back(image.cols);
          heights.push_back(image.rows);
          original_widths.push_back(orig_width);
          original_heights.push_back(orig_height);
          preprocessed_widths.push_back(preprocessed_width);
          preprocessed_heights.push_back(preprocessed_height);
        }

      TensorWriteStats tensor_stats;
      APIData batch;
      batch.add("kind", std::string("tensor_batch"));
      batch.add("inputs", std::vector<APIData>{ pull_image_tensor_ref(
                              values, static_cast<int>(count), channels, rows,
                              cols, batch_id, tensor_stats) });
      batch.add("targets", keypoint_targets(targets));
      batch.add("meta",
                keypoint_meta(sample_ids, paths, target_paths, original_widths,
                              original_heights, preprocessed_widths,
                              preprocessed_heights, widths, heights,
                              nkeypoints));
      return PullBatchBuildResult{ batch, tensor_stats };
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
                                  int cols, const std::string &batch_id,
                                  TensorWriteStats &stats)
    {
      stats.nbytes = static_cast<long long int>(values.size() * sizeof(float));
      if (_pull_transport == "inline")
        return inline_image_tensor_ref(values, batch_size, channels, rows,
                                       cols, "connector_tensor_pull");
      return shared_memory_image_tensor_ref(values, batch_size, channels, rows,
                                            cols, batch_id, stats);
    }

    APIData shared_memory_image_tensor_ref(const std::vector<double> &values,
                                           int batch_size, int channels,
                                           int rows, int cols,
                                           const std::string &batch_id,
                                           TensorWriteStats &stats)
    {
      if (_pull_shm_dir.empty())
        throw InputConnectorBadParamException(
            "connector_tensor_pull shared memory session is not initialized");
      const std::filesystem::path path
          = _pull_shm_dir / ("batch-" + batch_id + "-input0.bin");
      const auto write_start = std::chrono::steady_clock::now();
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
      const auto write_end = std::chrono::steady_clock::now();
      stats.shared_memory_write_ms = elapsed_ms(write_start, write_end);
      _pull_batch_files[batch_id].push_back(path);

      APIData storage;
      storage.add("type", std::string("shared_memory"));
      storage.add("name", path.string());
      storage.add("offset", 0);
      storage.add("nbytes", static_cast<int>(stats.nbytes));

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

    std::vector<KeypointInstance>
    read_keypoints(const std::filesystem::path &keypoints_path, int nkeypoints,
                   int orig_width, int orig_height, int width,
                   int height) const
    {
      std::ifstream input(keypoints_path);
      if (!input.is_open())
        throw InputConnectorBadParamException("Could not open keypoints file: "
                                              + keypoints_path.string());
      if (nkeypoints <= 0)
        throw InputConnectorBadParamException(
            "mllib.nkeypoints must be positive for keypoint tensor input");
      if (orig_width <= 0 || orig_height <= 0)
        throw InputConnectorBadParamException(
            "Could not determine original image size for keypoints file: "
            + keypoints_path.string());
      const double wfactor = width > 0 ? static_cast<double>(width)
                                             / static_cast<double>(orig_width)
                                       : 1.0;
      const double hfactor = height > 0
                                 ? static_cast<double>(height)
                                       / static_cast<double>(orig_height)
                                 : 1.0;
      std::vector<KeypointInstance> instances;
      std::string line;
      int line_number = 0;
      while (std::getline(input, line))
        {
          ++line_number;
          if (line.empty())
            continue;
          std::istringstream row(line);
          std::vector<double> values;
          double value = 0.0;
          while (row >> value)
            values.push_back(value);
          if (!row.eof())
            throw InputConnectorBadParamException(
                "Invalid numeric keypoint value in: "
                + keypoints_path.string());
          if (values.size() != static_cast<size_t>(2 * nkeypoints))
            throw InputConnectorBadParamException(
                "Invalid keypoints line in: " + keypoints_path.string()
                + " line " + std::to_string(line_number) + ": expected "
                + std::to_string(2 * nkeypoints) + " fields");
          KeypointInstance instance;
          instance.reserve(static_cast<size_t>(nkeypoints));
          for (int index = 0; index < nkeypoints; ++index)
            {
              const double x = values[static_cast<size_t>(2 * index)];
              const double y = values[static_cast<size_t>(2 * index + 1)];
              if (!std::isfinite(x) || !std::isfinite(y))
                throw InputConnectorBadParamException(
                    "Invalid non-finite keypoint in: "
                    + keypoints_path.string());
              Keypoint keypoint;
              if (x == -1.0 && y == -1.0)
                {
                  keypoint.valid = false;
                }
              else
                {
                  if (x < 0.0 || y < 0.0)
                    throw InputConnectorBadParamException(
                        "Invalid keypoint sentinel in: "
                        + keypoints_path.string()
                        + "; missing keypoints must be -1 -1");
                  keypoint.x = x * wfactor;
                  keypoint.y = y * hfactor;
                  keypoint.valid = true;
                }
              instance.push_back(keypoint);
            }
          instances.push_back(instance);
        }
      return instances;
    }

    APIData
    keypoint_targets(const std::vector<KeypointInstance> &instances) const
    {
      APIData sample;
      sample.add("instances", keypoint_instances(instances));

      APIData targets;
      targets.add("samples", std::vector<APIData>{ sample });
      return targets;
    }

    APIData keypoint_targets(
        const std::vector<std::vector<KeypointInstance>> &items) const
    {
      std::vector<APIData> samples;
      samples.reserve(items.size());
      for (const auto &instances : items)
        {
          APIData sample;
          sample.add("instances", keypoint_instances(instances));
          samples.push_back(sample);
        }
      APIData targets;
      targets.add("samples", samples);
      return targets;
    }

    std::vector<APIData>
    keypoint_instances(const std::vector<KeypointInstance> &instances) const
    {
      std::vector<APIData> out;
      out.reserve(instances.size());
      for (const KeypointInstance &instance : instances)
        {
          std::vector<APIData> keypoints;
          keypoints.reserve(instance.size());
          for (const Keypoint &keypoint : instance)
            {
              APIData item;
              item.add("x", keypoint.valid ? keypoint.x : -1.0);
              item.add("y", keypoint.valid ? keypoint.y : -1.0);
              item.add("valid", keypoint.valid);
              keypoints.push_back(item);
            }
          APIData entry;
          entry.add("keypoints", keypoints);
          out.push_back(entry);
        }
      return out;
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
                           const std::string &target_path, int original_width,
                           int original_height, int width, int height) const
    {
      APIData meta;
      meta.add("sample_ids", std::vector<int>{ sample_index });
      meta.add("paths", std::vector<std::string>{ path });
      meta.add("target_paths", std::vector<std::string>{ target_path });
      meta.add("widths", std::vector<int>{ width });
      meta.add("heights", std::vector<int>{ height });
      meta.add("original_widths", std::vector<int>{ original_width });
      meta.add("original_heights", std::vector<int>{ original_height });
      meta.add("preprocessed_widths", std::vector<int>{ width });
      meta.add("preprocessed_heights", std::vector<int>{ height });
      add_detection_augmentation_meta(meta, false);
      return meta;
    }

    APIData detection_meta(const std::vector<int> &sample_ids,
                           const std::vector<std::string> &paths,
                           const std::vector<std::string> &target_paths,
                           const std::vector<int> &original_widths,
                           const std::vector<int> &original_heights,
                           const std::vector<int> &preprocessed_widths,
                           const std::vector<int> &preprocessed_heights,
                           const std::vector<int> &widths,
                           const std::vector<int> &heights,
                           bool augmentation_applied) const
    {
      APIData meta;
      meta.add("sample_ids", sample_ids);
      meta.add("paths", paths);
      meta.add("target_paths", target_paths);
      meta.add("widths", widths);
      meta.add("heights", heights);
      meta.add("original_widths", original_widths);
      meta.add("original_heights", original_heights);
      meta.add("preprocessed_widths", preprocessed_widths);
      meta.add("preprocessed_heights", preprocessed_heights);
      add_detection_augmentation_meta(meta, augmentation_applied);
      return meta;
    }

    APIData keypoint_meta(int sample_index, const std::string &path,
                          const std::string &target_path, int original_width,
                          int original_height, int width, int height,
                          int nkeypoints) const
    {
      APIData meta;
      meta.add("task", std::string("keypoint"));
      meta.add("nkeypoints", nkeypoints);
      meta.add("sample_ids", std::vector<int>{ sample_index });
      meta.add("paths", std::vector<std::string>{ path });
      meta.add("target_paths", std::vector<std::string>{ target_path });
      meta.add("widths", std::vector<int>{ width });
      meta.add("heights", std::vector<int>{ height });
      meta.add("original_widths", std::vector<int>{ original_width });
      meta.add("original_heights", std::vector<int>{ original_height });
      meta.add("preprocessed_widths", std::vector<int>{ width });
      meta.add("preprocessed_heights", std::vector<int>{ height });
      add_detection_augmentation_meta(meta, false);
      return meta;
    }

    APIData keypoint_meta(const std::vector<int> &sample_ids,
                          const std::vector<std::string> &paths,
                          const std::vector<std::string> &target_paths,
                          const std::vector<int> &original_widths,
                          const std::vector<int> &original_heights,
                          const std::vector<int> &preprocessed_widths,
                          const std::vector<int> &preprocessed_heights,
                          const std::vector<int> &widths,
                          const std::vector<int> &heights,
                          int nkeypoints) const
    {
      APIData meta;
      meta.add("task", std::string("keypoint"));
      meta.add("nkeypoints", nkeypoints);
      meta.add("sample_ids", sample_ids);
      meta.add("paths", paths);
      meta.add("target_paths", target_paths);
      meta.add("widths", widths);
      meta.add("heights", heights);
      meta.add("original_widths", original_widths);
      meta.add("original_heights", original_heights);
      meta.add("preprocessed_widths", preprocessed_widths);
      meta.add("preprocessed_heights", preprocessed_heights);
      add_detection_augmentation_meta(meta, false);
      return meta;
    }

    void add_detection_augmentation_meta(APIData &meta,
                                         bool augmentation_applied) const
    {
      meta.add("augmentation_applied", augmentation_applied);
      if (augmentation_applied && _seed >= 0)
        meta.add("augmentation_seed", _seed);
      else
        meta.add("augmentation_seed", APINull());
      meta.add("augmentation_policy", augmentation_applied
                                          ? _pull_augmentation_policy
                                          : std::string("none"));
    }

    void validate_keypoint_connector_config(const APIData &input_params,
                                            const APIData &mllib,
                                            const std::string &mode) const
    {
      if (input_params.has("keypoints")
          && !input_params.get("keypoints").get<bool>())
        throw InputConnectorBadParamException(mode
                                              + " requires input "
                                                "keypoints=true");
      if (_bw || _unchanged_data)
        throw InputConnectorBadParamException(
            mode + " keypoint input requires 3-channel image tensors");
      if (_crop_width != 0 || _crop_height != 0 || _aspect_ratio_pad
          || positive_int_param(input_params, "crop_width")
          || positive_int_param(input_params, "crop_height"))
        throw InputConnectorBadParamException(
            mode
            + " keypoint input does not support crop or aspect_ratio_pad yet");
      if (keypoint_augmentation_requested(mllib))
        throw InputConnectorBadParamException(
            mode + " keypoint input does not support C++ augmentation yet");
    }

    static bool positive_int_param(const APIData &ad, const std::string &key)
    {
      return ad.has(key) && ad.get(key).get<int>() > 0;
    }

    static bool positive_double_param(const APIData &ad,
                                      const std::string &key)
    {
      return ad.has(key) && ad.get(key).get<double>() > 0.0;
    }

    static bool true_bool_param(const APIData &ad, const std::string &key)
    {
      return ad.has(key) && ad.get(key).get<bool>();
    }

    static bool keypoint_augmentation_requested(const APIData &mllib)
    {
      if (true_bool_param(mllib, "mirror") || true_bool_param(mllib, "rotate")
          || positive_int_param(mllib, "crop_size")
          || positive_double_param(mllib, "cutout"))
        return true;
      APIData geometry = mllib.getobj("geometry");
      if (!geometry.empty() && positive_double_param(geometry, "prob"))
        return true;
      APIData noise = mllib.getobj("noise");
      if (!noise.empty() && positive_double_param(noise, "prob"))
        return true;
      APIData distort = mllib.getobj("distort");
      if (!distort.empty() && positive_double_param(distort, "prob"))
        return true;
      return false;
    }

    static std::string tensor_task(const APIData &mllib)
    {
      if (mllib.has("task"))
        return mllib.get("task").get<std::string>();
      return "detection";
    }

    static bool is_keypoint_task(const std::string &task)
    {
      return task == "keypoint" || task == "keypoints" || task == "pose";
    }

    static int keypoint_count(const APIData &mllib)
    {
      if (!mllib.has("nkeypoints"))
        throw InputConnectorBadParamException(
            "mllib.nkeypoints is required for keypoint tensor input");
      int nkeypoints = mllib.get("nkeypoints").get<int>();
      if (nkeypoints <= 0)
        throw InputConnectorBadParamException(
            "mllib.nkeypoints must be positive for keypoint tensor input");
      return nkeypoints;
    }

    static bool debug_enabled()
    {
      const char *debug = std::getenv("DEEPDETECT_DEBUG");
      const char *worker_debug = std::getenv("DEEPDETECT_WORKER_DEBUG");
      return (debug && *debug) || (worker_debug && *worker_debug);
    }

    static double
    elapsed_ms(const std::chrono::steady_clock::time_point &start,
               const std::chrono::steady_clock::time_point &end)
    {
      return std::chrono::duration<double, std::milli>(end - start).count();
    }

    static void debug_log_connector_batch(
        const std::string &batch_id, const std::string &split,
        size_t sample_count, double preprocessing_packing_ms,
        double shared_memory_write_ms, long long int bytes_written,
        const std::string &transport, double total_ms)
    {
      if (!debug_enabled())
        return;
      std::cerr << "[deepdetect-debug][connector_tensor_pull] "
                << "connector_batch_next" << " batch_id=" << batch_id
                << " split=" << split << " sample_count=" << sample_count
                << " transport=" << transport << " total_ms=" << total_ms
                << " preprocessing_packing_ms=" << preprocessing_packing_ms
                << " shared_memory_write_ms=" << shared_memory_write_ms
                << " bytes_written=" << bytes_written << std::endl;
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
    ImgRandAugCV _pull_img_rand_aug_cv;
    std::string _pull_augmentation_policy = "none";
    std::string _pull_task = "detection";
    int _pull_nkeypoints = 0;
    int _pull_next_batch_id = 0;
    int _pull_epoch = 0;
    bool _pull_augmentation_enabled = false;
    bool _pull_active = false;
  };
}

#endif

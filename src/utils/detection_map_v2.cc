/**
 * DeepDetect
 * Copyright (c) 2026 Jolibrain
 *
 * This file is part of deepdetect.
 */

#include "detection_map_v2.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <set>

namespace dd
{
  namespace
  {
    const std::vector<DetectionMapV2Threshold> default_thresholds{
      { "map-05", 0.05 },
      { "map-50", 0.50 },
      { "map-90", 0.90 },
    };

    bool decimal_integer(const std::string &value)
    {
      return !value.empty()
             && std::all_of(value.begin(), value.end(), [](unsigned char c) {
                  return c >= '0' && c <= '9';
                });
    }
  }

  bool DetectionMapV2::add_target(std::int64_t image_id, int label,
                                  const Box &box)
  {
    if (label <= 0 || !valid_box(box))
      return false;
    _targets.push_back({ image_id, label, box, 1.0 });
    return true;
  }

  bool DetectionMapV2::add_prediction(std::int64_t image_id, int label,
                                      const Box &box, double score)
  {
    if (label <= 0 || !valid_box(box) || !std::isfinite(score))
      return false;
    _predictions.push_back({ image_id, label, box, score });
    return true;
  }

  std::vector<DetectionMapV2Threshold>
  DetectionMapV2::thresholds_from_measures(
      const std::vector<std::string> &measures)
  {
    std::vector<DetectionMapV2Threshold> thresholds;
    std::set<std::string> names;
    for (const std::string &measure : measures)
      {
        constexpr const char prefix[] = "map-";
        if (measure.rfind(prefix, 0) != 0)
          continue;
        const std::string suffix = measure.substr(sizeof(prefix) - 1);
        if (!decimal_integer(suffix))
          continue;
        int percent = 0;
        for (const char digit : suffix)
          {
            percent = percent * 10 + (digit - '0');
            if (percent > 100)
              break;
          }
        if (percent <= 0 || percent > 100 || !names.insert(measure).second)
          continue;
        thresholds.push_back(
            { measure, static_cast<double>(percent) / 100.0 });
      }
    return thresholds.empty() ? default_thresholds : thresholds;
  }

  std::vector<DetectionMapV2Metric> DetectionMapV2::compute(
      const std::vector<DetectionMapV2Threshold> &requested_thresholds) const
  {
    const std::vector<DetectionMapV2Threshold> &thresholds
        = requested_thresholds.empty() ? default_thresholds
                                       : requested_thresholds;
    std::vector<DetectionMapV2Metric> threshold_metrics;
    threshold_metrics.reserve(thresholds.size());
    double sum = 0.0;
    for (const DetectionMapV2Threshold &threshold : thresholds)
      {
        const double value = mean_ap_at_iou(threshold.iou);
        threshold_metrics.push_back({ threshold.name, value });
        sum += value;
      }

    std::vector<DetectionMapV2Metric> metrics;
    metrics.reserve(threshold_metrics.size() + 1);
    metrics.push_back(
        { "map", threshold_metrics.empty()
                     ? 0.0
                     : sum / static_cast<double>(threshold_metrics.size()) });
    metrics.insert(metrics.end(), threshold_metrics.begin(),
                   threshold_metrics.end());
    return metrics;
  }

  void DetectionMapV2::clear()
  {
    _predictions.clear();
    _targets.clear();
  }

  bool DetectionMapV2::valid_box(const Box &box)
  {
    return std::all_of(box.begin(), box.end(),
                       [](double value) { return std::isfinite(value); })
           && box[2] > box[0] && box[3] > box[1];
  }

  double DetectionMapV2::box_iou(const Box &left, const Box &right)
  {
    const double left_area
        = std::max(0.0, left[2] - left[0])
          * std::max(0.0, left[3] - left[1]);
    const double right_area
        = std::max(0.0, right[2] - right[0])
          * std::max(0.0, right[3] - right[1]);
    if (left_area <= 0.0 || right_area <= 0.0)
      return 0.0;
    const double xmin = std::max(left[0], right[0]);
    const double ymin = std::max(left[1], right[1]);
    const double xmax = std::min(left[2], right[2]);
    const double ymax = std::min(left[3], right[3]);
    const double intersection
        = std::max(0.0, xmax - xmin) * std::max(0.0, ymax - ymin);
    const double union_area = left_area + right_area - intersection;
    return union_area > 0.0 ? intersection / union_area : 0.0;
  }

  double DetectionMapV2::average_precision(
      const std::vector<int> &true_positives,
      const std::vector<int> &false_positives, int num_targets)
  {
    if (num_targets <= 0 || true_positives.empty())
      return 0.0;

    int cumulative_tp = 0;
    int cumulative_fp = 0;
    std::vector<double> recalls;
    std::vector<double> precisions;
    recalls.reserve(true_positives.size());
    precisions.reserve(true_positives.size());
    for (size_t i = 0; i < true_positives.size(); ++i)
      {
        cumulative_tp += true_positives[i];
        cumulative_fp += false_positives[i];
        recalls.push_back(static_cast<double>(cumulative_tp)
                          / static_cast<double>(num_targets));
        precisions.push_back(static_cast<double>(cumulative_tp)
                             / static_cast<double>(cumulative_tp
                                                   + cumulative_fp));
      }

    std::vector<double> recall_points;
    std::vector<double> precision_points;
    recall_points.reserve(recalls.size() + 2);
    precision_points.reserve(precisions.size() + 2);
    recall_points.push_back(0.0);
    precision_points.push_back(0.0);
    recall_points.insert(recall_points.end(), recalls.begin(), recalls.end());
    precision_points.insert(precision_points.end(), precisions.begin(),
                            precisions.end());
    recall_points.push_back(1.0);
    precision_points.push_back(0.0);

    for (size_t i = precision_points.size() - 1; i-- > 0;)
      precision_points[i]
          = std::max(precision_points[i], precision_points[i + 1]);

    double ap = 0.0;
    for (size_t i = 1; i < recall_points.size(); ++i)
      {
        const double delta = recall_points[i] - recall_points[i - 1];
        if (delta > 0.0)
          ap += delta * precision_points[i];
      }
    return ap;
  }

  double DetectionMapV2::average_precision_for_label(
      const std::vector<Record> &predictions,
      const std::vector<Record> &targets, double threshold)
  {
    if (targets.empty())
      return 0.0;

    std::map<std::int64_t, std::vector<Record>> targets_by_image;
    for (const Record &target : targets)
      targets_by_image[target.image_id].push_back(target);
    std::map<std::int64_t, std::vector<bool>> matched;
    for (const auto &entry : targets_by_image)
      matched[entry.first] = std::vector<bool>(entry.second.size(), false);

    std::vector<Record> sorted_predictions = predictions;
    std::stable_sort(sorted_predictions.begin(), sorted_predictions.end(),
                     [](const Record &left, const Record &right) {
                       return left.score > right.score;
                     });

    std::vector<int> true_positives;
    std::vector<int> false_positives;
    true_positives.reserve(sorted_predictions.size());
    false_positives.reserve(sorted_predictions.size());
    for (const Record &prediction : sorted_predictions)
      {
        int best_index = -1;
        double best_iou = 0.0;
        auto image_it = targets_by_image.find(prediction.image_id);
        if (image_it != targets_by_image.end())
          {
            const std::vector<Record> &image_targets = image_it->second;
            std::vector<bool> &image_matched = matched[prediction.image_id];
            for (size_t i = 0; i < image_targets.size(); ++i)
              {
                if (image_matched[i])
                  continue;
                const double iou = box_iou(prediction.box,
                                           image_targets[i].box);
                if (iou > best_iou)
                  {
                    best_iou = iou;
                    best_index = static_cast<int>(i);
                  }
              }
          }

        if (best_index >= 0 && best_iou >= threshold)
          {
            matched[prediction.image_id][static_cast<size_t>(best_index)]
                = true;
            true_positives.push_back(1);
            false_positives.push_back(0);
          }
        else
          {
            true_positives.push_back(0);
            false_positives.push_back(1);
          }
      }
    return average_precision(true_positives, false_positives,
                             static_cast<int>(targets.size()));
  }

  double DetectionMapV2::mean_ap_at_iou(double threshold) const
  {
    std::set<int> labels;
    for (const Record &target : _targets)
      labels.insert(target.label);
    if (labels.empty())
      return 0.0;

    double sum = 0.0;
    for (int label : labels)
      {
        std::vector<Record> label_targets;
        std::vector<Record> label_predictions;
        for (const Record &target : _targets)
          if (target.label == label)
            label_targets.push_back(target);
        for (const Record &prediction : _predictions)
          if (prediction.label == label)
            label_predictions.push_back(prediction);
        sum += average_precision_for_label(label_predictions, label_targets,
                                           threshold);
      }
    return sum / static_cast<double>(labels.size());
  }
}

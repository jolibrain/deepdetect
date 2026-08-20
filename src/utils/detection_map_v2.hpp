/**
 * DeepDetect
 * Copyright (c) 2026 Jolibrain
 *
 * This file is part of deepdetect.
 */

#ifndef DD_UTILS_DETECTION_MAP_V2_HPP
#define DD_UTILS_DETECTION_MAP_V2_HPP

#include <array>
#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace dd
{
  struct DetectionMapV2Threshold
  {
    std::string name;
    double iou = 0.0;
  };

  struct DetectionMapV2Metric
  {
    std::string name;
    double value = 0.0;
  };

  /** Dataset-level detection AP implementation matching the Python worker
   * evaluator in builtin/vision/detection/common.py. */
  class DetectionMapV2
  {
  public:
    using Box = std::array<double, 4>;

    bool add_target(std::int64_t image_id, int label, const Box &box);
    bool add_prediction(std::int64_t image_id, int label, const Box &box,
                        double score);

    static std::vector<DetectionMapV2Threshold>
    thresholds_from_measures(const std::vector<std::string> &measures);

    std::vector<DetectionMapV2Metric>
    compute(const std::vector<DetectionMapV2Threshold> &thresholds) const;

    void clear();

  private:
    struct Record
    {
      std::int64_t image_id = 0;
      int label = 0;
      Box box{ 0.0, 0.0, 0.0, 0.0 };
      double score = 1.0;
    };

    static bool valid_box(const Box &box);
    static double box_iou(const Box &left, const Box &right);
    static double average_precision(const std::vector<int> &true_positives,
                                    const std::vector<int> &false_positives,
                                    int num_targets);
    static double
    average_precision_for_label(const std::vector<Record> &predictions,
                                const std::vector<Record> &targets,
                                double threshold);
    std::map<int, double> average_precisions_at_iou(double threshold) const;

    std::vector<Record> _predictions;
    std::vector<Record> _targets;
  };
}

#endif

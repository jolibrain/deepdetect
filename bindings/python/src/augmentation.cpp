// LGPL-3.0-or-later. Python buffer boundary for the shared native algorithms.
#include <deepdetect/imgdataaug.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cstring>
namespace py = pybind11;

namespace
{
  template <class T> T get(const py::dict &d, const char *key, T fallback)
  {
    return d.contains(key) ? py::cast<T>(d[key]) : fallback;
  }
  cv::Mat read_array(const py::array &input, bool mask)
  {
    if (input.ndim() != 2
        && (mask || input.ndim() != 3 || input.shape(2) != 3))
      throw py::value_error(
          "expected HW mask/grayscale or HWC three-channel image");
    bool byte = input.dtype().is(py::dtype::of<uint8_t>()),
         word = input.dtype().is(py::dtype::of<uint16_t>());
    if (!byte && !(mask && word))
      throw py::type_error(
          "images require uint8; masks require uint8 or uint16");
    if (input.shape(0) <= 0 || input.shape(1) <= 0 || input.shape(0) > INT_MAX
        || input.shape(1) > INT_MAX)
      throw py::value_error("invalid spatial dimensions");
    auto contiguous = py::array::ensure(input, py::array::c_style);
    cv::Mat out(input.shape(0), input.shape(1),
                CV_MAKETYPE(byte ? CV_8U : CV_16U, input.ndim() == 3 ? 3 : 1));
    std::memcpy(out.data, contiguous.data(), out.total() * out.elemSize());
    return out;
  }
  py::array array(const cv::Mat &m)
  {
    std::vector<py::ssize_t> shape = { m.rows, m.cols };
    if (m.channels() != 1)
      shape.push_back(m.channels());
    py::array out(m.depth() == CV_16U ? py::dtype::of<uint16_t>()
                                      : py::dtype::of<uint8_t>(),
                  shape);
    std::memcpy(out.mutable_data(), m.data, m.total() * m.elemSize());
    return out;
  }

  struct Runner : dd::ImgRandAugCV
  {
    using dd::ImgRandAugCV::ImgRandAugCV;
    cv::Mat H, valid, occluded, clean;
    std::vector<int> crop_info, cutout_info;
    int rotation = 0;
    bool mirrored = false;
    void run(cv::Mat &image, std::vector<cv::Mat> &masks, bool appearance)
    {
      int sw = image.cols, sh = image.rows;
      valid = cv::Mat(sh, sw, CV_8U, cv::Scalar(255));
      occluded = cv::Mat::zeros(sh, sw, CV_8U);
      H = cv::Mat::eye(3, 3, CV_64F);
      auto gp = _geometry_params;
      applyGeometry(image, gp, true, true);
      if (!gp._lambda.empty())
        {
          for (auto &mask : masks)
            applyGeometry(mask, gp, false, false);
          cv::Mat translation
              = (cv::Mat_<double>(3, 3) << 1, 0, sw, 0, 1, sh, 0, 0, 1);
          H = gp._lambda * translation;
          // Validity is zero outside the true image, even for reflected image
          // padding.
          cv::warpPerspective(valid, valid, H, cv::Size(sw, sh),
                              cv::INTER_NEAREST, cv::BORDER_CONSTANT, 0);
        }
      clean = image.clone();
      auto cp = _cutout_params;
      applyCutout(image, cp, true);
      cutout_info = { cp._rect_x, cp._rect_y, cp._w, cp._h };
      if (cp._w > 0 && cp._h > 0)
        occluded(cv::Rect(cp._rect_x, cp._rect_y, cp._w, cp._h)).setTo(255);
      int x = 0, y = 0;
      bool cropped = applyCrop(image, _crop_params, x, y);
      if (cropped)
        {
          for (auto &m : masks)
            applyCrop(m, _crop_params, x, y, false);
          applyCrop(valid, _crop_params, x, y, false);
          applyCrop(occluded, _crop_params, x, y, false);
          applyCrop(clean, _crop_params, x, y, false);
          cv::Mat t = (cv::Mat_<double>(3, 3) << 1, 0, -x, 0, 1, -y, 0, 0, 1);
          H = t * H;
        }
      crop_info = { x, y, image.cols, image.rows };
      mirrored = applyMirror(image);
      if (mirrored)
        {
          for (auto &m : masks)
            applyMirror(m, false);
          applyMirror(valid, false);
          applyMirror(occluded, false);
          applyMirror(clean, false);
          cv::Mat t = (cv::Mat_<double>(3, 3) << -1, 0, image.cols - 1, 0, 1,
                       0, 0, 0, 1);
          H = t * H;
        }
      int w = image.cols, h = image.rows;
      rotation = applyRotate(image);
      if (rotation > 0)
        {
          for (auto &m : masks)
            applyRotate(m, false, rotation);
          applyRotate(valid, false, rotation);
          applyRotate(occluded, false, rotation);
          applyRotate(clean, false, rotation);
          cv::Mat t;
          if (rotation == 1)
            t = (cv::Mat_<double>(3, 3) << 0, 1, 0, -1, 0, w - 1, 0, 0, 1);
          else if (rotation == 2)
            t = (cv::Mat_<double>(3, 3) << -1, 0, w - 1, 0, -1, h - 1, 0, 0,
                 1);
          else
            t = (cv::Mat_<double>(3, 3) << 0, -1, h - 1, 1, 0, 0, 0, 0, 1);
          H = t * H;
        }
      applyNoise(image);
      applyDistort(image);
      if (!appearance)
        {
          image = clean;
          occluded.setTo(0);
        }
    }
  };

  py::dict run(py::array pixels, py::dict targets, py::dict config,
               unsigned int seed, py::object replay, bool appearance)
  {
    cv::Mat image = read_array(pixels, false);
    std::vector<cv::Mat> masks;
    std::vector<std::string> names;
    for (auto item : targets)
      {
        names.push_back(py::cast<std::string>(item.first));
        masks.push_back(read_array(py::cast<py::array>(item.second), true));
        if (masks.back().size() != image.size())
          throw py::value_error("mask/image shape mismatch");
      }
    bool legacy = py::cast<std::string>(config["semantics"]) == "legacy_v1";
    bool bw = image.channels() == 1;
    bool rgb = py::cast<std::string>(config["color_order"]) == "RGB" && !bw;
    auto noise = py::cast<py::dict>(config["noise"]),
         distort = py::cast<py::dict>(config["distort"]),
         geometry = py::cast<py::dict>(config["geometry"]);
    dd::NoiseParams np(bw);
    np._prob = get<double>(noise, "prob", 0);
    np._rgb = rgb;
    dd::DistortParams dp(bw);
    dp._prob = get<double>(distort, "prob", 0);
    dp._rgb = rgb;
    if (!legacy)
      {
#define FLAG(field, key) np.field = get<bool>(noise, key, false)
        FLAG(_decolorize, "decolorize");
        FLAG(_gauss_blur, "gauss_blur");
        FLAG(_hist_eq, "hist_eq");
        FLAG(_clahe, "clahe");
        FLAG(_jpg, "jpg");
        FLAG(_erosion, "erosion");
        FLAG(_posterize, "posterize");
        FLAG(_inverse, "inverse");
        FLAG(_saltpepper, "saltpepper");
        FLAG(_convert_to_hsv, "convert_to_hsv");
        FLAG(_convert_to_lab, "convert_to_lab");
#undef FLAG
        np._saltpepper_fraction
            = get<double>(noise, "saltpepper_fraction", .1);
        dp._brightness = get<bool>(distort, "brightness", false);
        dp._contrast = get<bool>(distort, "contrast", false);
        dp._saturation = get<bool>(distort, "saturation", false);
        dp._hue = get<bool>(distort, "hue", false);
        dp._channel_order = get<bool>(distort, "channel_order", false);
        auto range = [&](const char *key) {
          auto values = py::cast<std::vector<float>>(distort[key]);
          if (values.size() != 2 || !std::isfinite(values[0])
              || !std::isfinite(values[1]) || values[0] > values[1])
            throw py::value_error("invalid distortion range");
          return values;
        };
        auto b = range("brightness_range"), c = range("contrast_range"),
             s = range("saturation_range"), h = range("hue_range");
        dp._uniform_real_brightness
            = std::uniform_real_distribution<float>(b[0], b[1]);
        dp._uniform_real_contrast
            = std::uniform_real_distribution<float>(c[0], c[1]);
        dp._uniform_real_saturation
            = std::uniform_real_distribution<float>(s[0], s[1]);
        dp._uniform_real_hue
            = std::uniform_real_distribution<float>(h[0], h[1]);
      }
    dd::GeometryParams gp(get<double>(geometry, "prob", 0),
                          get<bool>(geometry, "persp_horizontal", true),
                          get<bool>(geometry, "persp_vertical", true),
                          get<bool>(geometry, "transl_horizontal", false),
                          get<bool>(geometry, "transl_vertical", false),
                          get<bool>(geometry, "zoom_out", true),
                          get<bool>(geometry, "zoom_in", true),
                          get<std::string>(geometry, "pad_mode", "constant"));
    gp._geometry_zoom_factor = get<double>(geometry, "zoom_factor", .25);
    gp._geometry_persp_factor = get<double>(geometry, "persp_factor", .25);
    gp._geometry_transl_factor = get<double>(geometry, "transl_factor", .5);
    int crop = get<int>(config, "crop_size", -1);
    if (crop > image.cols || crop > image.rows)
      throw py::value_error("crop exceeds image dimensions");
    bool rotate = get<bool>(config, "rotate", false);
    if (rotate && image.cols != image.rows)
      {
        if (legacy)
          rotate = false;
        else
          throw py::value_error("rotation requires square input");
      }
    if (bw
        && ((np._prob > 0
             && (np._hist_eq || np._decolorize || np._jpg || np._convert_to_hsv
                 || np._convert_to_lab))
            || (dp._prob > 0
                && (dp._saturation || dp._hue || dp._channel_order))))
      throw py::value_error(
          "enabled colour operations require three channels");
    Runner aug(get<bool>(config, "mirror", false), rotate,
               dd::CropParams(crop),
               dd::CutoutParams(get<double>(config, "cutout", 0)), gp, np, dp);
    if (!legacy)
      {
        auto probs = py::cast<std::map<std::string, float>>(
            config["operation_probabilities"]);
        aug._operation_probabilities = probs;
      }
    dd::AugmentationTrace trace;
    if (!replay.is_none())
      {
        auto r = py::cast<py::dict>(replay);
        trace.replay = true;
        trace.draws = py::cast<decltype(trace.draws)>(r["draws"]);
        trace.blocks = py::cast<decltype(trace.blocks)>(r["random_blocks"]);
      }
    cv::RNG local_rng(static_cast<uint64_t>(
        static_cast<int64_t>(static_cast<int32_t>(seed))));
    aug._trace = &trace;
    aug._cutout_rng = &local_rng;
    aug._signed_brightness = !legacy;
    aug.seed_rnd_gen(seed);
    {
      py::gil_scoped_release release;
      aug.run(image, masks, appearance);
      if (trace.replay
          && (trace.position != trace.draws.size()
              || trace.block_position != trace.blocks.size()))
        throw std::invalid_argument("unused augmentation replay entries");
    }
    py::dict result, mask_result, record;
    result["image"] = array(image);
    for (size_t i = 0; i < masks.size(); i++)
      mask_result[py::str(names[i])] = array(masks[i]);
    result["masks"] = mask_result;
    result["in_image"] = array(aug.valid);
    result["occluded"] = array(aug.occluded);
    record["draws"] = trace.draws;
    record["random_blocks"] = trace.blocks;
    record["operations"] = trace.operations;
    record["crop"] = aug.crop_info;
    record["cutout"] = aug.cutout_info;
    std::vector<std::vector<int>> permutations;
    for (const auto &entry : trace.blocks)
      if (entry.first == "channel_order")
        {
          std::default_random_engine engine;
          std::istringstream input(entry.second);
          input >> engine;
          std::vector<int> order{ 0, 1, 2 };
          std::shuffle(order.begin(), order.end(), engine);
          permutations.push_back(order);
        }
    record["channel_permutations"] = permutations;
    record["mirrored"] = aug.mirrored;
    record["rotation"] = aug.rotation;
    std::vector<std::vector<double>> matrix(3, std::vector<double>(3));
    for (int y = 0; y < 3; y++)
      for (int x = 0; x < 3; x++)
        matrix[y][x] = aug.H.at<double>(y, x);
    record["source_to_output"] = matrix;
    result["record"] = record;
    return result;
  }
}
PYBIND11_MODULE(_augmentation, m)
{
  m.doc() = "DeepDetect CPU image augmentation";
  m.def("run", &run);
  m.def("build_info", [] {
    py::dict d;
    d["opencv"] = CV_VERSION;
    d["compiler"] = __VERSION__;
    d["implementation"] = DD_AUGMENTATION_FINGERPRINT;
    return d;
  });
}

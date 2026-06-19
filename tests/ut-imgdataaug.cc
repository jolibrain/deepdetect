#include "imgdataaug.h"

#include "apidata.h"

#include <gtest/gtest.h>

namespace dd
{
  class TestImgRandAugCV : public ImgRandAugCV
  {
  public:
    using ImgRandAugCV::applyCropBBox;
    using ImgRandAugCV::applyMirror;
    using ImgRandAugCV::applyMirrorBBox;
    using ImgRandAugCV::applyRotate;
    using ImgRandAugCV::applyRotateBBox;
  };
}

TEST(imgdataaug, mirror_updates_image_and_bbox)
{
  dd::TestImgRandAugCV aug;
  aug._mirror = true;
  cv::Mat image(2, 4, CV_8UC1);
  image.at<uchar>(0, 0) = 1;
  image.at<uchar>(0, 3) = 9;
  std::vector<std::vector<float>> bboxes = { { 1.0f, 0.0f, 3.0f, 1.0f } };

  ASSERT_TRUE(aug.applyMirror(image, false));
  aug.applyMirrorBBox(bboxes, static_cast<float>(image.cols));

  EXPECT_EQ(9, image.at<uchar>(0, 0));
  EXPECT_FLOAT_EQ(1.0f, bboxes[0][0]);
  EXPECT_FLOAT_EQ(3.0f, bboxes[0][2]);
}

TEST(imgdataaug, rotate_updates_bbox)
{
  dd::TestImgRandAugCV aug;
  aug._rotate = true;
  cv::Mat image(4, 4, CV_8UC1, cv::Scalar(0));
  std::vector<std::vector<float>> bboxes = { { 1.0f, 1.0f, 3.0f, 2.0f } };

  ASSERT_EQ(2, aug.applyRotate(image, false, 2));
  aug.applyRotateBBox(bboxes, 4.0f, 4.0f, 2);

  EXPECT_FLOAT_EQ(1.0f, bboxes[0][0]);
  EXPECT_FLOAT_EQ(2.0f, bboxes[0][1]);
  EXPECT_FLOAT_EQ(3.0f, bboxes[0][2]);
  EXPECT_FLOAT_EQ(3.0f, bboxes[0][3]);
}

TEST(imgdataaug, crop_without_remaining_bbox_emits_dummy)
{
  dd::TestImgRandAugCV aug;
  dd::CropParams crop(2);
  std::vector<std::vector<float>> bboxes = { { 3.0f, 3.0f, 4.0f, 4.0f } };
  std::vector<int> classes = { 1 };

  aug.applyCropBBox(bboxes, classes, crop, 2.0f, 2.0f, 0.0f, 0.0f);

  ASSERT_EQ(1U, bboxes.size());
  ASSERT_EQ(1U, classes.size());
  EXPECT_EQ(0, classes[0]);
  EXPECT_FLOAT_EQ(0.0f, bboxes[0][0]);
  EXPECT_FLOAT_EQ(0.0f, bboxes[0][1]);
  EXPECT_FLOAT_EQ(0.0f, bboxes[0][2]);
  EXPECT_FLOAT_EQ(0.0f, bboxes[0][3]);
}

TEST(imgdataaug, parse_empty_config_is_disabled)
{
  dd::APIData mllib;

  dd::ImgRandAugCVConfig config
      = dd::parse_img_rand_aug_cv_config(mllib, 224, 224, false, true);

  EXPECT_FALSE(config.enabled);
  EXPECT_FALSE(config.mirror);
  EXPECT_FALSE(config.rotate);
  EXPECT_FALSE(config.rotate_disabled_for_shape);
}

TEST(imgdataaug, parse_full_config_populates_params)
{
  dd::APIData geometry;
  geometry.add("prob", 0.3);
  geometry.add("persp_vertical", false);
  geometry.add("persp_horizontal", true);
  geometry.add("transl_vertical", true);
  geometry.add("transl_horizontal", false);
  geometry.add("zoom_out", false);
  geometry.add("zoom_in", true);
  geometry.add("persp_factor", 0.2);
  geometry.add("transl_factor", 0.4);
  geometry.add("zoom_factor", 0.6);
  geometry.add("pad_mode", std::string("mirrored"));

  dd::APIData noise;
  noise.add("prob", 0.2);
  dd::APIData distort;
  distort.add("prob", 0.4);

  dd::APIData mllib;
  mllib.add("mirror", true);
  mllib.add("rotate", true);
  mllib.add("crop_size", 32);
  mllib.add("test_crop_samples", 5);
  mllib.add("cutout", 0.25);
  mllib.add("geometry", geometry);
  mllib.add("noise", noise);
  mllib.add("distort", distort);

  dd::ImgRandAugCVConfig config
      = dd::parse_img_rand_aug_cv_config(mllib, 64, 64, false, true);

  ASSERT_TRUE(config.enabled);
  EXPECT_TRUE(config.mirror);
  EXPECT_TRUE(config.rotate);
  EXPECT_FALSE(config.rotate_disabled_for_shape);
  EXPECT_TRUE(config.has_crop_size);
  EXPECT_EQ(32, config.crop_size);
  EXPECT_EQ(32, config.crop_params._crop_size);
  EXPECT_EQ(5, config.crop_params._test_crop_samples);
  EXPECT_TRUE(config.has_cutout);
  EXPECT_FLOAT_EQ(0.25f, config.cutout);
  EXPECT_FLOAT_EQ(0.25f, config.cutout_params._prob);
  EXPECT_TRUE(config.has_geometry);
  EXPECT_FLOAT_EQ(0.3f, config.geometry_params._prob);
  EXPECT_FALSE(config.geometry_params._geometry_persp_vertical);
  EXPECT_TRUE(config.geometry_params._geometry_transl_vertical);
  EXPECT_FALSE(config.geometry_params._geometry_zoom_out);
  EXPECT_FLOAT_EQ(0.2f, config.geometry_params._geometry_persp_factor);
  EXPECT_FLOAT_EQ(0.4f, config.geometry_params._geometry_transl_factor);
  EXPECT_FLOAT_EQ(0.6f, config.geometry_params._geometry_zoom_factor);
  EXPECT_EQ(2, config.geometry_params._geometry_pad_mode);
  EXPECT_TRUE(config.has_noise);
  EXPECT_FLOAT_EQ(0.2f, config.noise_params._prob);
  EXPECT_TRUE(config.has_distort);
  EXPECT_FLOAT_EQ(0.4f, config.distort_params._prob);
  EXPECT_TRUE(config.noise_params._rgb);
  EXPECT_TRUE(config.distort_params._rgb);
}

TEST(imgdataaug, parse_non_square_config_disables_rotate)
{
  dd::APIData mllib;
  mllib.add("rotate", true);

  dd::ImgRandAugCVConfig config
      = dd::parse_img_rand_aug_cv_config(mllib, 64, 32, false, false);

  ASSERT_TRUE(config.enabled);
  EXPECT_FALSE(config.rotate);
  EXPECT_TRUE(config.rotate_disabled_for_shape);
}

TEST(imgdataaug, parse_bw_config_disables_color_noise_defaults)
{
  dd::APIData noise;
  noise.add("prob", 0.2);
  dd::APIData distort;
  distort.add("prob", 0.4);
  dd::APIData mllib;
  mllib.add("noise", noise);
  mllib.add("distort", distort);

  dd::ImgRandAugCVConfig config
      = dd::parse_img_rand_aug_cv_config(mllib, 64, 64, true, false);

  ASSERT_TRUE(config.enabled);
  EXPECT_FALSE(config.noise_params._decolorize);
  EXPECT_FALSE(config.noise_params._jpg);
  EXPECT_FALSE(config.noise_params._convert_to_hsv);
  EXPECT_FALSE(config.distort_params._saturation);
  EXPECT_FALSE(config.distort_params._hue);
  EXPECT_FALSE(config.distort_params._channel_order);
  EXPECT_FALSE(config.noise_params._rgb);
  EXPECT_FALSE(config.distort_params._rgb);
}

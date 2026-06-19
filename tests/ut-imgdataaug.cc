#include "imgdataaug.h"

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

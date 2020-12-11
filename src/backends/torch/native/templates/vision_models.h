
#ifndef DD_VISION_FACTORY_H
#define DD_VISION_FACTORY_H

#include <functional>

#include <torchvision/vision.h>

#include "../native_wrapper.h"

namespace dd
{
  class VisionModelsFactory
  {
  public:
    template <typename TModule>
    static NativeModule *create_wrapper(const APIData &template_params)
    {
      if (template_params.has("nclasses"))
        {
          return new NativeModuleWrapper<TModule>(
              template_params.get("nclasses").get<int>());
        }
      else
        {
          return new NativeModuleWrapper<TModule>();
        }
    }

    static bool is_vision_template(const std::string tdef)
    {
      auto &ctor_map = get_constructor_map();
      return ctor_map.find(tdef) != ctor_map.end();
    }

    template <class TInputConnectorStrategy>
    static NativeModule *from_template(const std::string tdef,
                                       const APIData &template_params,
                                       const TInputConnectorStrategy &inputc)
    {
      (void)(inputc);
      auto &ctor_map = get_constructor_map();
      auto it = ctor_map.find(tdef);

      if (it != ctor_map.end())
        {
          return it->second(template_params);
        }
      else
        return nullptr;
    }

  private:
    static std::map<std::string,
                    std::function<NativeModule *(const APIData &)>> &
    get_constructor_map()
    {
      // XXX: Inception V3 is has a different output format and can not
      // be wrapped normally
      // Generic MNASNET & Shufflenet have more initialization parameters
      // and need to have their custom create_wrapper function
      static std::map<std::string,
                      std::function<NativeModule *(const APIData &)>>
          ctor_map{
            { "resnet18", create_wrapper<vision::models::ResNet18> },
            { "resnet34", create_wrapper<vision::models::ResNet34> },
            { "resnet50", create_wrapper<vision::models::ResNet50> },
            { "resnet101", create_wrapper<vision::models::ResNet101> },
            { "resnet152", create_wrapper<vision::models::ResNet152> },
            { "resnext50_32x4d",
              create_wrapper<vision::models::ResNext50_32x4d> },
            { "resnext101_32x8d",
              create_wrapper<vision::models::ResNext101_32x8d> },
            { "wideresnet50_2",
              create_wrapper<vision::models::WideResNet50_2> },
            { "wideresnet101_2",
              create_wrapper<vision::models::WideResNet101_2> },
            // { "inceptionv3", create_wrapper<vision::models::InceptionV3> },
            { "alexnet", create_wrapper<vision::models::AlexNet> },
            { "vgg11", create_wrapper<vision::models::VGG11> },
            { "vgg13", create_wrapper<vision::models::VGG13> },
            { "vgg16", create_wrapper<vision::models::VGG16> },
            { "vgg19", create_wrapper<vision::models::VGG19> },
            { "vgg11bn", create_wrapper<vision::models::VGG11BN> },
            { "vgg13bn", create_wrapper<vision::models::VGG13BN> },
            { "vgg16bn", create_wrapper<vision::models::VGG16BN> },
            { "vgg19bn", create_wrapper<vision::models::VGG19BN> },
            { "mobilenetv2", create_wrapper<vision::models::MobileNetV2> },
            { "densenet121", create_wrapper<vision::models::DenseNet121> },
            { "densenet169", create_wrapper<vision::models::DenseNet169> },
            { "densenet201", create_wrapper<vision::models::DenseNet201> },
            { "densenet161", create_wrapper<vision::models::DenseNet161> },
            // { "mnasnet", create_wrapper<vision::models::MNASNet> },
            { "mnasnet0_5", create_wrapper<vision::models::MNASNet0_5> },
            { "mnasnet0_75", create_wrapper<vision::models::MNASNet0_75> },
            { "mnasnet1_0", create_wrapper<vision::models::MNASNet1_0> },
            { "mnasnet1_3", create_wrapper<vision::models::MNASNet1_3> },
            // { "shufflenetv2", create_wrapper<vision::models::MobileNetV2> }
            { "shufflenetv2_x0_5",
              create_wrapper<vision::models::ShuffleNetV2_x0_5> },
            { "shufflenetv2_x1_0",
              create_wrapper<vision::models::ShuffleNetV2_x1_0> },
            { "shufflenetv2_x1_5",
              create_wrapper<vision::models::ShuffleNetV2_x1_5> },
            { "shufflenetv2_x2_0",
              create_wrapper<vision::models::ShuffleNetV2_x2_0> },
            { "squeezenet1_0", create_wrapper<vision::models::SqueezeNet1_0> },
            { "squeezenet1_1", create_wrapper<vision::models::SqueezeNet1_1> },
          };
      return ctor_map;
    }
  };
}

#endif // DD_VISION_FACTORY_H

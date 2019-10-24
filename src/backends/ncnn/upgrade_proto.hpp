#ifndef CAFFE_UTIL_UPGRADE_PROTO_H_
#define CAFFE_UTIL_UPGRADE_PROTO_H_

#include <string>

#include "src/caffe.pb.h"


// Return true iff the net is not the current version.
bool NetNeedsUpgrade(const caffe::NetParameter& net_param);

// Check for deprecations and upgrade the NetParameter as needed.
bool UpgradeNetAsNeeded(caffe::NetParameter* param);

// Return true iff any layer contains parameters specified using
// deprecated V0LayerParameter.
bool NetNeedsV0ToV1Upgrade(const caffe::NetParameter& net_param);

// Perform all necessary transformations to upgrade a V0NetParameter into a
// NetParameter (including upgrading padding layers and LayerParameters).
bool UpgradeV0Net(const caffe::NetParameter& v0_net_param, caffe::NetParameter* net_param);

// Upgrade NetParameter with padding layers to pad-aware conv layers.
// For any padding layer, remove it and put its pad parameter in any layers
// taking its top blob as input.
// Error if any of these above layers are not-conv layers.
void UpgradeV0PaddingLayers(const caffe::NetParameter& param,
                            caffe::NetParameter* param_upgraded_pad);

// Upgrade a single V0LayerConnection to the V1LayerParameter format.
bool UpgradeV0LayerParameter(const caffe::V1LayerParameter& v0_layer_connection,
                             caffe::V1LayerParameter* layer_param);

caffe::V1LayerParameter_LayerType UpgradeV0LayerType(const std::string& type);

// Return true iff any layer contains deprecated data transformation parameters.
bool NetNeedsDataUpgrade(const caffe::NetParameter& net_param);

// Perform all necessary transformations to upgrade old transformation fields
// into a TransformationParameter.
void UpgradeNetDataTransformation(caffe::NetParameter* net_param);

// Return true iff the Net contains any layers specified as V1LayerParameters.
bool NetNeedsV1ToV2Upgrade(const caffe::NetParameter& net_param);

// Perform all necessary transformations to upgrade a NetParameter with
// deprecated V1LayerParameters.
bool UpgradeV1Net(const caffe::NetParameter& v1_net_param, caffe::NetParameter* net_param);

bool UpgradeV1LayerParameter(const caffe::V1LayerParameter& v1_layer_param,
                             caffe::LayerParameter* layer_param);

const char* UpgradeV1LayerType(const caffe::V1LayerParameter_LayerType type);

// Return true iff the Net contains input fields.
bool NetNeedsInputUpgrade(const caffe::NetParameter& net_param);

// Perform all necessary transformations to upgrade input fields into layers.
void UpgradeNetInput(caffe::NetParameter* net_param);

// Return true iff the Net contains batch norm layers with manual local LRs.
bool NetNeedsBatchNormUpgrade(const caffe::NetParameter& net_param);

// Perform all necessary transformations to upgrade batch norm layers.
void UpgradeNetBatchNorm(caffe::NetParameter* net_param);



#endif   // CAFFE_UTIL_UPGRADE_PROTO_H_

#include "documentation.hpp"

namespace dd
{
  // TODO This will be obsolete once training params will use DTO
  std::string GET_TRAIN_PARAMETERS()
  {
    static std::string TRAIN_PARAMS = R"trainparams(
Documentation source: [documentation.cc](https://github.com/jolibrain/deepdetect/blob/master/src/http/documentation.cc)

Parameter | Type   | Optional | Default | Description
--------- | ----   | -------- | ------- | -----------
service   | string | No       | N/A     | service resource identifier
async     | bool   | No       | true    | whether to start a non-blocking training call
data      | object | yes      | empty   | input dataset for training, in some cases can be handled by the input connectors, in general non optional though

## Output connector

Parameter         | Type   | Optional | Default | Description
---------         | ----   | -------- | ------- | -----------
best              | int    | yes      | 1       | Number of top predictions returned by data URI (supervised)
measure           | array  | yes      | empty   | Output measures requested, from `acc`: accuracy, `acc-k`: top-k accuracy, replace k with number (e.g. `acc-5`), `f1`: f1, precision and recall, `mcll`: multi-class log loss, `auc`: area under the curve, `cmdiag`: diagonal of confusion matrix (requires `f1`), `cmfull`: full confusion matrix (requires `f1`), `mcc`: Matthews correlation coefficient, `eucll`: euclidean distance (e.g. for regression tasks),`l1`: l1 distance (e.g. for regression tasks), `percent`: mean relative error in percent,  `kl`: KL_divergence, `js`: JS divergence, `was`: Wasserstein, `ks`: Kolmogorov Smirnov, `dc`: distance correlation, `r2`: R2, `deltas`: delta scores, 'raw': ouput raw results, in case of predict call, this requires a special deploy.prototxt that is a test network (to have ground truth)
target_repository | string | yes      | empty   | target directory to which to copy the best model files once training has completed

Problem type | Default | Possible values | Description
------------ | ------- | --------------- | -----------
timeserie    |   L1    | L1, L2, mase, mape, smape, mase, owa, mae, mse; L1_all, L2_all, mase_all, mape_all, smape_all, mase_all, owa_all, mae_all, mse_all | L1: mean error, L2: mean squared error, mase : mean absolute scaled error, mape: mean absolute percentage error, smape: symetric mean absolute percentage error, owa: overall weighted average, mae: mean absolute error, mse: mean squarred error; ; versions with "\_all" also show metrics per dimension/serie, and not only average.

## Machine learning libraries

Parameter     | Type           | Optional | Default        | Description
---------     | ----           | -------- | -------        | -----------
gpu           | bool           | yes      | false          | Whether to use GPU
gpuid         | int or array   | yes      | 0              | GPU id, use single int for single GPU, `-1` for using all GPUs, and array e.g. `[1,3]` for selecting among multiple GPUs
resume        | bool           | yes      | false          | Whether to resume training from .solverstate and .caffemodel files
class_weights | array of float | yes      | 1.0 everywhere | Whether to weight some classes more / less than others, e.g. [1.0,0.1,1.0]
ignore_label  | int            | yes      | N/A            | A single label to be ignored by the loss (i.e. no gradients)
timesteps     | int            | yes      | N/A            | Number of timesteps for recurrence ('csvts', `ctc` OCR) models (in case of csvts, used only at train time)
offset        | int            | yes      | N/A            | Offset beween start point of sequences with connector `cvsts`, defining the overlap of input series. For [0, n] steps a timestep of t and an offset of k, series [0..t-1], [k..t+k-1], [2k, 2k+t-1] ... will be cosntructed. If some elements at the end could not be taken using this, it will add a final [n-t+1..n] sequence (used only at train time).
freeze_traced   | bool   | yes      | false   | Freeze the traced part of the net during finetuning (e.g. for classification)
retain_graph	| bool	 | yes	    | false   | Whether to use `retain_graph` with torch autograd
template        | string | yes      | ""      | e.g. "bert", "gpt2", "recurrent", "nbeats", "vit", "visformer", "ttransformer", "resnet50", ... All templates are listed in the [Model Templates](#model-templates) section.
template_params | dict   | yes      | template dependent | Model parameter for templates. All parameters are listed in the [Model Templates](#model-templates) section.
regression | bool            | yes                      | false   | Whether the model is a regressor
timesteps     | int            | yes      | N/A            | Number of timesteps for time models (LSTM/NBEATS...) : this sets the length of sequences that will be given for learning, every timestep contains inputs and outputs as defined by the csv/csvts connector
forecast_timesteps      | int            | yes      | N/A       | for nbeats model, this gives the length of the forecast
backcast_timesteps      | int            | yes      | N/A       | for nbeats model, this gives the length of the backcast
datatype      | string | yes       | fp32 | Datatype used at prediction time, possible values are "fp16" (only if inference is done on GPU) , "fp32" and "fp64" (double)
dataloader_threads | int | yes | 1 | How many threads should be used to load data. 0 means no prefetch.

### Solver

Parameter            | Type         | Optional | Default | Description
---------            | ----         | -------- | ------- | -----------
iterations           | int          | yes      | N/A     | Max number of solver's iterations
snapshot             | int          | yes      | N/A     | Iterations between model snapshots
snapshot_prefix      | string       | yes      | empty   | Prefix to snapshot file, supports repository
solver_type          | string       | yes      | SGD     | from "SGD", "ADAGRAD", "NESTEROV", "RMSPROP", "ADADELTA", "ADAM",  "AMSGRAD",  "RANGER", "RANGER_PLUS", "ADAMW", "SGDW", "AMSGRADW" (\*W version for decoupled weight decay, RANGER_PLUS is ranger + adabelief + centralized_gradient)
clip                 | bool         | yes      | false (true if RANGER\* selected) | gradients with absolute value greater than clip_value will be clipped to below values
clip_value           | real         | yes      | 5.0     | gradients with absolute value greater than clip_value will be clipped to this value
clip_norm            | real         | yes      | 100.0   | gradients with euclidean norm greater than clip_norm will be clipped to this value
adabelief            | bool         | yes      | false   | adabelief mod for ADAM https://arxiv.org/abs/2010.07468
gradient_centralization | bool         | yes      | false   | centralized gradient mod for ADAM ie https://arxiv.org/abs/2004.01461v2
beta1         | real   | yes      | 0.9     | for RANGER\* : beta1 param
beta2         | real   | yes      | 0.999   | for RANGER\* : beta2 param
weight_decay  | real   | yes      | 0.0     | for RANGER\* : weight decay
rectified     | bool   | yes      | true    | for RANGER\* : enable/disable rectified ADAM
lookahead     | bool   | yes      | true    | for RANGER\* and MADGRAD : enable/disable lookahead
lookahead_steps | int  | yes      | 6       | for RANGER\* and MADGRAD : if lookahead enabled, number of steps
lookahead_alpha | real | yes      | 0.5     | for RANGER\* and MADGRAD : if lookahead enables, alpha param
adabelief     | bool   | yes      | false for RANGER, true for RANGER_PLUS   | for RANGER\* : enable/disable adabelief
gradient_centralization | bool | yes | false for RANGER, true for RANGER_PLUS| for RANGER\* : enable/disable gradient centralization
sam           | bool   | yes      | false   | Sharpness Aware Minimization (https://arxiv.org/abs/2010.01412)
sam_rho       | real   | yes      | 0.05    | neighborhood size for SAM (see above)
swa           | bool   | yes      | false   | SWA https://arxiv.org/abs/1803.05407 , implemented  only for  RANGER / RANGER_PLUS / MADGRAD  solver types.
test_interval        | int          | yes      | N/A     | Number of iterations between testing phases
test_initialization  | bool         | true     | N/A     | Whether to start training by testing the network
lr_policy            | string       | yes      | N/A     | learning rate policy ("step", "inv", "fixed", "sgdr", ...)
base_lr              | real         | yes      | N/A     | Initial learning rate
warmup_lr            | real         | yes      | N/A     | warmup starting learning rate (linearly goes to base_lr)
warmup_iter          | int          | yes      | 0       | number of warmup iterations
gamma                | real         | yes      | N/A     | Learning rate drop factor
stepsize             | int          | yes      | N/A     | Number of iterations between the dropping of the learning rate
stepvalue            | array of int | yes      | N/A     | Iterations at which a learning rate change takes place, with `multistep` `lr_policy`
momentum             | real         | yes      | N/A     | Learning momentum
period               | int          | yes      | -1      | N/A | Period in number of iterations with SGDR, best to use ncycles instead
ncycles              | int          | yes      | 1       | Number of restart cycles with SGDR
weight_decay         | real         | yes      | N/A     | Weight decay
power                | real         | yes      | N/A     | Power applicable to some learning rate policies
iter_size            | int          | yes      | 1       | Number of passes (iter_size * batch_size) at every iteration
rand_skip            | int          | yes      | 0       | Max number of images to skip when resuming training (only with segmentation or multilabel and Caffe backend)
lookahead            | bool         | yes      | false   | weither to use lookahead strategy from  https://arxiv.org/abs/1907.08610v1
lookahead_steps      | int          | yes      | 6       | number of lookahead steps for lookahead strategy
lookahead_alpha      | real         | yes      | 0.5     | size of step towards full lookahead
decoupled_wd_periods | int          | yes      | 4       | number of search periods for SGDW ADAMW AMSGRADW (periods end with a restart)
decoupled_wd_mult    | real         | yes      | 2.0     | muliplier of period for SGDW ADAMW AMSGRADW
lr_dropout           | real         | yes      | 1.0     | learning rate dropout, as in https://arxiv.org/abs/1912.00144 1.0 means no dropout, 0.0 means no learning at all (this value is the probability of keeping computed value and not putting zero)

Note: most of the default values for the parameters above are to be found in the Caffe files describing a given neural network architecture, or within Caffe library, therefore regarded as N/A at DeepDetect level.

### Net

Parameter       | Type | Optional | Default | Description
---------       | ---- | -------- | ------- | -----------
batch_size      | int  | yes      | N/A     | Training batch size
test_batch_size | int  | yes      | N/A     | Testing batch size

### Data augmentation

Noise (images only):

Parameter      | Type   | Optional | Default | Description
---------      | ----   | -------- | ------- | -----------
prob           | double | yes      | 0.0     | Probability of each effect occurence
all_effects    | bool   | yes      | false   | Apply all effects below, randomly
decolorize     | bool   | yes      | N/A     | Whether to decolorize image
hist_eq        | bool   | yes      | N/A     | Whether to equalize histogram
inverse        | bool   | yes      | N/A     | Whether to inverse image
gauss_blur     | bool   | yes      | N/A     | Whether to apply Gaussian blur
posterize      | bool   | yes      | N/A     | Whether to posterize image
erode          | bool   | yes      | N/A     | Whether to erode image
saltpepper     | bool   | yes      | N/A     | Whether to apply salt & pepper effect to image
clahe          | bool   | yes      | N/A     | Whether to apply CLAHE
convert_to_hsv | bool   | yes      | N/A     | Whether to convert to HSV
convert_to_lab | bool   | yes      | N/A     | Whether to convert to LAB

Distort (images only):

Parameter       | Type   | Optional | Default | Description
---------       | ----   | -------- | ------- | -----------
prob            | double | yes      | 0.0     | Probability of each effect occurence
all_effects     | bool   | yes      | false   | Apply all effects below, randomly
brightness      | bool   | yes      | N/A     | Whether to distort image brightness
contrast        | bool   | yes      | N/A     | Whether to distort image contrast
saturation      | bool   | yes      | N/A     | Whether to distort image saturation
HUE             | bool   | yes      | N/A     | Whether to distort image HUE
random ordering | bool   | yes      | N/A     | Whether to randomly reorder the image channels

Geometry (images only):

Parameter        | Type   | Optional | Default  | Description
---------        | ----   | -------- | -------  | -----------
prob             | double | yes      | 0.0      | Probability of each effect occurence
all_effects      | bool   | yes      | false    | Apply all effects below, randomly
persp_horizontal | bool   | yes      | true     | Whether to distort the perspective horizontally
persp_vertical   | bool   | yes      | true     | Whether to distort the perspective vertically
zoom_out         | bool   | yes      | true     | distance change, look further away
zoom_in          | bool   | yes      | true     | distance changee, look from closer by
zoom_factor      | float  | yes      | 0.25     | 0.25 means that image can be \*1.25 or /1.25
persp_factor     | float  | yes      | 0.25     | 0.25 means that new image corners  be in \*1.25 or 0.75
pad_mode         | string | yes      | mirrored | filling around image, from `mirrored` / `constant` (black) / `repeat_nearest`


### XGBoost

Parameter     | Type   | Optional | Default                | Description
---------     | ----   | -------- | -------                | -----------
objective     | string | yes      | multi:softprob         | objective function, among multi:softprob, binary:logistic, reg:linear, reg:logistic
booster       | string | yes      | gbtree                 | which booster to use, gbtree or gblinear
num_feature   | int    | yes      | set by xgbbost         | maximum dimension of the feature
eval_metric   | string | yes      | according to objective | evaluation metric internal to xgboost
base_score    | double | yes      | 0.5                    | initial prediction score, global bias
seed          | int    | yes      | 0                      | random number seed
iterations    | int    | no       | N/A                    | number of boosting iterations
test_interval | int    | yes      | 1                      | number of iterations between each testing pass
save_period   | int    | yes      | 0                      | number of iterations between model saving to disk

Booster_params:

Parameter        | Type   | Optional | Default | Description
---------        | ----   | -------- | ------- | -----------
eta              | double | yes      | 0.3     | step size shrinkage
gamma            | double | yes      | 0       | minimum loss reduction
max_depth        | int    | yes      | 6       | maximum depth of a tree
min_child_weight | int    | yes      | 1       | minimum sum of instance weight
max_delta_step   | int    | yes      | 0       | maximum delta step
subsample        | double | yes      | 1.0     | subsample ratio of traning instance
colsample        | double | yes      | 1.0     | subsample ratio of columns when contructing each tree
lambda           | double | yes      | 1.0     | L2 regularization term on weights
alpha            | double | yes      | 0.0     | L1 regularization term on weights
lambda_bias      | double | yes      | 0.0     | L2 regularization for linear booster
tree_method      | string | yes      | auto    | tree construction algorithm, from auto, exact, approx
scale_pos_weight | double | yes      | 1.0     | control the balance of positive and negative weights

For more details on all XGBoost parameters see the dedicated page at https://github.com/dmlc/xgboost/blob/master/doc/parameter.md
        )trainparams";
    return TRAIN_PARAMS;
  }
}

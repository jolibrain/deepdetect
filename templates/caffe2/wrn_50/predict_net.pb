
 =
data
conv1_wconv1"Conv*

stride*
pad*

kernel|
conv1
conv1_scale

conv1_bias

conv1_mean
	conv1_varconv1_unique"	SpatialBN*
is_test*
epsilon��'7*
order"NCHWL
conv1_unique
conv1_w_secondconv1_internal"Mul*
axis*
	broadcast>
conv1_internal
conv1_bconv1"Add*
axis*
	broadcast
conv1conv1"ReluW
conv1pool1"MaxPool*

stride*
pad*

kernel*
order"NCHW*

legacy_padH
pool1
res1_conv1_w
res1_conv1"Conv*

stride*
pad *

kernel�

res1_conv1
res1_conv1_scale
res1_conv1_bias
res1_conv1_mean
res1_conv1_varres1_conv1_unique"	SpatialBN*
is_test*
epsilon��'7*
order"NCHW[
res1_conv1_unique
res1_conv1_w_secondres1_conv1_internal"Mul*
axis*
	broadcastM
res1_conv1_internal
res1_conv1_b
res1_conv1"Add*
axis*
	broadcast

res1_conv1
res1_conv1"ReluM

res1_conv1
res1_conv2_w
res1_conv2"Conv*

stride*
pad*

kernel�

res1_conv2
res1_conv2_scale
res1_conv2_bias
res1_conv2_mean
res1_conv2_varres1_conv2_unique"	SpatialBN*
is_test*
epsilon��'7*
order"NCHW[
res1_conv2_unique
res1_conv2_w_secondres1_conv2_internal"Mul*
axis*
	broadcastM
res1_conv2_internal
res1_conv2_b
res1_conv2"Add*
axis*
	broadcast

res1_conv2
res1_conv2"ReluM

res1_conv2
res1_conv3_w
res1_conv3"Conv*

stride*
pad *

kernel�

res1_conv3
res1_conv3_scale
res1_conv3_bias
res1_conv3_mean
res1_conv3_varres1_conv3_unique"	SpatialBN*
is_test*
epsilon��'7*
order"NCHW[
res1_conv3_unique
res1_conv3_w_secondres1_conv3_internal"Mul*
axis*
	broadcastM
res1_conv3_internal
res1_conv3_b
res1_conv3"Add*
axis*
	broadcastR
pool1
res1_match_conv_wres1_match_conv"Conv*

stride*
pad *

kernel�
res1_match_conv
res1_match_conv_scale
res1_match_conv_bias
res1_match_conv_mean
res1_match_conv_varres1_match_conv_unique"	SpatialBN*
is_test*
epsilon��'7*
order"NCHWj
res1_match_conv_unique
res1_match_conv_w_secondres1_match_conv_internal"Mul*
axis*
	broadcast\
res1_match_conv_internal
res1_match_conv_bres1_match_conv"Add*
axis*
	broadcast0
res1_match_conv

res1_conv3res1_elewise"Sum"
res1_elewiseres1_elewise"ReluO
res1_elewise
res2_conv1_w
res2_conv1"Conv*

stride*
pad *

kernel�

res2_conv1
res2_conv1_scale
res2_conv1_bias
res2_conv1_mean
res2_conv1_varres2_conv1_unique"	SpatialBN*
is_test*
epsilon��'7*
order"NCHW[
res2_conv1_unique
res2_conv1_w_secondres2_conv1_internal"Mul*
axis*
	broadcastM
res2_conv1_internal
res2_conv1_b
res2_conv1"Add*
axis*
	broadcast

res2_conv1
res2_conv1"ReluM

res2_conv1
res2_conv2_w
res2_conv2"Conv*

stride*
pad*

kernel�

res2_conv2
res2_conv2_scale
res2_conv2_bias
res2_conv2_mean
res2_conv2_varres2_conv2_unique"	SpatialBN*
is_test*
epsilon��'7*
order"NCHW[
res2_conv2_unique
res2_conv2_w_secondres2_conv2_internal"Mul*
axis*
	broadcastM
res2_conv2_internal
res2_conv2_b
res2_conv2"Add*
axis*
	broadcast

res2_conv2
res2_conv2"ReluM

res2_conv2
res2_conv3_w
res2_conv3"Conv*

stride*
pad *

kernel�

res2_conv3
res2_conv3_scale
res2_conv3_bias
res2_conv3_mean
res2_conv3_varres2_conv3_unique"	SpatialBN*
is_test*
epsilon��'7*
order"NCHW[
res2_conv3_unique
res2_conv3_w_secondres2_conv3_internal"Mul*
axis*
	broadcastM
res2_conv3_internal
res2_conv3_b
res2_conv3"Add*
axis*
	broadcast-
res1_elewise

res2_conv3res2_elewise"Sum"
res2_elewiseres2_elewise"ReluO
res2_elewise
res3_conv1_w
res3_conv1"Conv*

stride*
pad *

kernel�

res3_conv1
res3_conv1_scale
res3_conv1_bias
res3_conv1_mean
res3_conv1_varres3_conv1_unique"	SpatialBN*
is_test*
epsilon��'7*
order"NCHW[
res3_conv1_unique
res3_conv1_w_secondres3_conv1_internal"Mul*
axis*
	broadcastM
res3_conv1_internal
res3_conv1_b
res3_conv1"Add*
axis*
	broadcast

res3_conv1
res3_conv1"ReluM

res3_conv1
res3_conv2_w
res3_conv2"Conv*

stride*
pad*

kernel�

res3_conv2
res3_conv2_scale
res3_conv2_bias
res3_conv2_mean
res3_conv2_varres3_conv2_unique"	SpatialBN*
is_test*
epsilon��'7*
order"NCHW[
res3_conv2_unique
res3_conv2_w_secondres3_conv2_internal"Mul*
axis*
	broadcastM
res3_conv2_internal
res3_conv2_b
res3_conv2"Add*
axis*
	broadcast

res3_conv2
res3_conv2"ReluM

res3_conv2
res3_conv3_w
res3_conv3"Conv*

stride*
pad *

kernel�

res3_conv3
res3_conv3_scale
res3_conv3_bias
res3_conv3_mean
res3_conv3_varres3_conv3_unique"	SpatialBN*
is_test*
epsilon��'7*
order"NCHW[
res3_conv3_unique
res3_conv3_w_secondres3_conv3_internal"Mul*
axis*
	broadcastM
res3_conv3_internal
res3_conv3_b
res3_conv3"Add*
axis*
	broadcast-
res2_elewise

res3_conv3res3_elewise"Sum"
res3_elewiseres3_elewise"ReluO
res3_elewise
res4_conv1_w
res4_conv1"Conv*

stride*
pad *

kernel�

res4_conv1
res4_conv1_scale
res4_conv1_bias
res4_conv1_mean
res4_conv1_varres4_conv1_unique"	SpatialBN*
is_test*
epsilon��'7*
order"NCHW[
res4_conv1_unique
res4_conv1_w_secondres4_conv1_internal"Mul*
axis*
	broadcastM
res4_conv1_internal
res4_conv1_b
res4_conv1"Add*
axis*
	broadcast

res4_conv1
res4_conv1"ReluM

res4_conv1
res4_conv2_w
res4_conv2"Conv*

stride*
pad*

kernel�

res4_conv2
res4_conv2_scale
res4_conv2_bias
res4_conv2_mean
res4_conv2_varres4_conv2_unique"	SpatialBN*
is_test*
epsilon��'7*
order"NCHW[
res4_conv2_unique
res4_conv2_w_secondres4_conv2_internal"Mul*
axis*
	broadcastM
res4_conv2_internal
res4_conv2_b
res4_conv2"Add*
axis*
	broadcast

res4_conv2
res4_conv2"ReluM

res4_conv2
res4_conv3_w
res4_conv3"Conv*

stride*
pad *

kernel�

res4_conv3
res4_conv3_scale
res4_conv3_bias
res4_conv3_mean
res4_conv3_varres4_conv3_unique"	SpatialBN*
is_test*
epsilon��'7*
order"NCHW[
res4_conv3_unique
res4_conv3_w_secondres4_conv3_internal"Mul*
axis*
	broadcastM
res4_conv3_internal
res4_conv3_b
res4_conv3"Add*
axis*
	broadcastY
res3_elewise
res4_match_conv_wres4_match_conv"Conv*

stride*
pad *

kernel�
res4_match_conv
res4_match_conv_scale
res4_match_conv_bias
res4_match_conv_mean
res4_match_conv_varres4_match_conv_unique"	SpatialBN*
is_test*
epsilon��'7*
order"NCHWj
res4_match_conv_unique
res4_match_conv_w_secondres4_match_conv_internal"Mul*
axis*
	broadcast\
res4_match_conv_internal
res4_match_conv_bres4_match_conv"Add*
axis*
	broadcast0
res4_match_conv

res4_conv3res4_elewise"Sum"
res4_elewiseres4_elewise"ReluO
res4_elewise
res5_conv1_w
res5_conv1"Conv*

stride*
pad *

kernel�

res5_conv1
res5_conv1_scale
res5_conv1_bias
res5_conv1_mean
res5_conv1_varres5_conv1_unique"	SpatialBN*
is_test*
epsilon��'7*
order"NCHW[
res5_conv1_unique
res5_conv1_w_secondres5_conv1_internal"Mul*
axis*
	broadcastM
res5_conv1_internal
res5_conv1_b
res5_conv1"Add*
axis*
	broadcast

res5_conv1
res5_conv1"ReluM

res5_conv1
res5_conv2_w
res5_conv2"Conv*

stride*
pad*

kernel�

res5_conv2
res5_conv2_scale
res5_conv2_bias
res5_conv2_mean
res5_conv2_varres5_conv2_unique"	SpatialBN*
is_test*
epsilon��'7*
order"NCHW[
res5_conv2_unique
res5_conv2_w_secondres5_conv2_internal"Mul*
axis*
	broadcastM
res5_conv2_internal
res5_conv2_b
res5_conv2"Add*
axis*
	broadcast

res5_conv2
res5_conv2"ReluM

res5_conv2
res5_conv3_w
res5_conv3"Conv*

stride*
pad *

kernel�

res5_conv3
res5_conv3_scale
res5_conv3_bias
res5_conv3_mean
res5_conv3_varres5_conv3_unique"	SpatialBN*
is_test*
epsilon��'7*
order"NCHW[
res5_conv3_unique
res5_conv3_w_secondres5_conv3_internal"Mul*
axis*
	broadcastM
res5_conv3_internal
res5_conv3_b
res5_conv3"Add*
axis*
	broadcast-
res4_elewise

res5_conv3res5_elewise"Sum"
res5_elewiseres5_elewise"ReluO
res5_elewise
res6_conv1_w
res6_conv1"Conv*

stride*
pad *

kernel�

res6_conv1
res6_conv1_scale
res6_conv1_bias
res6_conv1_mean
res6_conv1_varres6_conv1_unique"	SpatialBN*
is_test*
epsilon��'7*
order"NCHW[
res6_conv1_unique
res6_conv1_w_secondres6_conv1_internal"Mul*
axis*
	broadcastM
res6_conv1_internal
res6_conv1_b
res6_conv1"Add*
axis*
	broadcast

res6_conv1
res6_conv1"ReluM

res6_conv1
res6_conv2_w
res6_conv2"Conv*

stride*
pad*

kernel�

res6_conv2
res6_conv2_scale
res6_conv2_bias
res6_conv2_mean
res6_conv2_varres6_conv2_unique"	SpatialBN*
is_test*
epsilon��'7*
order"NCHW[
res6_conv2_unique
res6_conv2_w_secondres6_conv2_internal"Mul*
axis*
	broadcastM
res6_conv2_internal
res6_conv2_b
res6_conv2"Add*
axis*
	broadcast

res6_conv2
res6_conv2"ReluM

res6_conv2
res6_conv3_w
res6_conv3"Conv*

stride*
pad *

kernel�

res6_conv3
res6_conv3_scale
res6_conv3_bias
res6_conv3_mean
res6_conv3_varres6_conv3_unique"	SpatialBN*
is_test*
epsilon��'7*
order"NCHW[
res6_conv3_unique
res6_conv3_w_secondres6_conv3_internal"Mul*
axis*
	broadcastM
res6_conv3_internal
res6_conv3_b
res6_conv3"Add*
axis*
	broadcast-
res5_elewise

res6_conv3res6_elewise"Sum"
res6_elewiseres6_elewise"ReluO
res6_elewise
res7_conv1_w
res7_conv1"Conv*

stride*
pad *

kernel�

res7_conv1
res7_conv1_scale
res7_conv1_bias
res7_conv1_mean
res7_conv1_varres7_conv1_unique"	SpatialBN*
is_test*
epsilon��'7*
order"NCHW[
res7_conv1_unique
res7_conv1_w_secondres7_conv1_internal"Mul*
axis*
	broadcastM
res7_conv1_internal
res7_conv1_b
res7_conv1"Add*
axis*
	broadcast

res7_conv1
res7_conv1"ReluM

res7_conv1
res7_conv2_w
res7_conv2"Conv*

stride*
pad*

kernel�

res7_conv2
res7_conv2_scale
res7_conv2_bias
res7_conv2_mean
res7_conv2_varres7_conv2_unique"	SpatialBN*
is_test*
epsilon��'7*
order"NCHW[
res7_conv2_unique
res7_conv2_w_secondres7_conv2_internal"Mul*
axis*
	broadcastM
res7_conv2_internal
res7_conv2_b
res7_conv2"Add*
axis*
	broadcast

res7_conv2
res7_conv2"ReluM

res7_conv2
res7_conv3_w
res7_conv3"Conv*

stride*
pad *

kernel�

res7_conv3
res7_conv3_scale
res7_conv3_bias
res7_conv3_mean
res7_conv3_varres7_conv3_unique"	SpatialBN*
is_test*
epsilon��'7*
order"NCHW[
res7_conv3_unique
res7_conv3_w_secondres7_conv3_internal"Mul*
axis*
	broadcastM
res7_conv3_internal
res7_conv3_b
res7_conv3"Add*
axis*
	broadcast-
res6_elewise

res7_conv3res7_elewise"Sum"
res7_elewiseres7_elewise"ReluO
res7_elewise
res8_conv1_w
res8_conv1"Conv*

stride*
pad *

kernel�

res8_conv1
res8_conv1_scale
res8_conv1_bias
res8_conv1_mean
res8_conv1_varres8_conv1_unique"	SpatialBN*
is_test*
epsilon��'7*
order"NCHW[
res8_conv1_unique
res8_conv1_w_secondres8_conv1_internal"Mul*
axis*
	broadcastM
res8_conv1_internal
res8_conv1_b
res8_conv1"Add*
axis*
	broadcast

res8_conv1
res8_conv1"ReluM

res8_conv1
res8_conv2_w
res8_conv2"Conv*

stride*
pad*

kernel�

res8_conv2
res8_conv2_scale
res8_conv2_bias
res8_conv2_mean
res8_conv2_varres8_conv2_unique"	SpatialBN*
is_test*
epsilon��'7*
order"NCHW[
res8_conv2_unique
res8_conv2_w_secondres8_conv2_internal"Mul*
axis*
	broadcastM
res8_conv2_internal
res8_conv2_b
res8_conv2"Add*
axis*
	broadcast

res8_conv2
res8_conv2"ReluM

res8_conv2
res8_conv3_w
res8_conv3"Conv*

stride*
pad *

kernel�

res8_conv3
res8_conv3_scale
res8_conv3_bias
res8_conv3_mean
res8_conv3_varres8_conv3_unique"	SpatialBN*
is_test*
epsilon��'7*
order"NCHW[
res8_conv3_unique
res8_conv3_w_secondres8_conv3_internal"Mul*
axis*
	broadcastM
res8_conv3_internal
res8_conv3_b
res8_conv3"Add*
axis*
	broadcastY
res7_elewise
res8_match_conv_wres8_match_conv"Conv*

stride*
pad *

kernel�
res8_match_conv
res8_match_conv_scale
res8_match_conv_bias
res8_match_conv_mean
res8_match_conv_varres8_match_conv_unique"	SpatialBN*
is_test*
epsilon��'7*
order"NCHWj
res8_match_conv_unique
res8_match_conv_w_secondres8_match_conv_internal"Mul*
axis*
	broadcast\
res8_match_conv_internal
res8_match_conv_bres8_match_conv"Add*
axis*
	broadcast0

res8_conv3
res8_match_convres8_elewise"Sum"
res8_elewiseres8_elewise"ReluO
res8_elewise
res9_conv1_w
res9_conv1"Conv*

stride*
pad *

kernel�

res9_conv1
res9_conv1_scale
res9_conv1_bias
res9_conv1_mean
res9_conv1_varres9_conv1_unique"	SpatialBN*
is_test*
epsilon��'7*
order"NCHW[
res9_conv1_unique
res9_conv1_w_secondres9_conv1_internal"Mul*
axis*
	broadcastM
res9_conv1_internal
res9_conv1_b
res9_conv1"Add*
axis*
	broadcast

res9_conv1
res9_conv1"ReluM

res9_conv1
res9_conv2_w
res9_conv2"Conv*

stride*
pad*

kernel�

res9_conv2
res9_conv2_scale
res9_conv2_bias
res9_conv2_mean
res9_conv2_varres9_conv2_unique"	SpatialBN*
is_test*
epsilon��'7*
order"NCHW[
res9_conv2_unique
res9_conv2_w_secondres9_conv2_internal"Mul*
axis*
	broadcastM
res9_conv2_internal
res9_conv2_b
res9_conv2"Add*
axis*
	broadcast

res9_conv2
res9_conv2"ReluM

res9_conv2
res9_conv3_w
res9_conv3"Conv*

stride*
pad *

kernel�

res9_conv3
res9_conv3_scale
res9_conv3_bias
res9_conv3_mean
res9_conv3_varres9_conv3_unique"	SpatialBN*
is_test*
epsilon��'7*
order"NCHW[
res9_conv3_unique
res9_conv3_w_secondres9_conv3_internal"Mul*
axis*
	broadcastM
res9_conv3_internal
res9_conv3_b
res9_conv3"Add*
axis*
	broadcast-
res8_elewise

res9_conv3res9_elewise"Sum"
res9_elewiseres9_elewise"ReluQ
res9_elewise
res10_conv1_wres10_conv1"Conv*

stride*
pad *

kernel�
res10_conv1
res10_conv1_scale
res10_conv1_bias
res10_conv1_mean
res10_conv1_varres10_conv1_unique"	SpatialBN*
is_test*
epsilon��'7*
order"NCHW^
res10_conv1_unique
res10_conv1_w_secondres10_conv1_internal"Mul*
axis*
	broadcastP
res10_conv1_internal
res10_conv1_bres10_conv1"Add*
axis*
	broadcast 
res10_conv1res10_conv1"ReluP
res10_conv1
res10_conv2_wres10_conv2"Conv*

stride*
pad*

kernel�
res10_conv2
res10_conv2_scale
res10_conv2_bias
res10_conv2_mean
res10_conv2_varres10_conv2_unique"	SpatialBN*
is_test*
epsilon��'7*
order"NCHW^
res10_conv2_unique
res10_conv2_w_secondres10_conv2_internal"Mul*
axis*
	broadcastP
res10_conv2_internal
res10_conv2_bres10_conv2"Add*
axis*
	broadcast 
res10_conv2res10_conv2"ReluP
res10_conv2
res10_conv3_wres10_conv3"Conv*

stride*
pad *

kernel�
res10_conv3
res10_conv3_scale
res10_conv3_bias
res10_conv3_mean
res10_conv3_varres10_conv3_unique"	SpatialBN*
is_test*
epsilon��'7*
order"NCHW^
res10_conv3_unique
res10_conv3_w_secondres10_conv3_internal"Mul*
axis*
	broadcastP
res10_conv3_internal
res10_conv3_bres10_conv3"Add*
axis*
	broadcast/
res9_elewise
res10_conv3res10_elewise"Sum$
res10_elewiseres10_elewise"ReluR
res10_elewise
res11_conv1_wres11_conv1"Conv*

stride*
pad *

kernel�
res11_conv1
res11_conv1_scale
res11_conv1_bias
res11_conv1_mean
res11_conv1_varres11_conv1_unique"	SpatialBN*
is_test*
epsilon��'7*
order"NCHW^
res11_conv1_unique
res11_conv1_w_secondres11_conv1_internal"Mul*
axis*
	broadcastP
res11_conv1_internal
res11_conv1_bres11_conv1"Add*
axis*
	broadcast 
res11_conv1res11_conv1"ReluP
res11_conv1
res11_conv2_wres11_conv2"Conv*

stride*
pad*

kernel�
res11_conv2
res11_conv2_scale
res11_conv2_bias
res11_conv2_mean
res11_conv2_varres11_conv2_unique"	SpatialBN*
is_test*
epsilon��'7*
order"NCHW^
res11_conv2_unique
res11_conv2_w_secondres11_conv2_internal"Mul*
axis*
	broadcastP
res11_conv2_internal
res11_conv2_bres11_conv2"Add*
axis*
	broadcast 
res11_conv2res11_conv2"ReluP
res11_conv2
res11_conv3_wres11_conv3"Conv*

stride*
pad *

kernel�
res11_conv3
res11_conv3_scale
res11_conv3_bias
res11_conv3_mean
res11_conv3_varres11_conv3_unique"	SpatialBN*
is_test*
epsilon��'7*
order"NCHW^
res11_conv3_unique
res11_conv3_w_secondres11_conv3_internal"Mul*
axis*
	broadcastP
res11_conv3_internal
res11_conv3_bres11_conv3"Add*
axis*
	broadcast0
res10_elewise
res11_conv3res11_elewise"Sum$
res11_elewiseres11_elewise"ReluR
res11_elewise
res12_conv1_wres12_conv1"Conv*

stride*
pad *

kernel�
res12_conv1
res12_conv1_scale
res12_conv1_bias
res12_conv1_mean
res12_conv1_varres12_conv1_unique"	SpatialBN*
is_test*
epsilon��'7*
order"NCHW^
res12_conv1_unique
res12_conv1_w_secondres12_conv1_internal"Mul*
axis*
	broadcastP
res12_conv1_internal
res12_conv1_bres12_conv1"Add*
axis*
	broadcast 
res12_conv1res12_conv1"ReluP
res12_conv1
res12_conv2_wres12_conv2"Conv*

stride*
pad*

kernel�
res12_conv2
res12_conv2_scale
res12_conv2_bias
res12_conv2_mean
res12_conv2_varres12_conv2_unique"	SpatialBN*
is_test*
epsilon��'7*
order"NCHW^
res12_conv2_unique
res12_conv2_w_secondres12_conv2_internal"Mul*
axis*
	broadcastP
res12_conv2_internal
res12_conv2_bres12_conv2"Add*
axis*
	broadcast 
res12_conv2res12_conv2"ReluP
res12_conv2
res12_conv3_wres12_conv3"Conv*

stride*
pad *

kernel�
res12_conv3
res12_conv3_scale
res12_conv3_bias
res12_conv3_mean
res12_conv3_varres12_conv3_unique"	SpatialBN*
is_test*
epsilon��'7*
order"NCHW^
res12_conv3_unique
res12_conv3_w_secondres12_conv3_internal"Mul*
axis*
	broadcastP
res12_conv3_internal
res12_conv3_bres12_conv3"Add*
axis*
	broadcast0
res11_elewise
res12_conv3res12_elewise"Sum$
res12_elewiseres12_elewise"ReluR
res12_elewise
res13_conv1_wres13_conv1"Conv*

stride*
pad *

kernel�
res13_conv1
res13_conv1_scale
res13_conv1_bias
res13_conv1_mean
res13_conv1_varres13_conv1_unique"	SpatialBN*
is_test*
epsilon��'7*
order"NCHW^
res13_conv1_unique
res13_conv1_w_secondres13_conv1_internal"Mul*
axis*
	broadcastP
res13_conv1_internal
res13_conv1_bres13_conv1"Add*
axis*
	broadcast 
res13_conv1res13_conv1"ReluP
res13_conv1
res13_conv2_wres13_conv2"Conv*

stride*
pad*

kernel�
res13_conv2
res13_conv2_scale
res13_conv2_bias
res13_conv2_mean
res13_conv2_varres13_conv2_unique"	SpatialBN*
is_test*
epsilon��'7*
order"NCHW^
res13_conv2_unique
res13_conv2_w_secondres13_conv2_internal"Mul*
axis*
	broadcastP
res13_conv2_internal
res13_conv2_bres13_conv2"Add*
axis*
	broadcast 
res13_conv2res13_conv2"ReluP
res13_conv2
res13_conv3_wres13_conv3"Conv*

stride*
pad *

kernel�
res13_conv3
res13_conv3_scale
res13_conv3_bias
res13_conv3_mean
res13_conv3_varres13_conv3_unique"	SpatialBN*
is_test*
epsilon��'7*
order"NCHW^
res13_conv3_unique
res13_conv3_w_secondres13_conv3_internal"Mul*
axis*
	broadcastP
res13_conv3_internal
res13_conv3_bres13_conv3"Add*
axis*
	broadcast0
res12_elewise
res13_conv3res13_elewise"Sum$
res13_elewiseres13_elewise"ReluR
res13_elewise
res14_conv1_wres14_conv1"Conv*

stride*
pad *

kernel�
res14_conv1
res14_conv1_scale
res14_conv1_bias
res14_conv1_mean
res14_conv1_varres14_conv1_unique"	SpatialBN*
is_test*
epsilon��'7*
order"NCHW^
res14_conv1_unique
res14_conv1_w_secondres14_conv1_internal"Mul*
axis*
	broadcastP
res14_conv1_internal
res14_conv1_bres14_conv1"Add*
axis*
	broadcast 
res14_conv1res14_conv1"ReluP
res14_conv1
res14_conv2_wres14_conv2"Conv*

stride*
pad*

kernel�
res14_conv2
res14_conv2_scale
res14_conv2_bias
res14_conv2_mean
res14_conv2_varres14_conv2_unique"	SpatialBN*
is_test*
epsilon��'7*
order"NCHW^
res14_conv2_unique
res14_conv2_w_secondres14_conv2_internal"Mul*
axis*
	broadcastP
res14_conv2_internal
res14_conv2_bres14_conv2"Add*
axis*
	broadcast 
res14_conv2res14_conv2"ReluP
res14_conv2
res14_conv3_wres14_conv3"Conv*

stride*
pad *

kernel�
res14_conv3
res14_conv3_scale
res14_conv3_bias
res14_conv3_mean
res14_conv3_varres14_conv3_unique"	SpatialBN*
is_test*
epsilon��'7*
order"NCHW^
res14_conv3_unique
res14_conv3_w_secondres14_conv3_internal"Mul*
axis*
	broadcastP
res14_conv3_internal
res14_conv3_bres14_conv3"Add*
axis*
	broadcast\
res13_elewise
res14_match_conv_wres14_match_conv"Conv*

stride*
pad *

kernel�
res14_match_conv
res14_match_conv_scale
res14_match_conv_bias
res14_match_conv_mean
res14_match_conv_varres14_match_conv_unique"	SpatialBN*
is_test*
epsilon��'7*
order"NCHWm
res14_match_conv_unique
res14_match_conv_w_secondres14_match_conv_internal"Mul*
axis*
	broadcast_
res14_match_conv_internal
res14_match_conv_bres14_match_conv"Add*
axis*
	broadcast3
res14_match_conv
res14_conv3res14_elewise"Sum$
res14_elewiseres14_elewise"ReluR
res14_elewise
res15_conv1_wres15_conv1"Conv*

stride*
pad *

kernel�
res15_conv1
res15_conv1_scale
res15_conv1_bias
res15_conv1_mean
res15_conv1_varres15_conv1_unique"	SpatialBN*
is_test*
epsilon��'7*
order"NCHW^
res15_conv1_unique
res15_conv1_w_secondres15_conv1_internal"Mul*
axis*
	broadcastP
res15_conv1_internal
res15_conv1_bres15_conv1"Add*
axis*
	broadcast 
res15_conv1res15_conv1"ReluP
res15_conv1
res15_conv2_wres15_conv2"Conv*

stride*
pad*

kernel�
res15_conv2
res15_conv2_scale
res15_conv2_bias
res15_conv2_mean
res15_conv2_varres15_conv2_unique"	SpatialBN*
is_test*
epsilon��'7*
order"NCHW^
res15_conv2_unique
res15_conv2_w_secondres15_conv2_internal"Mul*
axis*
	broadcastP
res15_conv2_internal
res15_conv2_bres15_conv2"Add*
axis*
	broadcast 
res15_conv2res15_conv2"ReluP
res15_conv2
res15_conv3_wres15_conv3"Conv*

stride*
pad *

kernel�
res15_conv3
res15_conv3_scale
res15_conv3_bias
res15_conv3_mean
res15_conv3_varres15_conv3_unique"	SpatialBN*
is_test*
epsilon��'7*
order"NCHW^
res15_conv3_unique
res15_conv3_w_secondres15_conv3_internal"Mul*
axis*
	broadcastP
res15_conv3_internal
res15_conv3_bres15_conv3"Add*
axis*
	broadcast0
res14_elewise
res15_conv3res15_elewise"Sum$
res15_elewiseres15_elewise"ReluR
res15_elewise
res16_conv1_wres16_conv1"Conv*

stride*
pad *

kernel�
res16_conv1
res16_conv1_scale
res16_conv1_bias
res16_conv1_mean
res16_conv1_varres16_conv1_unique"	SpatialBN*
is_test*
epsilon��'7*
order"NCHW^
res16_conv1_unique
res16_conv1_w_secondres16_conv1_internal"Mul*
axis*
	broadcastP
res16_conv1_internal
res16_conv1_bres16_conv1"Add*
axis*
	broadcast 
res16_conv1res16_conv1"ReluP
res16_conv1
res16_conv2_wres16_conv2"Conv*

stride*
pad*

kernel�
res16_conv2
res16_conv2_scale
res16_conv2_bias
res16_conv2_mean
res16_conv2_varres16_conv2_unique"	SpatialBN*
is_test*
epsilon��'7*
order"NCHW^
res16_conv2_unique
res16_conv2_w_secondres16_conv2_internal"Mul*
axis*
	broadcastP
res16_conv2_internal
res16_conv2_bres16_conv2"Add*
axis*
	broadcast 
res16_conv2res16_conv2"ReluP
res16_conv2
res16_conv3_wres16_conv3"Conv*

stride*
pad *

kernel�
res16_conv3
res16_conv3_scale
res16_conv3_bias
res16_conv3_mean
res16_conv3_varres16_conv3_unique"	SpatialBN*
is_test*
epsilon��'7*
order"NCHW^
res16_conv3_unique
res16_conv3_w_secondres16_conv3_internal"Mul*
axis*
	broadcastP
res16_conv3_internal
res16_conv3_bres16_conv3"Add*
axis*
	broadcast0
res15_elewise
res16_conv3res16_elewise"Sum$
res16_elewiseres16_elewise"Reluz
res16_elewisepool_ave"AveragePool*

stride*
pad *

kernel *
order"NCHW*

legacy_pad*
global_pooling6
pool_ave
classifier_w
classifier_b
classifier"FC

classifierprob"Softmax:data:conv1_w:conv1_scale:
conv1_bias:
conv1_mean:	conv1_var:conv1_w_second:conv1_b:res1_conv1_w:res1_conv1_scale:res1_conv1_bias:res1_conv1_mean:res1_conv1_var:res1_conv1_w_second:res1_conv1_b:res1_conv2_w:res1_conv2_scale:res1_conv2_bias:res1_conv2_mean:res1_conv2_var:res1_conv2_w_second:res1_conv2_b:res1_conv3_w:res1_conv3_scale:res1_conv3_bias:res1_conv3_mean:res1_conv3_var:res1_conv3_w_second:res1_conv3_b:res1_match_conv_w:res1_match_conv_scale:res1_match_conv_bias:res1_match_conv_mean:res1_match_conv_var:res1_match_conv_w_second:res1_match_conv_b:res2_conv1_w:res2_conv1_scale:res2_conv1_bias:res2_conv1_mean:res2_conv1_var:res2_conv1_w_second:res2_conv1_b:res2_conv2_w:res2_conv2_scale:res2_conv2_bias:res2_conv2_mean:res2_conv2_var:res2_conv2_w_second:res2_conv2_b:res2_conv3_w:res2_conv3_scale:res2_conv3_bias:res2_conv3_mean:res2_conv3_var:res2_conv3_w_second:res2_conv3_b:res3_conv1_w:res3_conv1_scale:res3_conv1_bias:res3_conv1_mean:res3_conv1_var:res3_conv1_w_second:res3_conv1_b:res3_conv2_w:res3_conv2_scale:res3_conv2_bias:res3_conv2_mean:res3_conv2_var:res3_conv2_w_second:res3_conv2_b:res3_conv3_w:res3_conv3_scale:res3_conv3_bias:res3_conv3_mean:res3_conv3_var:res3_conv3_w_second:res3_conv3_b:res4_conv1_w:res4_conv1_scale:res4_conv1_bias:res4_conv1_mean:res4_conv1_var:res4_conv1_w_second:res4_conv1_b:res4_conv2_w:res4_conv2_scale:res4_conv2_bias:res4_conv2_mean:res4_conv2_var:res4_conv2_w_second:res4_conv2_b:res4_conv3_w:res4_conv3_scale:res4_conv3_bias:res4_conv3_mean:res4_conv3_var:res4_conv3_w_second:res4_conv3_b:res4_match_conv_w:res4_match_conv_scale:res4_match_conv_bias:res4_match_conv_mean:res4_match_conv_var:res4_match_conv_w_second:res4_match_conv_b:res5_conv1_w:res5_conv1_scale:res5_conv1_bias:res5_conv1_mean:res5_conv1_var:res5_conv1_w_second:res5_conv1_b:res5_conv2_w:res5_conv2_scale:res5_conv2_bias:res5_conv2_mean:res5_conv2_var:res5_conv2_w_second:res5_conv2_b:res5_conv3_w:res5_conv3_scale:res5_conv3_bias:res5_conv3_mean:res5_conv3_var:res5_conv3_w_second:res5_conv3_b:res6_conv1_w:res6_conv1_scale:res6_conv1_bias:res6_conv1_mean:res6_conv1_var:res6_conv1_w_second:res6_conv1_b:res6_conv2_w:res6_conv2_scale:res6_conv2_bias:res6_conv2_mean:res6_conv2_var:res6_conv2_w_second:res6_conv2_b:res6_conv3_w:res6_conv3_scale:res6_conv3_bias:res6_conv3_mean:res6_conv3_var:res6_conv3_w_second:res6_conv3_b:res7_conv1_w:res7_conv1_scale:res7_conv1_bias:res7_conv1_mean:res7_conv1_var:res7_conv1_w_second:res7_conv1_b:res7_conv2_w:res7_conv2_scale:res7_conv2_bias:res7_conv2_mean:res7_conv2_var:res7_conv2_w_second:res7_conv2_b:res7_conv3_w:res7_conv3_scale:res7_conv3_bias:res7_conv3_mean:res7_conv3_var:res7_conv3_w_second:res7_conv3_b:res8_conv1_w:res8_conv1_scale:res8_conv1_bias:res8_conv1_mean:res8_conv1_var:res8_conv1_w_second:res8_conv1_b:res8_conv2_w:res8_conv2_scale:res8_conv2_bias:res8_conv2_mean:res8_conv2_var:res8_conv2_w_second:res8_conv2_b:res8_conv3_w:res8_conv3_scale:res8_conv3_bias:res8_conv3_mean:res8_conv3_var:res8_conv3_w_second:res8_conv3_b:res8_match_conv_w:res8_match_conv_scale:res8_match_conv_bias:res8_match_conv_mean:res8_match_conv_var:res8_match_conv_w_second:res8_match_conv_b:res9_conv1_w:res9_conv1_scale:res9_conv1_bias:res9_conv1_mean:res9_conv1_var:res9_conv1_w_second:res9_conv1_b:res9_conv2_w:res9_conv2_scale:res9_conv2_bias:res9_conv2_mean:res9_conv2_var:res9_conv2_w_second:res9_conv2_b:res9_conv3_w:res9_conv3_scale:res9_conv3_bias:res9_conv3_mean:res9_conv3_var:res9_conv3_w_second:res9_conv3_b:res10_conv1_w:res10_conv1_scale:res10_conv1_bias:res10_conv1_mean:res10_conv1_var:res10_conv1_w_second:res10_conv1_b:res10_conv2_w:res10_conv2_scale:res10_conv2_bias:res10_conv2_mean:res10_conv2_var:res10_conv2_w_second:res10_conv2_b:res10_conv3_w:res10_conv3_scale:res10_conv3_bias:res10_conv3_mean:res10_conv3_var:res10_conv3_w_second:res10_conv3_b:res11_conv1_w:res11_conv1_scale:res11_conv1_bias:res11_conv1_mean:res11_conv1_var:res11_conv1_w_second:res11_conv1_b:res11_conv2_w:res11_conv2_scale:res11_conv2_bias:res11_conv2_mean:res11_conv2_var:res11_conv2_w_second:res11_conv2_b:res11_conv3_w:res11_conv3_scale:res11_conv3_bias:res11_conv3_mean:res11_conv3_var:res11_conv3_w_second:res11_conv3_b:res12_conv1_w:res12_conv1_scale:res12_conv1_bias:res12_conv1_mean:res12_conv1_var:res12_conv1_w_second:res12_conv1_b:res12_conv2_w:res12_conv2_scale:res12_conv2_bias:res12_conv2_mean:res12_conv2_var:res12_conv2_w_second:res12_conv2_b:res12_conv3_w:res12_conv3_scale:res12_conv3_bias:res12_conv3_mean:res12_conv3_var:res12_conv3_w_second:res12_conv3_b:res13_conv1_w:res13_conv1_scale:res13_conv1_bias:res13_conv1_mean:res13_conv1_var:res13_conv1_w_second:res13_conv1_b:res13_conv2_w:res13_conv2_scale:res13_conv2_bias:res13_conv2_mean:res13_conv2_var:res13_conv2_w_second:res13_conv2_b:res13_conv3_w:res13_conv3_scale:res13_conv3_bias:res13_conv3_mean:res13_conv3_var:res13_conv3_w_second:res13_conv3_b:res14_conv1_w:res14_conv1_scale:res14_conv1_bias:res14_conv1_mean:res14_conv1_var:res14_conv1_w_second:res14_conv1_b:res14_conv2_w:res14_conv2_scale:res14_conv2_bias:res14_conv2_mean:res14_conv2_var:res14_conv2_w_second:res14_conv2_b:res14_conv3_w:res14_conv3_scale:res14_conv3_bias:res14_conv3_mean:res14_conv3_var:res14_conv3_w_second:res14_conv3_b:res14_match_conv_w:res14_match_conv_scale:res14_match_conv_bias:res14_match_conv_mean:res14_match_conv_var:res14_match_conv_w_second:res14_match_conv_b:res15_conv1_w:res15_conv1_scale:res15_conv1_bias:res15_conv1_mean:res15_conv1_var:res15_conv1_w_second:res15_conv1_b:res15_conv2_w:res15_conv2_scale:res15_conv2_bias:res15_conv2_mean:res15_conv2_var:res15_conv2_w_second:res15_conv2_b:res15_conv3_w:res15_conv3_scale:res15_conv3_bias:res15_conv3_mean:res15_conv3_var:res15_conv3_w_second:res15_conv3_b:res16_conv1_w:res16_conv1_scale:res16_conv1_bias:res16_conv1_mean:res16_conv1_var:res16_conv1_w_second:res16_conv1_b:res16_conv2_w:res16_conv2_scale:res16_conv2_bias:res16_conv2_mean:res16_conv2_var:res16_conv2_w_second:res16_conv2_b:res16_conv3_w:res16_conv3_scale:res16_conv3_bias:res16_conv3_mean:res16_conv3_var:res16_conv3_w_second:res16_conv3_b:classifier_w:classifier_bBprob
��
��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
�
BiasAdd

value"T	
bias"T
output"T""
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
�
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

$
DisableCopyOnRead
resource�
.
Identity

input"T
output"T"	
Ttype
u
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
2	
�
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
�
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( �

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
L

StringJoin
inputs*N

output"

Nint("
	separatorstring 
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.13.02v2.13.0-rc2-7-g1cb1a030a628��
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
�
$Adam/v/mnist_classifier/dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*5
shared_name&$Adam/v/mnist_classifier/dense_1/bias
�
8Adam/v/mnist_classifier/dense_1/bias/Read/ReadVariableOpReadVariableOp$Adam/v/mnist_classifier/dense_1/bias*
_output_shapes
:
*
dtype0
�
$Adam/m/mnist_classifier/dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*5
shared_name&$Adam/m/mnist_classifier/dense_1/bias
�
8Adam/m/mnist_classifier/dense_1/bias/Read/ReadVariableOpReadVariableOp$Adam/m/mnist_classifier/dense_1/bias*
_output_shapes
:
*
dtype0
�
&Adam/v/mnist_classifier/dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�
*7
shared_name(&Adam/v/mnist_classifier/dense_1/kernel
�
:Adam/v/mnist_classifier/dense_1/kernel/Read/ReadVariableOpReadVariableOp&Adam/v/mnist_classifier/dense_1/kernel*
_output_shapes
:	�
*
dtype0
�
&Adam/m/mnist_classifier/dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�
*7
shared_name(&Adam/m/mnist_classifier/dense_1/kernel
�
:Adam/m/mnist_classifier/dense_1/kernel/Read/ReadVariableOpReadVariableOp&Adam/m/mnist_classifier/dense_1/kernel*
_output_shapes
:	�
*
dtype0
�
"Adam/v/mnist_classifier/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*3
shared_name$"Adam/v/mnist_classifier/dense/bias
�
6Adam/v/mnist_classifier/dense/bias/Read/ReadVariableOpReadVariableOp"Adam/v/mnist_classifier/dense/bias*
_output_shapes	
:�*
dtype0
�
"Adam/m/mnist_classifier/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*3
shared_name$"Adam/m/mnist_classifier/dense/bias
�
6Adam/m/mnist_classifier/dense/bias/Read/ReadVariableOpReadVariableOp"Adam/m/mnist_classifier/dense/bias*
_output_shapes	
:�*
dtype0
�
$Adam/v/mnist_classifier/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*5
shared_name&$Adam/v/mnist_classifier/dense/kernel
�
8Adam/v/mnist_classifier/dense/kernel/Read/ReadVariableOpReadVariableOp$Adam/v/mnist_classifier/dense/kernel* 
_output_shapes
:
��*
dtype0
�
$Adam/m/mnist_classifier/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*5
shared_name&$Adam/m/mnist_classifier/dense/kernel
�
8Adam/m/mnist_classifier/dense/kernel/Read/ReadVariableOpReadVariableOp$Adam/m/mnist_classifier/dense/kernel* 
_output_shapes
:
��*
dtype0
�
%Adam/v/mnist_classifier/conv2d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*6
shared_name'%Adam/v/mnist_classifier/conv2d_1/bias
�
9Adam/v/mnist_classifier/conv2d_1/bias/Read/ReadVariableOpReadVariableOp%Adam/v/mnist_classifier/conv2d_1/bias*
_output_shapes
:@*
dtype0
�
%Adam/m/mnist_classifier/conv2d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*6
shared_name'%Adam/m/mnist_classifier/conv2d_1/bias
�
9Adam/m/mnist_classifier/conv2d_1/bias/Read/ReadVariableOpReadVariableOp%Adam/m/mnist_classifier/conv2d_1/bias*
_output_shapes
:@*
dtype0
�
'Adam/v/mnist_classifier/conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*8
shared_name)'Adam/v/mnist_classifier/conv2d_1/kernel
�
;Adam/v/mnist_classifier/conv2d_1/kernel/Read/ReadVariableOpReadVariableOp'Adam/v/mnist_classifier/conv2d_1/kernel*&
_output_shapes
: @*
dtype0
�
'Adam/m/mnist_classifier/conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*8
shared_name)'Adam/m/mnist_classifier/conv2d_1/kernel
�
;Adam/m/mnist_classifier/conv2d_1/kernel/Read/ReadVariableOpReadVariableOp'Adam/m/mnist_classifier/conv2d_1/kernel*&
_output_shapes
: @*
dtype0
�
#Adam/v/mnist_classifier/conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *4
shared_name%#Adam/v/mnist_classifier/conv2d/bias
�
7Adam/v/mnist_classifier/conv2d/bias/Read/ReadVariableOpReadVariableOp#Adam/v/mnist_classifier/conv2d/bias*
_output_shapes
: *
dtype0
�
#Adam/m/mnist_classifier/conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *4
shared_name%#Adam/m/mnist_classifier/conv2d/bias
�
7Adam/m/mnist_classifier/conv2d/bias/Read/ReadVariableOpReadVariableOp#Adam/m/mnist_classifier/conv2d/bias*
_output_shapes
: *
dtype0
�
%Adam/v/mnist_classifier/conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *6
shared_name'%Adam/v/mnist_classifier/conv2d/kernel
�
9Adam/v/mnist_classifier/conv2d/kernel/Read/ReadVariableOpReadVariableOp%Adam/v/mnist_classifier/conv2d/kernel*&
_output_shapes
: *
dtype0
�
%Adam/m/mnist_classifier/conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *6
shared_name'%Adam/m/mnist_classifier/conv2d/kernel
�
9Adam/m/mnist_classifier/conv2d/kernel/Read/ReadVariableOpReadVariableOp%Adam/m/mnist_classifier/conv2d/kernel*&
_output_shapes
: *
dtype0
n
learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namelearning_rate
g
!learning_rate/Read/ReadVariableOpReadVariableOplearning_rate*
_output_shapes
: *
dtype0
f
	iterationVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	iteration
_
iteration/Read/ReadVariableOpReadVariableOp	iteration*
_output_shapes
: *
dtype0	
�
mnist_classifier/dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*.
shared_namemnist_classifier/dense_1/bias
�
1mnist_classifier/dense_1/bias/Read/ReadVariableOpReadVariableOpmnist_classifier/dense_1/bias*
_output_shapes
:
*
dtype0
�
mnist_classifier/dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�
*0
shared_name!mnist_classifier/dense_1/kernel
�
3mnist_classifier/dense_1/kernel/Read/ReadVariableOpReadVariableOpmnist_classifier/dense_1/kernel*
_output_shapes
:	�
*
dtype0
�
mnist_classifier/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*,
shared_namemnist_classifier/dense/bias
�
/mnist_classifier/dense/bias/Read/ReadVariableOpReadVariableOpmnist_classifier/dense/bias*
_output_shapes	
:�*
dtype0
�
mnist_classifier/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*.
shared_namemnist_classifier/dense/kernel
�
1mnist_classifier/dense/kernel/Read/ReadVariableOpReadVariableOpmnist_classifier/dense/kernel* 
_output_shapes
:
��*
dtype0
�
mnist_classifier/conv2d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*/
shared_name mnist_classifier/conv2d_1/bias
�
2mnist_classifier/conv2d_1/bias/Read/ReadVariableOpReadVariableOpmnist_classifier/conv2d_1/bias*
_output_shapes
:@*
dtype0
�
 mnist_classifier/conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*1
shared_name" mnist_classifier/conv2d_1/kernel
�
4mnist_classifier/conv2d_1/kernel/Read/ReadVariableOpReadVariableOp mnist_classifier/conv2d_1/kernel*&
_output_shapes
: @*
dtype0
�
mnist_classifier/conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_namemnist_classifier/conv2d/bias
�
0mnist_classifier/conv2d/bias/Read/ReadVariableOpReadVariableOpmnist_classifier/conv2d/bias*
_output_shapes
: *
dtype0
�
mnist_classifier/conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: */
shared_name mnist_classifier/conv2d/kernel
�
2mnist_classifier/conv2d/kernel/Read/ReadVariableOpReadVariableOpmnist_classifier/conv2d/kernel*&
_output_shapes
: *
dtype0
�
serving_default_input_1Placeholder*/
_output_shapes
:���������*
dtype0*$
shape:���������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1mnist_classifier/conv2d/kernelmnist_classifier/conv2d/bias mnist_classifier/conv2d_1/kernelmnist_classifier/conv2d_1/biasmnist_classifier/dense/kernelmnist_classifier/dense/biasmnist_classifier/dense_1/kernelmnist_classifier/dense_1/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *,
f'R%
#__inference_signature_wrapper_18274

NoOpNoOp
�B
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�B
value�AB�A B�A
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	conv1
		pool1
	
conv2
	pool2
flatten

dense1
outputs
	optimizer

signatures*
<
0
1
2
3
4
5
6
7*
<
0
1
2
3
4
5
6
7*
* 
�
non_trainable_variables

layers
metrics
layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

trace_0* 

trace_0* 
* 
�
 	variables
!trainable_variables
"regularization_losses
#	keras_api
$__call__
*%&call_and_return_all_conditional_losses

kernel
bias
 &_jit_compiled_convolution_op*
�
'	variables
(trainable_variables
)regularization_losses
*	keras_api
+__call__
*,&call_and_return_all_conditional_losses* 
�
-	variables
.trainable_variables
/regularization_losses
0	keras_api
1__call__
*2&call_and_return_all_conditional_losses

kernel
bias
 3_jit_compiled_convolution_op*
�
4	variables
5trainable_variables
6regularization_losses
7	keras_api
8__call__
*9&call_and_return_all_conditional_losses* 
�
:	variables
;trainable_variables
<regularization_losses
=	keras_api
>__call__
*?&call_and_return_all_conditional_losses* 
�
@	variables
Atrainable_variables
Bregularization_losses
C	keras_api
D__call__
*E&call_and_return_all_conditional_losses

kernel
bias*
�
F	variables
Gtrainable_variables
Hregularization_losses
I	keras_api
J__call__
*K&call_and_return_all_conditional_losses

kernel
bias*
�
L
_variables
M_iterations
N_learning_rate
O_index_dict
P
_momentums
Q_velocities
R_update_step_xla*

Sserving_default* 
^X
VARIABLE_VALUEmnist_classifier/conv2d/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEmnist_classifier/conv2d/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUE mnist_classifier/conv2d_1/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEmnist_classifier/conv2d_1/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEmnist_classifier/dense/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEmnist_classifier/dense/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEmnist_classifier/dense_1/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEmnist_classifier/dense_1/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
* 
5
0
	1

2
3
4
5
6*

T0
U1*
* 
* 
* 
* 

0
1*

0
1*
* 
�
Vnon_trainable_variables

Wlayers
Xmetrics
Ylayer_regularization_losses
Zlayer_metrics
 	variables
!trainable_variables
"regularization_losses
$__call__
*%&call_and_return_all_conditional_losses
&%"call_and_return_conditional_losses*

[trace_0* 

\trace_0* 
* 
* 
* 
* 
�
]non_trainable_variables

^layers
_metrics
`layer_regularization_losses
alayer_metrics
'	variables
(trainable_variables
)regularization_losses
+__call__
*,&call_and_return_all_conditional_losses
&,"call_and_return_conditional_losses* 

btrace_0* 

ctrace_0* 

0
1*

0
1*
* 
�
dnon_trainable_variables

elayers
fmetrics
glayer_regularization_losses
hlayer_metrics
-	variables
.trainable_variables
/regularization_losses
1__call__
*2&call_and_return_all_conditional_losses
&2"call_and_return_conditional_losses*

itrace_0* 

jtrace_0* 
* 
* 
* 
* 
�
knon_trainable_variables

llayers
mmetrics
nlayer_regularization_losses
olayer_metrics
4	variables
5trainable_variables
6regularization_losses
8__call__
*9&call_and_return_all_conditional_losses
&9"call_and_return_conditional_losses* 

ptrace_0* 

qtrace_0* 
* 
* 
* 
�
rnon_trainable_variables

slayers
tmetrics
ulayer_regularization_losses
vlayer_metrics
:	variables
;trainable_variables
<regularization_losses
>__call__
*?&call_and_return_all_conditional_losses
&?"call_and_return_conditional_losses* 

wtrace_0* 

xtrace_0* 

0
1*

0
1*
* 
�
ynon_trainable_variables

zlayers
{metrics
|layer_regularization_losses
}layer_metrics
@	variables
Atrainable_variables
Bregularization_losses
D__call__
*E&call_and_return_all_conditional_losses
&E"call_and_return_conditional_losses*

~trace_0* 

trace_0* 

0
1*

0
1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
F	variables
Gtrainable_variables
Hregularization_losses
J__call__
*K&call_and_return_all_conditional_losses
&K"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
�
M0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16*
SM
VARIABLE_VALUE	iteration0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUElearning_rate3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
D
�0
�1
�2
�3
�4
�5
�6
�7*
D
�0
�1
�2
�3
�4
�5
�6
�7*
* 
* 
<
�	variables
�	keras_api

�total

�count*
M
�	variables
�	keras_api

�total

�count
�
_fn_kwargs*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
pj
VARIABLE_VALUE%Adam/m/mnist_classifier/conv2d/kernel1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUE%Adam/v/mnist_classifier/conv2d/kernel1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE#Adam/m/mnist_classifier/conv2d/bias1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE#Adam/v/mnist_classifier/conv2d/bias1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUE'Adam/m/mnist_classifier/conv2d_1/kernel1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUE'Adam/v/mnist_classifier/conv2d_1/kernel1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUE%Adam/m/mnist_classifier/conv2d_1/bias1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUE%Adam/v/mnist_classifier/conv2d_1/bias1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUE$Adam/m/mnist_classifier/dense/kernel1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUE$Adam/v/mnist_classifier/dense/kernel2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/m/mnist_classifier/dense/bias2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/v/mnist_classifier/dense/bias2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUE&Adam/m/mnist_classifier/dense_1/kernel2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUE&Adam/v/mnist_classifier/dense_1/kernel2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUE$Adam/m/mnist_classifier/dense_1/bias2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUE$Adam/v/mnist_classifier/dense_1/bias2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUE*

�0
�1*

�	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

�0
�1*

�	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�	
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenamemnist_classifier/conv2d/kernelmnist_classifier/conv2d/bias mnist_classifier/conv2d_1/kernelmnist_classifier/conv2d_1/biasmnist_classifier/dense/kernelmnist_classifier/dense/biasmnist_classifier/dense_1/kernelmnist_classifier/dense_1/bias	iterationlearning_rate%Adam/m/mnist_classifier/conv2d/kernel%Adam/v/mnist_classifier/conv2d/kernel#Adam/m/mnist_classifier/conv2d/bias#Adam/v/mnist_classifier/conv2d/bias'Adam/m/mnist_classifier/conv2d_1/kernel'Adam/v/mnist_classifier/conv2d_1/kernel%Adam/m/mnist_classifier/conv2d_1/bias%Adam/v/mnist_classifier/conv2d_1/bias$Adam/m/mnist_classifier/dense/kernel$Adam/v/mnist_classifier/dense/kernel"Adam/m/mnist_classifier/dense/bias"Adam/v/mnist_classifier/dense/bias&Adam/m/mnist_classifier/dense_1/kernel&Adam/v/mnist_classifier/dense_1/kernel$Adam/m/mnist_classifier/dense_1/bias$Adam/v/mnist_classifier/dense_1/biastotal_1count_1totalcountConst*+
Tin$
"2 *
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *'
f"R 
__inference__traced_save_18587
�	
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamemnist_classifier/conv2d/kernelmnist_classifier/conv2d/bias mnist_classifier/conv2d_1/kernelmnist_classifier/conv2d_1/biasmnist_classifier/dense/kernelmnist_classifier/dense/biasmnist_classifier/dense_1/kernelmnist_classifier/dense_1/bias	iterationlearning_rate%Adam/m/mnist_classifier/conv2d/kernel%Adam/v/mnist_classifier/conv2d/kernel#Adam/m/mnist_classifier/conv2d/bias#Adam/v/mnist_classifier/conv2d/bias'Adam/m/mnist_classifier/conv2d_1/kernel'Adam/v/mnist_classifier/conv2d_1/kernel%Adam/m/mnist_classifier/conv2d_1/bias%Adam/v/mnist_classifier/conv2d_1/bias$Adam/m/mnist_classifier/dense/kernel$Adam/v/mnist_classifier/dense/kernel"Adam/m/mnist_classifier/dense/bias"Adam/v/mnist_classifier/dense/bias&Adam/m/mnist_classifier/dense_1/kernel&Adam/v/mnist_classifier/dense_1/kernel$Adam/m/mnist_classifier/dense_1/bias$Adam/v/mnist_classifier/dense_1/biastotal_1count_1totalcount**
Tin#
!2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� **
f%R#
!__inference__traced_restore_18686��
�
d
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_18096

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
f
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_18106

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
��
�
__inference__traced_save_18587
file_prefixO
5read_disablecopyonread_mnist_classifier_conv2d_kernel: C
5read_1_disablecopyonread_mnist_classifier_conv2d_bias: S
9read_2_disablecopyonread_mnist_classifier_conv2d_1_kernel: @E
7read_3_disablecopyonread_mnist_classifier_conv2d_1_bias:@J
6read_4_disablecopyonread_mnist_classifier_dense_kernel:
��C
4read_5_disablecopyonread_mnist_classifier_dense_bias:	�K
8read_6_disablecopyonread_mnist_classifier_dense_1_kernel:	�
D
6read_7_disablecopyonread_mnist_classifier_dense_1_bias:
,
"read_8_disablecopyonread_iteration:	 0
&read_9_disablecopyonread_learning_rate: Y
?read_10_disablecopyonread_adam_m_mnist_classifier_conv2d_kernel: Y
?read_11_disablecopyonread_adam_v_mnist_classifier_conv2d_kernel: K
=read_12_disablecopyonread_adam_m_mnist_classifier_conv2d_bias: K
=read_13_disablecopyonread_adam_v_mnist_classifier_conv2d_bias: [
Aread_14_disablecopyonread_adam_m_mnist_classifier_conv2d_1_kernel: @[
Aread_15_disablecopyonread_adam_v_mnist_classifier_conv2d_1_kernel: @M
?read_16_disablecopyonread_adam_m_mnist_classifier_conv2d_1_bias:@M
?read_17_disablecopyonread_adam_v_mnist_classifier_conv2d_1_bias:@R
>read_18_disablecopyonread_adam_m_mnist_classifier_dense_kernel:
��R
>read_19_disablecopyonread_adam_v_mnist_classifier_dense_kernel:
��K
<read_20_disablecopyonread_adam_m_mnist_classifier_dense_bias:	�K
<read_21_disablecopyonread_adam_v_mnist_classifier_dense_bias:	�S
@read_22_disablecopyonread_adam_m_mnist_classifier_dense_1_kernel:	�
S
@read_23_disablecopyonread_adam_v_mnist_classifier_dense_1_kernel:	�
L
>read_24_disablecopyonread_adam_m_mnist_classifier_dense_1_bias:
L
>read_25_disablecopyonread_adam_v_mnist_classifier_dense_1_bias:
+
!read_26_disablecopyonread_total_1: +
!read_27_disablecopyonread_count_1: )
read_28_disablecopyonread_total: )
read_29_disablecopyonread_count: 
savev2_const
identity_61��MergeV2Checkpoints�Read/DisableCopyOnRead�Read/ReadVariableOp�Read_1/DisableCopyOnRead�Read_1/ReadVariableOp�Read_10/DisableCopyOnRead�Read_10/ReadVariableOp�Read_11/DisableCopyOnRead�Read_11/ReadVariableOp�Read_12/DisableCopyOnRead�Read_12/ReadVariableOp�Read_13/DisableCopyOnRead�Read_13/ReadVariableOp�Read_14/DisableCopyOnRead�Read_14/ReadVariableOp�Read_15/DisableCopyOnRead�Read_15/ReadVariableOp�Read_16/DisableCopyOnRead�Read_16/ReadVariableOp�Read_17/DisableCopyOnRead�Read_17/ReadVariableOp�Read_18/DisableCopyOnRead�Read_18/ReadVariableOp�Read_19/DisableCopyOnRead�Read_19/ReadVariableOp�Read_2/DisableCopyOnRead�Read_2/ReadVariableOp�Read_20/DisableCopyOnRead�Read_20/ReadVariableOp�Read_21/DisableCopyOnRead�Read_21/ReadVariableOp�Read_22/DisableCopyOnRead�Read_22/ReadVariableOp�Read_23/DisableCopyOnRead�Read_23/ReadVariableOp�Read_24/DisableCopyOnRead�Read_24/ReadVariableOp�Read_25/DisableCopyOnRead�Read_25/ReadVariableOp�Read_26/DisableCopyOnRead�Read_26/ReadVariableOp�Read_27/DisableCopyOnRead�Read_27/ReadVariableOp�Read_28/DisableCopyOnRead�Read_28/ReadVariableOp�Read_29/DisableCopyOnRead�Read_29/ReadVariableOp�Read_3/DisableCopyOnRead�Read_3/ReadVariableOp�Read_4/DisableCopyOnRead�Read_4/ReadVariableOp�Read_5/DisableCopyOnRead�Read_5/ReadVariableOp�Read_6/DisableCopyOnRead�Read_6/ReadVariableOp�Read_7/DisableCopyOnRead�Read_7/ReadVariableOp�Read_8/DisableCopyOnRead�Read_8/ReadVariableOp�Read_9/DisableCopyOnRead�Read_9/ReadVariableOpw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: �
Read/DisableCopyOnReadDisableCopyOnRead5read_disablecopyonread_mnist_classifier_conv2d_kernel"/device:CPU:0*
_output_shapes
 �
Read/ReadVariableOpReadVariableOp5read_disablecopyonread_mnist_classifier_conv2d_kernel^Read/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
: *
dtype0q
IdentityIdentityRead/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
: i

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0*&
_output_shapes
: �
Read_1/DisableCopyOnReadDisableCopyOnRead5read_1_disablecopyonread_mnist_classifier_conv2d_bias"/device:CPU:0*
_output_shapes
 �
Read_1/ReadVariableOpReadVariableOp5read_1_disablecopyonread_mnist_classifier_conv2d_bias^Read_1/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0i

Identity_2IdentityRead_1/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_2/DisableCopyOnReadDisableCopyOnRead9read_2_disablecopyonread_mnist_classifier_conv2d_1_kernel"/device:CPU:0*
_output_shapes
 �
Read_2/ReadVariableOpReadVariableOp9read_2_disablecopyonread_mnist_classifier_conv2d_1_kernel^Read_2/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
: @*
dtype0u

Identity_4IdentityRead_2/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
: @k

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0*&
_output_shapes
: @�
Read_3/DisableCopyOnReadDisableCopyOnRead7read_3_disablecopyonread_mnist_classifier_conv2d_1_bias"/device:CPU:0*
_output_shapes
 �
Read_3/ReadVariableOpReadVariableOp7read_3_disablecopyonread_mnist_classifier_conv2d_1_bias^Read_3/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0i

Identity_6IdentityRead_3/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@_

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_4/DisableCopyOnReadDisableCopyOnRead6read_4_disablecopyonread_mnist_classifier_dense_kernel"/device:CPU:0*
_output_shapes
 �
Read_4/ReadVariableOpReadVariableOp6read_4_disablecopyonread_mnist_classifier_dense_kernel^Read_4/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0o

Identity_8IdentityRead_4/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��e

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0* 
_output_shapes
:
���
Read_5/DisableCopyOnReadDisableCopyOnRead4read_5_disablecopyonread_mnist_classifier_dense_bias"/device:CPU:0*
_output_shapes
 �
Read_5/ReadVariableOpReadVariableOp4read_5_disablecopyonread_mnist_classifier_dense_bias^Read_5/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0k
Identity_10IdentityRead_5/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_6/DisableCopyOnReadDisableCopyOnRead8read_6_disablecopyonread_mnist_classifier_dense_1_kernel"/device:CPU:0*
_output_shapes
 �
Read_6/ReadVariableOpReadVariableOp8read_6_disablecopyonread_mnist_classifier_dense_1_kernel^Read_6/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�
*
dtype0o
Identity_12IdentityRead_6/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�
f
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0*
_output_shapes
:	�
�
Read_7/DisableCopyOnReadDisableCopyOnRead6read_7_disablecopyonread_mnist_classifier_dense_1_bias"/device:CPU:0*
_output_shapes
 �
Read_7/ReadVariableOpReadVariableOp6read_7_disablecopyonread_mnist_classifier_dense_1_bias^Read_7/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:
*
dtype0j
Identity_14IdentityRead_7/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:
a
Identity_15IdentityIdentity_14:output:0"/device:CPU:0*
T0*
_output_shapes
:
v
Read_8/DisableCopyOnReadDisableCopyOnRead"read_8_disablecopyonread_iteration"/device:CPU:0*
_output_shapes
 �
Read_8/ReadVariableOpReadVariableOp"read_8_disablecopyonread_iteration^Read_8/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0	f
Identity_16IdentityRead_8/ReadVariableOp:value:0"/device:CPU:0*
T0	*
_output_shapes
: ]
Identity_17IdentityIdentity_16:output:0"/device:CPU:0*
T0	*
_output_shapes
: z
Read_9/DisableCopyOnReadDisableCopyOnRead&read_9_disablecopyonread_learning_rate"/device:CPU:0*
_output_shapes
 �
Read_9/ReadVariableOpReadVariableOp&read_9_disablecopyonread_learning_rate^Read_9/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0f
Identity_18IdentityRead_9/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_19IdentityIdentity_18:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_10/DisableCopyOnReadDisableCopyOnRead?read_10_disablecopyonread_adam_m_mnist_classifier_conv2d_kernel"/device:CPU:0*
_output_shapes
 �
Read_10/ReadVariableOpReadVariableOp?read_10_disablecopyonread_adam_m_mnist_classifier_conv2d_kernel^Read_10/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
: *
dtype0w
Identity_20IdentityRead_10/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
: m
Identity_21IdentityIdentity_20:output:0"/device:CPU:0*
T0*&
_output_shapes
: �
Read_11/DisableCopyOnReadDisableCopyOnRead?read_11_disablecopyonread_adam_v_mnist_classifier_conv2d_kernel"/device:CPU:0*
_output_shapes
 �
Read_11/ReadVariableOpReadVariableOp?read_11_disablecopyonread_adam_v_mnist_classifier_conv2d_kernel^Read_11/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
: *
dtype0w
Identity_22IdentityRead_11/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
: m
Identity_23IdentityIdentity_22:output:0"/device:CPU:0*
T0*&
_output_shapes
: �
Read_12/DisableCopyOnReadDisableCopyOnRead=read_12_disablecopyonread_adam_m_mnist_classifier_conv2d_bias"/device:CPU:0*
_output_shapes
 �
Read_12/ReadVariableOpReadVariableOp=read_12_disablecopyonread_adam_m_mnist_classifier_conv2d_bias^Read_12/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_24IdentityRead_12/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_25IdentityIdentity_24:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_13/DisableCopyOnReadDisableCopyOnRead=read_13_disablecopyonread_adam_v_mnist_classifier_conv2d_bias"/device:CPU:0*
_output_shapes
 �
Read_13/ReadVariableOpReadVariableOp=read_13_disablecopyonread_adam_v_mnist_classifier_conv2d_bias^Read_13/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_26IdentityRead_13/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_27IdentityIdentity_26:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_14/DisableCopyOnReadDisableCopyOnReadAread_14_disablecopyonread_adam_m_mnist_classifier_conv2d_1_kernel"/device:CPU:0*
_output_shapes
 �
Read_14/ReadVariableOpReadVariableOpAread_14_disablecopyonread_adam_m_mnist_classifier_conv2d_1_kernel^Read_14/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
: @*
dtype0w
Identity_28IdentityRead_14/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
: @m
Identity_29IdentityIdentity_28:output:0"/device:CPU:0*
T0*&
_output_shapes
: @�
Read_15/DisableCopyOnReadDisableCopyOnReadAread_15_disablecopyonread_adam_v_mnist_classifier_conv2d_1_kernel"/device:CPU:0*
_output_shapes
 �
Read_15/ReadVariableOpReadVariableOpAread_15_disablecopyonread_adam_v_mnist_classifier_conv2d_1_kernel^Read_15/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
: @*
dtype0w
Identity_30IdentityRead_15/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
: @m
Identity_31IdentityIdentity_30:output:0"/device:CPU:0*
T0*&
_output_shapes
: @�
Read_16/DisableCopyOnReadDisableCopyOnRead?read_16_disablecopyonread_adam_m_mnist_classifier_conv2d_1_bias"/device:CPU:0*
_output_shapes
 �
Read_16/ReadVariableOpReadVariableOp?read_16_disablecopyonread_adam_m_mnist_classifier_conv2d_1_bias^Read_16/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_32IdentityRead_16/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_33IdentityIdentity_32:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_17/DisableCopyOnReadDisableCopyOnRead?read_17_disablecopyonread_adam_v_mnist_classifier_conv2d_1_bias"/device:CPU:0*
_output_shapes
 �
Read_17/ReadVariableOpReadVariableOp?read_17_disablecopyonread_adam_v_mnist_classifier_conv2d_1_bias^Read_17/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_34IdentityRead_17/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_35IdentityIdentity_34:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_18/DisableCopyOnReadDisableCopyOnRead>read_18_disablecopyonread_adam_m_mnist_classifier_dense_kernel"/device:CPU:0*
_output_shapes
 �
Read_18/ReadVariableOpReadVariableOp>read_18_disablecopyonread_adam_m_mnist_classifier_dense_kernel^Read_18/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0q
Identity_36IdentityRead_18/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��g
Identity_37IdentityIdentity_36:output:0"/device:CPU:0*
T0* 
_output_shapes
:
���
Read_19/DisableCopyOnReadDisableCopyOnRead>read_19_disablecopyonread_adam_v_mnist_classifier_dense_kernel"/device:CPU:0*
_output_shapes
 �
Read_19/ReadVariableOpReadVariableOp>read_19_disablecopyonread_adam_v_mnist_classifier_dense_kernel^Read_19/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0q
Identity_38IdentityRead_19/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��g
Identity_39IdentityIdentity_38:output:0"/device:CPU:0*
T0* 
_output_shapes
:
���
Read_20/DisableCopyOnReadDisableCopyOnRead<read_20_disablecopyonread_adam_m_mnist_classifier_dense_bias"/device:CPU:0*
_output_shapes
 �
Read_20/ReadVariableOpReadVariableOp<read_20_disablecopyonread_adam_m_mnist_classifier_dense_bias^Read_20/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_40IdentityRead_20/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_41IdentityIdentity_40:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_21/DisableCopyOnReadDisableCopyOnRead<read_21_disablecopyonread_adam_v_mnist_classifier_dense_bias"/device:CPU:0*
_output_shapes
 �
Read_21/ReadVariableOpReadVariableOp<read_21_disablecopyonread_adam_v_mnist_classifier_dense_bias^Read_21/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_42IdentityRead_21/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_43IdentityIdentity_42:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_22/DisableCopyOnReadDisableCopyOnRead@read_22_disablecopyonread_adam_m_mnist_classifier_dense_1_kernel"/device:CPU:0*
_output_shapes
 �
Read_22/ReadVariableOpReadVariableOp@read_22_disablecopyonread_adam_m_mnist_classifier_dense_1_kernel^Read_22/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�
*
dtype0p
Identity_44IdentityRead_22/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�
f
Identity_45IdentityIdentity_44:output:0"/device:CPU:0*
T0*
_output_shapes
:	�
�
Read_23/DisableCopyOnReadDisableCopyOnRead@read_23_disablecopyonread_adam_v_mnist_classifier_dense_1_kernel"/device:CPU:0*
_output_shapes
 �
Read_23/ReadVariableOpReadVariableOp@read_23_disablecopyonread_adam_v_mnist_classifier_dense_1_kernel^Read_23/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�
*
dtype0p
Identity_46IdentityRead_23/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�
f
Identity_47IdentityIdentity_46:output:0"/device:CPU:0*
T0*
_output_shapes
:	�
�
Read_24/DisableCopyOnReadDisableCopyOnRead>read_24_disablecopyonread_adam_m_mnist_classifier_dense_1_bias"/device:CPU:0*
_output_shapes
 �
Read_24/ReadVariableOpReadVariableOp>read_24_disablecopyonread_adam_m_mnist_classifier_dense_1_bias^Read_24/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:
*
dtype0k
Identity_48IdentityRead_24/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:
a
Identity_49IdentityIdentity_48:output:0"/device:CPU:0*
T0*
_output_shapes
:
�
Read_25/DisableCopyOnReadDisableCopyOnRead>read_25_disablecopyonread_adam_v_mnist_classifier_dense_1_bias"/device:CPU:0*
_output_shapes
 �
Read_25/ReadVariableOpReadVariableOp>read_25_disablecopyonread_adam_v_mnist_classifier_dense_1_bias^Read_25/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:
*
dtype0k
Identity_50IdentityRead_25/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:
a
Identity_51IdentityIdentity_50:output:0"/device:CPU:0*
T0*
_output_shapes
:
v
Read_26/DisableCopyOnReadDisableCopyOnRead!read_26_disablecopyonread_total_1"/device:CPU:0*
_output_shapes
 �
Read_26/ReadVariableOpReadVariableOp!read_26_disablecopyonread_total_1^Read_26/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_52IdentityRead_26/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_53IdentityIdentity_52:output:0"/device:CPU:0*
T0*
_output_shapes
: v
Read_27/DisableCopyOnReadDisableCopyOnRead!read_27_disablecopyonread_count_1"/device:CPU:0*
_output_shapes
 �
Read_27/ReadVariableOpReadVariableOp!read_27_disablecopyonread_count_1^Read_27/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_54IdentityRead_27/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_55IdentityIdentity_54:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_28/DisableCopyOnReadDisableCopyOnReadread_28_disablecopyonread_total"/device:CPU:0*
_output_shapes
 �
Read_28/ReadVariableOpReadVariableOpread_28_disablecopyonread_total^Read_28/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_56IdentityRead_28/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_57IdentityIdentity_56:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_29/DisableCopyOnReadDisableCopyOnReadread_29_disablecopyonread_count"/device:CPU:0*
_output_shapes
 �
Read_29/ReadVariableOpReadVariableOpread_29_disablecopyonread_count^Read_29/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_58IdentityRead_29/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_59IdentityIdentity_58:output:0"/device:CPU:0*
T0*
_output_shapes
: �
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Q
valueHBFB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0Identity_15:output:0Identity_17:output:0Identity_19:output:0Identity_21:output:0Identity_23:output:0Identity_25:output:0Identity_27:output:0Identity_29:output:0Identity_31:output:0Identity_33:output:0Identity_35:output:0Identity_37:output:0Identity_39:output:0Identity_41:output:0Identity_43:output:0Identity_45:output:0Identity_47:output:0Identity_49:output:0Identity_51:output:0Identity_53:output:0Identity_55:output:0Identity_57:output:0Identity_59:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *-
dtypes#
!2	�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 i
Identity_60Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: U
Identity_61IdentityIdentity_60:output:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_10/DisableCopyOnRead^Read_10/ReadVariableOp^Read_11/DisableCopyOnRead^Read_11/ReadVariableOp^Read_12/DisableCopyOnRead^Read_12/ReadVariableOp^Read_13/DisableCopyOnRead^Read_13/ReadVariableOp^Read_14/DisableCopyOnRead^Read_14/ReadVariableOp^Read_15/DisableCopyOnRead^Read_15/ReadVariableOp^Read_16/DisableCopyOnRead^Read_16/ReadVariableOp^Read_17/DisableCopyOnRead^Read_17/ReadVariableOp^Read_18/DisableCopyOnRead^Read_18/ReadVariableOp^Read_19/DisableCopyOnRead^Read_19/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_20/DisableCopyOnRead^Read_20/ReadVariableOp^Read_21/DisableCopyOnRead^Read_21/ReadVariableOp^Read_22/DisableCopyOnRead^Read_22/ReadVariableOp^Read_23/DisableCopyOnRead^Read_23/ReadVariableOp^Read_24/DisableCopyOnRead^Read_24/ReadVariableOp^Read_25/DisableCopyOnRead^Read_25/ReadVariableOp^Read_26/DisableCopyOnRead^Read_26/ReadVariableOp^Read_27/DisableCopyOnRead^Read_27/ReadVariableOp^Read_28/DisableCopyOnRead^Read_28/ReadVariableOp^Read_29/DisableCopyOnRead^Read_29/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp^Read_6/DisableCopyOnRead^Read_6/ReadVariableOp^Read_7/DisableCopyOnRead^Read_7/ReadVariableOp^Read_8/DisableCopyOnRead^Read_8/ReadVariableOp^Read_9/DisableCopyOnRead^Read_9/ReadVariableOp*
_output_shapes
 "#
identity_61Identity_61:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints20
Read/DisableCopyOnReadRead/DisableCopyOnRead2*
Read/ReadVariableOpRead/ReadVariableOp24
Read_1/DisableCopyOnReadRead_1/DisableCopyOnRead2.
Read_1/ReadVariableOpRead_1/ReadVariableOp26
Read_10/DisableCopyOnReadRead_10/DisableCopyOnRead20
Read_10/ReadVariableOpRead_10/ReadVariableOp26
Read_11/DisableCopyOnReadRead_11/DisableCopyOnRead20
Read_11/ReadVariableOpRead_11/ReadVariableOp26
Read_12/DisableCopyOnReadRead_12/DisableCopyOnRead20
Read_12/ReadVariableOpRead_12/ReadVariableOp26
Read_13/DisableCopyOnReadRead_13/DisableCopyOnRead20
Read_13/ReadVariableOpRead_13/ReadVariableOp26
Read_14/DisableCopyOnReadRead_14/DisableCopyOnRead20
Read_14/ReadVariableOpRead_14/ReadVariableOp26
Read_15/DisableCopyOnReadRead_15/DisableCopyOnRead20
Read_15/ReadVariableOpRead_15/ReadVariableOp26
Read_16/DisableCopyOnReadRead_16/DisableCopyOnRead20
Read_16/ReadVariableOpRead_16/ReadVariableOp26
Read_17/DisableCopyOnReadRead_17/DisableCopyOnRead20
Read_17/ReadVariableOpRead_17/ReadVariableOp26
Read_18/DisableCopyOnReadRead_18/DisableCopyOnRead20
Read_18/ReadVariableOpRead_18/ReadVariableOp26
Read_19/DisableCopyOnReadRead_19/DisableCopyOnRead20
Read_19/ReadVariableOpRead_19/ReadVariableOp24
Read_2/DisableCopyOnReadRead_2/DisableCopyOnRead2.
Read_2/ReadVariableOpRead_2/ReadVariableOp26
Read_20/DisableCopyOnReadRead_20/DisableCopyOnRead20
Read_20/ReadVariableOpRead_20/ReadVariableOp26
Read_21/DisableCopyOnReadRead_21/DisableCopyOnRead20
Read_21/ReadVariableOpRead_21/ReadVariableOp26
Read_22/DisableCopyOnReadRead_22/DisableCopyOnRead20
Read_22/ReadVariableOpRead_22/ReadVariableOp26
Read_23/DisableCopyOnReadRead_23/DisableCopyOnRead20
Read_23/ReadVariableOpRead_23/ReadVariableOp26
Read_24/DisableCopyOnReadRead_24/DisableCopyOnRead20
Read_24/ReadVariableOpRead_24/ReadVariableOp26
Read_25/DisableCopyOnReadRead_25/DisableCopyOnRead20
Read_25/ReadVariableOpRead_25/ReadVariableOp26
Read_26/DisableCopyOnReadRead_26/DisableCopyOnRead20
Read_26/ReadVariableOpRead_26/ReadVariableOp26
Read_27/DisableCopyOnReadRead_27/DisableCopyOnRead20
Read_27/ReadVariableOpRead_27/ReadVariableOp26
Read_28/DisableCopyOnReadRead_28/DisableCopyOnRead20
Read_28/ReadVariableOpRead_28/ReadVariableOp26
Read_29/DisableCopyOnReadRead_29/DisableCopyOnRead20
Read_29/ReadVariableOpRead_29/ReadVariableOp24
Read_3/DisableCopyOnReadRead_3/DisableCopyOnRead2.
Read_3/ReadVariableOpRead_3/ReadVariableOp24
Read_4/DisableCopyOnReadRead_4/DisableCopyOnRead2.
Read_4/ReadVariableOpRead_4/ReadVariableOp24
Read_5/DisableCopyOnReadRead_5/DisableCopyOnRead2.
Read_5/ReadVariableOpRead_5/ReadVariableOp24
Read_6/DisableCopyOnReadRead_6/DisableCopyOnRead2.
Read_6/ReadVariableOpRead_6/ReadVariableOp24
Read_7/DisableCopyOnReadRead_7/DisableCopyOnRead2.
Read_7/ReadVariableOpRead_7/ReadVariableOp24
Read_8/DisableCopyOnReadRead_8/DisableCopyOnRead2.
Read_8/ReadVariableOpRead_8/ReadVariableOp24
Read_9/DisableCopyOnReadRead_9/DisableCopyOnRead2.
Read_9/ReadVariableOpRead_9/ReadVariableOp:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:>:
8
_user_specified_name mnist_classifier/conv2d/kernel:<8
6
_user_specified_namemnist_classifier/conv2d/bias:@<
:
_user_specified_name" mnist_classifier/conv2d_1/kernel:>:
8
_user_specified_name mnist_classifier/conv2d_1/bias:=9
7
_user_specified_namemnist_classifier/dense/kernel:;7
5
_user_specified_namemnist_classifier/dense/bias:?;
9
_user_specified_name!mnist_classifier/dense_1/kernel:=9
7
_user_specified_namemnist_classifier/dense_1/bias:)	%
#
_user_specified_name	iteration:-
)
'
_user_specified_namelearning_rate:EA
?
_user_specified_name'%Adam/m/mnist_classifier/conv2d/kernel:EA
?
_user_specified_name'%Adam/v/mnist_classifier/conv2d/kernel:C?
=
_user_specified_name%#Adam/m/mnist_classifier/conv2d/bias:C?
=
_user_specified_name%#Adam/v/mnist_classifier/conv2d/bias:GC
A
_user_specified_name)'Adam/m/mnist_classifier/conv2d_1/kernel:GC
A
_user_specified_name)'Adam/v/mnist_classifier/conv2d_1/kernel:EA
?
_user_specified_name'%Adam/m/mnist_classifier/conv2d_1/bias:EA
?
_user_specified_name'%Adam/v/mnist_classifier/conv2d_1/bias:D@
>
_user_specified_name&$Adam/m/mnist_classifier/dense/kernel:D@
>
_user_specified_name&$Adam/v/mnist_classifier/dense/kernel:B>
<
_user_specified_name$"Adam/m/mnist_classifier/dense/bias:B>
<
_user_specified_name$"Adam/v/mnist_classifier/dense/bias:FB
@
_user_specified_name(&Adam/m/mnist_classifier/dense_1/kernel:FB
@
_user_specified_name(&Adam/v/mnist_classifier/dense_1/kernel:D@
>
_user_specified_name&$Adam/m/mnist_classifier/dense_1/bias:D@
>
_user_specified_name&$Adam/v/mnist_classifier/dense_1/bias:'#
!
_user_specified_name	total_1:'#
!
_user_specified_name	count_1:%!

_user_specified_nametotal:%!

_user_specified_namecount:=9

_output_shapes
: 

_user_specified_nameConst
�
K
/__inference_max_pooling2d_1_layer_call_fn_18329

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_18106�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�

�
B__inference_dense_1_layer_call_and_return_conditional_losses_18385

inputs1
matmul_readvariableop_resource:	�
-
biasadd_readvariableop_resource:

identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:���������
`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:���������
S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
�
0__inference_mnist_classifier_layer_call_fn_18209
input_1!
unknown: 
	unknown_0: #
	unknown_1: @
	unknown_2:@
	unknown_3:
��
	unknown_4:	�
	unknown_5:	�

	unknown_6:

identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_mnist_classifier_layer_call_and_return_conditional_losses_18188o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:���������: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:���������
!
_user_specified_name	input_1:%!

_user_specified_name18191:%!

_user_specified_name18193:%!

_user_specified_name18195:%!

_user_specified_name18197:%!

_user_specified_name18199:%!

_user_specified_name18201:%!

_user_specified_name18203:%!

_user_specified_name18205
�8
�
 __inference__wrapped_model_18091
input_1P
6mnist_classifier_conv2d_conv2d_readvariableop_resource: E
7mnist_classifier_conv2d_biasadd_readvariableop_resource: R
8mnist_classifier_conv2d_1_conv2d_readvariableop_resource: @G
9mnist_classifier_conv2d_1_biasadd_readvariableop_resource:@I
5mnist_classifier_dense_matmul_readvariableop_resource:
��E
6mnist_classifier_dense_biasadd_readvariableop_resource:	�J
7mnist_classifier_dense_1_matmul_readvariableop_resource:	�
F
8mnist_classifier_dense_1_biasadd_readvariableop_resource:

identity��.mnist_classifier/conv2d/BiasAdd/ReadVariableOp�-mnist_classifier/conv2d/Conv2D/ReadVariableOp�0mnist_classifier/conv2d_1/BiasAdd/ReadVariableOp�/mnist_classifier/conv2d_1/Conv2D/ReadVariableOp�-mnist_classifier/dense/BiasAdd/ReadVariableOp�,mnist_classifier/dense/MatMul/ReadVariableOp�/mnist_classifier/dense_1/BiasAdd/ReadVariableOp�.mnist_classifier/dense_1/MatMul/ReadVariableOp�
-mnist_classifier/conv2d/Conv2D/ReadVariableOpReadVariableOp6mnist_classifier_conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
mnist_classifier/conv2d/Conv2DConv2Dinput_15mnist_classifier/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
paddingSAME*
strides
�
.mnist_classifier/conv2d/BiasAdd/ReadVariableOpReadVariableOp7mnist_classifier_conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
mnist_classifier/conv2d/BiasAddBiasAdd'mnist_classifier/conv2d/Conv2D:output:06mnist_classifier/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� �
mnist_classifier/conv2d/ReluRelu(mnist_classifier/conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:��������� �
&mnist_classifier/max_pooling2d/MaxPoolMaxPool*mnist_classifier/conv2d/Relu:activations:0*/
_output_shapes
:��������� *
ksize
*
paddingVALID*
strides
�
/mnist_classifier/conv2d_1/Conv2D/ReadVariableOpReadVariableOp8mnist_classifier_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0�
 mnist_classifier/conv2d_1/Conv2DConv2D/mnist_classifier/max_pooling2d/MaxPool:output:07mnist_classifier/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
�
0mnist_classifier/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp9mnist_classifier_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
!mnist_classifier/conv2d_1/BiasAddBiasAdd)mnist_classifier/conv2d_1/Conv2D:output:08mnist_classifier/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@�
mnist_classifier/conv2d_1/ReluRelu*mnist_classifier/conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:���������@�
(mnist_classifier/max_pooling2d_1/MaxPoolMaxPool,mnist_classifier/conv2d_1/Relu:activations:0*/
_output_shapes
:���������@*
ksize
*
paddingVALID*
strides
o
mnist_classifier/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"����@  �
 mnist_classifier/flatten/ReshapeReshape1mnist_classifier/max_pooling2d_1/MaxPool:output:0'mnist_classifier/flatten/Const:output:0*
T0*(
_output_shapes
:�����������
,mnist_classifier/dense/MatMul/ReadVariableOpReadVariableOp5mnist_classifier_dense_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
mnist_classifier/dense/MatMulMatMul)mnist_classifier/flatten/Reshape:output:04mnist_classifier/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
-mnist_classifier/dense/BiasAdd/ReadVariableOpReadVariableOp6mnist_classifier_dense_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
mnist_classifier/dense/BiasAddBiasAdd'mnist_classifier/dense/MatMul:product:05mnist_classifier/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������
mnist_classifier/dense/ReluRelu'mnist_classifier/dense/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
.mnist_classifier/dense_1/MatMul/ReadVariableOpReadVariableOp7mnist_classifier_dense_1_matmul_readvariableop_resource*
_output_shapes
:	�
*
dtype0�
mnist_classifier/dense_1/MatMulMatMul)mnist_classifier/dense/Relu:activations:06mnist_classifier/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
�
/mnist_classifier/dense_1/BiasAdd/ReadVariableOpReadVariableOp8mnist_classifier_dense_1_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0�
 mnist_classifier/dense_1/BiasAddBiasAdd)mnist_classifier/dense_1/MatMul:product:07mnist_classifier/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
�
 mnist_classifier/dense_1/SoftmaxSoftmax)mnist_classifier/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:���������
y
IdentityIdentity*mnist_classifier/dense_1/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:���������
�
NoOpNoOp/^mnist_classifier/conv2d/BiasAdd/ReadVariableOp.^mnist_classifier/conv2d/Conv2D/ReadVariableOp1^mnist_classifier/conv2d_1/BiasAdd/ReadVariableOp0^mnist_classifier/conv2d_1/Conv2D/ReadVariableOp.^mnist_classifier/dense/BiasAdd/ReadVariableOp-^mnist_classifier/dense/MatMul/ReadVariableOp0^mnist_classifier/dense_1/BiasAdd/ReadVariableOp/^mnist_classifier/dense_1/MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:���������: : : : : : : : 2`
.mnist_classifier/conv2d/BiasAdd/ReadVariableOp.mnist_classifier/conv2d/BiasAdd/ReadVariableOp2^
-mnist_classifier/conv2d/Conv2D/ReadVariableOp-mnist_classifier/conv2d/Conv2D/ReadVariableOp2d
0mnist_classifier/conv2d_1/BiasAdd/ReadVariableOp0mnist_classifier/conv2d_1/BiasAdd/ReadVariableOp2b
/mnist_classifier/conv2d_1/Conv2D/ReadVariableOp/mnist_classifier/conv2d_1/Conv2D/ReadVariableOp2^
-mnist_classifier/dense/BiasAdd/ReadVariableOp-mnist_classifier/dense/BiasAdd/ReadVariableOp2\
,mnist_classifier/dense/MatMul/ReadVariableOp,mnist_classifier/dense/MatMul/ReadVariableOp2b
/mnist_classifier/dense_1/BiasAdd/ReadVariableOp/mnist_classifier/dense_1/BiasAdd/ReadVariableOp2`
.mnist_classifier/dense_1/MatMul/ReadVariableOp.mnist_classifier/dense_1/MatMul/ReadVariableOp:X T
/
_output_shapes
:���������
!
_user_specified_name	input_1:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�

�
@__inference_dense_layer_call_and_return_conditional_losses_18365

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:����������S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
d
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_18304

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
��
�
!__inference__traced_restore_18686
file_prefixI
/assignvariableop_mnist_classifier_conv2d_kernel: =
/assignvariableop_1_mnist_classifier_conv2d_bias: M
3assignvariableop_2_mnist_classifier_conv2d_1_kernel: @?
1assignvariableop_3_mnist_classifier_conv2d_1_bias:@D
0assignvariableop_4_mnist_classifier_dense_kernel:
��=
.assignvariableop_5_mnist_classifier_dense_bias:	�E
2assignvariableop_6_mnist_classifier_dense_1_kernel:	�
>
0assignvariableop_7_mnist_classifier_dense_1_bias:
&
assignvariableop_8_iteration:	 *
 assignvariableop_9_learning_rate: S
9assignvariableop_10_adam_m_mnist_classifier_conv2d_kernel: S
9assignvariableop_11_adam_v_mnist_classifier_conv2d_kernel: E
7assignvariableop_12_adam_m_mnist_classifier_conv2d_bias: E
7assignvariableop_13_adam_v_mnist_classifier_conv2d_bias: U
;assignvariableop_14_adam_m_mnist_classifier_conv2d_1_kernel: @U
;assignvariableop_15_adam_v_mnist_classifier_conv2d_1_kernel: @G
9assignvariableop_16_adam_m_mnist_classifier_conv2d_1_bias:@G
9assignvariableop_17_adam_v_mnist_classifier_conv2d_1_bias:@L
8assignvariableop_18_adam_m_mnist_classifier_dense_kernel:
��L
8assignvariableop_19_adam_v_mnist_classifier_dense_kernel:
��E
6assignvariableop_20_adam_m_mnist_classifier_dense_bias:	�E
6assignvariableop_21_adam_v_mnist_classifier_dense_bias:	�M
:assignvariableop_22_adam_m_mnist_classifier_dense_1_kernel:	�
M
:assignvariableop_23_adam_v_mnist_classifier_dense_1_kernel:	�
F
8assignvariableop_24_adam_m_mnist_classifier_dense_1_bias:
F
8assignvariableop_25_adam_v_mnist_classifier_dense_1_bias:
%
assignvariableop_26_total_1: %
assignvariableop_27_count_1: #
assignvariableop_28_total: #
assignvariableop_29_count: 
identity_31��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Q
valueHBFB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes~
|:::::::::::::::::::::::::::::::*-
dtypes#
!2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOp/assignvariableop_mnist_classifier_conv2d_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOp/assignvariableop_1_mnist_classifier_conv2d_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp3assignvariableop_2_mnist_classifier_conv2d_1_kernelIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp1assignvariableop_3_mnist_classifier_conv2d_1_biasIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp0assignvariableop_4_mnist_classifier_dense_kernelIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp.assignvariableop_5_mnist_classifier_dense_biasIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp2assignvariableop_6_mnist_classifier_dense_1_kernelIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOp0assignvariableop_7_mnist_classifier_dense_1_biasIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_8AssignVariableOpassignvariableop_8_iterationIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOp assignvariableop_9_learning_rateIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOp9assignvariableop_10_adam_m_mnist_classifier_conv2d_kernelIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOp9assignvariableop_11_adam_v_mnist_classifier_conv2d_kernelIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOp7assignvariableop_12_adam_m_mnist_classifier_conv2d_biasIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOp7assignvariableop_13_adam_v_mnist_classifier_conv2d_biasIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOp;assignvariableop_14_adam_m_mnist_classifier_conv2d_1_kernelIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOp;assignvariableop_15_adam_v_mnist_classifier_conv2d_1_kernelIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOp9assignvariableop_16_adam_m_mnist_classifier_conv2d_1_biasIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOp9assignvariableop_17_adam_v_mnist_classifier_conv2d_1_biasIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOp8assignvariableop_18_adam_m_mnist_classifier_dense_kernelIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOp8assignvariableop_19_adam_v_mnist_classifier_dense_kernelIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOp6assignvariableop_20_adam_m_mnist_classifier_dense_biasIdentity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOp6assignvariableop_21_adam_v_mnist_classifier_dense_biasIdentity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOp:assignvariableop_22_adam_m_mnist_classifier_dense_1_kernelIdentity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOp:assignvariableop_23_adam_v_mnist_classifier_dense_1_kernelIdentity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_24AssignVariableOp8assignvariableop_24_adam_m_mnist_classifier_dense_1_biasIdentity_24:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_25AssignVariableOp8assignvariableop_25_adam_v_mnist_classifier_dense_1_biasIdentity_25:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_26AssignVariableOpassignvariableop_26_total_1Identity_26:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_27AssignVariableOpassignvariableop_27_count_1Identity_27:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_28AssignVariableOpassignvariableop_28_totalIdentity_28:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_29AssignVariableOpassignvariableop_29_countIdentity_29:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 �
Identity_30Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_31IdentityIdentity_30:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
_output_shapes
 "#
identity_31Identity_31:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:>:
8
_user_specified_name mnist_classifier/conv2d/kernel:<8
6
_user_specified_namemnist_classifier/conv2d/bias:@<
:
_user_specified_name" mnist_classifier/conv2d_1/kernel:>:
8
_user_specified_name mnist_classifier/conv2d_1/bias:=9
7
_user_specified_namemnist_classifier/dense/kernel:;7
5
_user_specified_namemnist_classifier/dense/bias:?;
9
_user_specified_name!mnist_classifier/dense_1/kernel:=9
7
_user_specified_namemnist_classifier/dense_1/bias:)	%
#
_user_specified_name	iteration:-
)
'
_user_specified_namelearning_rate:EA
?
_user_specified_name'%Adam/m/mnist_classifier/conv2d/kernel:EA
?
_user_specified_name'%Adam/v/mnist_classifier/conv2d/kernel:C?
=
_user_specified_name%#Adam/m/mnist_classifier/conv2d/bias:C?
=
_user_specified_name%#Adam/v/mnist_classifier/conv2d/bias:GC
A
_user_specified_name)'Adam/m/mnist_classifier/conv2d_1/kernel:GC
A
_user_specified_name)'Adam/v/mnist_classifier/conv2d_1/kernel:EA
?
_user_specified_name'%Adam/m/mnist_classifier/conv2d_1/bias:EA
?
_user_specified_name'%Adam/v/mnist_classifier/conv2d_1/bias:D@
>
_user_specified_name&$Adam/m/mnist_classifier/dense/kernel:D@
>
_user_specified_name&$Adam/v/mnist_classifier/dense/kernel:B>
<
_user_specified_name$"Adam/m/mnist_classifier/dense/bias:B>
<
_user_specified_name$"Adam/v/mnist_classifier/dense/bias:FB
@
_user_specified_name(&Adam/m/mnist_classifier/dense_1/kernel:FB
@
_user_specified_name(&Adam/v/mnist_classifier/dense_1/kernel:D@
>
_user_specified_name&$Adam/m/mnist_classifier/dense_1/bias:D@
>
_user_specified_name&$Adam/v/mnist_classifier/dense_1/bias:'#
!
_user_specified_name	total_1:'#
!
_user_specified_name	count_1:%!

_user_specified_nametotal:%!

_user_specified_namecount
�
�
%__inference_dense_layer_call_fn_18354

inputs
unknown:
��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_18165p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs:%!

_user_specified_name18348:%!

_user_specified_name18350
�
f
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_18334

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
�
'__inference_dense_1_layer_call_fn_18374

inputs
unknown:	�

	unknown_0:

identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_18181o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs:%!

_user_specified_name18368:%!

_user_specified_name18370
�
�
&__inference_conv2d_layer_call_fn_18283

inputs!
unknown: 
	unknown_0: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_18124w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:��������� <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs:%!

_user_specified_name18277:%!

_user_specified_name18279
�
I
-__inference_max_pooling2d_layer_call_fn_18299

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_18096�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
�
C__inference_conv2d_1_layer_call_and_return_conditional_losses_18141

inputs8
conv2d_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������@i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������@S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�

�
@__inference_dense_layer_call_and_return_conditional_losses_18165

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:����������S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
�
A__inference_conv2d_layer_call_and_return_conditional_losses_18124

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:��������� i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:��������� S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�

�
B__inference_dense_1_layer_call_and_return_conditional_losses_18181

inputs1
matmul_readvariableop_resource:	�
-
biasadd_readvariableop_resource:

identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:���������
`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:���������
S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
^
B__inference_flatten_layer_call_and_return_conditional_losses_18345

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"����@  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�!
�
K__inference_mnist_classifier_layer_call_and_return_conditional_losses_18188
input_1&
conv2d_18125: 
conv2d_18127: (
conv2d_1_18142: @
conv2d_1_18144:@
dense_18166:
��
dense_18168:	� 
dense_1_18182:	�

dense_1_18184:

identity��conv2d/StatefulPartitionedCall� conv2d_1/StatefulPartitionedCall�dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�
conv2d/StatefulPartitionedCallStatefulPartitionedCallinput_1conv2d_18125conv2d_18127*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_18124�
max_pooling2d/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_18096�
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0conv2d_1_18142conv2d_1_18144*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_18141�
max_pooling2d_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_18106�
flatten/PartitionedCallPartitionedCall(max_pooling2d_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_18153�
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_18166dense_18168*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_18165�
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_18182dense_1_18184*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_18181w
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
�
NoOpNoOp^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:���������: : : : : : : : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:X T
/
_output_shapes
:���������
!
_user_specified_name	input_1:%!

_user_specified_name18125:%!

_user_specified_name18127:%!

_user_specified_name18142:%!

_user_specified_name18144:%!

_user_specified_name18166:%!

_user_specified_name18168:%!

_user_specified_name18182:%!

_user_specified_name18184
�
�
#__inference_signature_wrapper_18274
input_1!
unknown: 
	unknown_0: #
	unknown_1: @
	unknown_2:@
	unknown_3:
��
	unknown_4:	�
	unknown_5:	�

	unknown_6:

identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *)
f$R"
 __inference__wrapped_model_18091o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:���������: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:���������
!
_user_specified_name	input_1:%!

_user_specified_name18256:%!

_user_specified_name18258:%!

_user_specified_name18260:%!

_user_specified_name18262:%!

_user_specified_name18264:%!

_user_specified_name18266:%!

_user_specified_name18268:%!

_user_specified_name18270
�
�
A__inference_conv2d_layer_call_and_return_conditional_losses_18294

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:��������� i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:��������� S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
^
B__inference_flatten_layer_call_and_return_conditional_losses_18153

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"����@  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
(__inference_conv2d_1_layer_call_fn_18313

inputs!
unknown: @
	unknown_0:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_18141w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:��������� : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs:%!

_user_specified_name18307:%!

_user_specified_name18309
�
C
'__inference_flatten_layer_call_fn_18339

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_18153a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
C__inference_conv2d_1_layer_call_and_return_conditional_losses_18324

inputs8
conv2d_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������@i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������@S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
C
input_18
serving_default_input_1:0���������<
output_10
StatefulPartitionedCall:0���������
tensorflow/serving/predict:ޡ
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	conv1
		pool1
	
conv2
	pool2
flatten

dense1
outputs
	optimizer

signatures"
_tf_keras_model
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
 "
trackable_list_wrapper
�
non_trainable_variables

layers
metrics
layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
trace_02�
0__inference_mnist_classifier_layer_call_fn_18209�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 ztrace_0
�
trace_02�
K__inference_mnist_classifier_layer_call_and_return_conditional_losses_18188�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 ztrace_0
�B�
 __inference__wrapped_model_18091input_1"�
���
FullArgSpec
args� 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�
 	variables
!trainable_variables
"regularization_losses
#	keras_api
$__call__
*%&call_and_return_all_conditional_losses

kernel
bias
 &_jit_compiled_convolution_op"
_tf_keras_layer
�
'	variables
(trainable_variables
)regularization_losses
*	keras_api
+__call__
*,&call_and_return_all_conditional_losses"
_tf_keras_layer
�
-	variables
.trainable_variables
/regularization_losses
0	keras_api
1__call__
*2&call_and_return_all_conditional_losses

kernel
bias
 3_jit_compiled_convolution_op"
_tf_keras_layer
�
4	variables
5trainable_variables
6regularization_losses
7	keras_api
8__call__
*9&call_and_return_all_conditional_losses"
_tf_keras_layer
�
:	variables
;trainable_variables
<regularization_losses
=	keras_api
>__call__
*?&call_and_return_all_conditional_losses"
_tf_keras_layer
�
@	variables
Atrainable_variables
Bregularization_losses
C	keras_api
D__call__
*E&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
�
F	variables
Gtrainable_variables
Hregularization_losses
I	keras_api
J__call__
*K&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
�
L
_variables
M_iterations
N_learning_rate
O_index_dict
P
_momentums
Q_velocities
R_update_step_xla"
experimentalOptimizer
,
Sserving_default"
signature_map
8:6 2mnist_classifier/conv2d/kernel
*:( 2mnist_classifier/conv2d/bias
::8 @2 mnist_classifier/conv2d_1/kernel
,:*@2mnist_classifier/conv2d_1/bias
1:/
��2mnist_classifier/dense/kernel
*:(�2mnist_classifier/dense/bias
2:0	�
2mnist_classifier/dense_1/kernel
+:)
2mnist_classifier/dense_1/bias
 "
trackable_list_wrapper
Q
0
	1

2
3
4
5
6"
trackable_list_wrapper
.
T0
U1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
0__inference_mnist_classifier_layer_call_fn_18209input_1"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
K__inference_mnist_classifier_layer_call_and_return_conditional_losses_18188input_1"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Vnon_trainable_variables

Wlayers
Xmetrics
Ylayer_regularization_losses
Zlayer_metrics
 	variables
!trainable_variables
"regularization_losses
$__call__
*%&call_and_return_all_conditional_losses
&%"call_and_return_conditional_losses"
_generic_user_object
�
[trace_02�
&__inference_conv2d_layer_call_fn_18283�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z[trace_0
�
\trace_02�
A__inference_conv2d_layer_call_and_return_conditional_losses_18294�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z\trace_0
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
]non_trainable_variables

^layers
_metrics
`layer_regularization_losses
alayer_metrics
'	variables
(trainable_variables
)regularization_losses
+__call__
*,&call_and_return_all_conditional_losses
&,"call_and_return_conditional_losses"
_generic_user_object
�
btrace_02�
-__inference_max_pooling2d_layer_call_fn_18299�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zbtrace_0
�
ctrace_02�
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_18304�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zctrace_0
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
dnon_trainable_variables

elayers
fmetrics
glayer_regularization_losses
hlayer_metrics
-	variables
.trainable_variables
/regularization_losses
1__call__
*2&call_and_return_all_conditional_losses
&2"call_and_return_conditional_losses"
_generic_user_object
�
itrace_02�
(__inference_conv2d_1_layer_call_fn_18313�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zitrace_0
�
jtrace_02�
C__inference_conv2d_1_layer_call_and_return_conditional_losses_18324�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zjtrace_0
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
knon_trainable_variables

llayers
mmetrics
nlayer_regularization_losses
olayer_metrics
4	variables
5trainable_variables
6regularization_losses
8__call__
*9&call_and_return_all_conditional_losses
&9"call_and_return_conditional_losses"
_generic_user_object
�
ptrace_02�
/__inference_max_pooling2d_1_layer_call_fn_18329�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zptrace_0
�
qtrace_02�
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_18334�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zqtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
rnon_trainable_variables

slayers
tmetrics
ulayer_regularization_losses
vlayer_metrics
:	variables
;trainable_variables
<regularization_losses
>__call__
*?&call_and_return_all_conditional_losses
&?"call_and_return_conditional_losses"
_generic_user_object
�
wtrace_02�
'__inference_flatten_layer_call_fn_18339�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zwtrace_0
�
xtrace_02�
B__inference_flatten_layer_call_and_return_conditional_losses_18345�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zxtrace_0
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
ynon_trainable_variables

zlayers
{metrics
|layer_regularization_losses
}layer_metrics
@	variables
Atrainable_variables
Bregularization_losses
D__call__
*E&call_and_return_all_conditional_losses
&E"call_and_return_conditional_losses"
_generic_user_object
�
~trace_02�
%__inference_dense_layer_call_fn_18354�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z~trace_0
�
trace_02�
@__inference_dense_layer_call_and_return_conditional_losses_18365�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 ztrace_0
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
F	variables
Gtrainable_variables
Hregularization_losses
J__call__
*K&call_and_return_all_conditional_losses
&K"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
'__inference_dense_1_layer_call_fn_18374�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
B__inference_dense_1_layer_call_and_return_conditional_losses_18385�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
M0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16"
trackable_list_wrapper
:	 2	iteration
: 2learning_rate
 "
trackable_dict_wrapper
`
�0
�1
�2
�3
�4
�5
�6
�7"
trackable_list_wrapper
`
�0
�1
�2
�3
�4
�5
�6
�7"
trackable_list_wrapper
�2��
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
�B�
#__inference_signature_wrapper_18274input_1"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
R
�	variables
�	keras_api

�total

�count"
_tf_keras_metric
c
�	variables
�	keras_api

�total

�count
�
_fn_kwargs"
_tf_keras_metric
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
&__inference_conv2d_layer_call_fn_18283inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
A__inference_conv2d_layer_call_and_return_conditional_losses_18294inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
-__inference_max_pooling2d_layer_call_fn_18299inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_18304inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
(__inference_conv2d_1_layer_call_fn_18313inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_conv2d_1_layer_call_and_return_conditional_losses_18324inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
/__inference_max_pooling2d_1_layer_call_fn_18329inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_18334inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
'__inference_flatten_layer_call_fn_18339inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
B__inference_flatten_layer_call_and_return_conditional_losses_18345inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
%__inference_dense_layer_call_fn_18354inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
@__inference_dense_layer_call_and_return_conditional_losses_18365inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
'__inference_dense_1_layer_call_fn_18374inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
B__inference_dense_1_layer_call_and_return_conditional_losses_18385inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
=:; 2%Adam/m/mnist_classifier/conv2d/kernel
=:; 2%Adam/v/mnist_classifier/conv2d/kernel
/:- 2#Adam/m/mnist_classifier/conv2d/bias
/:- 2#Adam/v/mnist_classifier/conv2d/bias
?:= @2'Adam/m/mnist_classifier/conv2d_1/kernel
?:= @2'Adam/v/mnist_classifier/conv2d_1/kernel
1:/@2%Adam/m/mnist_classifier/conv2d_1/bias
1:/@2%Adam/v/mnist_classifier/conv2d_1/bias
6:4
��2$Adam/m/mnist_classifier/dense/kernel
6:4
��2$Adam/v/mnist_classifier/dense/kernel
/:-�2"Adam/m/mnist_classifier/dense/bias
/:-�2"Adam/v/mnist_classifier/dense/bias
7:5	�
2&Adam/m/mnist_classifier/dense_1/kernel
7:5	�
2&Adam/v/mnist_classifier/dense_1/kernel
0:.
2$Adam/m/mnist_classifier/dense_1/bias
0:.
2$Adam/v/mnist_classifier/dense_1/bias
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper�
 __inference__wrapped_model_18091y8�5
.�+
)�&
input_1���������
� "3�0
.
output_1"�
output_1���������
�
C__inference_conv2d_1_layer_call_and_return_conditional_losses_18324s7�4
-�*
(�%
inputs��������� 
� "4�1
*�'
tensor_0���������@
� �
(__inference_conv2d_1_layer_call_fn_18313h7�4
-�*
(�%
inputs��������� 
� ")�&
unknown���������@�
A__inference_conv2d_layer_call_and_return_conditional_losses_18294s7�4
-�*
(�%
inputs���������
� "4�1
*�'
tensor_0��������� 
� �
&__inference_conv2d_layer_call_fn_18283h7�4
-�*
(�%
inputs���������
� ")�&
unknown��������� �
B__inference_dense_1_layer_call_and_return_conditional_losses_18385d0�-
&�#
!�
inputs����������
� ",�)
"�
tensor_0���������

� �
'__inference_dense_1_layer_call_fn_18374Y0�-
&�#
!�
inputs����������
� "!�
unknown���������
�
@__inference_dense_layer_call_and_return_conditional_losses_18365e0�-
&�#
!�
inputs����������
� "-�*
#� 
tensor_0����������
� �
%__inference_dense_layer_call_fn_18354Z0�-
&�#
!�
inputs����������
� ""�
unknown�����������
B__inference_flatten_layer_call_and_return_conditional_losses_18345h7�4
-�*
(�%
inputs���������@
� "-�*
#� 
tensor_0����������
� �
'__inference_flatten_layer_call_fn_18339]7�4
-�*
(�%
inputs���������@
� ""�
unknown�����������
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_18334�R�O
H�E
C�@
inputs4������������������������������������
� "O�L
E�B
tensor_04������������������������������������
� �
/__inference_max_pooling2d_1_layer_call_fn_18329�R�O
H�E
C�@
inputs4������������������������������������
� "D�A
unknown4�������������������������������������
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_18304�R�O
H�E
C�@
inputs4������������������������������������
� "O�L
E�B
tensor_04������������������������������������
� �
-__inference_max_pooling2d_layer_call_fn_18299�R�O
H�E
C�@
inputs4������������������������������������
� "D�A
unknown4�������������������������������������
K__inference_mnist_classifier_layer_call_and_return_conditional_losses_18188r8�5
.�+
)�&
input_1���������
� ",�)
"�
tensor_0���������

� �
0__inference_mnist_classifier_layer_call_fn_18209g8�5
.�+
)�&
input_1���������
� "!�
unknown���������
�
#__inference_signature_wrapper_18274�C�@
� 
9�6
4
input_1)�&
input_1���������"3�0
.
output_1"�
output_1���������

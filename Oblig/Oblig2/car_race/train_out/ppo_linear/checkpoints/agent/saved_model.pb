ЄШ
тБ
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 

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
$
DisableCopyOnRead
resource
Ў
GatherV2
params"Tparams
indices"Tindices
axis"Taxis
output"Tparams"

batch_dimsint "
Tparamstype"
Tindicestype:
2	"
Taxistype:
2	
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

MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( 
Ј
Multinomial
logits"T
num_samples
output"output_dtype"
seedint "
seed2int "
Ttype:
2	" 
output_dtypetype0	:
2	
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
dtypetype
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
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
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
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
С
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
executor_typestring Ј
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

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.14.02v2.14.0-rc1-21-g4dacf3f368e8ск

ConstConst*
_output_shapes

:*
dtype0*U
valueLBJ"<  П                      ?              ?              ?

agent/policy_network/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!agent/policy_network/dense/bias

3agent/policy_network/dense/bias/Read/ReadVariableOpReadVariableOpagent/policy_network/dense/bias*
_output_shapes
:*
dtype0
 
!agent/policy_network/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
и*2
shared_name#!agent/policy_network/dense/kernel

5agent/policy_network/dense/kernel/Read/ReadVariableOpReadVariableOp!agent/policy_network/dense/kernel* 
_output_shapes
:
и*
dtype0

serving_default_input_1Placeholder*/
_output_shapes
:џџџџџџџџџ``*
dtype0*$
shape:џџџџџџџџџ``
џ
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1!agent/policy_network/dense/kernelagent/policy_network/dense/biasConst*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 **
f%R#
!__inference_signature_wrapper_213

NoOpNoOp
к
Const_1Const"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB Bџ
б
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
policy_network
	
signatures*


0
1*


0
1*
* 
А
non_trainable_variables

layers
metrics
layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

trace_0* 

trace_0* 

	capture_2* 
В
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
feature_extractor
	dense*

serving_default* 
a[
VARIABLE_VALUE!agent/policy_network/dense/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEagent/policy_network/dense/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
* 

0*
* 
* 
* 

	capture_2* 

	capture_2* 
* 


0
1*


0
1*
* 

non_trainable_variables

layers
metrics
 layer_regularization_losses
!layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

"trace_0* 

#trace_0* 

$	variables
%trainable_variables
&regularization_losses
'	keras_api
(__call__
*)&call_and_return_all_conditional_losses
*flatten* 
І
+	variables
,trainable_variables
-regularization_losses
.	keras_api
/__call__
*0&call_and_return_all_conditional_losses


kernel
bias*

	capture_2* 
* 

0
1*
* 
* 
* 
* 
* 
* 
* 
* 

1non_trainable_variables

2layers
3metrics
4layer_regularization_losses
5layer_metrics
$	variables
%trainable_variables
&regularization_losses
(__call__
*)&call_and_return_all_conditional_losses
&)"call_and_return_conditional_losses* 

6trace_0* 

7trace_0* 

8	variables
9trainable_variables
:regularization_losses
;	keras_api
<__call__
*=&call_and_return_all_conditional_losses* 


0
1*


0
1*
* 

>non_trainable_variables

?layers
@metrics
Alayer_regularization_losses
Blayer_metrics
+	variables
,trainable_variables
-regularization_losses
/__call__
*0&call_and_return_all_conditional_losses
&0"call_and_return_conditional_losses*

Ctrace_0* 

Dtrace_0* 
* 
	
*0* 
* 
* 
* 
* 
* 
* 
* 
* 

Enon_trainable_variables

Flayers
Gmetrics
Hlayer_regularization_losses
Ilayer_metrics
8	variables
9trainable_variables
:regularization_losses
<__call__
*=&call_and_return_all_conditional_losses
&="call_and_return_conditional_losses* 

Jtrace_0* 

Ktrace_0* 
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
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
р
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename!agent/policy_network/dense/kernelagent/policy_network/dense/biasConst_1*
Tin
2*
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
GPU 2J 8 *%
f R
__inference__traced_save_278
й
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename!agent/policy_network/dense/kernelagent/policy_network/dense/bias*
Tin
2*
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
GPU 2J 8 *(
f#R!
__inference__traced_restore_293ЭБ
љ	
ё
>__inference_dense_layer_call_and_return_conditional_losses_232

inputs2
matmul_readvariableop_resource:
и-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
и*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџS
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:џџџџџџџџџи: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:Q M
)
_output_shapes
:џџџџџџџџџи
 
_user_specified_nameinputs
љ	
ё
>__inference_dense_layer_call_and_return_conditional_losses_150

inputs2
matmul_readvariableop_resource:
и-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
и*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџS
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:џџџџџџџџџи: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:Q M
)
_output_shapes
:џџџџџџџџџи
 
_user_specified_nameinputs
М
Ђ
#__inference_agent_layer_call_fn_201
input_1
unknown:
и
	unknown_0:
	unknown_1
identityЂStatefulPartitionedCallр
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *G
fBR@
>__inference_agent_layer_call_and_return_conditional_losses_190o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):џџџџџџџџџ``: : :22
StatefulPartitionedCallStatefulPartitionedCall:$ 

_output_shapes

::#

_user_specified_name195:#

_user_specified_name193:X T
/
_output_shapes
:џџџџџџџџџ``
!
_user_specified_name	input_1
ћ

,__inference_policy_network_layer_call_fn_169
input_1
unknown:
и
	unknown_0:
identity	ЂStatefulPartitionedCallй
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0*
Tin
2*
Tout
2	*
_collective_manager_ids
 *#
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_policy_network_layer_call_and_return_conditional_losses_160k
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0	*#
_output_shapes
:џџџџџџџџџ<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ``: : 22
StatefulPartitionedCallStatefulPartitionedCall:#

_user_specified_name165:#

_user_specified_name163:X T
/
_output_shapes
:џџџџџџџџџ``
!
_user_specified_name	input_1
б
ц
>__inference_agent_layer_call_and_return_conditional_losses_190
input_1&
policy_network_181:
и 
policy_network_183:
gatherv2_params
identityЂ&policy_network/StatefulPartitionedCallќ
&policy_network/StatefulPartitionedCallStatefulPartitionedCallinput_1policy_network_181policy_network_183*
Tin
2*
Tout
2	*
_collective_manager_ids
 *#
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_policy_network_layer_call_and_return_conditional_losses_160O
GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : У
GatherV2GatherV2gatherv2_params/policy_network/StatefulPartitionedCall:output:0GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*'
_output_shapes
:џџџџџџџџџ`
IdentityIdentityGatherV2:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџK
NoOpNoOp'^policy_network/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):џџџџџџџџџ``: : :2P
&policy_network/StatefulPartitionedCall&policy_network/StatefulPartitionedCall:$ 

_output_shapes

::#

_user_specified_name183:#

_user_specified_name181:X T
/
_output_shapes
:џџџџџџџџџ``
!
_user_specified_name	input_1
Ф
\
@__inference_flatten_layer_call_and_return_conditional_losses_124

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ l  ^
ReshapeReshapeinputsConst:output:0*
T0*)
_output_shapes
:џџџџџџџџџиZ
IdentityIdentityReshape:output:0*
T0*)
_output_shapes
:џџџџџџџџџи"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ``:W S
/
_output_shapes
:џџџџџџџџџ``
 
_user_specified_nameinputs
Љ
A
%__inference_flatten_layer_call_fn_237

inputs
identity­
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:џџџџџџџџџи* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_flatten_layer_call_and_return_conditional_losses_124b
IdentityIdentityPartitionedCall:output:0*
T0*)
_output_shapes
:џџџџџџџџџи"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ``:W S
/
_output_shapes
:џџџџџџџџџ``
 
_user_specified_nameinputs
Р
L
/__inference_feature_extractor_layer_call_fn_132
input_1
identityИ
PartitionedCallPartitionedCallinput_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:џџџџџџџџџи* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_feature_extractor_layer_call_and_return_conditional_losses_127b
IdentityIdentityPartitionedCall:output:0*
T0*)
_output_shapes
:џџџџџџџџџи"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ``:X T
/
_output_shapes
:џџџџџџџџџ``
!
_user_specified_name	input_1
Ф
\
@__inference_flatten_layer_call_and_return_conditional_losses_243

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ l  ^
ReshapeReshapeinputsConst:output:0*
T0*)
_output_shapes
:џџџџџџџџџиZ
IdentityIdentityReshape:output:0*
T0*)
_output_shapes
:џџџџџџџџџи"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ``:W S
/
_output_shapes
:џџџџџџџџџ``
 
_user_specified_nameinputs
т

#__inference_dense_layer_call_fn_222

inputs
unknown:
и
	unknown_0:
identityЂStatefulPartitionedCallг
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *G
fBR@
>__inference_dense_layer_call_and_return_conditional_losses_150o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:џџџџџџџџџи: : 22
StatefulPartitionedCallStatefulPartitionedCall:#

_user_specified_name218:#

_user_specified_name216:Q M
)
_output_shapes
:џџџџџџџџџи
 
_user_specified_nameinputs
с
g
J__inference_feature_extractor_layer_call_and_return_conditional_losses_127
input_1
identityЖ
flatten/PartitionedCallPartitionedCallinput_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:џџџџџџџџџи* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_flatten_layer_call_and_return_conditional_losses_124j
IdentityIdentity flatten/PartitionedCall:output:0*
T0*)
_output_shapes
:џџџџџџџџџи"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ``:X T
/
_output_shapes
:џџџџџџџџџ``
!
_user_specified_name	input_1
№
й
__inference__wrapped_model_116
input_1M
9agent_policy_network_dense_matmul_readvariableop_resource:
иH
:agent_policy_network_dense_biasadd_readvariableop_resource:
agent_gatherv2_params
identityЂ1agent/policy_network/dense/BiasAdd/ReadVariableOpЂ0agent/policy_network/dense/MatMul/ReadVariableOp
4agent/policy_network/feature_extractor/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ l  Н
6agent/policy_network/feature_extractor/flatten/ReshapeReshapeinput_1=agent/policy_network/feature_extractor/flatten/Const:output:0*
T0*)
_output_shapes
:џџџџџџџџџиЌ
0agent/policy_network/dense/MatMul/ReadVariableOpReadVariableOp9agent_policy_network_dense_matmul_readvariableop_resource* 
_output_shapes
:
и*
dtype0и
!agent/policy_network/dense/MatMulMatMul?agent/policy_network/feature_extractor/flatten/Reshape:output:08agent/policy_network/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџЈ
1agent/policy_network/dense/BiasAdd/ReadVariableOpReadVariableOp:agent_policy_network_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ч
"agent/policy_network/dense/BiasAddBiasAdd+agent/policy_network/dense/MatMul:product:09agent/policy_network/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџz
8agent/policy_network/categorical/Multinomial/num_samplesConst*
_output_shapes
: *
dtype0*
value	B :н
,agent/policy_network/categorical/MultinomialMultinomial+agent/policy_network/dense/BiasAdd:output:0Aagent/policy_network/categorical/Multinomial/num_samples:output:0*
T0*'
_output_shapes
:џџџџџџџџџЌ
agent/policy_network/SqueezeSqueeze5agent/policy_network/categorical/Multinomial:output:0*
T0	*#
_output_shapes
:џџџџџџџџџ*
squeeze_dims

џџџџџџџџџU
agent/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : Ы
agent/GatherV2GatherV2agent_gatherv2_params%agent/policy_network/Squeeze:output:0agent/GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*'
_output_shapes
:џџџџџџџџџf
IdentityIdentityagent/GatherV2:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
NoOpNoOp2^agent/policy_network/dense/BiasAdd/ReadVariableOp1^agent/policy_network/dense/MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):џџџџџџџџџ``: : :2f
1agent/policy_network/dense/BiasAdd/ReadVariableOp1agent/policy_network/dense/BiasAdd/ReadVariableOp2d
0agent/policy_network/dense/MatMul/ReadVariableOp0agent/policy_network/dense/MatMul/ReadVariableOp:$ 

_output_shapes

::($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:X T
/
_output_shapes
:џџџџџџџџџ``
!
_user_specified_name	input_1

 
!__inference_signature_wrapper_213
input_1
unknown:
и
	unknown_0:
	unknown_1
identityЂStatefulPartitionedCallР
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *'
f"R 
__inference__wrapped_model_116o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):џџџџџџџџџ``: : :22
StatefulPartitionedCallStatefulPartitionedCall:$ 

_output_shapes

::#

_user_specified_name207:#

_user_specified_name205:X T
/
_output_shapes
:џџџџџџџџџ``
!
_user_specified_name	input_1
А
П
G__inference_policy_network_layer_call_and_return_conditional_losses_160
input_1
	dense_151:
и
	dense_153:
identity	Ђdense/StatefulPartitionedCallЪ
!feature_extractor/PartitionedCallPartitionedCallinput_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:џџџџџџџџџи* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_feature_extractor_layer_call_and_return_conditional_losses_127џ
dense/StatefulPartitionedCallStatefulPartitionedCall*feature_extractor/PartitionedCall:output:0	dense_151	dense_153*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *G
fBR@
>__inference_dense_layer_call_and_return_conditional_losses_150e
#categorical/Multinomial/num_samplesConst*
_output_shapes
: *
dtype0*
value	B :Ў
categorical/MultinomialMultinomial&dense/StatefulPartitionedCall:output:0,categorical/Multinomial/num_samples:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
SqueezeSqueeze categorical/Multinomial:output:0*
T0	*#
_output_shapes
:џџџџџџџџџ*
squeeze_dims

џџџџџџџџџ[
IdentityIdentitySqueeze:output:0^NoOp*
T0	*#
_output_shapes
:џџџџџџџџџB
NoOpNoOp^dense/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ``: : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:#

_user_specified_name153:#

_user_specified_name151:X T
/
_output_shapes
:џџџџџџџџџ``
!
_user_specified_name	input_1
п 
у
__inference__traced_save_278
file_prefixL
8read_disablecopyonread_agent_policy_network_dense_kernel:
иF
8read_1_disablecopyonread_agent_policy_network_dense_bias:
savev2_const_1

identity_5ЂMergeV2CheckpointsЂRead/DisableCopyOnReadЂRead/ReadVariableOpЂRead_1/DisableCopyOnReadЂRead_1/ReadVariableOpw
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
_temp/part
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
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 
Read/DisableCopyOnReadDisableCopyOnRead8read_disablecopyonread_agent_policy_network_dense_kernel"/device:CPU:0*
_output_shapes
 Ж
Read/ReadVariableOpReadVariableOp8read_disablecopyonread_agent_policy_network_dense_kernel^Read/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
и*
dtype0k
IdentityIdentityRead/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
иc

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0* 
_output_shapes
:
и
Read_1/DisableCopyOnReadDisableCopyOnRead8read_1_disablecopyonread_agent_policy_network_dense_bias"/device:CPU:0*
_output_shapes
 Д
Read_1/ReadVariableOpReadVariableOp8read_1_disablecopyonread_agent_policy_network_dense_bias^Read_1/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0i

Identity_2IdentityRead_1/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:_

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*
_output_shapes
:и
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valuexBvB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHs
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B 
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0savev2_const_1"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtypes
2
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:Г
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 h

Identity_4Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: S

Identity_5IdentityIdentity_4:output:0^NoOp*
T0*
_output_shapes
: 
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp*
_output_shapes
 "!

identity_5Identity_5:output:0*(
_construction_contextkEagerRuntime*
_input_shapes

: : : : 2(
MergeV2CheckpointsMergeV2Checkpoints20
Read/DisableCopyOnReadRead/DisableCopyOnRead2*
Read/ReadVariableOpRead/ReadVariableOp24
Read_1/DisableCopyOnReadRead_1/DisableCopyOnRead2.
Read_1/ReadVariableOpRead_1/ReadVariableOp:?;

_output_shapes
: 
!
_user_specified_name	Const_1:?;
9
_user_specified_name!agent/policy_network/dense/bias:A=
;
_user_specified_name#!agent/policy_network/dense/kernel:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
ж
ї
__inference__traced_restore_293
file_prefixF
2assignvariableop_agent_policy_network_dense_kernel:
и@
2assignvariableop_1_agent_policy_network_dense_bias:

identity_3ЂAssignVariableOpЂAssignVariableOp_1л
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valuexBvB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHv
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B ­
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0* 
_output_shapes
:::*
dtypes
2[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:Х
AssignVariableOpAssignVariableOp2assignvariableop_agent_policy_network_dense_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:Щ
AssignVariableOp_1AssignVariableOp2assignvariableop_1_agent_policy_network_dense_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 

Identity_2Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^NoOp"/device:CPU:0*
T0*
_output_shapes
: U

Identity_3IdentityIdentity_2:output:0^NoOp_1*
T0*
_output_shapes
: L
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1*
_output_shapes
 "!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : 2(
AssignVariableOp_1AssignVariableOp_12$
AssignVariableOpAssignVariableOp:?;
9
_user_specified_name!agent/policy_network/dense/bias:A=
;
_user_specified_name#!agent/policy_network/dense/kernel:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix"ЇL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Г
serving_default
C
input_18
serving_default_input_1:0џџџџџџџџџ``<
output_10
StatefulPartitionedCall:0џџџџџџџџџtensorflow/serving/predict:ђX
ц
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
policy_network
	
signatures"
_tf_keras_model
.

0
1"
trackable_list_wrapper
.

0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
Ъ
non_trainable_variables

layers
metrics
layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
т
trace_02Х
#__inference_agent_layer_call_fn_201
В
FullArgSpec
args
jobservation
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0
§
trace_02р
>__inference_agent_layer_call_and_return_conditional_losses_190
В
FullArgSpec
args
jobservation
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0
ч
	capture_2BЦ
__inference__wrapped_model_116input_1"
В
FullArgSpec
args

jargs_0
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 z	capture_2
Ч
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
feature_extractor
	dense"
_tf_keras_model
,
serving_default"
signature_map
5:3
и2!agent/policy_network/dense/kernel
-:+2agent/policy_network/dense/bias
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ё
	capture_2Bа
#__inference_agent_layer_call_fn_201input_1"
В
FullArgSpec
args
jobservation
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 z	capture_2

	capture_2Bы
>__inference_agent_layer_call_and_return_conditional_losses_190input_1"
В
FullArgSpec
args
jobservation
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 z	capture_2
J
Constjtf.TrackableConstant
.

0
1"
trackable_list_wrapper
.

0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
non_trainable_variables

layers
metrics
 layer_regularization_losses
!layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
с
"trace_02Ф
,__inference_policy_network_layer_call_fn_169
В
FullArgSpec
args
jx
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 z"trace_0
ќ
#trace_02п
G__inference_policy_network_layer_call_and_return_conditional_losses_160
В
FullArgSpec
args
jx
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 z#trace_0
В
$	variables
%trainable_variables
&regularization_losses
'	keras_api
(__call__
*)&call_and_return_all_conditional_losses
*flatten"
_tf_keras_model
Л
+	variables
,trainable_variables
-regularization_losses
.	keras_api
/__call__
*0&call_and_return_all_conditional_losses


kernel
bias"
_tf_keras_layer
ы
	capture_2BЪ
!__inference_signature_wrapper_213input_1"
В
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs
	jinput_1
kwonlydefaults
 
annotationsЊ *
 z	capture_2
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
вBЯ
,__inference_policy_network_layer_call_fn_169input_1"
В
FullArgSpec
args
jx
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
эBъ
G__inference_policy_network_layer_call_and_return_conditional_losses_160input_1"
В
FullArgSpec
args
jx
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
1non_trainable_variables

2layers
3metrics
4layer_regularization_losses
5layer_metrics
$	variables
%trainable_variables
&regularization_losses
(__call__
*)&call_and_return_all_conditional_losses
&)"call_and_return_conditional_losses"
_generic_user_object
њ
6trace_02н
/__inference_feature_extractor_layer_call_fn_132Љ
ЂВ
FullArgSpec!
args
jx
jsample_action
varargs
 
varkw
 
defaultsЂ
p

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 z6trace_0

7trace_02ј
J__inference_feature_extractor_layer_call_and_return_conditional_losses_127Љ
ЂВ
FullArgSpec!
args
jx
jsample_action
varargs
 
varkw
 
defaultsЂ
p

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 z7trace_0
Ѕ
8	variables
9trainable_variables
:regularization_losses
;	keras_api
<__call__
*=&call_and_return_all_conditional_losses"
_tf_keras_layer
.

0
1"
trackable_list_wrapper
.

0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
>non_trainable_variables

?layers
@metrics
Alayer_regularization_losses
Blayer_metrics
+	variables
,trainable_variables
-regularization_losses
/__call__
*0&call_and_return_all_conditional_losses
&0"call_and_return_conditional_losses"
_generic_user_object
н
Ctrace_02Р
#__inference_dense_layer_call_fn_222
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zCtrace_0
ј
Dtrace_02л
>__inference_dense_layer_call_and_return_conditional_losses_232
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zDtrace_0
 "
trackable_list_wrapper
'
*0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
цBу
/__inference_feature_extractor_layer_call_fn_132input_1"Є
В
FullArgSpec!
args
jx
jsample_action
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Bў
J__inference_feature_extractor_layer_call_and_return_conditional_losses_127input_1"Є
В
FullArgSpec!
args
jx
jsample_action
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
Enon_trainable_variables

Flayers
Gmetrics
Hlayer_regularization_losses
Ilayer_metrics
8	variables
9trainable_variables
:regularization_losses
<__call__
*=&call_and_return_all_conditional_losses
&="call_and_return_conditional_losses"
_generic_user_object
п
Jtrace_02Т
%__inference_flatten_layer_call_fn_237
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zJtrace_0
њ
Ktrace_02н
@__inference_flatten_layer_call_and_return_conditional_losses_243
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zKtrace_0
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
ЭBЪ
#__inference_dense_layer_call_fn_222inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
шBх
>__inference_dense_layer_call_and_return_conditional_losses_232inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
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
ЯBЬ
%__inference_flatten_layer_call_fn_237inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ъBч
@__inference_flatten_layer_call_and_return_conditional_losses_243inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
__inference__wrapped_model_116t
8Ђ5
.Ђ+
)&
input_1џџџџџџџџџ``
Њ "3Њ0
.
output_1"
output_1џџџџџџџџџЏ
>__inference_agent_layer_call_and_return_conditional_losses_190m
8Ђ5
.Ђ+
)&
input_1џџџџџџџџџ``
Њ ",Ђ)
"
tensor_0џџџџџџџџџ
 
#__inference_agent_layer_call_fn_201b
8Ђ5
.Ђ+
)&
input_1џџџџџџџџџ``
Њ "!
unknownџџџџџџџџџЇ
>__inference_dense_layer_call_and_return_conditional_losses_232e
1Ђ.
'Ђ$
"
inputsџџџџџџџџџи
Њ ",Ђ)
"
tensor_0џџџџџџџџџ
 
#__inference_dense_layer_call_fn_222Z
1Ђ.
'Ђ$
"
inputsџџџџџџџџџи
Њ "!
unknownџџџџџџџџџМ
J__inference_feature_extractor_layer_call_and_return_conditional_losses_127n<Ђ9
2Ђ/
)&
input_1џџџџџџџџџ``
p
Њ ".Ђ+
$!
tensor_0џџџџџџџџџи
 
/__inference_feature_extractor_layer_call_fn_132c<Ђ9
2Ђ/
)&
input_1џџџџџџџџџ``
p
Њ "# 
unknownџџџџџџџџџи­
@__inference_flatten_layer_call_and_return_conditional_losses_243i7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ``
Њ ".Ђ+
$!
tensor_0џџџџџџџџџи
 
%__inference_flatten_layer_call_fn_237^7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ``
Њ "# 
unknownџџџџџџџџџиГ
G__inference_policy_network_layer_call_and_return_conditional_losses_160h
8Ђ5
.Ђ+
)&
input_1џџџџџџџџџ``
Њ "(Ђ%

tensor_0џџџџџџџџџ	
 
,__inference_policy_network_layer_call_fn_169]
8Ђ5
.Ђ+
)&
input_1џџџџџџџџџ``
Њ "
unknownџџџџџџџџџ	Є
!__inference_signature_wrapper_213
CЂ@
Ђ 
9Њ6
4
input_1)&
input_1џџџџџџџџџ``"3Њ0
.
output_1"
output_1џџџџџџџџџ
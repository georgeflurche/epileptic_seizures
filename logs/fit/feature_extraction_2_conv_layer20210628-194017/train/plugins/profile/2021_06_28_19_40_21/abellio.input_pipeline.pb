	?@?9w2b@?@?9w2b@!?@?9w2b@	??o??????o????!??o????"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:?@?9w2b@?
a5????AZ??mb@Y?/?x????rEagerKernelExecute 0*	Zd;?ck@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat {?\???!?K?,X0D@)ȴ6?????1?Vhd?uB@:Preprocessing2U
Iterator::Model::ParallelMapV2<?2T?T??!/).9?V0@)<?2T?T??1/).9?V0@:Preprocessing2F
Iterator::ModelF??\???!?%bG??=@)?R%??R??1r?gf+@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSlice?R)v4??!(xQK?*@)?R)v4??1(xQK?*@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate?H?<?+??!??h??S7@)?&P?"??1zY??#@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip???????!?v'?Z?Q@)??<????1j7?^@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?sF???!+O????@)?sF???1+O????@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap??G??5??!R~??$9@)Q?\?mOp?1???
???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.1% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9??o????I?
$???X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?
a5?????
a5????!?
a5????      ??!       "      ??!       *      ??!       2	Z??mb@Z??mb@!Z??mb@:      ??!       B      ??!       J	?/?x?????/?x????!?/?x????R      ??!       Z	?/?x?????/?x????!?/?x????b      ??!       JCPU_ONLYY??o????b q?
$???X@
	|??˙V@|??˙V@!|??˙V@	? ?4?T??? ?4?T??!? ?4?T??"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:|??˙V@?C5%Y???A	?/??U@Y>^H??0??rEagerKernelExecute 0*	??? ?Jb@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat????1ʫ?!??1???B@)??Co???1nz??T?@@:Preprocessing2U
Iterator::Model::ParallelMapV2?Ü?M??!??ڨ?=@)?Ü?M??1??ڨ?=@:Preprocessing2F
Iterator::Model#ظ?]???!䠠eV?G@)-]?6???1'??F1@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSlice?\??X3??!b?K??J@)?\??X3??1b?K??J@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::ConcatenateF??(&o??!HJN?K?%@)~b??U}?1.?P输@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor???"?v?!$?xne@)???"?v?1$?xne@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip? ?!?ֳ?!__??zJ@)??P??dv?1?/nDF?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapC???-??!?T?[&C(@)ҏ?S??[?1Th?Ӟ??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.1% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9? ?4?T??I??e???X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?C5%Y????C5%Y???!?C5%Y???      ??!       "      ??!       *      ??!       2		?/??U@	?/??U@!	?/??U@:      ??!       B      ??!       J	>^H??0??>^H??0??!>^H??0??R      ??!       Z	>^H??0??>^H??0??!>^H??0??b      ??!       JCPU_ONLYY? ?4?T??b q??e???X@
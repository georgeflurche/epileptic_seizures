	?W????P@?W????P@!?W????P@	?k?}????k?}???!?k?}???"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:?W????P@??ƠB??A??????P@YA)Z????rEagerKernelExecute 0*	?t??d@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::ZՒ??!??Jc,?A@)?O??????1???B?>@:Preprocessing2U
Iterator::Model::ParallelMapV2?@׾???!~A?H??1@)?@׾???1~A?H??1@:Preprocessing2F
Iterator::Model???????!r??j@@)???-I??1ˮO???-@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip??)r???!?y?.??P@)?H.?!???1?????+@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate???e?i??!???"0@)_????=??1Yl??!@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSliceg?????!uMp?B?@)g?????1uMp?B?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??Z?a/??!m8?@)??Z?a/??1m8?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap???????![?Þ)?1@)??p?d?1=??HQ??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.1% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9?k?}???IJ1A|??X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??ƠB????ƠB??!??ƠB??      ??!       "      ??!       *      ??!       2	??????P@??????P@!??????P@:      ??!       B      ??!       J	A)Z????A)Z????!A)Z????R      ??!       Z	A)Z????A)Z????!A)Z????b      ??!       JCPU_ONLYY?k?}???b qJ1A|??X@
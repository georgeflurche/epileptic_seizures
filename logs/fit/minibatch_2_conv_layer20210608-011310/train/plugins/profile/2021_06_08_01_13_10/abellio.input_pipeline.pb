	Kt?Y??T@Kt?Y??T@!Kt?Y??T@	?? ?P???? ?P??!?? ?P??"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:Kt?Y??T@?n?l???A?????T@Y?H?}8??rEagerKernelExecute 0*	??ʡ?V@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?G????!r??_?
@@)?>rk?m??1tĭ?";@:Preprocessing2U
Iterator::Model::ParallelMapV2?IF???!W???n8@)?IF???1W???n8@:Preprocessing2F
Iterator::Model??qS??!1q??=?D@)??_ ??1
`??1@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::TensorSlice???_???!??U?T'@)???_???1??U?T'@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapvOjM??!?b? Д4@)֨?ht??1?ּfK("@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipHnM?-???!ΎK:?dM@)??	j?v?1????~@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor!#???r?!?I?S??@)!#???r?1?I?S??@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.1% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9?? ?P??I????+?X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?n?l????n?l???!?n?l???      ??!       "      ??!       *      ??!       2	?????T@?????T@!?????T@:      ??!       B      ??!       J	?H?}8???H?}8??!?H?}8??R      ??!       Z	?H?}8???H?}8??!?H?}8??b      ??!       JCPU_ONLYY?? ?P??b q????+?X@
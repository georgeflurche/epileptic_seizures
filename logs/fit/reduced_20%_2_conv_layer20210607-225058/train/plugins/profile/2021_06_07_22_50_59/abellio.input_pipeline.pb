	?v??&@?v??&@!?v??&@	?P?d????P?d???!?P?d???"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:?v??&@4?s?????Ar????%@Y<?H??ڲ?rEagerKernelExecute 0*	T㥛ĠZ@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?7??
*??!
?l?@@)???H?1?R???3<@:Preprocessing2F
Iterator::Model??0??!?ܘy??A@)???c?ғ?1t1???,2@:Preprocessing2U
Iterator::Model::ParallelMapV2?.?e????!??7\?1@)?.?e????1??7\?1@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?Ǚ&l???!,?#b&7@)x??!S??1?8>??+@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?/g?+??!I??b~"@)?/g?+??1I??b~"@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?$??W???!??3?,4P@)Oʤ?6 {?1??Jg??@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor!??q4Gv?! ?*?m@)!??q4Gv?1 ?*?m@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?H?<???!ʍ?gzQ9@)oJy???b?1???+?[@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.7% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9?P?d???I^??7!?X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	4?s?????4?s?????!4?s?????      ??!       "      ??!       *      ??!       2	r????%@r????%@!r????%@:      ??!       B      ??!       J	<?H??ڲ?<?H??ڲ?!<?H??ڲ?R      ??!       Z	<?H??ڲ?<?H??ڲ?!<?H??ڲ?b      ??!       JCPU_ONLYY?P?d???b q^??7!?X@
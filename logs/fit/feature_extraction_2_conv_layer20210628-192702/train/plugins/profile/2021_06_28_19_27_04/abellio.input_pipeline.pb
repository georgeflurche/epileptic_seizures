	?Ƃ ,Y@?Ƃ ,Y@!?Ƃ ,Y@	?s?ʩ???s?ʩ??!?s?ʩ??"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:?Ƃ ,Y@???cZ???A??U?ZY@Y???O???rEagerKernelExecute 0*	@5^?I?b@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?????!??π??@@)?P??dV??1???rjB>@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate?: ??^??!g??M?;@){-??1??1^??S?4@:Preprocessing2U
Iterator::Model::ParallelMapV2???1???!?scڑ1@)???1???1?scڑ1@:Preprocessing2F
Iterator::Model?ЕT???!U?*??4@@)??X?????1?-@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSlice?4`??i??!'??{??@)?4`??i??1'??{??@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip??_#I??!Պ?$??P@)??ֈ`|?1???l9@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??f?|u?!??ud?@)??f?|u?1??ud?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?5"???!b???JJ=@)ɯb??c?1?ߩn?O??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.1% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9?s?ʩ??Icx??X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	???cZ??????cZ???!???cZ???      ??!       "      ??!       *      ??!       2	??U?ZY@??U?ZY@!??U?ZY@:      ??!       B      ??!       J	???O??????O???!???O???R      ??!       Z	???O??????O???!???O???b      ??!       JCPU_ONLYY?s?ʩ??b qcx??X@
	??a??9V@??a??9V@!??a??9V@	K??ǲ???K??ǲ???!K??ǲ???"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:??a??9V@??֥F??A9_???"V@Y??"[A??rEagerKernelExecute 0*?????j@)       =2F
Iterator::Model???i??!???{?~I@)b?qm???1?E????<@:Preprocessing2U
Iterator::Model::ParallelMapV2 Й?????!8%3?6@) Й?????18%3?6@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat6??Ϸ??!3???24@)???%??1?U?4?0@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSlice)? ???!??.gB?-@))? ???1??.gB?-@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate?p!??F??!{8mp8@)ԀAҧU??1B???"@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??#?|?!r8?
@)??#?|?1r8?
@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipKs+??X??!A? ?H@)?V???|?1k???\
@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap???*?w??!?зQ??9@)??ƽ?c?13R?GԸ??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.2% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9L??ǲ???I"??+?X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??֥F????֥F??!??֥F??      ??!       "      ??!       *      ??!       2	9_???"V@9_???"V@!9_???"V@:      ??!       B      ??!       J	??"[A????"[A??!??"[A??R      ??!       Z	??"[A????"[A??!??"[A??b      ??!       JCPU_ONLYYL??ǲ???b q"??+?X@
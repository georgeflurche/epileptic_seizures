?	?B:<?}5@?B:<?}5@!?B:<?}5@	???y???????y????!???y????"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:?B:<?}5@l?`q8??A	?/??+5@YL???j???rEagerKernelExecute 0*	?E????k@2F
Iterator::Model??s?/??!T;?i?P@)??J???1???j?(M@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatc??K?A??!A?P?1?0@)?CP5z5??1H?n-?,@:Preprocessing2U
Iterator::Model::ParallelMapV2????oҔ?!(?E?YN"@)????oҔ?1(?E?YN"@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?ٮ?˘?!F?e??%@)??S??1Ȱ?xCf@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlicejM????!????1@)jM????1????1@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipDܜJ???!Y??,?C@@)\*?????1??xjg4@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?s|?8cx?!? 9??p@)?s|?8cx?1? 9??p@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap????Y.??!?????'@)?'??c?1=??????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.9% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9???y????Iv???X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	l?`q8??l?`q8??!l?`q8??      ??!       "      ??!       *      ??!       2		?/??+5@	?/??+5@!	?/??+5@:      ??!       B      ??!       J	L???j???L???j???!L???j???R      ??!       Z	L???j???L???j???!L???j???b      ??!       JCPU_ONLYY???y????b qv???X@Y      Y@q(??u???"?
device?Your program is NOT input-bound because only 0.9% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2M
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono2no:
Refer to the TF2 Profiler FAQ2"CPU: B 
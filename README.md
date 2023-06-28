# pymica (python for mineral identification and characterization using imaging spectrometer data)
自20世纪90年代起由Roger N. Clark等发展起来的Tetracorder矿物识别专家系统不断成熟完善（Clark等，2003），为高光谱矿物识别提供了强大的算法框架和基础工具。Kokaly等（2011）在Tetracorder算法框架的基础上对光谱拟合过程进行了改进，而且增加了用户对光谱诊断特征的权重、匹配阈值的调控余地，提出了MICA算法，并使用它首次完成了阿富汗全境的Hymap高光谱矿物填图工作。

从2018年起，国内陆续发射了高分五号卫星（2018.05）、资源一号02D星（2019.09）、高分五号02星（2021.09）、资源一号02E星（2021.12）、高分五号01A（2022.12）等一系列技术参数达到国际领先水平的高光谱遥感卫星，为国内正如火如荼展开的新一轮找矿突破战略行动提供了海量优质高光谱遥感数据，彻底解决了国内高光谱遥感矿产勘查应用“无米下锅”的窘境。

由于Kokaly提供的MICA算法实现是内嵌于基于IDL开发的PRISM(以ENVI插件方式运行)中的，其运行环境依赖商业授权，且运行效率并不是很高（一景高分五号影像处理超过一个小时，Z840实测），给国内高光谱遥感应用带来了困扰。鉴于此，本工程给出了mica算法的python生态实现。


![Uploading image.png…]()


简介
-------------

GeoAHSI (Geological mapper for AHSI)是一个基于纯Python生态的国产（中国）高光谱遥感卫星矿物填图工具包，它继承了由Roger N. Clark等开发的Tetracorder系统的理论基础和算法，并吸纳了Kokaly等（2011）对Tetracorder所做的部分简化。GeoAHSI的输入数据为经大气校正得到的地表反射率影像（ENVI Header格式），我们称其为Image Cube（m×n×k matrix，m代表行数，n代表列数，k代表波段数）。GeoAHSI执行矿物填图分为三个步骤：首先，通过将Image Cube与从USGS Spectral Library 06（Clark et al. 2007）中筛选的参考矿物光谱进行匹配来计算它们的诊断光谱特征在Image Cube中的信号强度（相关性+吸收深度），得到Fit Cube（m×n×s matrix，s代表参考矿物光谱数量）和Depth Cube（m×n×s matrix）；然后，计算Fit Cube中沿第三维度（维数s）最大值对应的索引，降维得到Mineral Map（m×n matrix），并使用该索引检索Depth Cube得到Depth Map（m×n matrix）；最后，根据预设的颜色表为Mineral Map“赋色”，生成Color-Coded Map。在赋色的过程中，可以使用Depth Map经拉伸处理作为明度分量，从而呈现更加细腻的填图效果。

<div align="center">
    <img src="https://github.com/leecugb/GeoAHSI/blob/main/workflow.jpg" width="300px" alt="GeoAHSI work flow">
</div>

示例
-------------

1.打开影像文件，并执行光谱分析

将经大气校正影像（ENVI格式，含路径）作为参数调用geo_map()。

        from ahsi import *
        geo_map('./data/ZY1E_AHSI_E96.59_N41.04_20220929_015969_L1A0000509267/flaash.hdr') # load reflectance image cube and execute spectrum analysis

geo_map()会在源影像文件的目录下生成三个GeoTiff文件：mineral_map.tiff, mineral_color_enhanced.tiff, muscovite_wv.tiff。它们分别代表光谱主导矿物类别分布图、吸收深度（经归一化拉伸）做明度分量的光谱主导矿物类别分布图和白云母2200nm吸收特征波长分布图。

2.快速可视化

        with gdal.Open('./data/ZY1E_AHSI_E96.59_N41.04_20220929_015969_L1A0000509267/mineral_map.tiff') as f:
            img = f.ReadAsArray()
        plt.figure(figsize=(16, 15), dpi=300)
        plt.imshow(image)
        patches = [mpatches.Patch(color=[i/255 for i in value], label="{l}".format(l=key)) for key, value in colors_dic.items()]
        plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


<div align="center">
    <img src="https://github.com/leecugb/pymica/assets/38849659/2a9aa593-e85f-4d86-8dd1-7ac3eb2671e0" width="800px" alt="GeoAHSI work flow">
</div>

行业应用
-------------

我们使用该工具完成了东天山-北山成矿带、阿尔泰-准噶尔北缘成矿带和阿尔金成矿带近100万平方千米的国产高光谱遥感卫星矿物填图工作，相关数据成果参见[地质云](https://geocloud.cgs.gov.cn/topic/view?id=1686360171308183554)。

<div align="center">
    <img src="https://github.com/leecugb/GeoAHSI/blob/main/Mineral_Map_of_Dongtianshan-Beishan_Metallogenic_Belts.jpg" width="900px" alt="GeoAHSI work flow">
</div>



*出于简约展示光谱分析算法的目的，删减了地理空间处理部分代码。待后续更新*

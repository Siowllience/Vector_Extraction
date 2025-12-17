论文[Increasing alignment of large language models with language processing in the human brain](https://www.nature.com/articles/s43588-025-00863-0)的眼动向量和fMRI向量提取代码。

# words.csv和words_list.p的构建

所述两个文件记录了句子id和句子词数的索引，用于构建初始眼动矩阵和fMRI矩阵，对后续处理眼动向量和fMRI向量至关重要。但原论文未给出相关文件，需要从原数据集的`text_data.xlsx`文件中提取（参考原论文提供的相关代码`reference/heads_vs_fmri.py`以及`reference/heads_vs_saccade.py`总结得出）。

设置好`textdata_to_words.py`的输入输出路径后执行：
```python
python textdata_to_words.py
```
（处理好的文件已给出，可以不用重复处理）

除此之外，所述两个文件决定了最后提取向量的维度。维度计算公式为：

$$
\text{总维度} = \sum_{i=1}^{K} \frac{n_i \times (n_i - 1)}{2} 
$$

变量说明：
- $K$：实验刺激的总句子数（论文中 $K=148$，包含133个训练句+15个测试句）；
- $n_i$：第i个句子的单词数量（论文中单个句子平均单词数为 $10.33$）；
- $\frac{n_i \times (n_i - 1)}{2}$：单个句子对应的 $n_i \times n_i$ 注意力矩阵/眼动矩阵/ fMRI矩阵中，**下三角部分（不含对角线）的元素个数**（即右到左的回归型眼动或注意力关联的有效计算单元）。

然而，现在提取出的向量维度为7421维，与原论文提到的7388维不符，需要检查`textdata_to_words.py`的处理逻辑是否有问题，是否导致每个句子单词数量的变化，从而影响到了维度。


# 眼动向量提取

在`saccadeVector_Extraction.py`中修改原数据集路径、向量输出路径后，执行：

```python
python saccadeVector_Extraction.py
```

即可提取所有眼动向量，默认保存到`saccadeVector/`文件夹中。该脚本参考原论文代码`reference/heads_vs_saccade.py`。

# fMRI向量提取

fMRI向量提取分为两步，首先要使用fMRIPrep工具处理原始数据集的fMRI数据，以期获得`{subj}_task-read_run-{run}_hemi-{hem}_space-fsaverage5_bold.func.gii`形式的文件（参考原论文提供的相关代码`reference/heads_vs_fmri.py`得出）。然后执行用于向量处理的脚本。

## fMRIPrep工具配置
参考[fMRIPrep](https://fmriprep.org/en/stable/.)官方文档，采用docker运行fMRIPrep工具最为稳定。在确保docker配置正常的情况下，终端执行以下指令以拉取25.0.0版本的fMRIPrep：
```python
docker pull nipreps/fmriprep:25.0.0
```

如果服务器无法连接到docker hub，可从本地上传相关docker压缩包到服务器（在此提供我下载好的docker压缩包，请点击该链接进行下载[百度网盘](https://pan.baidu.com/s/12VW8ZYd8lAURu08HKi7lzQ?pwd=wv5f)）。上传到服务器后，执行以下指令构建docker：
```python
docker load -i fmriprep_25.0.0.tar
```

确认docker无误后即可利用fMRIPrep工具处理原始fMRI数据：
```python
docker run --rm -it \
  -v /data/home/jinyuehan/LLMBrain/11.25/reading_brain_datalad/ds003974:/data/input:ro \
  -v fmriprep:/data/output \
  -v /data/home/jinyuehan/LLMBrain/12.03/license.txt:/opt/freesurfer/license.txt \
  nipreps/fmriprep:25.0.0 \
  /data/input /data/output participant \
  --participant-label 16 17 18 19 20 \
  --output-spaces MNI152NLin2009cAsym fsaverage5 \
  --n_cpus 4 \
  --mem_mb 16000 \
  --skip-bids-validation \
  >"/data/home/jinyuehan/LLMBrain/12.11/fmriprep16-20.log" 2>&1
```
其中需要根据实际情况设置原始数据集路径、处理数据输出路径以及freesurfer许可证路径。默认输出路径为`fmriprep/`,freesurfer许可证需要去[FreeSurfer官网](https://surfer.nmr.mgh.harvard.edu/registration.html )申请后，在注册邮箱中下载。

## fMRI向量提取

在`fMRIVector_Extraction.py`中修改原数据集路径、向量输出路径、处理的受试者编号后，执行：

```python
python fMRIVector_Extraction.py
```

即可提取特定受试者的fMRI向量，默认保存到`fmriVector/`文件夹中。该脚本参考原论文代码`reference/heads_vs_saccade.py`。
### task 3

在 hw3/ 下 git clone fairseq, mosesdecoder 以及 subword-nmt 三个工具包，与 nmt 目录同级
在 hw3/nmt/data/v16news 下放置中英文数据集
在 hw3/nmt/models/v16news 下放置模型 checkpoints

主要的脚本位于 hw3/nmt/scripts 以及 hw3/nmt/utils，脚本运行顺序为：
initpath.sh -> train1.sh or train2.sh -> generate.sh -> postprocess.sh

### task 4

在 hw4/data 下放置数据集，主程序位于 fine_tune.py
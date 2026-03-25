# 快速开始
```
# 镜像 registry.mthreads.com/mcconline/musa-train:4.3.5_tele3_105b_alinux
# 驱动 http://swci.mthreads.com:8848/ui/buckets/sw-daily/objects?prefix=musa%2Frelease_musa_4.3.0%2F2026-01-26%2F
# tokenizer modelscope download --model TeleAI/TeleChat3-36B-Thinking --local_dir TeleChat3-36B-Thinking
# 数据集链接 https://www.modelscope.cn/datasets/TeleAI/TeleChat-PTD/resolve/master/data
# 数据集 使用镜像中/home/Megatron-LM/tools/process_test_data.py 进行处理。
# 修改 megatron-lm-musa-patch/examples/telechat3 中 hostfile和run_telechatv3_105B_musa.sh路径后
# 执行run_telechatv3_105B_musa.sh开始训练
```





# telechat3-105B 训练方法(原电信提供)

本次提供素材包括数据处理脚本、训练脚本、环境信息

请按照下述步骤进行环境配置和运行测试。

## 环境信息与素材获取
其它环境基础信息: python==3.12.3,cuda==12.8,nccl==2.27.5 nvidia Driver Version: 580.105.08 

pip 相关信息请参考 requirements.txt

核心代码来源于开源代码 NVIDIA Megatron Core v0.15.2 https://github.com/NVIDIA/Megatron-LM/archive/refs/tags/core_v0.15.2.zip 如果需要打patch,请在获取该代码后进行patch

原始数据文件来源于 
- https://www.modelscope.cn/datasets/TeleAI/TeleChat-PTD/resolve/master/data/0.jsonl.gz
- https://www.modelscope.cn/datasets/TeleAI/TeleChat-PTD/resolve/master/data/1.jsonl.gz

使用脚本 Megatron-LM-core_v0.15.2/tools/process_test_data.py 进行处理。

词表采用

```shell
modelscope download --model TeleAI/TeleChat3-36B-Thinking --local_dir TeleChat3-36B-Thinking 
```

## 多机训练脚本

请参考 train_telechat3_105B_multi_node.sh 

其中的的主机地址、并行策略、优化方法均需要根据机器修改，只作参考使用。

不要修改其中的 学习率、batch-size、训练步数、位置编码、负载均衡算法等信息。

如不明确某个参数是否可进行修改，请进行直接确认

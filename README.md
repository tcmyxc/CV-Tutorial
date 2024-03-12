# CV Tutorial——CV方向入门指南

> 代码在 `src` 文件夹下面 :coffee: 

## 编程环境和IDE

> 工欲善其事，必先利其器

Python下载：https://www.python.org/downloads/

VSCode：推荐

- 下载安装：https://code.visualstudio.com/Download
- 相关设置（字体大小、禁止更新、图标、Python拓展）
- SSH远程链接
	- 密钥生成及保存（涉及git）

PyCharm（代码提示好）

## git安装以及生成公私钥

git安装地址：

- 国内中文网站：https://git.p2hp.com/
- 官网：https://git-scm.com/

git相关配置

```bash
git config --global user.name '你的用户名'
git config --global user.email '你的邮箱'
```

生成ssh密钥：

```bash
ssh-keygen -t rsa -C "你的邮箱"
```



## 入门案例

案例：https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html

讲解：https://www.bilibili.com/video/BV1hu4y1g7VP/

### 数据增强

推荐仓库：

1. [AutoAugment](https://github.com/DeepVoltaire/AutoAugment)
2. [Cutout](https://github.com/uoguelph-mlrg/Cutout)
3. https://albumentations.ai/docs/


## 训练的trick

基础配置：
- 学习率：0.01
- batch size：128
- epoch：200
- momentum：0.9
- weight decay：5e-4
- 学习率调度器：余弦
- 优化器：SGD


ResNet18 + CIFAR-10 实验结果
- 基础配置: 86.91
	- +预热5轮: 87.20 (+0.29)
		- +label smoothing: 87.21 (+0.01)
		- +AA: 89.41 (+2.21)
			- +label smoothing: 89.24 (-0.17)
			- +Mixup: 89.67 (+0.26)
				- +Cutmix: 89.78 (+0.11)
					- +label smoothing: 89.32 (-0.46)
			- +Cutout: 90.02 (+0.61)
				- +label smoothing: 89.21 (-0.81)
			

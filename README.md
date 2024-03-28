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


## 服务器使用本地梯子

```bash
# 下面这行在本地运行
ssh -CqTnN -R 127.0.0.1:45588:127.0.0.1:7890 `主机名`

# 下面的在服务器运行
export http_proxy=http://127.0.0.1:45588
export https_proxy=http://127.0.0.1:45588
```

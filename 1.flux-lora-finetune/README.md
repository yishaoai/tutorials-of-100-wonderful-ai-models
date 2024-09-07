## 1. 童装lora的效果展示
本部分以182张童装数据作为训练数据，训练flux.1 模型的lora，生成的图片适合买家秀和商品种草场景，从以下的图片可以看出，生成的图片非常真实且质量很高。本部分实验仅为了进行效果展示，如果训练图片数量更多，质量更高，将会获得更好的生成结果。本部分实验的效果展示如下：

![kid-clothes-lora](https://github.com/yishaoai/tutorials-of-100-wonderful-ai-models/tree/main/1.flux-lora-finetune/assets/kid-clothes-lora.png)

## 2. Flux.1 的lora微调实验

**代码和环境准备**

首先我们下载代码和安装 python 依赖库：
```shell
git clone https://github.com/yishaoai/tutorials-of-100-wonderful-ai-models
cd tutorials-of-100-wonderful-ai-models/1.flux-lora-finetune/x-flux/
git submodule update --init .

pip install -r requirements.txt
cd ../..

```

**数据准备**
本部分实验主要从互联网搜集了182张童装的照片，包括男孩和女孩。这些童装主要是买家秀种草场景。如果您需要训练自己的数据，可以按这个 png/json 成对的格式准备数据。
我们将所有的图像都放在 images 文件夹下。并且每个图片有一个对应的json文件，json文件中内容都是一样的，如下所示：

```shell
├── images/
│    ├── 1.png
│    ├── 1.json
│    ├── 2.png
│    ├── 2.json
│    ├── ...
```


**实验脚本**
本实验使用单卡L20 机器，最大显存占用为。
```shell
export PYTHONPATH=$PWD/x-flux:$PYTHONPATH
accelerate launch --config_file "default_config.yaml" x-flux/train_flux_lora_deepspeed.py --config "x-flux/train_configs/test_lora.yaml"
```

## 3. 基于童装lora的推断

可以在huggingface下载本实验训练好的 lora，地址是 [YishaoAI/flux-dev-lora-kid-clothes](https://huggingface.co/YishaoAI/flux-dev-lora-kid-clothes/tree/main).

**x-flux 的推断方法**

```shell
bash infer_xlabs.sh
```

**diffusers 的推断方法**
```shell
cd 1.flux-lora-finetune/diffusers/
git submodule update --init .
cd ../..
python infer_diffusers.py
```


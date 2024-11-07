



## Introduction

![Overview](D:\wjl\postgraduate\记事本\Typora_Y_image\Overview.jpg)

## Data Preparation

By default, for training, testing and demo, we use  [HAKE](https://github.com/DirtyHarryLYL/HAKE) dataset.

Instance-level part state annotations on [HICO-DET](http://www-personal.umich.edu/~ywchao/hico/) are also available.

The labels are packaged in **Annotations/hico-det-instance-level.tar.gz**, you could use:

```
cd Annotations
tar zxvf hico-det-instance-level.tar.gz
```

to unzip them and get hico-det-training-set-instance-level.json for train set of HICO-DET respectively. More details about the format are shown in [Dataset format](https://github.com/DirtyHarryLYL/HAKE/blob/master/Annotations/README.md).

The HICO-DET dataset can be found here: [HICO-DET](http://www-personal.umich.edu/~ywchao/hico/).

## Model-Zoo

[MODEL -ViT-B-16]: https://github.com/Yangless/AlignCLIP

## Inference

Stanford40：

demo_Stanford40.py

ILSVRC2012:

demo.py

HAKE:

main_singel_hake.py

## Evaluation

see utils/options.py

```shell
train:
python main_single.py  --visual_model ViT-B-16 --batch_size_test 16  --test_dataset imagenet  --test_data_path   E:\Datasets\ILSVRC2012_img_train_split2  

val:
python main_single.py  --visual_model ViT-B-16 --batch_size_test 16  --test_dataset imagenet  --test_data_path   E:\Datasets\ILSVRC2012_img_train_split2  --evaluate ckpts/checkpoint_epoch_1.pth
```






## Introduction

![创新点1](D:\wjl\postgraduate\记事本\Typora_Y_image\创新点1.jpg)

## Model-Zoo

[MODEL -ViT-B-16]: https://github.com/Yangless/AlignCLIP

## Inference

demo.py

## Evaluation

see utils/options.py

```shell
train:
python main_single.py  --visual_model ViT-B-16 --batch_size_test 16  --test_dataset imagenet  --test_data_path   E:\Datasets\ILSVRC2012_img_train_split2  

val:
python main_single.py  --visual_model ViT-B-16 --batch_size_test 16  --test_dataset imagenet  --test_data_path   E:\Datasets\ILSVRC2012_img_train_split2  --evaluate ckpts/checkpoint_epoch_1.pth
```


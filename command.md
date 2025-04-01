https://github.com/mindee/doctr/tree/main/references/detection#readme


## Train Detection Model with PyTorch for fast_base

```shell
python references/detection/train_pytorch.py fast_base --train_path ./dataset_dir/text_det_test_word --val_path ./dataset_dir/text_det_test_word --epochs 20 --batch_size 16 --input_size 320 --pretrained --freeze-backbone
```

## Train Detection Model with PyTorch for db_resnet50
```shell
python references/detection/train_pytorch.py db_resnet50 --train_path ./dataset_dir/text_det_test_word --val_path ./dataset_dir/text_det_test_word --epochs 20 --batch_size 16 --input_size 320 --pretrained --freeze-backbone
```

More info to the available model you can finde here: https://mindee.github.io/doctr/using_doctr/using_models.html

Available models:
```python
ARCHS = [
    "db_resnet34",
    "db_resnet50",
    "db_mobilenet_v3_large",
    "linknet_resnet18",
    "linknet_resnet34",
    "linknet_resnet50",
    "fast_tiny",
    "fast_small",
    "fast_base",
]
```

### My custom train command (only for testing)

```shell
python references/detection/train_pytorch.py db_resnet50 --train_path ./dataset_dir/text_det_test_word --val_path ./dataset_dir/text_det_test_word --epochs 20 --batch_size 16 --input_size 320 --pretrained  --show-sample
```

```shell
python references/detection/train_pytorch.py db_resnet50 --train_path ./dataset_dir/output_doctr/train --val_path ./dataset_dir/output_doctr/val --epochs 30 --batch_size 8 --input_size 640 --pretrained  --lr 0.001  --freeze-backbone
```

```shell
python references/detection/train_pytorch.py fast_tiny --train_path ./dataset_dir/output_doctr/train --val_path ./dataset_dir/output_doctr/val --epochs=30 --batch_size=5 --lr=0.0001  --workers=2  --weight-decay=0.01 --pretrained
```


## Train Orientation Classification Model with PyTorch based on page/image orientation
```shell
python references/classification/train_pytorch_orientation.py resnet50 --type page --train_path ./dataset_dir/images --val_path ./dataset_dir/images --epochs 100 --batch_size 32
```

## Train Orientation Classification Model with PyTorch based on crop/word orientation
```shell
python references/classification/train_pytorch_orientation.py resnet50 --type crop --train_path ./dataset_dir/crop_img --val_path ./dataset_dir/crop_img --epochs 200 --batch_size 40 --wb
```

Available models:
```python
ARCHS: list[str] = [
    "magc_resnet31",
    "mobilenet_v3_small",
    "mobilenet_v3_small_r",
    "mobilenet_v3_large",
    "mobilenet_v3_large_r",
    "resnet18",
    "resnet31",
    "resnet34",
    "resnet50",
    "resnet34_wide",
    "textnet_tiny",
    "textnet_small",
    "textnet_base",
    "vgg16_bn_r",
    "vit_s",
    "vit_b",
]
```
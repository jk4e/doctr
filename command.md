## Train Detection Model with PyTorch
```shell
python references/detection/train_pytorch.py fast_base --train_path ./dataset_dir/text_det_test_word --val_path ./dataset_dir/text_det_test_word --epochs 20 --batch_size 2 --input_size 512 --pretrained --freeze_backbone
```

## Train Orientation Classification Model with PyTorch
```shell
python references/classification/train_pytorch_orientation.py resnet18 --type page --train_path ./dataset_dir/images --val_path ./dataset_dir/images --epochs 20 --batch_size 2 --input_size 512 
```


```shell
python references/detection/train_pytorch.py fast_base --train_path ./dataset_dir/text_det_test_word --val_path ./dataset_dir/text_det_test_word --epochs 20 --batch_size 2 --input_size 512 --pretrained --freeze_backbone
```
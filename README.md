# deep-text-recognition-benchmark-mnn-ncnn
## Brief
- Rewrite Version of [clovaai/deep-text-recognition-benchmark](https://github.com/clovaai/deep-text-recognition-benchmark). Make attempts of deployment on edge devices.
- Based on Pytorch(training), MNN(Inference) and NCNN(Inference)

## Training
- Based on ```Pytorch1.3, Python3.6``` 
- Original repo please check [clovaai/deep-text-recognition-benchmark](https://github.com/clovaai/deep-text-recognition-benchmark).
- Default training the **TPS-ResNet-BiLSTM-Attn** model.
#### 1. Prepare datasets
- Please Check the [**"When you need to train on your own dataset or Non-Latin language datasets."**](https://github.com/clovaai/deep-text-recognition-benchmark#when-you-need-to-train-on-your-own-dataset-or-non-latin-language-datasets) in the original repo.
- Once when you prepare the datasets well, there should be "**```data.mdb```**" and "**```lock.mdb```**" under **the output folder**.
- Assign the value of "**train_data**" and "**valid_data**" as the absolute addresses of **the output folders**.
#### 2. Start to train
- Modify the values in ```train.py```:

Parameters | Position | Remarks
--- | --- | ---
```train_data``` | [```train.py```, Line:223](./train.py#L223) | lmdb trainset folder
```valid_data``` | [```train.py```, Line:224](./train.py#L224) | lmdb valset folder
```imgH``` | [```train.py```, Line:246](./train.py#L246) | height of the resized image 
```imgW``` | [```train.py```, Line:247](./train.py#L247) | width of the resized image
```character``` | [```train.py```, Line:249](./train.py#L249) | all your characters you want to recognize

- The output weights (end with ```.pth```) are saved in the ```./saved_models``` in default.
- Then click the "run" button (if you use IDE with Python interpreter) to train.

## TODOs
Not Finished yet!!!
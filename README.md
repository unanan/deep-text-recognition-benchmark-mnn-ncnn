# deep-text-recognition-benchmark-mnn-ncnn
## Brief
- Rewrite Version of [clovaai/deep-text-recognition-benchmark](https://github.com/clovaai/deep-text-recognition-benchmark). Make attempts of deployment on edge devices.
- Based on Pytorch(training), MNN(Inference) and NCNN(Inference)

## Training
- Based on ```Pytorch1.3, Python3.6``` 
- Original repo please check [clovaai/deep-text-recognition-benchmark](https://github.com/clovaai/deep-text-recognition-benchmark).
- Default training the **TPS-ResNet-BiLSTM-Attn** model.
#### 1. Prepare datasets
- Please check the [**"When you need to train on your own dataset or Non-Latin language datasets."**](https://github.com/clovaai/deep-text-recognition-benchmark#when-you-need-to-train-on-your-own-dataset-or-non-latin-language-datasets) in the README.md of the original repo.
- Once when you prepare the datasets well, there should be "**```data.mdb```**" and "**```lock.mdb```**" under **the output folder**.
- Assign the value of "**train_data**" and "**valid_data**" with the absolute addresses of **the output folders**.
#### 2. Start to train
- Modify the values in ```train.py```:

Parameters | Position | Remarks
--- | --- | ---
```train_data``` | [```train.py```, Line:223](./train.py#L223) | lmdb trainset folder
```valid_data``` | [```train.py```, Line:224](./train.py#L224) | lmdb valset folder
```batch_max_length``` | [```train.py```, Line:246](./train.py#L246) | width of the resized image
```imgH``` | [```train.py```, Line:247](./train.py#L247) | height of the resized image 
```imgW``` | [```train.py```, Line:248](./train.py#L248) | width of the resized image
```character``` | [```train.py```, Line:250](./train.py#L249) | all your characters of your vocabulary you want to recognize
- **IMPORTANT:If there're special characters in your vocabulary(e.g."-"), please modify the [```dataset.py``` Line:170](./dataset.py#L170).**

- The output weights (end with ```.pth```) are saved in the ```./saved_models``` by default.
- Then run ```python train.py```.

## Visualization
- You can use **netron** to open your ```*.pth``` to view the model architecture.

## Inference
- Please check the [README.md under the ```./inference```](./inference).

## Ideas & Issues
- Be free to open issues or pull requests.

## TODOs
Not Finished yet!!!
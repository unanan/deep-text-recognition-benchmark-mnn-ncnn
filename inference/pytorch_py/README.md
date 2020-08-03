# Inference (Based on Python Pytorch)

### Requirements
- Same environment with the training phase. (```Pytorch1.3, Python3.6```)

### Test
- Test the accuracy on the testset. Please explore [```test.py```](./test.py).
- Remember modify the parameters under the "imports".(```batch_max_length,imgW,imgH,character,model_path```)

### Inference
- Inference the model which can be used in the deploying phase. Please explore [```inference.py```](./inference.py).
- Remember modify the parameters under the "imports".(```batch_max_length,imgW,imgH,character,model_path```)

#### Model Conversion
* Attention: In the original repo, the saved model is in ```Data-Parallel``` form. You need to convert it first, if you wouldn't init the model as Data-parallel form in the inference phase.
* To convert the ```Data-Parallel``` model, please run
```
cd deep-text-recognition-benchmark-mnn-ncnn/inference/pytorch_py/
python convert_dpmodel.py --model_path TPS-ResNet-BiLSTM-Attn.pth --output_path TPS-ResNet-BiLSTM-Attn.pth
```

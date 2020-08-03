# Inference (Based on C++ MNN)

### 1. Build MNN Convert Tools (Refer to [MNNConvert](https://github.com/alibaba/MNN/tree/master/tools/converter))
##### Git clone MNN
```
git clone https://github.com/alibaba/MNN.git
```
##### Then build the tool
```
# Install protobuf first: https://github.com/protocolbuffers/protobuf/tree/master/src
cd MNN
./schema/generate.sh
mkdir build
cd build
cmake .. -DMNN_BUILD_CONVERTER=true
make
```

### 2 Convert the ```*.pth``` to ```*.mnn```
##### Run ```convert_pth_to_mnn_ncnn.py```
```
cd deep-text-recognition-benchmark-mnn-ncnn/   # go to the project root path
python convert_pth_to_mnn_ncnn.py --target mnn --model_path inference/pytorch_py/TPS-ResNet-BiLSTM-Attn.pth --output_path inference/mnn_cpp/TPS-ResNet-BiLSTM-Attn.mnn
```
_The output model will be located at the input model's path._


### TODOs
- Not finished yet


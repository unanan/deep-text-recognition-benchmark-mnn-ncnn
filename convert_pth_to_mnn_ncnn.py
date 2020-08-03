import os
import torch
import torch.nn as nn
import torch.onnx
import onnx
import fire

from inference.pytorch_py.model import Model
from inference.pytorch_py.utils import AttnLabelConverter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Global Parameters
#TODO: Unsafe global parameters
imgW=100
imgH=32
character = "0123456789abcdefghijklmnopqrstuvwxyz"
converter = AttnLabelConverter(character)
batch_max_length = 25
num_class = len(converter.character)
model_path = "./inference/pytorch_py/TPS-ResNet-BiLSTM-Attn.pth"
output_path = "./inference/mnn_cpp/TPS-ResNet-BiLSTM-Attn.mnn"

target = "mnn"


def pth2mnn(model_path,output_path):
    model_root_name = os.path.splitext(model_path)[0]
    model = Model(imgW, imgH, num_class, batch_max_length)
    model = model.to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    image_tensor = torch.randn(1, 1, imgH, imgW).to(device)
    text_for_pred = torch.LongTensor(1, batch_max_length + 1).fill_(0).to(device)

    torch.onnx.export(model, (image_tensor, text_for_pred), f"{model_root_name}.onnx", verbose=True)
    print("PTH converted to ONNX finished!")

    # model = onnx.load(f"{model_root_name}.onnx")
    #
    # onnx.checker.check_model(model)
    # onnx.helper.printable_graph(model.graph)
    # print("PTH converted to ONNX finished!")
    if os.path.isdir(output_path):
        if os.path.exists(output_path):
            os.system(f"MNN/build/MNNConvert -f ONNX --modelFile {model_root_name}.onnx --MNNModel {os.path.join(output_path,os.path.split(model_root_name)[-1])}.mnn --bizCode MNN")
        else:
            print(f"Error: {output_path} not exists! ")
            return
    else:
        os.system(f"MNN/build/MNNConvert -f ONNX --modelFile {model_root_name}.onnx --MNNModel {output_path} --bizCode MNN")

    print("ONNX converted to MNN finished!")
    return


def pth2ncnn(model_path):
    raise NotImplementedError()


def pth2model(target=target,model_path=model_path, output_path=output_path):
    if target=="mnn":
        pth2mnn(model_path,output_path)
    elif target=="ncnn":
        pth2ncnn(model_path)
    else:
        raise ValueError("Must assign target as 'mnn' or 'ncnn'.")


if __name__ == '__main__':
    fire.Fire(pth2model)
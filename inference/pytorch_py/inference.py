import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F
import torchvision.transforms as transforms

import cv2
from PIL import Image

from utils import AttnLabelConverter
from model import Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


model = None
converter = None
batch_max_length=25
imgW=100
imgH=32
transform = None
character = "0123456789abcdefghijklmnopqrstuvwxyz"
model_path = "./TPS-ResNet-BiLSTM-Attn.pth"

class ResizeNormalize(object):

    def __init__(self, size, interpolation=Image.BICUBIC):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        img = img.resize(self.size, self.interpolation)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img


def init_model(model_path):
    global model,converter,batch_max_length,imgW,imgH,transform

    cudnn.benchmark = True
    cudnn.deterministic = True

    """ model configuration """
    converter = AttnLabelConverter(character)
    num_class = len(converter.character)

    model = Model(imgW, imgH, num_class,batch_max_length)
    model = model = torch.nn.DataParallel(model).to(device)

    # load model
    model.load_state_dict(torch.load(model_path, map_location=device))


    transform = ResizeNormalize((imgW, imgH))


def tps_resnet_bilstm_attn(cv_img):
    global model,converter,batch_max_length,imgW,imgH,transform

    # predict
    model.eval()
    with torch.no_grad():
        image = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)).convert('L')

        image_tensor = transform(image).unsqueeze(0).to(device)

        # For max length prediction
        length_for_pred = torch.IntTensor([batch_max_length] * 1).to(device)  # *bs
        text_for_pred = torch.LongTensor(1, batch_max_length + 1).fill_(0).to(device)  #bs,


        preds = model(image_tensor, text_for_pred, is_train=False)

        # select max probabilty (greedy decoding) then decode index to character
        _, preds_index = preds.max(2)
        preds_str = converter.decode(preds_index, length_for_pred)


        preds_prob = F.softmax(preds, dim=2)
        preds_max_prob, _ = preds_prob.max(dim=2)

        pred, pred_max_prob = preds_str[0], preds_max_prob[0]
        try:
            pred_EOS = pred.find('[s]')
            pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
            pred_max_prob = pred_max_prob[:pred_EOS]

            # calculate confidence score (= multiply of pred_max_prob)
            confidence = pred_max_prob.cumprod(dim=0)[-1]
            confidence = confidence.item()

        except:
            print("Error")
            pred = ""
            confidence = 0.0

    return pred, confidence


if __name__ == '__main__':
    import time

    start_init = time.time()
    init_model(model_path)
    end_init = time.time()
    print(f"Init success! Costs time: {end_init-start_init} s")

    cv_img = cv2.imread("./underground_0.png")
    start_inference = time.time()
    pred, confidence = tps_resnet_bilstm_attn(cv_img)
    end_inference = time.time()
    print(f"pred:{pred}, confidence:{confidence}, costs time: {end_inference-start_inference} s.")
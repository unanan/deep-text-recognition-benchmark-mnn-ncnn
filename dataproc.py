import os
import glob

img_rootpath = "/media/data4T2/meterdataset/digital_dataset/otherdataset/val"
labels = []
# with open("/media/data4T2/meterdataset/digital_dataset/otherdataset/train.txt",'w') as f:
for imgpath in glob.glob(os.path.join(img_rootpath, "*.*")):
    imgnameext = os.path.split(imgpath)[-1]
    imgname, ext = os.path.splitext(imgnameext)

    label = imgname.split("_")[0]
    labels.extend(list(label))
    labels = list(set(labels))
    # f.write(f"train/{imgnameext}\t{label}\n")
# f.close()
# print(labels)
print(''.join(labels))
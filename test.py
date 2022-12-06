import cv2
import cupy as np
import os

def get_test_images(path, filename, image_h, image_w):
    with open(os.path.join(path, filename)) as file:
        filelist = file.readlines()
    img_tensor = np.zeros((len(filelist), 3, image_h, image_w))
    label_tensor = np.zeros((len(filelist),))
    for i in range(len(filelist)):
        data = filelist[i].strip()
        image = cv2.imread(os.path.join(path, data))
        image = cv2.resize(image, dsize=(image_h, image_w), interpolation=cv2.INTER_AREA)
        label = data.split("/")[0]
        image = image[:, :, ::-1].astype(np.float32)
        img_tensor[i] = np.asarray(image.transpose(2, 0, 1))
        label_tensor[i] = int(label)
    return img_tensor, label_tensor


def test(net, path, filename, image_h=64, image_w=64):
    net.eval()
    images, labels = get_test_images(path, filename, image_h, image_w)
    infers = np.zeros((images.shape[0],), dtype=np.int32)
    for i in range(images.shape[0]):
        infers[i] = int(net.inference(images[i].reshape(1, 3, image_h, image_w)))
    return np.sum(infers == (labels - 1)) / infers.shape[0]

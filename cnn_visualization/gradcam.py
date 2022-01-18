from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
import os
import glob
import tqdm
import cv2
import skimage.measure as measure
from model.deepgrading import BiResNet

from cnn_visualization.misc_fucntion import save_class_activation_images, preprocess_image
from cnn_visualization.misc_fucntion import get_mask, get_and_save_nerve_gradient

"""
When you need to code your own model using this script, you'd better change the module name (feature, classifier)
in CamExtractor()
    Usage:
        target_example=0 # the target class index
        original_image=*
        preprocessed_img=*
        target_class=*
        file_name_to_save=*
        pretrained_model=*
        grad_cam=GradCam(pretrained_model, target_layer=11)
        cam=grad_cam.generate_cam(preprocessed_img, target_class)
        # save the mask
        save_class_activation_images(original_image, cam, file_name_to_save)
"""


def generate_roi(cam, imgs, labels):
    """
    The function is aimed to find the cam region which we desire to crop
    :param features: the cnn features
    :return: a list of coordinates of each input tensor
    """
    size_upsample = (224, 224)
    heatmap = binary_img(cv2.resize(cam, size_upsample), threshold=0.5)

    max_region = findMaxConnectedComponent(heatmap)
    max_region = np.uint8(max_region)
    # find the index of the largest region
    contours, hierarchy = cv2.findContours(max_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area = []
    for j in range(len(contours)):
        area.append(cv2.contourArea(contours[j]))
    max_idx = np.argmax(area)
    # get the min rect of the corresponding contour
    x, y, w, h = cv2.boundingRect(contours[max_idx])
    cx, cy = x + w // 2, y + h // 2
    max_len = max(w, h) // 2
    x1 = cx - max_len if (cx - max_len) >= 0 else 0
    x2 = cx + max_len if (cx + max_len) <= 224 else 224
    y1 = cy - max_len if (cy - max_len) >= 0 else 0
    y2 = cy + max_len if (cy + max_len) <= 224 else 224

    img = imgs[x1:x2, y1:y2]
    # img = img.resize_(imgs.shape)
    seg = labels[x1:x2, y1:y2]
    # seg = seg.resize_(imgs.shape)

    return img, seg


def binary_img(heatmap, threshold):
    """
    performing a threshold on the cam heatmap to get the roi region
    """
    if threshold < 1:
        threshold = int(threshold * 255)
    # _, binary_heatmap = cv2.threshold(heatmap, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, binary_heatmap = cv2.threshold(heatmap, threshold, 255, cv2.THRESH_BINARY)
    return binary_heatmap


def findMaxConnectedComponent(binary_img):
    """
    find the largest connectivity component of he binary heatmap
    """
    labeled_img, num = measure.label(binary_img, neighbors=4, background=0, return_num=True)
    max_label = 0
    max_num = 0
    for i in range(1, num + 1):
        if np.sum(labeled_img == i) > max_num:
            max_num = np.sum(labeled_img == i)
            max_label = i
    # convert False/True to 0/1
    lcc = (labeled_img == max_label) * 1
    return lcc


class CamExtractor():
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None

    def save_gradients(self, grad):
        self.gradients = grad

    def forward_pass_on_convolutions(self, x):
        """
        Perform a forward pass on convolutions and hook the feature at given layer
        :param x:
        :return:
        """
        conv_output = None
        # self.model.features... One should change ("feature") the module name in the below line depending
        # on the definition of your own model
        for name, module in list(self.model._modules.items())[:-5]:
            # print(name, "---", module)
            x = module(x)
            if str(name) == self.target_layer:
                x.register_hook(self.save_gradients)
                # save the convolution output on the target layer
                conv_output = x

        return conv_output, x

    def forward_pass(self, x):
        """
        Perform full forward pass on the network
        :param x:
        :return:
        """
        # forward pass on the convolutions
        conv_output, x = self.forward_pass_on_convolutions(x)

        batch_size, C, W, H = x.size()
        xx = x.view(batch_size, C, W ** 2)
        xx = (torch.bmm(xx, torch.transpose(xx, 1, 2)) / W ** 2)
        xx = xx.view(batch_size, -1)
        xx = F.normalize(torch.sign(xx) * torch.sqrt(torch.abs(xx) + 1e-5))
        leb = self.model.fc_bi(xx)

        # the fc1 output
        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        x1 = self.model.fc1(x)

        # the attention output
        # x = x.view(batch_size, C)
        x = x + x * leb
        x2 = self.model.fc2(x)

        out = torch.cat((x1, x2), dim=1)
        out = self.model.classifier(out)

        return conv_output, out


class GradCam():
    """
    Generate class activatin map
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()
        self.extractor = CamExtractor(self.model, target_layer)

    def generate_cam(self, input_image, target_class=None):
        """
        Full forward pass;
        conv_output is the specific output of convolutions
        model_output is the final output of the model(1,1000)
        """
        conv_output, model_output = self.extractor.forward_pass(input_image)
        if target_class is None:
            target_class = np.argmax(model_output.data.cpu().numpy())
        # target for backprop
        one_hot = torch.FloatTensor(1, model_output.size()[-1]).zero_().cuda()
        one_hot[0][target_class] = 1
        # zero gradients------------------------------------------------------------------
        # self.model.features.zero_grad()
        # self.model.classifier.zero_grad()
        self.model.zero_grad()

        # backward pass with specific target----------------------------------------------
        model_output.backward(gradient=one_hot, retain_graph=True)
        # get hooked gradients
        guided_gradients = self.extractor.gradients.data.cpu().numpy()[0]
        target = conv_output.data.cpu().numpy()[0]

        # get weights form gradients
        weights = np.mean(guided_gradients, axis=(1, 2))

        cam = np.ones(target.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * target[i, :, :]
        cam = np.maximum(cam, 0)
        # Normalize to 0~1
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))
        cam = np.uint8(cam * 255)
        cam = np.uint8(Image.fromarray(cam).resize((input_image.shape[2], input_image.shape[3]), Image.ANTIALIAS)) / 255
        # ^ I am extremely unhappy with this line. Originally resizing was done in cv2 which
        # supports resizing numpy matrices with antialiasing, however,
        # when I moved the repository to PIL, this option was out of the window.
        # So, in order to use resizing with ANTIALIAS feature of PIL,
        # I briefly convert matrix to PIL image and then back.
        # If there is a more beautiful way, do not hesitate to send a PR.

        # You can also use the code below instead of the code line above, suggested by @ ptschandl
        # from scipy.ndimage.interpolation import zoom
        # cam = zoom(cam, np.array(input_image[0].shape[1:])/np.array(cam.shape))
        return cam


def load_model():
    model = torch.load('./checkpoint/BANet.pth')
    return model


if __name__ == '__main__':
    model = load_model()
    print(model)
    root_path = '/media/Data/test'
    for file in tqdm.tqdm(glob.glob(os.path.join(root_path, "img", "*.jpg"))):
        seg_file = os.path.join(root_path, "seg", os.path.basename(file)[:-4] + ".png")
        features_blobs = []
        img = Image.open(file).convert("L")
        input_image = preprocess_image(img, resize=304)
        seg = Image.open(seg_file)
        input_seg = preprocess_image(seg, resize=304)
        input = torch.cat((input_image, input_seg, input_image), dim=1)
        target_class = None
        threshold = 0.7
        output_path = root_path
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        file_name_to_save = os.path.join(output_path, "GradCAM" + str(threshold)) + "/"
        base_name = os.path.basename(file)[:-4]
        grad_cam = GradCam(model, target_layer="layer4")
        # generate cam mask
        cam = grad_cam.generate_cam(input, target_class)
        cam = cv2.resize(cam, img.size)
        save_class_activation_images(img, cam, file_name_to_save, base_name)

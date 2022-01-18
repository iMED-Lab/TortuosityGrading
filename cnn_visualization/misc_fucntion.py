import os
import copy
import numpy as np
from PIL import Image, ImageFilter
import matplotlib.cm as mpl_color_map
from torch.autograd import Variable
from torchvision import transforms

# path to save the cam activation maps
SAVE_OUTPUT_PATH = "/home/outputs/"


def convert_to_grayscale(im_as_arr):
    """
        Converts 3d image to grayscale
    Args:
        im_as_arr (numpy arr): RGB image with shape (D,W,H)
    returns:
        grayscale_im (numpy_arr): Grayscale image with shape (1,W,D)
    """
    grayscale_im = np.sum(np.abs(im_as_arr), axis=0)
    im_max = np.percentile(grayscale_im, 99)
    im_min = np.min(grayscale_im)
    grayscale_im = (np.clip((grayscale_im - im_min) / (im_max - im_min), 0, 1))
    grayscale_im = np.expand_dims(grayscale_im, axis=0)
    return grayscale_im


def save_image(img, path):
    if isinstance(img, (np.ndarray, np.generic)):
        img = format_numpy_output(img)
        img = Image.fromarray(img)
    img.save(path)


def format_numpy_output(np_arr):
    # Phase/Case 1: The np arr only has 2 dimensions
    # Result: Add a dimension at the beginning
    if len(np_arr.shape) == 2:
        np_arr = np.expand_dims(np_arr, axis=0)
    # Phase/Case 2: Np arr has only 1 channel (assuming first dim is channel)
    # Result: Repeat first channel and convert 1xWxH to 3xWxH
    if np_arr.shape[0] == 1:
        np_arr = np.repeat(np_arr, 3, axis=0)
    # Phase/Case 3: Np arr is of shape 3xWxH
    # Result: Convert it to WxHx3 in order to make it saveable by PIL
    if np_arr.shape[0] == 3:
        np_arr = np_arr.transpose(1, 2, 0)
    # Phase/Case 4: NP arr is normalized between 0-1
    # Result: Multiply with 255 and change type to make it saveable by PIL
    if np.max(np_arr) <= 1:
        np_arr = (np_arr * 255).astype(np.uint8)
    return np_arr


def apply_color_map_on_image(original_img, activaton_map, colormap_name):
    # get color map
    color_map = mpl_color_map.get_cmap(colormap_name)
    no_trans_heatmap = color_map(activaton_map)
    # change alpha channel in colormap to make sure the original image is displayed
    heatmap = copy.copy(no_trans_heatmap)
    heatmap[:, :, 3] = 0.4
    heatmap = Image.fromarray((heatmap * 255).astype(np.uint8))
    no_trans_heatmap = Image.fromarray((no_trans_heatmap * 255).astype(np.uint8))
    # apply heatmap on image
    heatmap_on_image = Image.new("RGBA", original_img.size)
    heatmap_on_image = Image.alpha_composite(heatmap_on_image, original_img.convert("RGBA"))
    heatmap_on_image = Image.alpha_composite(heatmap_on_image, heatmap)
    return no_trans_heatmap, heatmap_on_image


def preprocess_image(img, resize=304):
    preprocess = transforms.Compose([
        transforms.Resize((resize, resize)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.321, 0.224, 0.161], std=[0.262, 0.183, 0.132]),
        transforms.Normalize(mean=0.339, std=0.138),
    ])
    img_tensor = preprocess(img)
    img_variable = Variable(img_tensor.unsqueeze(0)).cuda()
    return img_variable


def get_positive_negative_saliency(gradient):
    """
    Generates postitive and negative saliency based on gradient
    :param gradient:
    :return:
    """
    positive_saliency = (np.maximum(0, gradient) / gradient.max())
    negative_saliency = (np.maximum(0, -gradient) / -gradient.min())
    return positive_saliency, negative_saliency


def save_gradient_images(gradient, file_name):
    """
    Exports the original gradient image
    :param gradient: numpy array of the gradient with shape (3,224,224)
    :param file_name: file name to be saved
    :return: None
    """
    if not os.path.exists(SAVE_OUTPUT_PATH):
        os.makedirs(SAVE_OUTPUT_PATH)
    # Normalize
    gradient = gradient - gradient.min()
    gradient /= gradient.max()
    # save image
    file_path = os.path.join(SAVE_OUTPUT_PATH, file_name + ".jpg")
    save_image(gradient, file_path)


def build_path(path):
    if not os.path.exists(path):
        os.makedirs(path)


def save_class_activation_images(original_img, activation_map, file_name, img_name):
    """
    save the cam activation on the original image
    :param original_img: original image
    :param activation_map: activation map (gray scale, 0~255)
    :param file_name: save file name
    :return: None
    """
    # gray-scale the activation map
    heatmap, heatmap_on_iamge = apply_color_map_on_image(original_img, activation_map, "jet")
    # save colored heatmap
    heatmap_file_path = os.path.join(file_name, "heatmap")
    build_path(heatmap_file_path)
    heatmap_name = os.path.join(heatmap_file_path, img_name + "_heatmap.png")
    save_image(heatmap, heatmap_name)

    # save heatmap on image
    cam_on_img_path = os.path.join(file_name, "cam_on_img")
    build_path(cam_on_img_path)
    cam_on_image_path = os.path.join(cam_on_img_path, img_name + "_cam_on_image.png")
    save_image(heatmap_on_iamge, cam_on_image_path)

    # # save gray-scale map
    # graymap_file_path = os.path.join(SAVE_OUTPUT_PATH, file_name + "_cam_grayscale.png")
    # save_image(activation_map, graymap_file_path

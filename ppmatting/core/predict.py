import cv2
import numpy as np
import paddle
import paddle.nn.functional as F

from ppmatting.utils import mkdir, estimate_foreground_ml


def reverse_transform(alpha, trans_info):
    """recover pred to origin shape"""
    for item in trans_info[::-1]:
        if item[0] == 'resize':
            h, w = item[1][0], item[1][1]
            alpha = F.interpolate(alpha, [h, w], mode='bilinear')
        elif item[0] == 'padding':
            h, w = item[1][0], item[1][1]
            alpha = alpha[:, :, 0:h, 0:w]
        else:
            raise Exception("Unexpected info '{}' in im_info".format(item[0]))
    return alpha


def preprocess(img, transforms, trimap=None):
    data = {}
    data['img'] = img
    if trimap is not None:
        data['trimap'] = trimap
        data['gt_fields'] = ['trimap']
    data['trans_info'] = []
    data = transforms(data)
    data['img'] = paddle.to_tensor(data['img'])
    data['img'] = data['img'].unsqueeze(0)
    if trimap is not None:
        data['trimap'] = paddle.to_tensor(data['trimap'])
        data['trimap'] = data['trimap'].unsqueeze((0, 1))

    return data


def predict(model,
            # model_path,
            transforms,
            image_list,
            image_dir=None,
            trimap_list=None,
            save_dir='output',
            fg_estimate=True):
    """
    predict and visualize the image_list.

    Args:
        model (nn.Layer): Used to predict for input image.
        model_path (str): The path of pretrained model.
        transforms (transforms.Compose): Preprocess for input image.
        image_list (list): A list of image path to be predicted.
        image_dir (str, optional): The root directory of the images predicted. Default: None.
        trimap_list (list, optional): A list of trimap of image_list. Default: None.
        save_dir (str, optional): The directory to save the visualized results. Default: 'output'.
    """
    # utils.utils.load_entire_model(model, model_path)
    # model.eval()

    with paddle.no_grad():
        for i, im_path in enumerate(image_list):
            from PIL import Image
            # img = Image.open(im_path)
            # img = np.asarray(img)
            img = im_path
            data = preprocess(img=img, transforms=transforms, trimap=None)

            alpha_pred = model(data)

            alpha_pred = reverse_transform(alpha_pred, data['trans_info'])
            alpha_pred = (alpha_pred.numpy()).squeeze()
            alpha_pred = (alpha_pred * 255).astype('uint8')

            cv2.imwrite("test_alpha.png", alpha_pred)

    return alpha_pred, 0

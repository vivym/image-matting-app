from typing import Optional

import numpy as np
import paddle
import paddle.nn.functional as F


def reverse_transform(alpha, trans_info):
    """recover pred to origin shape"""
    for item in trans_info[::-1]:
        if item[0] == "resize":
            h, w = item[1][0], item[1][1]
            alpha = F.interpolate(alpha, [h, w], mode="bilinear")
        elif item[0] == "padding":
            h, w = item[1][0], item[1][1]
            alpha = alpha[:, :, 0:h, 0:w]
        else:
            raise Exception(f"Unexpected info '{item[0]}' in im_info")

    return alpha


def preprocess(img, transforms, trimap=None):
    data = {}
    data["img"] = img
    if trimap is not None:
        data["trimap"] = trimap
        data["gt_fields"] = ["trimap"]
    data["trans_info"] = []
    data = transforms(data)
    data["img"] = paddle.to_tensor(data["img"])
    data["img"] = data["img"].unsqueeze(0)
    if trimap is not None:
        data["trimap"] = paddle.to_tensor(data["trimap"])
        data["trimap"] = data["trimap"].unsqueeze((0, 1))

    return data


def predict(
    model,
    transforms,
    image: np.ndarray,
    trimap: Optional[np.ndarray] = None,
):
    with paddle.no_grad():
        data = preprocess(img=image, transforms=transforms, trimap=None)

        alpha = model(data)

        alpha = reverse_transform(alpha, data["trans_info"])
        alpha = alpha.numpy().squeeze()

        if trimap is not None:
            alpha[trimap == 0] = 0
            alpha[trimap == 255] = 1.

    return alpha

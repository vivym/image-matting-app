from hashlib import sha1
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
from PIL import Image

from paddleseg.cvlibs import manager, Config
from paddleseg.utils import load_entire_model

manager.BACKBONES._components_dict.clear()
manager.TRANSFORMS._components_dict.clear()

import ppmatting as ppmatting
from ppmatting.core import predict
from ppmatting.utils import estimate_foreground_ml

model_names = [
    "modnet-mobilenetv2",
    "ppmatting-512",
    "ppmatting-1024",
    "ppmatting-2048",
    "modnet-hrnet_w18",
    "modnet-resnet50_vd",
]
model_dict = {
    name: None
    for name in model_names
}

last_result = {
    "cache_key": None,
    "algorithm": None,
}


def image_matting(
    image: np.ndarray,
    result_type: str,
    bg_color: str,
    algorithm: str,
    morph_op: str,
    morph_op_factor: float,
) -> np.ndarray:
    image = np.ascontiguousarray(image)
    cache_key = sha1(image).hexdigest()
    if cache_key == last_result["cache_key"] and algorithm == last_result["algorithm"]:
        alpha = last_result["alpha"]
    else:
        cfg = Config(f"configs/{algorithm}.yml")
        if model_dict[algorithm] is not None:
            model = model_dict[algorithm]
        else:
            model = cfg.model
            load_entire_model(model, f"models/{algorithm}.pdparams")
            model.eval()
            model_dict[algorithm] = model

        transforms = ppmatting.transforms.Compose(cfg.val_transforms)

        alpha = predict(
            model,
            transforms=transforms,
            image=image,
        )
        last_result["cache_key"] = cache_key
        last_result["algorithm"] = algorithm
        last_result["alpha"] = alpha

    alpha = (alpha * 255).astype(np.uint8)
    kernel = np.ones((5, 5), np.uint8)
    if morph_op == "dilate":
        alpha = cv2.dilate(alpha, kernel, iterations=int(morph_op_factor))
    else:
        alpha = cv2.erode(alpha, kernel, iterations=int(morph_op_factor))
    alpha = (alpha / 255).astype(np.float32)

    image = (image / 255.0).astype("float32")
    fg = estimate_foreground_ml(image, alpha)

    if result_type == "Remove BG":
        result = np.concatenate((fg, alpha[:, :, None]), axis=-1)
    elif result_type == "Replace BG":
        bg_r = int(bg_color[1:3], base=16)
        bg_g = int(bg_color[3:5], base=16)
        bg_b = int(bg_color[5:7], base=16)

        bg = np.zeros_like(fg)
        bg[:, :, 0] = bg_r / 255.
        bg[:, :, 1] = bg_g / 255.
        bg[:, :, 2] = bg_b / 255.

        result = alpha[:, :, None] * fg + (1 - alpha[:, :, None]) * bg
        result = np.clip(result, 0, 1)
    else:
        result = alpha

    return result


def main():
    images_path = Path("images")
    if not images_path.exists():
        images_path.mkdir()

    with gr.Blocks() as app:
        gr.Markdown("Image Matting Powered By AI")

        with gr.Row(variant="panel"):
            image_input = gr.Image()
            image_output = gr.Image()

        with gr.Row(variant="panel"):
            result_type = gr.Radio(
                label="Mode",
                show_label=True,
                choices=[
                    "Remove BG",
                    "Replace BG",
                    "Generate Mask",
                ],
                value="Remove BG",
            )
            bg_color = gr.ColorPicker(
                label="BG Color",
                show_label=True,
                value="#000000",
            )
            algorithm = gr.Dropdown(
                label="Algorithm",
                show_label=True,
                choices=model_names,
                value="modnet-hrnet_w18"
            )

        with gr.Row(variant="panel"):
            morph_op = gr.Radio(
                label="Post-process",
                show_label=True,
                choices=[
                    "Dilate",
                    "Erode",
                ],
                value="Dilate",
            )

            morph_op_factor = gr.Slider(
                label="Factor",
                show_label=True,
                minimum=0,
                maximum=20,
                value=0,
                step=1,
            )

        run_button = gr.Button("Run")

        run_button.click(
            image_matting,
            inputs=[
                image_input,
                result_type,
                bg_color,
                algorithm,
                morph_op,
                morph_op_factor,
            ],
            outputs=image_output,
        )

    app.launch(share=True)


if __name__ == "__main__":
    main()

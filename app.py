import cv2
import requests
import gradio as gr
import numpy as np
from PIL import Image

from paddleseg.cvlibs import manager, Config
from paddleseg.utils import load_entire_model

manager.BACKBONES._components_dict.clear()
manager.TRANSFORMS._components_dict.clear()

import ppmatting as ppmatting
from ppmatting.core import predict

model_names = [
    "modnet-mobilenetv2",
]
model_dict = {
    name: None
    for name in model_names
}


def modnet_mobilenetv2_matting(image: np.ndarray, algo: str, slider: float) -> np.ndarray:
    print(algo, slider)

    cfg = Config("./modnet-mobilenetv2.yml")
    if model_dict["modnet-mobilenetv2"] is not None:
        model = model_dict["modnet-mobilenetv2"]
    else:
        model = cfg.model
        load_entire_model(model, "models/modnet-mobilenetv2.pdparams")
        model.eval()
        model_dict["modnet-mobilenetv2"] = model

    transforms = ppmatting.transforms.Compose(cfg.val_transforms)

    alpha, fg = predict(
        model,
        transforms=transforms,
        # image=image,
        image_list=[image],
        fg_estimate=True,
    )
    # rgba = np.concatenate((fg, alpha[:, :, None]), axis=-1)

    return image


def main():
    with gr.Blocks() as app:
        gr.Markdown("智能抠图")
        with gr.Row():
            image_input = gr.Image()
            image_output = gr.Image()

        with gr.Row():
            alg = gr.Dropdown(
                choices=[
                    "modnet-mobilenetv2"
                ],
                value="modnet-mobilenetv2"
            )
            slider = gr.Slider(
                minimum=-1.,
                maximum=1.,
                value=0.,
                step=0.1,
            )

        image_button = gr.Button("Run")

        image_button.click(
            modnet_mobilenetv2_matting,
            inputs=[
                image_input,
                alg,
                slider,
            ],
            outputs=image_output,
        )

    app.launch()


if __name__ == "__main__":
    main()

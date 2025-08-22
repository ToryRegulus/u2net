import numpy as np
import onnxruntime as ort
import torch
from PIL import Image
from pillow_heif import register_heif_opener
from torchvision import transforms

from data_loader import Normalize, Resize


def renormalize(pred: np.ndarray) -> np.ndarray:
    value_max = np.max(pred)
    value_min = np.min(pred)
    pred = (pred - value_min) / (value_max - value_min)
    pred = pred * 255
    return pred


def load_image(file_path: str) -> np.ndarray:
    image = np.array(Image.open(file_path))
    image = image.transpose((2, 0, 1))  # HWC -> CHW
    image = image.astype(np.float32)
    image = np.expand_dims(image, 0)
    return image


def main(weight_name: str, device: str = "CPU") -> None:
    if device == "GPU":
        providers = ["CUDAExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    ort_session = ort.InferenceSession(
        f"weight/{weight_name}.onnx", sess_options, providers=providers
    )
    input_name = ort_session.get_inputs()[0].name
    output_name = ort_session.get_outputs()[0].name

    image_orin = load_image("./test.jpg")

    transform = transforms.Compose([Resize(288), Normalize()])
    image_tensor = torch.from_numpy(image_orin)
    image_tensor, _ = transform((image_tensor, None))
    image = image_tensor.numpy()

    pred_lst = ort_session.run([output_name], {input_name: image})
    pred = pred_lst[0][0, 0, :, :]  # type: ignore
    pred = 1 / (1 + np.exp(-pred))  # sigmoid
    pred = renormalize(pred)
    pred = pred.astype(np.uint8)

    mask = Image.fromarray(pred)
    mask = mask.resize((image_orin.shape[3], image_orin.shape[2]))
    mask = np.array(mask).astype(np.float32) / 255.0
    mask = np.expand_dims(mask, axis=-1)

    image_orin = image_orin.squeeze(0).transpose(1, 2, 0)
    background = np.ones_like(image_orin, dtype=np.uint8) * 255

    res = (image_orin * mask + background * (1 - mask)).astype(np.uint8)
    res = Image.fromarray(res)
    res.save("res.jpg")


if __name__ == "__main__":
    register_heif_opener()

    main("u2net-0.044858")

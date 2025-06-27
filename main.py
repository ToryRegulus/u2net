from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from pillow_heif import register_heif_opener
from torch import Tensor
from torchvision import transforms

from data_loader import Resize, Normalize
from model.u2net import U2NET


def renormalize(pred):
    value_max = torch.max(pred)
    value_min = torch.min(pred)

    pred = (pred - value_min) / (value_max - value_min)
    pred = pred * 255

    return pred


def load_image(file_path: str) -> Tensor:
    image = np.array(Image.open(file_path))
    image = image.transpose((2, 0, 1))
    image = torch.from_numpy(image).float()
    image.unsqueeze_(0)

    return image


def main(model_name: str):
    cfg = dict()
    if model_name == "u2net":
        cfg = yaml.safe_load(Path("model/u2net_full.yaml").read_text())
    elif model_name == "u2netp":
        cfg = yaml.safe_load(Path("model/u2net_lite.yaml").read_text())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = U2NET(cfg=cfg)
    net.load_state_dict(torch.load("weight/u2net-0.020238.pth"))
    net.to(device)
    net.eval()

    image_orin = load_image("./test.jpg")

    image = image_orin.to(device)
    transform = transforms.Compose([
        Resize(320),
        Normalize()
    ])
    image, _ = transform((image, None))

    with torch.no_grad():
        pred_lst = net(image)
        pred_lst = [torch.sigmoid(x) for x in pred_lst]
        pred = pred_lst[0][:, 0, :, :]
        pred = renormalize(pred)
        pred = pred.to(torch.uint8)
        pred = transforms.Resize((image_orin.shape[2], image_orin.shape[3]))(pred)

        if pred.device != "cpu":
            label = pred.to("cpu")
        else:
            label = pred

        label_np = label.squeeze().numpy()
        image_np = image_orin.squeeze().permute(1, 2, 0).numpy()
        image_np = image_np.astype(np.uint8)
        mask = (label_np > 0).astype(np.uint8)
        mask = np.stack([mask] * 3, axis=-1)
        background = np.ones_like(image_np) * 255

        res = image_np * mask + background * (1 - mask)
        res = Image.fromarray(res)

        res.save("res.png")


if __name__ == '__main__':
    register_heif_opener()

    main("u2net")

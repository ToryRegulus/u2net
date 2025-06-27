from pathlib import Path

import torch
import yaml
from torch import Tensor, nn, optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torchvision import transforms

from data_loader import Normalize, RandomFlipCrop, Resize, U2NETDataSet
from model.u2net import U2NET


def set_device() -> torch.device:
    """
    选择计算设备。

    Returns:
        device: 计算设备。
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    return device


def multi_bce_loss(
    sup0: Tensor,
    sup1: Tensor,
    sup2: Tensor,
    sup3: Tensor,
    sup4: Tensor,
    sup5: Tensor,
    sup6: Tensor,
    label: Tensor,
) -> tuple[Tensor, Tensor]:
    """
    多层级的二元交叉熵损失，输入为U2Net的每一层特征图，返回一个由融合图损失与所有图损失构成的一个元组。
    """
    bce_loss = nn.BCEWithLogitsLoss()

    loss0: Tensor = bce_loss(sup0, label)
    loss1: Tensor = bce_loss(sup1, label)
    loss2: Tensor = bce_loss(sup2, label)
    loss3: Tensor = bce_loss(sup3, label)
    loss4: Tensor = bce_loss(sup4, label)
    loss5: Tensor = bce_loss(sup5, label)
    loss6: Tensor = bce_loss(sup6, label)

    loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6

    return loss0, loss


def mkdirs(img_dir: Path, label_dir: Path, weight_dir: Path) -> None:
    """
    创建训练所需的文件夹。

    Args:
        img_dir (Path): 训练所需的图像文件夹。
        label_dir (Path): 训练所需的标签文件夹。
        weight_dir (Path): 训练结果权重文件夹。
    """
    img_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    weight_dir.mkdir(parents=True, exist_ok=True)

    missing = []
    if not any(img_dir.iterdir()):
        missing.append("img")
    if not any(label_dir.iterdir()):
        missing.append("label")

    assert (
        not missing
    ), f"训练目录下的以下子目录为空（必须包含文件）: {', '.join(missing)}"


def train(model_name: str, epoch_num: int) -> None:
    """
    训练过程。

    Args:
        model_name (str): 模型名字。
        epoch_num (int): 训练轮次。
    """
    # ----- 数据准备 -----
    img_dir = Path() / "train_data" / "img"
    label_dir = Path() / "train_data" / "label"
    weight_dir = Path() / "weight"
    mkdirs(img_dir, label_dir, weight_dir)

    img_paths = list(img_dir.glob(f"*.jpg"))
    label_paths = list(label_dir.glob(f"*.png"))
    img_dict = {p.stem: p for p in img_paths}
    label_dict = {p.stem: p for p in label_paths}
    common_stems = sorted(set(img_dict.keys()) & set(label_dict.keys()))
    img_lst = [img_dict[stem] for stem in common_stems]
    label_lst = [label_dict[stem] for stem in common_stems]

    # ----- 模型准备 -----
    cfg = dict()
    if model_name == "u2net":
        cfg = yaml.safe_load(Path("model/u2net_full.yaml").read_text())
    elif model_name == "u2netp":
        cfg = yaml.safe_load(Path("model/u2net_lite.yaml").read_text())
    net = U2NET(cfg=cfg)
    net.train()

    optimizer = optim.Adam(net.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-08)
    device = set_device()
    net.to(device)

    # ----- 载入DataLoader -----
    image_dataset = U2NETDataSet(
        img_lst,
        label_lst,
        transform=transforms.Compose(
            [Resize(320), RandomFlipCrop(0.5, 288), Normalize()]
        ),
    )

    dataloader = DataLoader(image_dataset, batch_size=16, shuffle=True, num_workers=1)
    scaler = GradScaler()

    # ----- 训练 -----
    best_loss = 1
    for epoch in range(epoch_num):
        epoch_loss = 0
        epoch_target_loss = 0
        iter_num = 0

        for idx, (img, label) in enumerate(dataloader):
            img = img.to(device)
            label = label.to(device)
            optimizer.zero_grad()

            with autocast(device_type="cuda"):
                sup0, sup1, sup2, sup3, sup4, sup5, sup6 = net(img)
                loss_fuse, loss = multi_bce_loss(
                    sup0, sup1, sup2, sup3, sup4, sup5, sup6, label
                )

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                epoch_loss += loss.item()
                epoch_target_loss += loss_fuse.item()
                iter_num += 1

        avg_loss = epoch_loss / iter_num

        if (epoch > epoch_num - 100) and (avg_loss < best_loss):
            torch.save(
                net.state_dict(), Path(f"weight/{model_name}-{avg_loss:.6f}.pth")
            )

            old_weight_path = Path(f"weight/{model_name}-{best_loss:.6f}.pth")
            if old_weight_path.exists():
                old_weight_path.unlink()

            best_loss = avg_loss

        print(
            f"Epoch: {epoch + 1}/{epoch_num}, "
            f"loss: {avg_loss:.6f}, "
            f"fuse_loss: {(epoch_target_loss / iter_num):.6f}"
        )


if __name__ == "__main__":
    model_name = "u2net"  # u2net/u2netp
    epoch_num = 1000

    train(model_name, epoch_num=epoch_num)

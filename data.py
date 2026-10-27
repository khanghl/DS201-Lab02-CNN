import os
import zipfile
from collections import Counter
from pathlib import Path
import gdown

from PIL import Image
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
DEFAULT_WORKERS = 0 if os.name == "nt" else 2
VINAFOOD_GDRIVE_ID = "1UpZOf0XlwvB4rKpyZ35iwTA8oWHqDBbR"


def rgb_loader(path: str) -> Image.Image:
    #Doc anh RGB an toan, xu ly anh Palette co kenh trong suot (trnah PIL UserWarning)
    with open(path, "rb") as f:
        img = Image.open(f)
        if img.mode == "P" and "transparency" in img.info:
            return img.convert("RGBA").convert("RGB")
        return img.convert("RGB")


def ensure_vinafood21(data_root: str | Path = "VinaFood21") -> Path:
    #Kiem tra, tai va unzip VinaFood21 tu Google Drive neu chua co
    target = Path(data_root)
    train_dir = target / "train"
    test_dir = target / "test"

    #kiem tra xem da co du lieu hay chua: co -> bo qua buoc tai va giai nen
    parent_dir = target.parent if target.parent != Path("") else Path(".")
    for cand in (target, parent_dir / "VinaFood21"):
        if (cand / "train").is_dir() and (cand / "test").is_dir():
            return cand

    #Tao thu muc chua du lieu de luu zip va giai nen
    parent_dir.mkdir(parents=True, exist_ok=True)
    zip_path = parent_dir / "VinaFood21.zip"

    #Tai file zip tu Google Drive neu chua co
    if not zip_path.is_file():
        download = gdown.download(
            id=VINAFOOD_GDRIVE_ID,
            output=str(zip_path),
            quiet=False,
        )
        if download is None or not zip_path.is_file():
            raise RuntimeError("Khong the tai VinaFood21 tu Google Drive")

    #Giai nen file zip
    print(f"Dang giai nen {zip_path} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(parent_dir)

    #Xac thuc cau truc thu muc sau khi giai nen (VinaFood21 tren Linux phan biet hoa/thuong)
    for cand in (target, parent_dir / "VinaFood21"):
        if (cand / "train").is_dir() and (cand / "test").is_dir():
            return cand
    
    raise FileNotFoundError(f"Khong tim thay thu muc train/test trong {target}" 
                            f"Vui long kiem tra file zip VinaFood21 tren Google Drive")


def get_class_distribution(data_root: str | Path):
    #Thong ke so luong anh theo tung lop tren tap train va test
    root = Path(data_root)
    train_ds = datasets.ImageFolder(root / "train", loader=rgb_loader)
    test_ds = datasets.ImageFolder(root / "test", loader=rgb_loader)
    assert train_ds.classes == test_ds.classes, "Thu tu lop giua train và test khong khop!"

    train_counts = Counter(y for _, y in train_ds.samples)
    test_counts = Counter(y for _, y in test_ds.samples)
    classes = train_ds.classes
    return (
        classes,
        [train_counts[i] for i in range(len(classes))],
        [test_counts[i] for i in range(len(classes))],
    )


def build_transforms(img_size: int, augment="full", mean=MEAN, std=STD):
    """Xay dung hai bo transform rieng biet: chi tap train co augmentation, val/test giu nguyen.

    Tham so augment:
    - False / 'none': khong augmentation (Resize + ToTensor + Normalize) -> dung o Bai 3 doi chung
    - flip: chi lat ngang (Resize + RandomHorizontalFlip) -> cau hinh Bai 1 & Bai 2
    - True / full: day du (RandomResizedCrop + Flip + Rotation + ColorJitter) -> Bai 3 & BBai 4
    """
    eval_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    if augment in (False, None, "none"):
        train_tf = eval_tf
    elif augment == "flip":
        train_tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    elif augment in (True, "full"):
        train_tf = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:
        raise ValueError(f"Che do augment ko hop le: {augment!r}")

    return train_tf, eval_tf


def build_train_val_loaders(
    data_root: str | Path,
    img_size: int,
    batch_size: int = 32,
    val_ratio: float = 0.1,
    augment="full",
    mean=MEAN,
    std=STD,
    num_workers: int = DEFAULT_WORKERS,
    seed: int = 42,
):
    #Tao DataLoader cho train va val voi cung mot phep chia ngau nhien co dinh theo seed
    root = Path(data_root)
    train_tf, eval_tf = build_transforms(img_size, augment=augment, mean=mean, std=std)

    if (root / "val").is_dir():
        train_ds = datasets.ImageFolder(root / "train", transform=train_tf, loader=rgb_loader)
        val_ds = datasets.ImageFolder(root / "val", transform=eval_tf, loader=rgb_loader)
        classes = train_ds.classes
    else:
        #Khong co thu muc val thi tach mot phan tu train voi seed co dinh cho ca 4 bai
        #Hai ImageFolder tro cung thu muc de tap val khong bi dinh augmentation cua train
        full_train = datasets.ImageFolder(root / "train", transform=train_tf, loader=rgb_loader)
        full_eval = datasets.ImageFolder(root / "train", transform=eval_tf, loader=rgb_loader)
        g = torch.Generator().manual_seed(seed)
        idx = torch.randperm(len(full_train), generator=g).tolist()
        n_val = int(len(idx) * val_ratio)
        train_ds = Subset(full_train, idx[n_val:])
        val_ds = Subset(full_eval, idx[:n_val])
        classes = full_train.classes

    pin = torch.cuda.is_available()
    train_dl = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin,
    )
    val_dl = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin,
    )
    return train_dl, val_dl, classes


def build_test_loader(
    data_root: str | Path,
    img_size: int,
    batch_size: int = 32,
    mean=MEAN,
    std=STD,
    num_workers: int = DEFAULT_WORKERS,
):
    #Tao DataLoader cho tap test (khong shuffle, khong augmentation)
    _, eval_tf = build_transforms(img_size, augment=False, mean=mean, std=std)
    test_ds = datasets.ImageFolder(Path(data_root) / "test", transform=eval_tf, loader=rgb_loader)
    test_dl = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
    )
    return test_dl, test_ds.classes

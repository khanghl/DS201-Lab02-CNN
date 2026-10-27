import torch
import torch.nn as nn


def n_params(module: nn.Module) -> int:
    #Tong so tham so cua module
    return sum(p.numel() for p in module.parameters())


def n_trainable_params(module: nn.Module) -> int:
    #Tong so tham so co requires_grad=True cua module
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


#Bai 1: LeNet-5 dieu chinh cho VinaFood21 (64x64x3 -> 21 lop, 339,421 tham so)
class LeNet5(nn.Module):
    def __init__(self, in_ch: int = 3, img: int = 64, num_classes: int = 21, pool_type: str = "avg"):
        super().__init__()
        if pool_type == "avg":
            pool_cls = nn.AvgPool2d
        elif pool_type == "max":
            pool_cls = nn.MaxPool2d
        else:
            raise ValueError(f"pool_type phai la 'avg' or 'max', nhan duoc: {pool_type!r}")

        self.pool_type = pool_type
        self.c1 = nn.Conv2d(in_ch, 6, kernel_size=5) #64 -> 60
        self.s2 = pool_cls(2, 2) #60 -> 30
        self.c3 = nn.Conv2d(6, 16, kernel_size=5) #30 -> 26
        self.s4 = pool_cls(2, 2) #26 -> 13

        n_flat = self._flat_dim(in_ch, img) # 16 * 13 * 13 = 2,704
        self.f5 = nn.Linear(n_flat, 120)
        self.f6 = nn.Linear(120, 84)
        self.out = nn.Linear(84, num_classes)

    def _flat_dim(self, in_ch: int, img: int) -> int:
        #Do so chieu sau S4 bang mot forward gia, tranh tinh nham sai
        with torch.no_grad():
            z = torch.zeros(1, in_ch, img, img)
            z = self.s4(self.c3(self.s2(self.c1(z))))
        return int(z.flatten(1).shape[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.s2(torch.relu(self.c1(x)))
        x = self.s4(torch.relu(self.c3(x)))
        x = x.flatten(1)
        x = torch.relu(self.f5(x))
        x = torch.relu(self.f6(x))
        return self.out(x)


#Bai 2: GoogLeNet rut gon (4 InceptionBlock + Global Average Pooling, 64x64)
class InceptionBlock(nn.Module):
    #Khoi Inception 4 nhanh song song
    def __init__(self, in_ch: int, f1: int, f3r: int, f3: int, f5r: int, f5: int, fp: int):
        super().__init__()
        self.config = (f1, f3r, f3, f5r, f5, fp)
        self.out_channels = f1 + f3 + f5 + fp  #f3r & f5r la trung gian, ko tinh vao output

        self.b1 = nn.Sequential(
            nn.Conv2d(in_ch, f1, kernel_size=1),
            nn.ReLU(),
        )
        self.b2 = nn.Sequential(
            nn.Conv2d(in_ch, f3r, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(f3r, f3, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.b3 = nn.Sequential(
            nn.Conv2d(in_ch, f5r, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(f5r, f5, kernel_size=5, padding=2),
            nn.ReLU(),
        )
        self.b4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_ch, fp, kernel_size=1),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], dim=1)


class GoogLeNetMini(nn.Module):
    #GoogLeNet rut gon voi 4 InceptionBlock va AdaptiveAvgPool2d(1) cho anh 64x64
    def __init__(self, in_ch: int = 3, num_classes: int = 21, dropout: float = 0.2):
        super().__init__()
        #stem: 64x64x3 -> 32x32x32
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        #4 inception blocks (ko can auxiliary classifier)
        self.inc1 = InceptionBlock(32,  16, 16,  24,  4,  8,  8) #out = 16 + 24 + 8 + 8 = 56
        self.inc2 = InceptionBlock(56,  32, 32,  48,  8, 16, 16) #out = 32 + 48 + 16 + 16 = 112
        self.pool1 = nn.MaxPool2d(2, 2) #32x32 -> 16x16

        self.inc3 = InceptionBlock(112, 64, 48,  96, 16, 32, 32) #out = 64 + 96 + 32 + 32 = 224
        self.pool2 = nn.MaxPool2d(2, 2) #16x16 -> 8x8

        self.inc4 = InceptionBlock(224, 64, 64, 128, 16, 32, 32) #out = 64 + 128 + 32 + 32 = 256

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(self.inc4.out_channels, num_classes)

    @torch.no_grad()
    def inspect_blocks(self, x: torch.Tensor):
        #Doi chieu so channel tinh theo cong thuc (f1+f3+f5+fp) va tensor.shape thuc te
        rows = []
        x = self.stem(x)
        rows.append(("stem", 32, tuple(x.shape)))
        for name, mod in [
            ("inc1", self.inc1),
            ("inc2", self.inc2),
            ("pool1", self.pool1),
            ("inc3", self.inc3),
            ("pool2", self.pool2),
            ("inc4", self.inc4),
        ]:
            x = mod(x)
            expected = mod.out_channels if isinstance(mod, InceptionBlock) else x.shape[1]
            rows.append((name, expected, tuple(x.shape)))
        x = self.gap(x).flatten(1)
        rows.append(("gap", x.shape[1], tuple(x.shape)))
        out = self.fc(x)
        rows.append(("fc", out.shape[1], tuple(out.shape)))
        return rows

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.inc1(x)
        x = self.inc2(x)
        x = self.pool1(x)
        x = self.inc3(x)
        x = self.pool2(x)
        x = self.inc4(x)
        x = self.gap(x).flatten(1)
        x = self.drop(x)
        return self.fc(x)


#Bai 3: ResNet-18 day du [2, 2, 2, 2] BasicBlock (11.187.285 tham so)
class BasicBlock(nn.Module):
    #BasicBlock chuan cua ResNet18
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

        self.shortcut = nn.Identity()
        if stride != 1 or in_ch != out_ch: #can projection de khop C, H, W
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x) #cong truoc
        return torch.relu(out) #ReLU sau


class ResNet18(nn.Module):
    #ResNet18 [2,2,2,2] train tu dau (21 lop output -> 11.187.285 tham so)
    def __init__(self, in_ch: int = 3, num_classes: int = 21):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_stage(64, 64, num_blocks=2, stride=1)
        self.layer2 = self._make_stage(64, 128, num_blocks=2, stride=2)
        self.layer3 = self._make_stage(128, 256, num_blocks=2, stride=2)
        self.layer4 = self._make_stage(256, 512, num_blocks=2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)

    def _make_stage(self, in_ch: int, out_ch: int, num_blocks: int, stride: int) -> nn.Sequential:
        blocks = [BasicBlock(in_ch, out_ch, stride=stride)]
        for _ in range(1, num_blocks):
            blocks.append(BasicBlock(out_ch, out_ch, stride=1))
        return nn.Sequential(*blocks)

    @torch.no_grad()
    def inspect_stages(self, x: torch.Tensor):
        #Theo doi shape va so tham so qua tung stage 
        rows = []
        x = torch.relu(self.bn1(self.conv1(x)))
        rows.append(("conv1 + bn1", tuple(x.shape), n_params(self.conv1) + n_params(self.bn1)))
        x = self.maxpool(x)
        rows.append(("maxpool", tuple(x.shape), 0))
        for name in ("layer1", "layer2", "layer3", "layer4"):
            stage = getattr(self, name)
            x = stage(x)
            rows.append((name, tuple(x.shape), n_params(stage)))
        x = self.avgpool(x).flatten(1)
        out = self.fc(x)
        rows.append(("avgpool + fc", tuple(out.shape), n_params(self.fc)))
        return rows

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x).flatten(1)
        return self.fc(x)


#Bai 4: ResNet-50 Pretrained (23,551,061 tham so)
class PretrainedResnet(nn.Module):
    """Mo hinh microsoft/resnet-50 pretrained cho ca 2 che do:
        - freeze_backbone=True: Feature extraction (chi hoc 43,029 tham so o classifier, lr=1e-3)
        - freeze_backbone=False: Fine-tuning toan mang (hoc 23,551,061 tham so, lr=1e-4)
    """

    def __init__(
        self,
        num_classes: int = 21,
        freeze_backbone: bool = False,
        model_name: str = "microsoft/resnet-50",
    ):
        super().__init__()
        from transformers import ResNetForImageClassification

        basemodel = ResNetForImageClassification.from_pretrained(model_name)
        self.resnet = basemodel.resnet
        self.classifier = nn.Linear(in_features=2048, out_features=num_classes, bias=True)

        if freeze_backbone:
            for p in self.resnet.parameters():
                p.requires_grad = False

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.resnet(images).pooler_output #[B, 2048, 1, 1]
        features = features.squeeze(-1).squeeze(-1) #[B, 2048]
        logits = self.classifier(features) #[B, 21]
        return logits


if __name__ == "__main__":
    #Kiem tra nhanh shape va tham so cho cac kien truc tu setup
    m1 = LeNet5(in_ch=3, img=64, num_classes=21, pool_type="avg")
    print(f"LeNet5 (64x64)       : logits={tuple(m1(torch.rand(2, 3, 64, 64)).shape)} | params={n_params(m1):,}")

    m2 = GoogLeNetMini(in_ch=3, num_classes=21)
    print(f"GoogLeNetMini (64x64): logits={tuple(m2(torch.rand(2, 3, 64, 64)).shape)} | params={n_params(m2):,}")

    m3 = ResNet18(in_ch=3, num_classes=21)
    print(f"ResNet18 (96x96)     : logits={tuple(m3(torch.rand(2, 3, 96, 96)).shape)} | params={n_params(m3):,}")

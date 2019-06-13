import torchvision as tv
from config import opt
from torch.utils import data

transforms=tv.transforms.Compose(
    [
        tv.transforms.Resize(opt.image_size),
        tv.transforms.CenterCrop(opt.image_size),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
)

dataset = tv.datasets.ImageFolder(opt.data_path,transform=transforms)
data_loader = data.DataLoader(
    dataset,
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=opt.num_workers,
    drop_last=True
)

# for image,label in data_loader:
#     print(label)
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage
from torch.utils.data import Dataset

from typing import Iterable, Optional, Callable
import glob


def get_manifold_image(images, im_size=(64, 64), manifold_size=(5, 5), mode='RGB'):
    assert images.shape[0] == manifold_size[0] * manifold_size[1]

    to_pil = ToPILImage()

    dst = Image.new(mode, (im_size[0] * manifold_size[0], im_size[1] * manifold_size[1]))
    for i in range(manifold_size[0]):
        dst_line = Image.new(mode, (im_size[0], im_size[1] * manifold_size[1]))
        for j in range(manifold_size[1]):
            dst_line.paste(to_pil(images[i * manifold_size[0] + j]), (0, im_size[1] * j))
        dst.paste(dst_line, (im_size[0] * i, 0))
    return dst


def square(im):
    max_size = max(im.size)
    dst = Image.new('RGB', (max_size, max_size))
    dst.paste(im, (0, 0))
    return dst


def resize(im, size=Iterable):
    return im.resize(size)


class AstroDataset(Dataset):
    def __init__(self, data_dir: str, transform:Optional[Callable] = None):
        self.image_paths = glob.glob(f"{data_dir}/*.jpg")
        self.transform = (lambda x: x) if transform is None else transform
        self.to_tensor = ToTensor()

    def __getitem__(self, item):
        im = Image.open(self.image_paths[item])
        im = self.transform(im)
        return self.to_tensor(im)

    def __len__(self):
        return len(self.image_paths)


def logging(log, path_to_save_dir):
    with open(f"{path_to_save_dir}/loss.log", 'a') as logger:
        logger.write(log + "\n")
import os
import random
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF


class KAIST_LLVIPDataset(Dataset):
    """
    Dataset class for loading unaligned/unpaired KAIST and LLVIP datasets.

    Requires two directories to host training images from domain A (infrared)
    and domain B (visible) respectively.
    """

    def __init__(self, LLVIP_PATH):
        """
        Initialize dataset class.

        Args:
            LLVIP_PATH: Path to LLVIP dataset

        """
        super(KAIST_LLVIPDataset, self).__init__()

        self.llvip_datapath = LLVIP_PATH
        self.train_num = 2000




        # Initialize paths
        self.A_paths = []  # Infrared images
        for idx in range(self.train_num):
            self.A_paths.append(os.listdir(os.path.join(LLVIP_PATH, "vis"))[idx])

        random.shuffle(self.A_paths)

        # Generate corresponding visible image paths
        self.B_paths = []
        for img_name in self.A_paths:
            if "LLVIP" in img_name:
                self.B_paths.append(img_name.replace('infrared', 'visible'))
            elif "KAIST" in img_name:
                self.B_paths.append(img_name.replace('lwir', 'visible'))
            else:
                self.B_paths.append(img_name.replace('ir', 'vis'))

        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)

        # Initialize transform
        self.transform = SyncTransforms(grayscale=True)

    def __getitem__(self, index):
        """
        Return a data point and its metadata information.

        Args:
            index (int): Random integer for data indexing

        Returns:
            tuple: (A, B, C) where
                A (tensor): Infrared image
                B (tensor): Visible image
                C (tensor): SOTA fusion result
        """
        A_path = self.A_paths[index % self.A_size]
        index_B = index % self.B_size
        B_path = self.B_paths[index_B]

        try:
            A_img = Image.open(A_path).convert('L')
            B_img = Image.open(B_path).convert('L')
        except Exception as e:
            print(e)

        # Apply synchronized transforms
        A, B= self.transform(A_img, B_img)

        return A, B

    def __len__(self):
        """Return the total number of images in the dataset."""
        return max(self.A_size, self.B_size)


class SyncTransforms:
    """
    Synchronized transformations for multiple inputs.
    Ensures same random crop coordinates are applied to all inputs.

    Args:
        crop_size (int): Size of the crop (default: 128)
        resize_size (list): Size to resize to before cropping (default: [286, 286])
        grayscale (bool): If True, convert to grayscale
        method: Interpolation method (default: Image.BICUBIC)
    """

    def __init__(self, crop_size=128, resize_size=None, grayscale=False, method=Image.BICUBIC):
        self.crop_size = crop_size
        self.resize_size = resize_size if resize_size is not None else [286, 286]
        self.grayscale = grayscale
        self.method = method

        # Normalization parameters
        self.norm_mean = (0.5,) if grayscale else (0.5, 0.5, 0.5)
        self.norm_std = (0.5,) if grayscale else (0.5, 0.5, 0.5)

    def __call__(self, *imgs):
        """
        Apply synchronized transforms to multiple inputs.

        Args:
            *imgs: Variable number of PIL Images

        Returns:
            tuple: Transformed tensors
        """
        # Ensure all inputs are PIL Images
        imgs = [img if isinstance(img, Image.Image) else TF.to_pil_image(img) for img in imgs]

        # 1. Resize
        imgs = [TF.resize(img, self.resize_size, self.method) for img in imgs]

        # 2. Generate random crop parameters
        i, j, h, w = transforms.RandomCrop.get_params(
            imgs[0], output_size=(self.crop_size, self.crop_size))

        # 3. Apply same crop parameters to all inputs
        imgs = [TF.crop(img, i, j, h, w) for img in imgs]

        # 4. Convert to tensor
        imgs = [TF.to_tensor(img) for img in imgs]

        # 5. Normalize
        imgs = [TF.normalize(img, self.norm_mean, self.norm_std) for img in imgs]

        return tuple(imgs)


def get_sync_transform(grayscale=False, crop_size=256, resize_size=None, method=Image.BICUBIC):
    """
    Factory function to create synchronized transforms.

    Args:
        grayscale (bool): If True, convert to grayscale
        crop_size (int): Size of the crop
        resize_size (list): Size to resize to before cropping
        method: Interpolation method

    Returns:
        SyncTransforms: Transform object that can be applied to multiple inputs
    """
    if resize_size is None:
        resize_size = [286, 286]

    return SyncTransforms(
        crop_size=crop_size,
        resize_size=resize_size,
        grayscale=grayscale,
        method=method
    )

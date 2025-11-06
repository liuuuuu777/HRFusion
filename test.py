# -*- coding: utf-8 -*-
import os
import time
import gc

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image

from nets.generate_model import ResnetGenerator_Encoder_decoder
from nets.FusionNet import Restormer_Encoder, Fusion_Decoder

class ImageFusionDataset(Dataset):
    def __init__(self, dataset_path, transform=None):
        """
        Custom Dataset for Image Fusion

        Args:
            dataset_path (str): Path to the dataset directory
            transform (callable, optional): Optional transform to be applied on a sample
        """
        self.ir_path = os.path.join(dataset_path, "ir")
        self.vi_path = os.path.join(dataset_path, "vis")

        # Get list of image names
        self.image_names = os.listdir(self.ir_path)
        self.transform = transform or self._default_transform()

    def _default_transform(self, grayscale=True):
        """
        Default transformation for images

        Args:
            grayscale (bool): Whether to apply grayscale normalization

        Returns:
            transforms.Compose: Transformation pipeline
        """
        transform_list = [transforms.ToTensor()]
        if grayscale:
            transform_list.append(transforms.Normalize((0.5,), (0.5,)))
        else:
            transform_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        return transforms.Compose(transform_list)

    def pad_to_divisible(self, img, k=8):
        """
        Pad image to be divisible by k using torch.nn.functional.pad

        Args:
            img (torch.Tensor): Input image tensor
            k (int): Divisibility factor

        Returns:
            tuple: Padded image, original width, original height
        """
        # If input is a PIL Image, convert to tensor
        if isinstance(img, Image.Image):
            img = transforms.ToTensor()(img)

        # Get original dimensions
        _, height, width = img.shape

        # Calculate padding
        pad_width = (k - (width % k)) % k
        pad_height = (k - (height % k)) % k

        # Pad the tensor using reflect padding
        # Pad order is: left, right, top, bottom
        padded_img = F.pad(img, (0, pad_width, 0, pad_height), mode='reflect')

        return padded_img, width, height

    def __len__(self):
        """
        Get total number of images in the dataset

        Returns:
            int: Number of images
        """
        return len(self.image_names)

    def __getitem__(self, idx):
        """
        Get transformed image pair

        Args:
            idx (int): Index of the image

        Returns:
            dict: Dictionary containing IR and VI images, original dimensions
        """
        img_name = self.image_names[idx]

        # Load and preprocess IR image
        ir_path = os.path.join(self.ir_path, img_name)
        ir_img = Image.open(ir_path).convert("L")
        ir_img, width, height = self.pad_to_divisible(ir_img)

        # Load and preprocess VI image
        vi_path = os.path.join(self.vi_path, img_name)
        vi_img = Image.open(vi_path).convert("L")
        vi_img, _, _ = self.pad_to_divisible(vi_img)

        return {
            'ir_img': ir_img,
            'vi_img': vi_img,
            'width': width,
            'height': height,
            'img_name': img_name
        }


def initialize_models(device):
    """
    Initialize and load models

    Args:
        device (torch.device): Device to load models on

    Returns:
        tuple: Initialized models
    """
    # Initialize models
    I_generate_Encoder = ResnetGenerator_Encoder_decoder().to(device)
    I_extract_Encoder = nn.DataParallel(Restormer_Encoder()).to(device)
    V_generate_Encoder = ResnetGenerator_Encoder_decoder().to(device)
    V_extract_Encoder = nn.DataParallel(Restormer_Encoder()).to(device)
    Decoder_Fusion = nn.DataParallel(Fusion_Decoder()).to(device)


    # a = [torch.rand((2,8,128,128)).cuda(),torch.rand((2,16,64,64)).cuda(),torch.rand((2,32,32,32)).cuda(),torch.rand((2,64,16,16)).cuda()]

    # print(profile(Decoder_Fusion,(a,a,a,a,)))


    # Load pre-trained weights
    model_paths = {
        'I_generate': "../model/generate/256_128_64_32_down1connect_C8_16_32_64/latest_net_G_A.pth",
        'I_extract': "../model/extract/C8_16_32_64/Encoder_ir.model",
        'V_generate': "../model/generate/256_128_64_32_down1connect_C8_16_32_64/latest_net_G_B.pth",
        'V_extract': "../model/extract/C8_16_32_64/Encoder_vi.model",
        'Decoder': "../model/Fusion/generate_down1_C8_16_32_64+extract_C8_16_32_64+vmd_FFT1_num2/Brain_finetune.model"
    }

    I_generate_Encoder.load_state_dict(torch.load(model_paths['I_generate'], map_location=device))
    I_extract_Encoder.load_state_dict(torch.load(model_paths['I_extract'], map_location=device).state_dict())
    V_generate_Encoder.load_state_dict(torch.load(model_paths['V_generate'], map_location=device))
    V_extract_Encoder.load_state_dict(torch.load(model_paths['V_extract'], map_location=device).state_dict())
    Decoder_Fusion.load_state_dict(torch.load(model_paths['Decoder'], map_location=device).state_dict())

    I_generate_Encoder = I_generate_Encoder.to(device)
    I_extract_Encoder = I_extract_Encoder.to(device)
    V_generate_Encoder = V_generate_Encoder.to(device)
    V_extract_Encoder = V_extract_Encoder.to(device)
    Decoder_Fusion = Decoder_Fusion.to(device)

    return I_generate_Encoder, I_extract_Encoder, V_generate_Encoder, V_extract_Encoder, Decoder_Fusion



def main():
    # Setup
    dataset_name = "MRI-PET"
    dataset_path = f"../deal_with_fusion/fusion_color/{dataset_name}"
    output_path = f"../deal_with_fusion/fusion_result/{dataset_name}"
    os.makedirs(output_path, exist_ok=True)

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize models
    I_generate_Encoder, I_extract_Encoder, V_generate_Encoder, V_extract_Encoder, Decoder_Fusion = initialize_models(
        device)

    # Create dataset and dataloader
    dataset = ImageFusionDataset(dataset_path)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    # Prepare transformation for saving
    to_pil = transforms.ToPILImage()

    # Timing and processing
    total_processing_time = 0

    # Disable gradient computation for inference
    with torch.no_grad():
        for batch_data in dataloader:

            # Unpack batch data
            ir_img = batch_data['ir_img'].to(device)
            vi_img = batch_data['vi_img'].to(device)
            width = batch_data['width'][0].item()
            height = batch_data['height'][0].item()
            img_name = batch_data['img_name'][0]


            start_time = time.time()


            # Feature extraction
            Feature_I_generate = I_generate_Encoder(ir_img)
            Feature_I_extract = I_extract_Encoder(ir_img)
            Feature_V_generate = V_generate_Encoder(vi_img)
            Feature_V_extract = V_extract_Encoder(vi_img)

            # Fusion
            img_Fu = Decoder_Fusion(Feature_I_generate, Feature_I_extract, Feature_V_generate,
                                    Feature_V_extract).squeeze()

            processing_time = time.time() - start_time


            # Post-processing
            fake_vi_unnormalized = (img_Fu + 1) / 2 * 255
            fake_vi_unnormalized = fake_vi_unnormalized[:height, :width]
            fake_vi_unnormalized = fake_vi_unnormalized.to(torch.uint8)

            # Save image
            fake_vi_img = to_pil(fake_vi_unnormalized.cpu())
            fake_vi_img.save(os.path.join(output_path, img_name))

            # Clean up
            del Feature_I_generate, Feature_I_extract, Feature_V_generate, Feature_V_extract, img_Fu
            torch.cuda.empty_cache()
            gc.collect()

            # Timing
            total_processing_time += processing_time
            print(f"Processed {img_name} in {processing_time:.4f} seconds")

    # Average processing time
    print(f"Average processing time: {total_processing_time / len(dataloader):.10f} seconds")


if __name__ == "__main__":
    main()
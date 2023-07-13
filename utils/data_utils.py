from typing import Tuple, List, Dict
import torchvision
from PIL import Image
import torch
import numpy as np
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader
from pathlib import Path
import json


class CustomDataset(torchvision.datasets.OxfordIIITPet):
    def __init__(
            self, 
            img_size: Tuple, 
            root: str, 
            split: str, 
            download: bool, 
            transform: torchvision.transforms, 
            target_types: List
            ):
        super().__init__(root=root, split=split, download=download, transform=transform, target_types=target_types)
        self.img_size = img_size

    def __getitem__(self, idx: int) -> Tuple[torch.tensor, torch.tensor]:
        image = Image.open(self._images[idx]).convert("RGB")
        # 1 = background, 2 = pet, 3 = border
        seg_mask = Image.open(self._segs[idx]).resize(self.img_size)
        array_seg_mask = np.array(seg_mask)
        array_seg_mask[array_seg_mask == 1] = 0
        array_seg_mask[array_seg_mask == 2] = 1
        array_seg_mask[array_seg_mask == 3] = 1

        if self.transforms:
            image, seg_mask = oxford_transform(image=image, seg_mask=array_seg_mask, img_size=self.img_size)

        return image, seg_mask


def oxford_transform(
        image: Image, 
        seg_mask: np.array, 
        img_size: Tuple
        ) -> Tuple[torch.tensor, torch.tensor]:
    '''
    transform image and segmentation mask into tensors and resize
    '''
    image = TF.to_tensor(pic=image)
    seg_mask = torch.tensor(seg_mask)
    image = TF.resize(image, size=img_size)

    return image, seg_mask


def create_dataloaders(batch_size: int, img_size: Tuple, data_root: str) -> Tuple[DataLoader, DataLoader, DataLoader]:
    '''
    creates and partitions train, validation and test data loaders
    '''
    trainval_dataset = CustomDataset(
        root=data_root,
        split="trainval",
        download=False,
        transform=True,
        target_types=["segmentation"],
        img_size=img_size
    )
    test_dataset = CustomDataset(
        root=data_root,
        split="test",
        download=False,
        transform=True,
        target_types=["segmentation"],
        img_size=img_size
    )
    # combine datasets
    combined_dataset = torch.utils.data.ConcatDataset([trainval_dataset, test_dataset])

    # assign dataset sizes
    train_amount = int(len(combined_dataset)*0.9)
    val_amount = int((len(combined_dataset) - train_amount) / 2)
    test_amount = len(combined_dataset) - train_amount - val_amount

    # randomly split datasets
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        combined_dataset, 
        lengths=[train_amount, val_amount, test_amount]
        )
    
    # create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def get_class_row(class_idx: int, mask: torch.tensor) -> List:
    '''
    
    '''
    class_row_list = []
    for row in mask:
        row = np.array(row)
        row[row == np.uint8(class_idx)] = 1.
        row[row != np.uint8(class_idx)] = 0.
        class_row_list.append(row)
    return class_row_list


def one_hot_mask(mask: torch.tensor, num_classes: int) -> torch.tensor:
    '''
    one hot encode segmentation mask
    '''
    class_row_list = get_class_row(class_idx=0, mask=mask)
    new_mask_array = np.expand_dims(a=np.array(class_row_list), axis=0)
    for class_idx in range(1, num_classes):
        class_row_list = get_class_row(class_idx=class_idx, mask=mask)
        new_mask_array = np.concatenate([new_mask_array, np.expand_dims(a=np.array(class_row_list), axis=0)], axis=0)
    return torch.tensor(new_mask_array)


def save_results(
        train_loss_list: List, 
        val_loss_list: List, 
        test_dict: Dict, 
        file_path: Path
        ) -> None:
    '''
    saves results from developing model into a JSON file
    '''
    with open(file_path, mode="w") as opened_json:
        json_obj = {
            "train loss": train_loss_list,
            "validation loss": val_loss_list,
            "test dice": test_dict["Dice"],
            "test iou": test_dict["IoU"]
        }
        json.dump(json_obj, opened_json) 


def measure_dice_and_iou(
        ground_truth_mask: torch.tensor, 
        prediction_mask: torch.tensor
        ) -> Tuple[torch.tensor, torch.tensor]:
    '''
    measures mean dice and mean IoU
    measures both to avoid computing tp, fp and fn twice
    '''
    dice_per_class = []
    iou_per_class = []
    num_classes = ground_truth_mask.shape[0]
    for class_idx in range(num_classes):

        prod = ground_truth_mask[class_idx] * prediction_mask[class_idx] # true positives will be 1, all else 0
        diff = ground_truth_mask[class_idx] - prediction_mask[class_idx] # fp will be -1, fn will be 1 
        counts = torch.unique(input=diff, return_counts=True)
        
        tp = torch.sum(prod.flatten())

        if -1. in counts[0]:
            neg_one_idx = list(counts[0]).index(-1.)
        else:
            neg_one_idx = None
        if 1. in counts[0]:
            one_idx = list(counts[0]).index(1.)
        else:
            one_idx = None

        if neg_one_idx != None:
            fp = counts[1][neg_one_idx]
        else:
            fp = 0
        if one_idx != None:
            fn = counts[1][1]
        else:
            fn = 0

        dice = (2*tp) / ((2*tp) + fn + fp) 
        dice_per_class.append(dice)
        
        iou = tp / (tp + fp + fn)
        iou_per_class.append(iou)

    return torch.mean(torch.tensor(dice_per_class)), torch.mean(torch.tensor(iou_per_class))


def determine_class(indexed_tensor: np.array) -> torch.tensor:
    '''
    determines the largest index of a tensor
    '''
    return torch.argmax(indexed_tensor)


def one_hot_encode(class_int: int, num_classes: int) -> torch.tensor:
    '''
    creates a one hot encodes tensor
    '''
    tensor_row = torch.tensor([0 for x in range(num_classes)])
    tensor_row[class_int-1] = 1
    return tensor_row


def update_tensor(one_hot_tensor: torch.tensor, tensor: torch.tensor, x: int, y: int):
    '''
    updates tensor values
    '''
    tensor[:, x, y] = one_hot_tensor
    return tensor


def binarize_array(tensor: torch.tensor, num_classes: int, device: torch.device) -> torch.tensor:
    '''
    updates each pixel in the mask as the one hot encoded vector along the channel dimension
    '''
    for chan1_idx in range(tensor.shape[1]):
        for chan2_idx in range(tensor.shape[2]):
            class_int = determine_class(torch.tensor([x[:][chan1_idx][chan2_idx] for x in tensor]).to(device))
            one_hot_tensor = one_hot_encode(
                class_int=class_int,
                num_classes=num_classes
            )
            array = update_tensor(one_hot_tensor=one_hot_tensor, tensor=tensor, x=chan1_idx, y=chan2_idx)
    return array

from typing import Tuple, List, Dict
import torchvision
from PIL import Image
import torch
import numpy as np
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader
from pathlib import Path
import json
import math



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
        super().__init__(
            root=root, 
            split=split, 
            download=download, 
            transform=transform, 
            target_types=target_types
            )
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
            image, seg_mask = oxford_transform(
                image=image, 
                seg_mask=array_seg_mask, 
                img_size=self.img_size
                )

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


def create_dataloaders(
        batch_size: int, 
        img_size: Tuple, 
        data_root: str
        ) -> Tuple[DataLoader, DataLoader, DataLoader]:
    '''
    creates and partitions train, validation and test data loaders
    '''
    trainval_dataset = CustomDataset(
        root=data_root,
        split="trainval",
        download=True, # will skip downloading if already downloaded
        transform=True,
        target_types=["segmentation"],
        img_size=img_size
    )
    test_dataset = CustomDataset(
        root=data_root,
        split="test",
        download=True, # will skip downloading if already downloaded
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
        new_mask_array = np.concatenate([
            new_mask_array, 
            np.expand_dims(a=np.array(class_row_list), axis=0)], 
            axis=0
            )
    return torch.tensor(new_mask_array)


def save_train_results(
        train_loss_list: List, 
        val_loss_list: List,
        file_path: Path,
        continue_bool: bool,
        batch_size: int,
        learning_rate: float
        ) -> None:
    '''
    saves results from developing model into a JSON file
    '''
    if continue_bool:
        with open(file_path, mode="r") as opened_json:
            json_dict = json.load(opened_json)
        with open(file_path, mode="w") as opened_json:
            if not "batch size" in list(json_dict.keys()):
                json_dict["batch size"] = batch_size
            if not "learning rate" in list(json_dict.keys()):
                json_dict["learning rate"] = learning_rate
            
            json_dict["train loss"] += train_loss_list
            json_dict["validation loss"] += val_loss_list
            json_obj = json.dumps(json_dict)
            opened_json.write(json_obj)
    else:
        with open(file_path, mode="w") as opened_json:
            json_dict = {
                "train loss": train_loss_list,
                "validation loss": val_loss_list,
                "batch size": batch_size,
                "learning rate": learning_rate
            }
            json_dict = json.dumps(json_dict)
            opened_json.write(json_dict) 


def save_test_results(
        test_dict: Dict, 
        file_path: Path
        ) -> None:
    '''
    saves results from developing model into a JSON file
    '''
    with open(file_path, mode="w") as opened_json:
        json_dict = {
            "test dice": test_dict["Dice"],
            "test iou": test_dict["IoU"]
        }
        json_obj = json.dumps(json_dict)
        opened_json.write(json_obj) 


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
        # true positives will be 1, all else 0
        prod = ground_truth_mask[class_idx] * prediction_mask[class_idx] 
        # fp will be -1, fn will be 1
        diff = ground_truth_mask[class_idx] - prediction_mask[class_idx]  
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
            fn = counts[1][one_idx]
        else:
            fn = 0

        dice = (2*tp) / ((2*tp) + fn + fp)
        if not math.isnan(dice):
            dice_per_class.append(dice)
        
        iou = tp / (tp + fp + fn)
        if not math.isnan(iou):
            iou_per_class.append(iou)

    mean_dice = torch.mean(torch.tensor(dice_per_class)).detach().numpy()
    mean_iou = torch.mean(torch.tensor(iou_per_class)).detach().numpy()

    return mean_dice, mean_iou


def onehot_3d_array(tensor, num_classes):
    return torch.nn.functional.one_hot(
        input=torch.argmax(tensor, dim=0),
        num_classes=num_classes
    ).permute(2, 0, 1)

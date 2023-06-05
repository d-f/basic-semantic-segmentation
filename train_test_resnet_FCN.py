from pathlib import Path
import json
from tqdm import tqdm
import numpy as np
import torchvision.transforms.functional as TF
import torchvision
from PIL import Image
import torch
import argparse
from torch.utils.data import DataLoader
from typing import Any, Tuple, Dict



def create_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-pretrained", action="store_true")
    parser.add_argument("-num_classes", type=int, default=37)
    parser.add_argument("-batch_size", default=24)
    parser.add_argument("-img_size", default=(256, 256))
    parser.add_argument("-patience", default=5)
    parser.add_argument("-result_dir", type=Path, default=Path("C:\\personal_ML\\fcn_segmentation\\results\\"))
    parser.add_argument("-result_filename", type=str, default="resnet_50_fcn_1_results.json")
    parser.add_argument("-lr", type=float, default=0.001)
    parser.add_argument("-model_save_name", default="resnet50_fcn_1.pth.tar")
    parser.add_argument("-num_epochs", default=64)
    return parser.parse_args()


def define_model(pre_trained: bool, num_classes: int) -> torchvision.models.segmentation.fcn_resnet50:
    model = torchvision.models.segmentation.fcn_resnet50(
        pretrained=pre_trained, 
        num_classes=num_classes, 
        weights=None
        )
    return model 


def print_model_summary(model):
    for param in model.named_parameters():
        print(param[0], param[1].size())


def oxford_transform(image, seg_mask, img_size):
    image = TF.to_tensor(pic=image)
    seg_mask = torch.tensor(seg_mask)

    image = TF.resize(image, size=img_size)

    return image, seg_mask


class CustomDataset(torchvision.datasets.OxfordIIITPet):
    def __init__(self, img_size, root, split, download, transform, target_types):
        super().__init__(root=root, split=split, download=download, transform=transform, target_types=target_types)
        self.img_size = img_size

    def __getitem__(self, idx: int) -> Tuple[Any, Any, Any]:
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


def create_datasets(batch_size, img_size):
    trainval_dataset = CustomDataset(
        root="C:\\personal_ML\\Oxford_PyTorch\\",
        split="trainval",
        download=False,
        transform=True,
        target_types=["segmentation"],
        img_size=img_size
    )
    test_dataset = CustomDataset(
        root="C:\\personal_ML\\Oxford_PyTorch\\",
        split="test",
        download=False,
        transform=True,
        target_types=["segmentation"],
        img_size=img_size
    )
    combined_dataset = torch.utils.data.ConcatDataset([trainval_dataset, test_dataset])

    train_amount = int(len(combined_dataset)*0.9)
    val_amount = int((len(combined_dataset) - train_amount) / 2)
    test_amount = len(combined_dataset) - train_amount - val_amount

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        combined_dataset, 
        lengths=[train_amount, val_amount, test_amount]
        )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def count_classes(loader):
    class_list = []

    for output in tqdm(loader):
        class_list.append(int(output[2]))
    
    for _ in range(np.max(class_list)):
        print(f"count of {_}", class_list.count(_))


def get_class_row(class_idx, mask):
    class_row_list = []
    for row in mask:
        row = np.array(row)
        row[row == np.uint8(class_idx)] = 1.
        row[row != np.uint8(class_idx)] = 0.
        class_row_list.append(row)
    return class_row_list


def one_hot_mask(mask, num_classes):
    class_row_list = get_class_row(class_idx=0, mask=mask)
    new_mask_array = np.expand_dims(a=np.array(class_row_list), axis=0)
    for class_idx in range(1, num_classes):
        class_row_list = get_class_row(class_idx=class_idx, mask=mask)
        new_mask_array = np.concatenate([new_mask_array, np.expand_dims(a=np.array(class_row_list), axis=0)], axis=0)
    return torch.tensor(new_mask_array)


def validate(val_loader, model, device, criterion, num_classes):
    with torch.no_grad():
        model.eval()
        val_dict = {
            "IoU": 0,
            "Dice": 0,
        }
        val_loss = 0
        for batch_image, batch_mask in tqdm(val_loader, desc="Validating"):
            batch_iou = 0
            batch_dice = 0
            batch_pred = model(batch_image.to(device))["out"]
            batch_onehot_segmask  = one_hot_mask(mask=batch_mask[0], num_classes=num_classes).unsqueeze(dim=0)
            for seg_mask in batch_mask[1:]:
                one_hot_seg = one_hot_mask(mask=seg_mask, num_classes=num_classes).unsqueeze(dim=0)
                batch_onehot_segmask = np.concatenate([batch_onehot_segmask, one_hot_seg], axis=0)
            batch_onehot_segmask = torch.tensor(batch_onehot_segmask).to(device).to(torch.float64)
            val_loss += criterion(batch_pred, batch_onehot_segmask).cpu().detach().numpy()
            for pred_idx in range(len(batch_pred)):
                pred = torch.nn.Softmax(dim=1)(batch_pred[pred_idx]).cpu().detach().numpy().astype("int32")
                mask = batch_onehot_segmask[pred_idx].cpu().detach().numpy().astype("int32")
                pred = binarize_array(array=pred, num_classes=num_classes)

                batch_iou += measure_iou(prediction_mask=pred, ground_truth_mask=mask)
                batch_dice += measure_dice(prediction_mask=pred, ground_truth_mask=mask)
            val_dict["IoU"] += batch_iou
            val_dict["Dice"] += batch_dice
        val_dict["IoU"] /= len(val_loader)
        val_dict["Dice"] /= len(val_loader)
        model.train()
    val_loss /= len(val_loader)
    return val_loss, val_dict, model


def save_checkpoint(state: Dict, filepath: Path) -> None:
    """
    saves the model state dictionary to a .pth.tar tile
    """
    print("saving...")
    torch.save(state, filepath)


def train(model, train_loader, val_loader, num_epochs, patience, num_classes, criterion, optimizer, device, result_dir, model_save_name):
    patience_counter = 0
    best_val_loss = np.inf
    train_loss_list = []
    val_loss_list = []
    val_iou_list = []
    val_dice_list = []

    for epoch_idx in range(num_epochs):
            if patience == patience_counter:
                break
            else:
                epoch_loss = 0
                for batch_image, batch_seg_mask in tqdm(train_loader, desc="Training"):
                    batch_pred = model(batch_image.to(device))["out"]
                    batch_onehot_segmask  = one_hot_mask(mask=batch_seg_mask[0], num_classes=num_classes).unsqueeze(dim=0)
                    for seg_mask in batch_seg_mask[1:]:
                        one_hot_seg = one_hot_mask(mask=seg_mask, num_classes=num_classes).unsqueeze(dim=0)
                        batch_onehot_segmask = np.concatenate([batch_onehot_segmask, one_hot_seg], axis=0)
                    batch_onehot_segmask = torch.tensor(batch_onehot_segmask).to(device).to(torch.float32)

                    loss = criterion(batch_pred, batch_onehot_segmask)
                    epoch_loss += loss.cpu().detach().numpy()
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                train_loss_list.append((epoch_idx+1, epoch_loss/len(train_loader)))
                val_loss, val_dict, model = validate(val_loader, model, criterion=criterion, device=device, num_classes=num_classes)
                print(f"epoch {epoch_idx+1} validation loss", val_loss)
                print(f"epoch {epoch_idx+1} validation performance", val_dict)
                val_loss_list.append((epoch_idx+1, val_loss))
                val_iou_list.append((epoch_idx+1, val_dict["IoU"]))
                val_dice_list.append((epoch_idx+1, val_dict["Dice"]))
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    save_checkpoint(state=model.state_dict(), filepath=result_dir.joinpath(model_save_name))
                else:
                    patience_counter += 1  
    
    return train_loss_list, val_loss_list, val_iou_list, val_dice_list       


def save_results(train_loss_list, val_loss_list, test_dict, file_path, val_iou, val_dice):
    with open(file_path, mode="w") as opened_json:
        json_obj = {
            "train loss": train_loss_list,
            "validation loss": val_loss_list,
            "validation IoU": val_iou,
            "validation dice": val_dice,
            "test dice": test_dict["Dice"],
            "test iou": test_dict["IoU"]
        }
        json.dump(json_obj, opened_json)


def measure_dice(ground_truth_mask: np.array, prediction_mask: np.array) -> float:
    """
    ensure that masks are 0 for background and 1 for segmentation
    """
    dice_list = []
    for chan_idx in range(ground_truth_mask.shape[0]):
        intersection = np.sum(ground_truth_mask[chan_idx]*prediction_mask[chan_idx]) # since masks are 0 and 1, multiplying reveals intersection
        dice_list.append((2 * intersection) / (np.sum(ground_truth_mask) + np.sum(prediction_mask)))
    return np.mean(dice_list)


def measure_iou(ground_truth_mask, prediction_mask):
    iou_list = []
    for chan_idx in range(ground_truth_mask.shape[0]):
        intersection = np.sum(ground_truth_mask[chan_idx]*prediction_mask[chan_idx]) # since masks are 0 and 1, multiplying reveals intersection
        union = len(np.union1d(ar1=ground_truth_mask[chan_idx], ar2=prediction_mask[chan_idx]))
        iou_list.append(abs(intersection) / abs(union))
    return np.mean(iou_list)


def load_model(weight_path: Path, model):
    '''
    loads all parameters of a model
    :param weight_path: path to the .pth.tar file with parameters to update
    :param model: model object
    :return: the model with updated parameters
    '''
    checkpoint = torch.load(weight_path)
    model.load_state_dict(checkpoint)
    return model


def determine_class(indexed_array: np.array) -> np.array:
    return np.argmax(indexed_array)


def one_hot_encode(class_int: int, num_classes: int) -> np.array:
    array = np.array([0 for x in range(num_classes)])
    array[class_int-1] = 1
    return array


def update_array(encoded_array, array, x, y):
    for channel_idx in range(array.shape[0]):
        array[channel_idx][x][y] = encoded_array[channel_idx]
    return array


def binarize_array(array: np.array, num_classes: int) -> np.array:
    for chan1_idx in range(array.shape[1]):
        for chan2_idx in range(array.shape[2]):
            one_hot_array = one_hot_encode(
                class_int=determine_class([x[chan1_idx][chan2_idx] for x in array]),
                num_classes=num_classes
            )
            array = update_array(encoded_array=one_hot_array, array=array, x=chan1_idx, y=chan2_idx)
    return array


def test_model(test_loader, model, device, num_classes, result_dir, model_save_name):
    with torch.no_grad():
        model = load_model(weight_path=result_dir.joinpath(model_save_name), model=model)
        model.eval()
        test_dict = {
            "IoU": 0,
            "Dice": 0
        }
        for batch_image, batch_seg_mask in tqdm(test_loader, desc="Testing"):
            batch_iou = 0
            batch_dice = 0
            batch_pred = model(batch_image.to(device))["out"]
            batch_onehot_segmask  = one_hot_mask(mask=batch_seg_mask[0], num_classes=num_classes).unsqueeze(dim=0)
            for seg_mask in batch_seg_mask[1:]:
                one_hot_seg = one_hot_mask(mask=seg_mask, num_classes=num_classes).unsqueeze(dim=0)
                batch_onehot_segmask = np.concatenate([batch_onehot_segmask, one_hot_seg], axis=0)
            batch_onehot_segmask = torch.tensor(batch_onehot_segmask).to(device).to(torch.float64)
            for pred_idx in range(len(batch_pred)):
                pred = torch.nn.Softmax(dim=1)(batch_pred[pred_idx]).cpu().detach().numpy().astype("int32")
                mask = batch_onehot_segmask[pred_idx].cpu().detach().numpy().astype("int32")
                pred = binarize_array(array=pred, num_classes=num_classes)

                batch_iou += measure_iou(prediction_mask=pred, ground_truth_mask=mask)
                batch_dice += measure_dice(prediction_mask=pred, ground_truth_mask=mask)
            test_dict["IoU"] += batch_iou
            test_dict["Dice"] += batch_dice
        test_dict["IoU"] /= len(test_loader)
        test_dict["Dice"] /= len(test_loader)
        print("test results", test_dict)
        return test_dict            

            
def main():
    args = create_argparser()
    model = define_model(pre_trained=args.pretrained, num_classes=args.num_classes)
    train_loader, val_loader, test_loader = create_datasets(batch_size=args.batch_size, img_size=args.img_size)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)
    device = torch.device("cuda")
    model = model.to(device)
    train_loss_list, val_loss_list, val_iou_list, val_dice_list = train(
        model=model, 
        train_loader=train_loader, 
        val_loader=val_loader, 
        num_epochs=args.num_epochs,
        patience=args.patience,
        num_classes=args.num_classes,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        result_dir=args.result_dir,
        model_save_name=args.model_save_name
        )
    test_dict = test_model(
        test_loader=test_loader, 
        model=model, 
        device=device, 
        num_classes=args.num_classes,
        result_dir=args.result_dir,
        model_save_name=args.model_save_name
        )
    save_results(
        train_loss_list=train_loss_list,
        val_loss_list=val_loss_list,
        file_path=args.result_dir.joinpath(args.result_filename),
        test_dict=test_dict,
        val_iou=val_iou_list,
        val_dice=val_dice_list
    )


if __name__ == "__main__":
    main()

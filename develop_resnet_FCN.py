import argparse
import torchvision
from torch.utils.data import DataLoader
import torch
from pathlib import Path
from typing import Tuple, List, Dict
import numpy as np
from tqdm import tqdm
from utils.data_utils import *
from utils.model_utils import *


def create_argparser() -> argparse.Namespace:
    '''
    defines the command line argument parser
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("-pretrained", action="store_true")
    parser.add_argument("-num_classes", type=int, default=37)
    parser.add_argument("-batch_size", default=24)
    parser.add_argument("-img_size", default=(256, 256))
    parser.add_argument("-patience", default=2)
    parser.add_argument("-result_dir", type=Path, default=Path("C:\\personal_ML\\fcn_segmentation\\results\\"))
    parser.add_argument("-result_filename", type=str, default="resnet_50_fcn_1_results.json")
    parser.add_argument("-lr", type=float, default=0.001)
    parser.add_argument("-model_save_name", default="resnet50_fcn_1.pth.tar")
    parser.add_argument("-num_epochs", default=64)
    parser.add_argument("-data_root", default="C:\\personal_ML\\Oxford_PyTorch\\")
    return parser.parse_args()


def validate(
        val_loader: DataLoader, 
        model: torchvision.models.segmentation.fcn_resnet50, 
        device: torch.device, 
        criterion: torch.optim, 
        num_classes: int
        ) -> torch.tensor:
    '''
    validates model
    '''
    with torch.no_grad():
        model.eval() 
        val_loss = 0
        for batch_image, batch_mask in tqdm(val_loader, desc="Validating"):
            batch_pred = model(batch_image.to(device))["out"]
            batch_onehot_segmask  = one_hot_mask(mask=batch_mask[0], num_classes=num_classes).unsqueeze(dim=0)
            for seg_mask in batch_mask[1:]:
                one_hot_seg = one_hot_mask(mask=seg_mask, num_classes=num_classes).unsqueeze(dim=0)
                batch_onehot_segmask = np.concatenate([batch_onehot_segmask, one_hot_seg], axis=0)
            batch_onehot_segmask = torch.tensor(batch_onehot_segmask).to(device).to(torch.float64)
            val_loss += criterion(batch_pred, batch_onehot_segmask).cpu().detach().numpy()
    val_loss /= len(val_loader)
    return val_loss


def train(
        model: torchvision.models.segmentation.fcn_resnet50, 
        train_loader: DataLoader, 
        val_loader: DataLoader, 
        num_epochs: int, 
        patience: int, 
        num_classes: int, 
        criterion: torch.nn.CrossEntropyLoss, 
        optimizer: torch.optim, 
        device: torch.device, 
        result_dir: Path, 
        model_save_name: str
        ) -> Tuple[List, List]:
    ''' 
    trains model and records training and validation loss throughout training
    '''
    patience_counter = 0
    best_val_loss = np.inf
    train_loss_list = []
    val_loss_list = []

    for epoch_idx in range(num_epochs):
            if patience == patience_counter:
                break
            else:
                epoch_loss = 0
                for batch_image, batch_seg_mask in tqdm(train_loader, desc="Training"):
                    model.train()
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
                val_loss = validate(val_loader, model, criterion=criterion, device=device, num_classes=num_classes)
                print(f"epoch {epoch_idx+1} validation loss", val_loss)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    save_checkpoint(state=model.state_dict(), filepath=result_dir.joinpath(model_save_name))
                else:
                    patience_counter += 1  
    
    return train_loss_list, val_loss_list   


def test_model(
        test_loader: DataLoader, 
        model: torchvision.models.segmentation.fcn_resnet50, 
        device: torch.device, 
        num_classes: int, 
        result_dir: Path, 
        model_save_name: str
        ) -> Dict:
    '''
    measures model performance on the test dataset
    '''
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
                pred = torch.nn.Softmax(dim=1)(batch_pred[pred_idx])#.cpu().detach().numpy().astype("int32")
                mask = batch_onehot_segmask[pred_idx]#.cpu().detach().numpy().astype("int32")
                pred = binarize_array(tensor=pred, num_classes=num_classes, device=device)

                mean_dice, mean_iou = measure_dice_and_iou(prediction_mask=pred, ground_truth_mask=mask)
                batch_iou += mean_iou
                batch_dice += mean_dice
            test_dict["IoU"] += batch_iou
            test_dict["Dice"] += batch_dice
        test_dict["IoU"] /= len(test_loader)
        test_dict["Dice"] /= len(test_loader)
        print("test results", test_dict)
        return test_dict 
    

def main():
    '''
    trains, validates and tests resnet FCN
    '''
    args = create_argparser()
    model = define_model(pre_trained=args.pretrained, num_classes=args.num_classes)
    train_loader, val_loader, test_loader = create_dataloaders(
        batch_size=args.batch_size, img_size=args.img_size, data_root=args.data_root
        )
    criterion = define_criterion()
    optimizer = define_optimizer(model=model, learning_rate=args.lr)
    device = define_device()
    model = model.to(device)
    train_loss_list, val_loss_list = train(
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
    print(test_dict)
    save_results(
        train_loss_list=train_loss_list,
        val_loss_list=val_loss_list,
        file_path=args.result_dir.joinpath(args.result_filename),
        test_dict=test_dict
    )


if __name__ == "__main__":
    main()

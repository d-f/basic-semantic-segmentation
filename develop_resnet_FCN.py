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
import matplotlib.pyplot as plt
from sklearn.metrics import jaccard_score


def create_argparser() -> argparse.Namespace:
    '''
    defines the command line argument parser
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("-pretrained", action="store_true", default=True) 
    parser.add_argument("-num_classes", type=int, default=37)
    parser.add_argument("-batch_size", default=32)
    parser.add_argument("-img_size", default=(128, 128))
    parser.add_argument("-patience", default=5)
    parser.add_argument("-result_dir", type=Path)
    parser.add_argument("-train_result_filename", type=str)
    parser.add_argument("-test_result_filename", type=str)
    parser.add_argument("-lr", type=float, default=1e-4)
    parser.add_argument("-model_save_name", default="model_1.pth.tar")
    parser.add_argument("-num_epochs", default=15)
    parser.add_argument("-data_root", default=Path("C:\\personal_ML\\Oxford_PyTorch\\"))
    parser.add_argument("-continue_bool", action="store_true", default=False)
    parser.add_argument("-start_epoch", type=int, default=0)
    parser.add_argument("-weight_path")
    return parser.parse_args()


def validate(
        val_loader: DataLoader, 
        model: torchvision.models.segmentation.fcn_resnet101, 
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
        for batch_image, batch_mask, batch_class in tqdm(val_loader, desc="Validating"):
            batch_pred = model(batch_image.to(device))["out"].softmax(dim=1)
            batch_onehot_segmask = onehot_segmask(batch_mask=batch_mask, batch_class=batch_class, num_classes=num_classes).to(device)
            loss = criterion(batch_pred, batch_onehot_segmask)
            val_loss += loss.item()
        val_loss /= len(val_loader)
    return val_loss


def train(
        model: torchvision.models.segmentation.fcn_resnet101, 
        train_loader: DataLoader, 
        val_loader: DataLoader, 
        num_epochs: int, 
        patience: int, 
        num_classes: int, 
        criterion: torch.nn.CrossEntropyLoss, 
        optimizer: torch.optim, 
        device: torch.device, 
        result_dir: Path, 
        model_save_name: str,
        continue_bool: bool,
        start_epoch: int
        ) -> Tuple[List, List]:
    ''' 
    trains model and records training and validation loss throughout training
    '''
    patience_counter = 0
    best_val_loss = np.inf
    train_loss_list = []
    val_loss_list = []

    if continue_bool:
        num_epochs += start_epoch
        print("epoch range", start_epoch, num_epochs)

    for epoch_idx in range(start_epoch, num_epochs):
        if patience == patience_counter:
            break
        else:
            epoch_loss = 0
            model.train()
            for batch_image, batch_seg_mask, batch_class in tqdm(train_loader, desc="Training"):

                    batch_pred = model(batch_image.to(device))["out"].softmax(dim=1)
                    batch_onehot_segmask = onehot_segmask(batch_mask=batch_seg_mask, batch_class=batch_class, num_classes=num_classes).to(device)

                    loss = criterion(batch_pred, batch_onehot_segmask)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()

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
        model: torchvision.models.segmentation.fcn_resnet101, 
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
        }
        for batch_image, batch_seg_mask, batch_class in tqdm(test_loader, desc="Testing"):
            batch_iou = 0
            batch_pred = model(batch_image.to(device))["out"].softmax(dim=1)

            batch_onehot_pred = onehot_pred(batch_pred=batch_pred, num_classes=num_classes)
            batch_onehot_pred = batch_onehot_pred.argmax(dim=1)

            for pred_idx in range(len(batch_pred)):
                mask = batch_seg_mask[pred_idx, :, :]
                mask *= batch_class[pred_idx]
                mask = mask.int().cpu().numpy()
                pred = batch_onehot_pred[pred_idx, :, :].int().cpu().numpy()
                # macro calculates metrics for each label and returns the unweighted mean
                batch_iou += jaccard_score(y_true=mask.flatten(), y_pred=pred.flatten(), average="macro") 

            batch_iou /= len(batch_pred)
            print("batch IoU", batch_iou)
            test_dict["IoU"] += batch_iou
        test_dict["IoU"] /= len(test_loader)
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
    if args.continue_bool:
        model = load_model(weight_path=args.weight_path, model=model)
    print_model_summary(model=model)
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
        model_save_name=args.model_save_name,
        continue_bool=args.continue_bool,
        start_epoch=args.start_epoch
        )
    save_train_results(
        train_loss_list=train_loss_list,
        val_loss_list=val_loss_list,
        file_path=args.result_dir.joinpath(args.train_result_filename),
        batch_size=args.batch_size,
        learning_rate=args.lr,
        continue_bool=args.continue_bool
    )
    test_dict = test_model(
        test_loader=test_loader, 
        model=model, 
        device=device, 
        num_classes=args.num_classes,
        result_dir=args.result_dir,
        model_save_name=args.model_save_name
        )
    save_test_results(
        file_path=args.result_dir.joinpath(args.test_result_filename),
        test_dict=test_dict
    )


if __name__ == "__main__":
    main()

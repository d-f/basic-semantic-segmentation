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
from sklearn.metrics import jaccard_score


def create_argparser() -> argparse.Namespace:
    '''
    defines the command line argument parser
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("-pretrained", action="store_true", default=True) 
    parser.add_argument("-num_classes", type=int, default=37)
    parser.add_argument("-batch_size", default=10)
    parser.add_argument("-img_size", default=(128, 128))
    parser.add_argument("-patience", default=5)
    parser.add_argument("-result_dir", type=Path, default=Path("C:\\personal_ML\\basic-semantic-segmentation\\results\\"))
    parser.add_argument("-train_result_filename", type=str, default="model_1_train_results.json")
    parser.add_argument("-test_result_filename", type=str, default="model_1_test_results.json")
    parser.add_argument("-lr", type=float, default=1e-3)
    parser.add_argument("-model_save_name", default="model_1.pth.tar")
    parser.add_argument("-num_epochs", default=256)
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
            batch_pred = model(batch_image.to(device))["out"]
            batch_onehot_segmask = batch_onehot(batch_mask=batch_mask, batch_class=batch_class, num_classes=num_classes).to(device)
            val_loss += criterion(batch_pred, batch_onehot_segmask).cpu().detach().numpy()
    val_loss /= len(val_loader)
    return val_loss


def onehot_tensor(tensor, num_classes, class_idx):
    new_ten = torch.empty(size=[num_classes, tensor.shape[0], tensor.shape[1]])
    for x_pos in range(tensor.shape[0]):
        for y_pos in range(tensor.shape[1]):
            new_ten[:, x_pos, y_pos] = torch.nn.functional.one_hot(input=class_idx, num_classes=num_classes)
    return new_ten


def onehot_multi(tensor, num_classes):
    for x_pos in range(tensor.shape[1]):
        for y_pos in range(tensor.shape[2]):
            oh_chan = tensor[:, x_pos, y_pos]
            oh_chan = torch.nn.functional.one_hot(input=torch.argmax(oh_chan), num_classes=num_classes)
            tensor[:, x_pos, y_pos] = oh_chan
    return tensor


def batch_onehot(batch_mask, batch_class, num_classes):
    new_batch = torch.empty(size=(batch_mask.shape[0], num_classes, batch_mask.shape[1], batch_mask.shape[2]))
    for batch_idx in range(batch_mask.shape[0]):
        new_batch[batch_idx, :, :, :] = onehot_tensor(batch_mask[batch_idx], num_classes, batch_class[batch_idx])    
    return new_batch


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

                    batch_onehot_segmask = batch_onehot(batch_mask=batch_seg_mask, batch_class=batch_class, num_classes=num_classes).to(device)

                    loss = criterion(batch_pred, batch_onehot_segmask)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.cpu().detach().numpy()


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
        # model = load_model(weight_path=result_dir.joinpath(model_save_name), model=model)
        model.eval()
        test_dict = {
            "IoU": 0,
        }
        for batch_image, batch_seg_mask, batch_class in tqdm(test_loader, desc="Testing"):
            batch_iou = 0
            batch_pred = model(batch_image.to(device))["out"]

            batch_onehot_segmask = batch_onehot(
                batch_mask=batch_seg_mask, batch_class=batch_class, num_classes=num_classes
            )
           
            for pred_idx in range(len(batch_pred)):
                if num_classes == 1:
                    pred = torch.nn.Sigmoid()(batch_pred[pred_idx]).cpu().numpy()
                    pred = np.round(pred)
                else:
                    pred = torch.nn.Softmax(dim=0)(batch_pred[pred_idx])
                    pred = onehot_multi(tensor=pred, num_classes=num_classes).int().cpu().numpy()

                mask = batch_onehot_segmask[pred_idx, :, :, :].int().cpu().numpy()

                avg_iou = 0
                for class_idx in range(num_classes):
                    avg_iou += jaccard_score(y_true=mask[class_idx].flatten(), y_pred=pred[class_idx].flatten(), average="binary")
                avg_iou /= num_classes
                batch_iou += avg_iou

            batch_iou /= len(batch_pred)
            print("batch IoU", batch_iou)
            test_dict["IoU"] += batch_iou
        test_dict["IoU"] /= len(test_loader)
        print("test results", test_dict)


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

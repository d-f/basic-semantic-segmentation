from pathlib import Path
import torchvision
import argparse


def create_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-dataset_dir", type=Path, default="C:\\Users\\danan\\personal_ML\\")
    return parser.parse_args()


def download_datasets(dataset_dir):
    trainval_dataset = torchvision.datasets.OxfordIIITPet(
        root=dataset_dir,
        split="trainval",
        download=True
    )
    


def main():
    args = create_argparser()
    download_datasets(dataset_dir=args.dataset_dir)


if __name__ == "__main__":
    main()

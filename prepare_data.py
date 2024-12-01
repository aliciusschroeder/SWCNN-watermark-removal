import argparse
from dataset import prepare_data
from utils import get_config

parser = argparse.ArgumentParser(description="SWCNN")
config = get_config('configs/config.yaml')

def main():
    prepare_data(data_path=config['train_data_path'], patch_size=256, stride=128, aug_times=1, mode='color')

if __name__ == "__main__":
    main()
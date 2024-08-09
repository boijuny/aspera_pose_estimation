import os
import sys
import shutil
import zipfile
import yaml
import boto3
import logging
import argparse
import subprocess

"""
AUTHOR: Matthieu Marchal (SII Internship)

LAST UPDATED: 2024-06-10

DESCRIPTION:
    This script trains a YOLO model using the Ultralytics library. It is the training script that will be executed on the SageMaker instance.
    
USAGE:
    This script is called by the SageMaker PyTorch estimator using the yolov8_train_sagemaker.py script.

CONFIGURATION:
    All the configurations parameters are passed as arguments in the other script (yolov8_train_sagemaker.py).
"""

logging.basicConfig(level=logging.INFO)
# install the required packages
subprocess.check_call([sys.executable, "-m", "pip", "install","-r", "requirements.txt"])
import yaml
from ultralytics import YOLO
from ultralytics.data.converter import convert_coco # for COCO dataset conversion to YOLO format
logging.info('Ultralytics has been installed successfully')


class PrepareData:
    def __init__(self, base_dir,data_bucket, data_zip, model_ckpt):
        self.s3 = boto3.client('s3')
        self.base_dir = base_dir
        self.data_bucket = data_bucket
        self.data_zip = data_zip
        self.data_folder = self.data_zip.split('.')[0]
        self.data_path = os.path.join(self.base_dir, self.data_folder)
        self.ckpt_s3 = model_ckpt
        if self.ckpt_s3 is not None : 
            self.ckpt = model_ckpt.split('/')[1]
            self.ckpt_path = os.path.join(self.base_dir, self.ckpt)
        

    def prepareData(self):
        # Extract the data from the zip file in the same directory
        logging.info(f'loading data {self.data_zip} at {self.base_dir}')
        zip_file = os.path.join(self.base_dir, self.data_zip)
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(self.base_dir)
        # remove the zip file
        os.remove(zip_file)
        logging.info(f'Data has been downloaded and unzipped successfully at {self.data_path}.')
        logging.info(f'Checking {self.data_folder} directory : {os.listdir(self.base_dir)}')

        # convert synthetic data to YOLO format
        logging.info(f'Converting synthetic data to YOLO format')
        annotations_dir = os.path.join(self.data_path, 'synthetic/annotations')
        labels_dir = os.path.join(self.data_path, 'synthetic')
        convert_coco(labels_dir=annotations_dir,save_dir='YOLO',use_segments=False,use_keypoints=True)
        os.system(f'mv YOLO/labels {labels_dir}')
        shutil.rmtree('YOLO')
        logging.info(f'Data has been converted to YOLO format successfully at {labels_dir}')

        if self.ckpt_s3 is None:
            logging.info('No checkpoint loaded.')
            return None
        else:
            self.s3.download_file(self.data_bucket, self.ckpt_s3, self.ckpt_path)
            logging.info(f'Downloaded ckpt file at : {self.ckpt_path}')
            return self.ckpt_path
    
    def createDataYaml(self):
        # Define the directory and file path
        yaml_path = os.path.join(self.data_path, "data.yaml")
        # Define the content to be written to the YAML file
        data = {
            'path': f'{self.data_path}',
            'train': 'synthetic/images/train',
            'val': 'synthetic/images/validation',
            'test': 'synthetic/images/test',
            'kpt_shape': [11, 3],
            #'flip_idx': [1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 10],
            'names': ['satellite']
        }

        # Write the content to the YAML file
        with open(yaml_path, 'w') as file:
            yaml.dump(data, file, default_flow_style=False)

        print(f"{yaml_path} has been created and written successfully.")
        logging.info(f'data.yaml file has been created successfully at {os.getcwd()}')
        return yaml_path

class TrainModel:
    def __init__(self, yaml_file, model_dir, model, ckpt_path, epochs, batch_size, img_size):
        self.yaml_file = yaml_file
        self.model_dir = model_dir
        self.model = model
        self.epochs = epochs
        self.batch_size = batch_size
        self.img_size = img_size
        self.ckpt_path = ckpt_path

    def trainModel(self):
        # load the model
        logging.info(f'Loading the model {self.model}')
        # check if the data.yaml file is available
        if not os.path.exists(self.yaml_file):
            logging.error('data.yaml file is not available in the data directory. Please check the data directory')
        # train the model
        if self.ckpt_path is None:
            try:
                model = YOLO(f'{self.model}.pt')
            except:
                logging.info(f'Model {self.model} is not available. Please check the model name')
            logging.info(f'Training the model using data from {self.yaml_file}')
            model.train(
                data=self.yaml_file, 
                epochs=self.epochs, 
                batch=self.batch_size, 
                imgsz=self.img_size,
                save=True,
                project =self.model_dir
                )
        else:
            try:
                model = YOLO(f'{self.ckpt_path}')
            except:
                logging.info(f'Model {self.model} is not available. Please check the model name')
            logging.info(f'Training the model with {self.ckpt_path}  using data from {self.yaml_file}')
            model.train(
                data=self.yaml_file,
                epochs=self.epochs,
                batch=self.batch_size,
                imgsz=self.img_size,
                save=True,
                project=self.model_dir
                )
        logging.info(f'Model is saved at {self.model_dir}')

def main(args):
    # Prepare the data
    prepare_data = PrepareData(args.base_dir,args.data_bucket, args.data_file, args.model_ckpt)
    ckpt_path = prepare_data.prepareData()
    yaml_file = prepare_data.createDataYaml()

    # Train the model
    train_model = TrainModel(
        yaml_file,
        model_dir=args.model_dir,
        model=args.model,
        ckpt_path=ckpt_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        img_size=args.img_size
    )
    train_model.trainModel()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train YOLO model on AWS')
    parser.add_argument('--base-dir', type=str, default=os.environ['SM_CHANNEL_TRAINING']) # SM_CHANNEL_TRAINING is the directory where the training data is downloaded on the SageMaker instance.
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR']) # SM_MODEL_DIR is the directory where the model artifacts are saved on the SageMaker instance. Use this directory to save the model and your results. (automatically uploaded to S3 by SageMaker)
    parser.add_argument('--data-bucket', type=str, required=True)
    parser.add_argument('--data-file', type=str, required=True)
    parser.add_argument('--model-ckpt', type=str, default=None,required=False)
    parser.add_argument('--model', type=str, default='yolov8s-pose')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--img-size', type=int, default=320)
    args = parser.parse_args()
    main(args)
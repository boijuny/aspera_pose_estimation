import sagemaker #ignore
from sagemaker.pytorch import PyTorch
import logging

"""
AUTHOR: Matthieu Marchal (SII Internship)

LAST UPDATED: 2024-07-14

DESCRIPTION:
    This script launch a SageMaker training job for YOLOv8 using the PyTorch estimator.

    ###! WARNING ! : Please carefully use an appropriate INSTANCE_TYPE and INSTANCE_COUNT to avoid UNEXPECTED COSTS.###
    ml.g5.16xlarge for training on GPU = 5$/hour, 
    ml.c5.xlarge for debugging on CPU < 1$/hour.
    
USAGE:
    On an AWS notebook instance (ex: heatmap-reg), run the script. Please make sure that the notebook instance contains the following:
        - yolov8_train_sagemaker.py
        - src/yolov8_train.py
        - src/requirements.txt (ultralytics + Logging)

CONFIGURATION:
    - data_bucket: The S3 bucket containing the training data. (str)
    - data_zip: The name of the zip file containing the training data. (in YOLO format)
    - output_bucket: The S3 bucket where the training output will be stored. (str)
    - job_name: The name of the training job. (should be unique) (str)
    - output_prefix: The prefix of the output path. (str)
    - model: The YOLO model to use. (str) (please refer to the YOLOv8 documentation for the available models)
    - model_ckpt: The path to the model checkpoint. (str)


MODIFICATIONS:
    For future modifications, adjust parameters in the "CONFIGURATION" section as needed.
"""

### CONFIGURATION ###
INSTANCE_TYPE = 'ml.g5.16xlarge'
INSTANCE_COUNT = 1

DATA_BUCKET = 'sagemaker-data-aspera'
DATA_ZIP  = 'spv2-COCO-s-eqhist.zip'

#YOLO
MODEL = 'yolov8s-pose' # no need to specify the .pt extension
MODEL_CKPT = 'ckpt/yolov8s-train-s-320-128-100-eqhist-ckpt.pt'

#Hyperparameters
EPOCHS = 50 
BATCH_SIZE = 128 
IMG_SIZE = 320

OUTPUT_BUCKET = 'sagemaker-output-aspera'
JOB_NAME = 'yolov8s-train-s-320-128-100-eqhist-finetuning-50-bis' # model-train-dataset-imgsize-batchsize-epochs
OUTPUT_PREFIX = f'yolov8/{JOB_NAME}'

TAGS = [
    {
        'Owner': 'matthieu.marchal@sii.fr',
        'Project': 'Aspera-Training-YOLOv8',
        'CostCenter': '25640',
        'RunningPolicy': '24/7'
    }
]
#####################

# configure the logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)
# Add a stream handler
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.DEBUG)
stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(stream_handler)

role = sagemaker.get_execution_role()

hyperparameters = {
    'data-bucket': DATA_BUCKET,
    'data-file': DATA_ZIP,
    'model': MODEL,
    'epochs': EPOCHS,
    'batch-size': BATCH_SIZE,
    'img-size': IMG_SIZE
}

if MODEL_CKPT is not None:
    hyperparameters['model-ckpt'] = MODEL_CKPT

estimator = PyTorch(entry_point='yolov8_train.py',
                    source_dir='src',
                    role=role,
                    instance_count=INSTANCE_COUNT,
                    instance_type=INSTANCE_TYPE,
                    framework_version='2.3',
                    py_version='py311',
                    hyperparameters=hyperparameters,
                    output_path=f's3://{OUTPUT_BUCKET}/{OUTPUT_PREFIX}',
                    #tags=TAGS

                    )

traininig_input = sagemaker.TrainingInput(s3_data=f's3://{DATA_BUCKET}/{DATA_ZIP}', content_type='application/zip')
estimator.fit({'training': traininig_input}, job_name=JOB_NAME)
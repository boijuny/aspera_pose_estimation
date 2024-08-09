## TO-DO

- Add AWS inference scripts for model testing.

- Replace average analysis by median analysis.
## About 
### Components 
This folder contains : 
- `aws/`: folder that contains the scripts to train a yolov8 model on AWS.
- `results/`: folder that contains training and inference results of the previously trained models.
- `yolov8_keypoints.ipynb`: Notebook to run inferences of trained yolov8 models on spv2-COCO format datasets.

## Model 
### Training with AWS

The `aws/` folder provides materials too either train a cycleGAN-turbo model or run inferences on spv2-COCO dataset (convert synthetic to sunlamp for exemple).

 To have more details on how to train or run inferences using AWS SageMaker and S3 please refer to the [README_AWS]().

### Testing
You can test the models using the notebook [yolov8_keypoints.ipynb](yolov8_keypoints.ipynb). No AWS inference scripts are available for now.

Despite GPU is recommended to perform tests on large datasets (>1000 images), the notebook provides CPU (local machine) running configuration.


### Model Details <a name = "model-details"></a>

**Custom Yolov8 Model**

- **Network**: Ultralytics Yolov8s
- **Training Dataset**: *spv2-COCO* dataset format (size s or xs)
- **Input/Output**: Image path / 11 keypoints in 2D coordinates

## Results
### Training
You can found the retrieved results of AWS YOLOv8 trainings in `results/xxx_train` folders. *Each folder corresponds to the use of cycleGAN-turbo enhancement or not (sy = no GAN, sy2li = synthetic 2 lightbox GAN, sy2su = synthetic 2 sunlamp GAN)*. Each folder contains the `weights` and `training metrics` accorded to yolov8s-pose training. 

Moreover, due to multiple finetunings, **a folder called `archives` is added**. It contains the precedent trainings that led to the final results.

For example *yolov8s-spv2-COCO-xs-sy2li* had 3 trainings : **base** (`results/sy2li_train/archives/base/`) --> **finetuning1** (`results/sy2li_train/archives/finetuning1/`) --> **finetuning2** (`results/sy2li_train/`)

### Tests

In `results/xxx_test`, you can find 4 kinds of files :

- `xxx_keypoints_xxx_xxx_image.png`: Illustrations
- `xxx_keypoints_avg_metrics.csv`: 1 line csv file of average metrics on a test images set.
- `xxx_keypoints_metrics.csv`: contains all the predicted keypoints, their MSE and PCK on a test images set.
- `xxx_keypoints_regression.json`: contains all the results on a test images set. **(This file is used for pose_estimation)**

## Documentation <a name = "documentation"></a>
- **Internship Report**: [Matthieu Marchal's report on ASPERA project (2024)]()
- **YOLOV8 Documentation**: [Ultralytics official documentation](https://docs.ultralytics.com/models/yolov8/)

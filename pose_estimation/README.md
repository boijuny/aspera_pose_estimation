## TO-DO

- Study in details the impact of Ransac parameters (n_loop, threshold). Found possible improvements.

## About 
### Components 
This folder contains : 
- `results/`: folder that contains results.
- `pose.ipynb`: Notebook to run pose estimation using  selected PnP solver and predicted values contained in `xxx_keypoints_regression.json` files.

## Testing
You can run tests using multiple PnP Ransac solvers :
- Iterative PnP
- EPnP

**Warning: These tests can run only if the prediction file `xxx_keypoints_regression.json` (predicted keypoints for each images) mathes with the correct dataset.**

- `sy_test/xxx_keypoints_regression.json` --> `spv2-COCO-s`
- `sy2li_test/xxx_keypoints_regression.json` --> `spv2-COCO-sy2li-xs`
- `sy2li_test/xxx_keypoints_regression.json` --> `spv2-COCO-sy2su-xs`


## Results

In `results/xxx_test`, you can find 4 kinds of files :

- `xxx_pose_xxx_distribution.png`: Illustrations of score distribution
- `xxx_pose_avg_metrics.csv`: 1 line csv file of average metrics on a test images set.
- `xxx_pose_metrics.csv`: contains all the predicted pose and their scores.


## Documentation <a name = "documentation"></a>
- **Internship Report**: [Matthieu Marchal's report on ASPERA project (2024)]()
- **OpenCV2 Documentation**: [Perspective-n-Points](https://docs.opencv.org/3.4/d5/d1f/calib3d_solvePnP.html)
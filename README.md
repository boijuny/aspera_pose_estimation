## 📝 Table of Contents

- [About](#-about)
- [Getting Started](#-getting-started)
- [Model Details](#-model-details)
- [Usage](#-usage)

## 🧐 About <a name = "about"></a>

The ASPERA Pose Estimation project leverages advanced computer vision techniques to tackle keypoints regression and pose estimation tasks. It utilizes the state-of-the-art Yolov8 model to deliver robust pose estimation solutions through keypoint detection.

### Components 
This repository contains : 
- `assets/`: files for README.md.
- `data/`: folder that should contain the datasets in spv2-COCO format. They can be generated using the [ASPERA_dataset_generation]() repository.
- `keypoints_regression/`: folder that contains the scripts and results of ai-based model keypoints regression.
- `pose_estimation/`: folder that contains the scripts and results for PnP solver pose estimation.
- `utils/`: useful classes and functions.

## 🏁 Getting Started <a name = "getting_started"></a>

Follow these steps to set up and start using the ASPERA Pose Estimation models:

### Prerequisites

Make sure you have the following installed:
- **Git**: For cloning the repository.
- **Python**: Programming language used, along with `pip` for package management.
- **Virtual Environment**: Recommended for managing Python dependencies.

### Installing

#### Cloning the Repository

1. Open your terminal.
2. Navigate to the directory where you want to set up the project.
3. Clone the repository using:
   ```bash
   git clone https://github.com/your-repository/ASPERA_pose_estimation.git
   ```
4. Navigate into the project directory:
   ```bash
   cd ASPERA_pose_estimation
   ```

#### Python and Virtual Environment

Set up your Python environment as follows:
1. Ensure Python and pip are installed. If not, download them from [Python's official site](https://www.python.org/downloads/).
2. Create and activate a virtual environment:
   - **Windows**:
     ```bash
     python -m venv venv
     venv\Scripts\activate
     ```
   - **macOS/Linux**:
     ```bash
     python -m venv venv
     source venv/bin/activate
     ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 🎈 Usage <a name = "usage"></a>

Navigate to specific directories for detailed steps and usage:

1. **Keypoints Regression**: Use the  [keypoints_regression/README.md](keypoints_regression/README.md) to have more informations on keypoints regression process.
2. **Pose Estimation**: Use the  [pose_estimation/README.md](pose_estimation/README.md) to have more informations on pose estimation process.
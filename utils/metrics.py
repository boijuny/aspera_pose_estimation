
import numpy as np
import matplotlib.pyplot as plt
import sys 
import os
import csv
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import visualizations

 ############################## 1. KEYPOINT REGRESSION METRICS ##############################
# The following functions are used to evaluate the keypoint regression performance of a model.

# compute PCKMatrix between two lists
def calculatePCKMat(gt_keypoints,pred_keypoints, threshold):
    """Calculate the percentage of correct keypoints (PCK) matrix between two lists of keypoints.
    Use it to search for matching keypoints between the ground truth and predicted keypoints.

    :param gt_keypoints: List of true keypoints in format [(x,y),(x,y)...].
    :param pred_keypoints: List of predicted keypoints [(x,y),(x,y)...]
    :param threshold: The distance threshold for a correct keypoint.
    :return: The PCK matrix between the true and predicted keypoints."""

    num_keypoints = len(gt_keypoints)
    pck_matrix = np.zeros((num_keypoints, num_keypoints))
    for i, gt in enumerate(gt_keypoints):
        for j, pred in enumerate(pred_keypoints):
            distance = np.linalg.norm(np.array(gt) - np.array(pred))
            if distance < threshold:
                pck_matrix[i, j] = 1
    return pck_matrix

def plotPCKMat(pck_matrix):
    plt.figure(figsize=(10, 8))
    plt.imshow(pck_matrix, cmap='viridis', interpolation='nearest')
    plt.colorbar(label='PCK')
    plt.xlabel('Predicted Keypoints')
    plt.ylabel('Ground Truth Keypoints')
    plt.title('Average PCK between Ground Truth and Predicted Keypoints')
    plt.xticks(range(len(pck_matrix)), range(len(pck_matrix)), rotation=90)
    plt.yticks(range(len(pck_matrix)), range(len(pck_matrix)))

    # Annotate maximum PCK values on the plot
    for i in range(len(pck_matrix)):
        max_value = np.max(pck_matrix[i])
        max_index = np.argmax(pck_matrix[i])
        plt.text(max_index, i, f'{max_value:.2f}', ha='center', va='center', color='red')
    
    plt.tight_layout()
    plt.show()


class KeypointsMetrics:
    def __init__(self, results_json,pck_threshold=None):
        """Initialize the KeypointMetrics object with the results of the keypoint regression model.

        :param results_json: List of dictionaries containing the results of the keypoint regression model."""

        self.results_json = results_json
        self.pck_threshold = pck_threshold
        self.images = []
        self.p_keypoints = []
        self.gt_keypoints = []
        self.skeletons = []
        self.pck_values = []
        self.mse_values = []
        self.avg_pck = None
        self.std_pck = None
        self.avg_mse = None
        self.std_mse = None
        self.img_avg_pck = None
        self.img_avg_mse = None
        self.img_fixed = None
        
        self.evaluate()

        self.fig_pck_dis = plt.figure(1,figsize=(10, 5))
        self.fig_mse_dis = plt.figure(2,figsize=(10, 5))
        self.fig_pck_img = plt.figure(3)
        self.fig_mse_img = plt.figure(4)
        self.fig_fixed_img = plt.figure(5)


    def computeMSE(self, y_pred, y_true):
        """Compute the mean squared error between two lists.

        :param y_true: List of true values.
        :param y_pred: List of predicted values.
        :return: The mean squared error between the true and predicted values."""

        # compute the mean squared error
        mse = np.mean((np.array(y_true) - np.array(y_pred))**2)
        return mse
    
    def computePCK(self, y_pred, y_true):
        """Compute the percentage of correct keypoints (PCK) between two lists.

        :param y_true: List of true keypoints.
        :param y_pred: List of predicted keypoints.
        :param threshold: The distance threshold for a correct keypoint.
        :return: The percentage of correct keypoints (PCK) between the true and predicted keypoints."""

        num_keypoints = len(y_true)
        correct_keypoints = 0
        for i in range(num_keypoints):
            distance = np.linalg.norm(np.array(y_true[i]) - np.array(y_pred[i]))
            if distance < self.pck_threshold:
                correct_keypoints += 1
        pck = correct_keypoints / num_keypoints
        return pck
    
    def evaluate(self):
        """Parse the results of the keypoint regression model to extract the images, ground truth keypoints, and predicted keypoints, and calculate the PCK and MSE values.

        :return: The images, ground truth keypoints, predicted keypoints, PCK values, and MSE values."""

        if self.pck_threshold is None:
            raise ValueError('No PCK threshold provided. Please provide a PCK threshold in object declaration: KeypointMetrics(args,pck_threshold=threshold)')
        skip = 0
        for element in self.results_json:
            if element['n_sat'] == 1 and len(element['gt_keypoints']) == len(element['p_keypoints']):
                self.images.append(element['filename'])
                self.gt_keypoints.append(element['gt_keypoints'])
                self.p_keypoints.append(element['p_keypoints'])
                self.skeletons.append(element['skeleton'])

                self.pck_values.append(self.computePCK(element['p_keypoints'], element['gt_keypoints']))
                self.mse_values.append(self.computeMSE(element['p_keypoints'], element['gt_keypoints']))
            else : 
                skip+=1

        self.avg_pck = np.mean(self.pck_values)
        self.std_pck = np.std(self.pck_values)
        self.avg_mse = np.mean(self.mse_values)
        self.std_mse = np.std(self.mse_values)
        print(f' Skipped : {skip/len(self.results_json)*100:.2f} % of the images due to wrong number of keypoints.')

    def plotPCKDis(self):
        """
        Plot a histogram of PCK values at specific sparse values like 0/11, 1/11, ..., 11/11,
        ensuring bars are centered on these values.

        :param pck_values: List of PCK values.
        """
        # Define the bin edges at each specific fraction and offset them so bars are centered
        bin_edges = [(i - 0.5)/11 for i in range(13)]  # Extending the range for proper binning

    
        values = [round(pck * 11) / 11 for pck in self.pck_values]  # Round each PCK value to nearest allowed fraction

        # Plotting the histogram with thin bars
        plt.figure(1)
        plt.hist(values, bins=bin_edges, color='yellow', edgecolor='black', alpha=0.5, rwidth=0.5)  # Thin bars
        plt.axvline(self.avg_pck, color='red', linestyle='dashed', linewidth=1, label='Average')
        plt.xlabel('PCK [u]')
        plt.ylabel('Frequency')
        plt.title('Distribution of PCK Values')
        plt.xticks([x/11 for x in range(12)], [f'{x/11:.2f}' for x in range(12)])  # Set x-ticks to show each specific bin edge value
        self.fig_pck_dis = plt.figure(1)

    def plotMSEDis(self):
        """
        Plot a histogram of PCK values in percentage and between 10% range intervals.
        
        :param pck_values: List of PCK values between 0 and 1.
        """
        # beautiful histogram
        plt.figure(2)
        plt.hist(self.mse_values, bins=50, color='yellow',edgecolor='black', alpha=0.5) # 10 bins for 10% intervals
        plt.axvline(self.avg_mse, color='red', linestyle='dashed', linewidth=1, label='Average')
        plt.xlabel('MSE [px²]')
        plt.ylabel('Frequency')
        plt.title('Distribution of MSE Values')
        self.fig_mse_dis = plt.figure(2)


    def plot(self,img_dir, verbose=True):
        if verbose:
            print('------- Keypoints Regression Scores -------')
            print(f'Average PCK [u]: {self.avg_pck}')
            print(f'Standard deviation of PCK [u]: {self.std_pck}')
            print(f'Average MSE [px²]: {self.avg_mse}')
            print(f'Standard deviation of MSE [px²]: {self.std_mse}')
            print('------------------------------------')
        self.plotPCKDis()
        self.plotMSEDis()
        plt.show()

        # Find the image with the average PCK and MSE values (not max or min)
        idx_avg_pck = np.argmin(np.abs(np.array(self.pck_values) - self.avg_pck))
        idx_avg_mse = np.argmin(np.abs(np.array(self.mse_values) - self.avg_mse))
        idx_fixed = 57
        self.img_avg_pck = self.images[idx_avg_pck]
        self.img_avg_mse = self.images[idx_avg_mse]

        plt.figure(3)
        visualizations.vizualizeAnns(os.path.join(img_dir,self.img_avg_pck),self.p_keypoints[idx_avg_pck],self.skeletons[idx_avg_pck])
        plt.title(f'Predicted Keypoints with Average PCK: {self.avg_pck:.2f} u ({self.img_avg_pck})')
        self.fig_pck_img = plt.figure(3)
        plt.figure(4)
        visualizations.vizualizeAnns(os.path.join(img_dir,self.img_avg_mse),self.p_keypoints[idx_avg_mse],self.skeletons[idx_avg_mse])
        plt.title(f'Predicted Keypoints with Average MSE: {self.avg_mse:.2f} px² ({self.img_avg_mse})')
        self.fig_mse_img = plt.figure(4)
        plt.figure(5)
        visualizations.vizualizeAnns(os.path.join(img_dir,self.images[idx_fixed]),self.p_keypoints[idx_fixed],self.skeletons[idx_fixed])
        plt.title(f'Predicted Keypoints: {self.images[idx_fixed]}')
        plt.legend([f"PCK: {self.pck_values[idx_fixed]:.2f} u, MSE: {self.mse_values[idx_fixed]:.2f} px²"])
        self.fig_fixed_img = plt.figure(5)

    def save(self,file=True,figure=True,img_dir=None, save_dir=None, data_type='synthetic'):
        """
        Save the evaluation results to a CSV file and/or save the figures in PNG format to output directory.
        The csv file base follows the format of the official SPEC2021 Challenge (https://kelvins.esa.int/pose-estimation-2021/submission-rules/).
        """

        if figure:
            self.plot(img_dir)
            self.fig_pck_dis.savefig(save_dir + f'/{data_type}_keypoints_pck_distribution.png')
            self.fig_mse_dis.savefig(save_dir + f'/{data_type}_keypoints_mse_distribution.png')
            self.fig_pck_img.savefig(save_dir + f'/{data_type}_keypoints_avg_pck_image.png')
            self.fig_mse_img.savefig(save_dir + f'/{data_type}_keypoints_avg_mse_image.png')
            self.fig_fixed_img.savefig(save_dir + f'/{data_type}_keypoints_fixed_image.png')


        if save_dir is None:
            raise ValueError('No save directory provided during object initialization. Please provide a save directory: KeypointMetrics(args,save_dir="path/to/save_dir")')
        if file : 
            with open(save_dir + f'/{data_type}_keypoints_metrics.csv', 'w') as file:
                csv_writer = csv.writer(file)
                csv_writer.writerow(['category','filename','p_keypoints','PCK','MSE'])
                for i in range(len(self.images)):
                    csv_writer.writerow([data_type,self.images[i],self.p_keypoints[i],self.pck_values[i],self.mse_values[i]])
            with open(save_dir + f'/{data_type}_keypoints_avg_metrics.csv', 'w') as file:
                file.write('Average PCK [u],Standard deviation of PCK [u],Average MSE [px²],Standard deviation of MSE [px²]\n')
                file.write(f'{self.avg_pck},{self.std_pck},{self.avg_mse},{self.std_mse}\n')

        print('Evaluation results saved successfully.')

        

 #########################################################################################

############################### POSE ESTIMATION METRICS ###############################
# The following functions are used to evaluate the pose estimation performance of a model.
# Based on the evaluation metrics used for the official SPEC2021 Challenge (https://kelvins.esa.int/pose-estimation-2021/scoring/).

class PoseMetrics:
    def __init__(self, images, pred_translations, pred_quaternions, gt_translations, gt_quaternions):
        """Initialize the PoseMetrics object with the estimated and ground truth poses.

        :param images: List of image names.
        :param
        :param pred_translations: List of numpy arrays representing the estimated translations.
        :param pred_quaternions: List of numpy arrays representing the estimated quaternions.
        :param gt_translations: List of numpy arrays representing the ground truth translations.
        :param gt_quaternions: List of numpy arrays representing the ground truth quaternions.
        :param save_dir: The directory to save the evaluation results and figures."""

        self.images = images
        self.p_tra_list = pred_translations
        self.p_quat_list = pred_quaternions
        self.gt_tra_list = gt_translations
        self.gt_quat_list = gt_quaternions

        if len(self.p_tra_list) != len(self.gt_tra_list) or len(self.p_quat_list) != len(self.gt_quat_list):
            raise ValueError('The number of estimated and ground truth poses must be the same.')
        if len(self.p_tra_list) != len(self.p_quat_list) or len(self.gt_tra_list) != len(self.gt_quat_list):
            raise Warning('The number of estimated positions and quaternions ARE NOT THE SAME. Consider checking the data.')
        
        self.p_pos_scr = []
        self.p_ori_scr = []
        self.avg_p_pos_scr = None
        self.avg_p_ori_scr = None
        self.std_p_pos_scr = None
        self.std_p_ori_scr = None
        self.p_tot_scr = None
        self.gt_tot_scr = None
        self.fig_pos = plt.figure(1,figsize=(10, 5))
        self.fig_ori = plt.figure(2,figsize=(10, 5))
        self.evaluate()

    def positionScr(self, est_pos, gt_pos):
        """Calculate the position error between estimated and ground truth positions considering a threshold of 0.002173 (machine precision error).

        :param est_pos: Numpy array of shape (3,) representing the estimated position.
        :param gt_pos: Numpy array of shape (3,) representing the ground truth position.
        :return: The position error between the estimated and ground truth positions."""

        position_error = np.linalg.norm(est_pos - gt_pos) / np.linalg.norm(gt_pos)
        position_threshold = 0.002173
        if position_error < position_threshold:
            return 0
        else:
            return position_error

    def orientationScr(self,est_quat, gt_quat):
        """Calculate the orientation error between estimated and ground truth quaternions considering a threshold of 0.169 degrees (machine precision error)

        :param est_quat: Numpy array of shape (4,) representing the estimated quaternion.
        :param gt_quat: Numpy array of shape (4,) representing the ground truth quaternion.
        :return: The orientation error between the estimated and ground truth quaternions."""

        # Normalize the quaternions to ensure they are unit quaternions
        est_quat = est_quat / np.linalg.norm(est_quat)
        gt_quat = gt_quat / np.linalg.norm(gt_quat)

        # Calculate the dot product and the absolute value of that dot product
        dot_product = np.dot(est_quat, gt_quat)
        abs_dot_product = abs(dot_product)

        # Calculate orientation error using arccos, safely handling edge cases
        if abs_dot_product > 1:
            abs_dot_product = 1  # Correcting numerical issues that can happen with floating point operations
        elif abs_dot_product < -1:
            abs_dot_product = -1

        orientation_error = 2 * np.arccos(abs_dot_product)
        orientation_threshold = 0.169

        if orientation_error < orientation_threshold:
            return 0
        else:
            return orientation_error # in radians

    def poseScr(self,est_pos, est_quat,gt_pos, gt_quat):
        """Calculate the pose score based on the position and orientation scores. The pose score is the sum of the position and orientation scores.

        :param est_pos: Numpy array of shape (3,) representing the estimated position.
        :param gt_pos: Numpy array of shape (3,) representing the ground truth position.
        :param est_quat: Numpy array of shape (4,) representing the estimated quaternion.
        :param gt_quat: Numpy array of shape (4,) representing the ground truth quaternion.
        :return: The pose score between the estimated and ground truth poses."""

        position_score = self.positionScr(est_pos, gt_pos)
        orientation_score = self.orientationScr(est_quat, gt_quat)
        return position_score + orientation_score

    def totalScr(self,estimated_positions, estimated_quaternions, ground_truth_positions, ground_truth_quaternions):
        """Calculate the total score based on the pose scores of all the estimated poses.

        :param estimated_positions: List of numpy arrays representing the estimated positions.
        :param ground_truth_positions: List of numpy arrays representing the ground truth positions.
        :param estimated_quaternions: List of numpy arrays representing the estimated quaternions.
        :param ground_truth_quaternions: List of numpy arrays representing the ground truth quaternions.
        :return: The total score based on the pose scores of all the estimated poses."""

        scores = []
        for est_pos, gt_pos, est_quat, gt_quat in zip(estimated_positions, ground_truth_positions, estimated_quaternions, ground_truth_quaternions):
            scores.append(self.poseScr(est_pos, est_quat, gt_pos, gt_quat))
        return np.mean(scores)

    def evaluate(self,verbose=True):
        """Evaluate the pose estimation performance of a model based on the estimated and ground truth poses.

        :return: a dictionary of the different scores."""
        # Compute the average and standard deviation of the position and orientation scores
        for i in range(len(self.p_tra_list)):
            self.p_pos_scr.append(self.positionScr(self.p_tra_list[i], self.gt_tra_list[i]))
            self.p_ori_scr.append(self.orientationScr(self.p_quat_list[i], self.gt_quat_list[i]))
        
        self.avg_p_pos_scr = np.mean(self.p_pos_scr)
        self.avg_p_ori_scr = np.mean(self.p_ori_scr)
        self.std_p_pos_scr = np.std(self.p_pos_scr)
        self.std_p_ori_scr = np.std(self.p_ori_scr)
        # Compute the total score
        self.p_tot_scr = self.totalScr(self.p_tra_list, self.p_quat_list, self.gt_tra_list, self.gt_quat_list)

    def plot(self,verbose=True):
        if verbose:
            print('------- Pose Estimation Scores -------')
            print(f'Average predicted position score [m]: {self.avg_p_pos_scr}')
            print(f'Standard deviation of predicted position score [m]: {self.std_p_pos_scr}')
            print(f'Average predicted orientation score [rad]: {self.avg_p_ori_scr}')
            print(f'Standard deviation of predicted orientation score [rad]: {self.std_p_ori_scr}')
            print('------- Total Score -------')
            print(f'Total score for predicted poses: {self.p_tot_scr}')
            print('------------------------------------')
        
        # View score distributions with vertical lines for the average scores
        plt.figure(1)

        plt.hist(self.p_pos_scr, bins=50, color='green', edgecolor='black', alpha=0.5)
        plt.axvline(self.avg_p_pos_scr, color='red', linestyle='dashed', linewidth=1, label='Average',)
        if self.avg_p_pos_scr - self.std_p_pos_scr > 0:
            plt.axvline(self.avg_p_pos_scr - self.std_p_pos_scr, color='red', linestyle='dotted', linewidth=0.7)
        else: 
            plt.axvline(0, color='red', linestyle='dotted', linewidth=0.7)
        plt.axvline(self.avg_p_pos_scr + self.std_p_pos_scr, color='red', linestyle='dotted', linewidth=0.7)
        plt.xlabel('Position Score [m]')
        plt.ylabel('Frequency')
        plt.title('Distribution of Position Scores')
        self.fig_pos = plt.figure(1)

    
        plt.figure(2)
        plt.hist(self.p_ori_scr, bins=50, color='green', edgecolor='black', alpha=0.5)
        plt.axvline(self.avg_p_ori_scr, color='red', linestyle='dashed', linewidth=1, label='Average')
        
        if self.avg_p_ori_scr - self.std_p_ori_scr > 0:
            plt.axvline(self.avg_p_ori_scr - self.std_p_ori_scr, color='red', linestyle='dotted', linewidth=0.7)
        else: 
            plt.axvline(0, color='red', linestyle='dotted', linewidth=0.7)
        plt.axvline(self.avg_p_ori_scr + self.std_p_ori_scr, color='red', linestyle='dotted', linewidth=0.7)
        plt.xlabel('Orientation Score [rad]')
        plt.ylabel('Frequency')
        plt.title('Distribution of Orientation Scores')
        self.fig_ori = plt.figure(2)
        if verbose:
            plt.show()
    
    def save(self,file=True,figure=True,save_dir=None, data_type='synthetic',solver_type=None):
        """
        Save the evaluation results to a CSV file and/or save the figures in PNG format to output directory.
        The csv file base follows the format of the official SPEC2021 Challenge (https://kelvins.esa.int/pose-estimation-2021/submission-rules/).

        :param csv: Boolean to save the evaluation results to a CSV file.
        :param figure: Boolean to save the figures in PNG format to output directory.
        """

        if save_dir is None:
            raise ValueError('No save directory provided during object initialization. Please provide a save directory: PoseMetrics(args,save_dir="path/to/save_dir")')

        # Save the evaluation results to a CSV file with csv library
        if file : 
            with open(save_dir + f'/{data_type}_{solver_type}_pose_metrics.csv', 'w') as file:
                csv_writer = csv.writer(file)
                csv_writer.writerow(['category','filename','position_score','orientation_score,total_score'])
                for i in range(len(self.images)):
                    csv_writer.writerow([data_type,self.images[i],self.p_pos_scr[i],self.p_ori_scr[i],(self.p_pos_scr[i]+self.p_ori_scr[i])])
            with open(save_dir + f'/{data_type}_{solver_type}_pose_avg_metrics.csv', 'w') as file:
                file.write('Average predicted position score [m],Standard deviation of predicted position score [m],Average predicted orientation score [rad],Standard deviation of predicted orientation score [rad],Total score for predicted poses\n')
                file.write(f'{self.avg_p_pos_scr},{self.std_p_pos_scr},{self.avg_p_ori_scr},{self.std_p_ori_scr},{self.p_tot_scr}\n')

        if figure:
            self.plot()
            self.fig_pos.savefig(save_dir + f'/{data_type}_{solver_type}_pose_position_score_distribution.png')
            self.fig_ori.savefig(save_dir + f'/{data_type}_{solver_type}_pose_orientation_score_distribution.png')
        print('Evaluation results saved successfully.')
#########################################################################################
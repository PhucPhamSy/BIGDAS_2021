
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import cv2
import pandas as pd
from glob import glob
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.utils import CustomObjectScope
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score
from metrics import dice_loss, dice_coef, iou


H = 256
W = 256

def read_image(path):
    x = cv2.imread(path, cv2.IMREAD_COLOR)  ## (H, W, 3)
    x = cv2.resize(x, (256, 256))
    ori_x = x
    x = x/255.0
    x = x.astype(np.float32)
    x = np.expand_dims(x, axis=0)
    return ori_x, x                                ## (1, 256, 256, 3)




def save_results(ori_x, ori_y, y_pred, save_image_path):
    line = np.ones((H, 10, 3)) * 255

    ori_y = np.expand_dims(ori_y, axis=-1)  ## (256, 256, 1)
    ori_y = np.concatenate([ori_y, ori_y, ori_y], axis=-1) ## (256, 256, 3)

    y_pred = np.expand_dims(y_pred, axis=-1)  ## (256, 256, 1)
    y_pred = np.concatenate([y_pred, y_pred, y_pred], axis=-1) ## (256, 256, 3)

    cat_images = np.concatenate([ori_x, line, ori_y, line, y_pred*255], axis=1)
    cv2.imwrite(save_image_path, cat_images)

def get_test_x(path):
    test_x_dir = os.path.join(path, "images")
    test_x = os.listdir(test_x_dir)
    return test_x


if __name__ == "__main__":
    """ Seeding """
    np.random.seed(42)
    tf.random.set_seed(42)

    """ Folder for saving results """


    """ Load the model """
    with CustomObjectScope({'iou': iou, 'dice_coef': dice_coef}):
        model = tf.keras.models.load_model("files_deeplabv3/model_deeplabv3_256.h5")

    """ Load the test data """
    # dataset_path = "D:/medical_challenge/segmentation/CVC-612/data/"
    path = 'data_test/'    
    test_x = get_test_x(path)

    print(len(test_x))


    SCORE = []
    # for x in tqdm(zip(test_x), total=len(test_x)):
    for x in test_x:
        name = os.path.join(os.getcwd(), "data_test\images", os.path.basename(x))
        """ Read the image and mask """
        ori_x, x = read_image(x)


        """ Predicting the mask """
        y_pred = model.predict(x)[0] > 0.5
        y_pred = np.squeeze(y_pred, axis=-1)
        y_pred = y_pred.astype(np.int32)

        """ Saving the predicted mask """
        save_image_path = os.path.join(os.getcwd(), 'results_test_deeplabv3_256', os.path.basename(name))
        ## os.getcwd: lấy path fiel hiện tại
        ## basename: lấy tên ảnh

        # print("*"*50, save_image_path)
        save_results(ori_x, y_pred, save_image_path)








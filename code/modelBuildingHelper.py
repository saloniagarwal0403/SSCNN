import sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Activation, Add, BatchNormalization, Concatenate, Conv2D, Dense, Flatten, GlobalMaxPooling2D, Lambda, MaxPooling2D, Reshape, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import CreateInputFeatureMaps_average_pooling_multiple_magnifications
from sklearn.model_selection import train_test_split
from tensorflow.python.framework import ops
import datetime, os
import random
import matplotlib
import pickle
import csv
import xlrd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings('ignore')
matplotlib.use('Agg')
random.seed(0)
from sklearn.decomposition import PCA
import gc
from sklearn.base import clone
from multiprocessing import Pool

def plot_training(history, acc_val_image_filename):
    acc = history.history['c_index']
    val_acc = history.history['val_c_index']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(acc))
    plt.figure()
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(epochs, acc, 'b.', label='Training accuracy')
    ax1.plot(epochs, val_acc, 'r-', label='Validation accuracy')
    ax1.set_title('Training and validation accuracy')

    ax2.plot(epochs, loss, 'b.', label='Training loss')
    ax2.plot(epochs, val_loss, 'r-', label='Validation loss')
    ax2.set_title('Training and validation loss')

    plt.legend()
    plt.savefig(acc_val_image_filename)

def loss4(y_true, y_pred):
    temp = y_true*y_pred
    valid_idx = tf.math.greater(0.0,temp)
    valid_y_pred = tf.where(valid_idx, y_pred, 0.0)
    valid_y_true = tf.where(valid_idx, y_true, 0.0)
    loss1 = tf.keras.losses.MSE(valid_y_true, valid_y_pred)

    y_pred2 = tf.where(valid_idx, 0.0, y_pred)
    y_true2 = tf.where(valid_idx, 0.0, y_true)
    valid_idx2 = tf.math.greater(tf.math.abs(y_true2),tf.math.abs(y_pred2))
    valid_y_pred2 = tf.where(valid_idx2, tf.math.abs(y_true2), 0.0)
    valid_y_true2 = tf.where(valid_idx2, tf.math.abs(y_pred2), 0.0)
    loss2 = tf.keras.losses.MSE(valid_y_true2, valid_y_pred2)

    valid_idx3 = tf.math.greater(365.0,tf.math.abs(y_pred))
    valid_loss3 = tf.where(valid_idx3, 1/(tf.math.abs(y_pred)+0.00000001), 0.0)
    loss3 = tf.math.reduce_sum(valid_loss3)
    return loss1+loss2+loss3

def loss5(y_true, y_pred):
    loss1 = tf.keras.losses.MSE(y_true, y_pred)
    valid_idx3 = tf.math.greater(365.0,tf.math.abs(y_pred))
    valid_loss3 = tf.where(valid_idx3, 1/(tf.math.abs(y_pred)+0.00000001), 0.0)
    loss3 = tf.math.reduce_sum(valid_loss3)
    return loss1+loss3

def loss6(y_true, y_pred):
    temp = y_true*y_pred
    valid_idx = tf.math.greater(0.0,temp)
    valid_y_pred = tf.where(valid_idx, y_pred, 0.0)
    valid_y_true = tf.where(valid_idx, y_true, 0.0)
    loss1 = tf.keras.losses.MSE(valid_y_true, valid_y_pred)

    y_pred2 = tf.where(valid_idx, 0.0, y_pred)
    y_true2 = tf.where(valid_idx, 0.0, y_true)
    valid_idx2 = tf.math.greater(tf.math.abs(y_true2),tf.math.abs(y_pred2))
    valid_y_pred2 = tf.where(valid_idx2, tf.math.abs(y_true2), 0.0)
    valid_y_true2 = tf.where(valid_idx2, tf.math.abs(y_pred2), 0.0)
    loss2 = tf.keras.losses.MSE(valid_y_true2, valid_y_pred2)

    # valid_idx3 = tf.math.greater(365.0,tf.math.abs(y_pred))
    # valid_loss3 = tf.where(valid_idx3, 1/(tf.math.abs(y_pred)+0.00000001), 0.0)
    loss3 = tf.math.reduce_sum(tf.math.log(1/(tf.math.abs(y_pred)+0.00000001)))
    return loss1+loss2+loss3

def get_X_Y_columns(this_df):
    Y = this_df[['Time','Occurence']]
    this_df = this_df.drop(columns= ['Time','Occurence'])
    return this_df, Y

def get_features(X):
    train_features = []
    X_filenames = X['filenames'].iloc[:]
    shuf = np.arange(0,len(X_filenames))
    random.shuffle(shuf)
    for i in shuf[0:100]:
        filepaths_i = X_filenames.iloc[i]
        for filepath_i in filepaths_i:
            # print("Working with file: ",i," with path ",filepath_i)
            train_features.extend(CreateInputFeatureMaps_average_pooling_multiple_magnifications.get_model_predictions(filepath_i))
    return train_features

def pca_features_extraction(X, pca, n_pca_f, tensors_size, saving_folder):
    X_filenames = X['filenames'].iloc[:]
    X = X.drop(columns= ["filenames"])
    arguments_for_pooling = []
    count = 0

    #For storing the PCA generated maps uncoment the following for loop and comment rest of the code in this fn
    for i in range(len(X_filenames)):
        ###In parallel store all the filesFeatureMaps
        filepaths_i = X_filenames.iloc[i]
        for filepath_i in filepaths_i:
            CreateInputFeatureMaps_average_pooling_multiple_magnifications.create_tensors(filepath_i, pca, n_pca_f, tensors_size, saving_folder)

def permissible_pairs(X, Y, DAYS_DIFF, tensors_size,saving_folder, count_i):
    permissible_pairs_set1 = []
    permissible_pairs_set2 = []
    image_features_set1 = []
    image_features_set2 = []
    y_true = []
    X_filenames = X['filenames'].iloc[:]
    X = X.drop(columns= ["filenames"])
    arguments_for_pooling = []
    count = 0
    i_j_pairs = []
    if count_i==-1:
        for i in range(len(X)):
            for j in range(i+1, len(X)):
                if Y["Occurence"].iloc[i]==True and (Y["Time"].iloc[i]<(Y["Time"].iloc[j]+DAYS_DIFF)):
                    filepaths_i = X_filenames.iloc[i]
                    filepaths_j = X_filenames.iloc[j]
                    for k in filepaths_i:
                        for l in filepaths_j:
                            # print("Working on file pair: ",filepath_i," and ",filepath_j)
                            img_a = tf.cast(X.iloc[i,:],tf.float32) ## retrieveing all the columns except last as it is for filename
                            img_b = tf.cast(X.iloc[j,:],tf.float32)
                            original_image_features_pickle_file_name = os.path.split(k)[-1]
                            with open(os.path.join(saving_folder,original_image_features_pickle_file_name), 'rb') as handle:
                                image_features_i = pickle.load(handle)
                            image_features_set1.append(image_features_i)
                            original_image_features_pickle_file_name = os.path.split(l)[-1]
                            with open(os.path.join(saving_folder,original_image_features_pickle_file_name), 'rb') as handle:
                                image_features_j = pickle.load(handle)
                            image_features_set2.append(image_features_j)
                            permissible_pairs_set1.append(img_a)
                            permissible_pairs_set2.append(img_b)
                            y_true.append(Y["Time"].iloc[i]-Y["Time"].iloc[j])
                            i_j_pairs.append([i,j])
                            count+=1
                            # print(count)
                        if Y["Occurence"].iloc[j]==True and ((Y["Time"].iloc[i]+DAYS_DIFF)>Y["Time"].iloc[j]):
                            img_a = tf.cast(X.iloc[i,:],tf.float32)
                            img_b = tf.cast(X.iloc[j,:],tf.float32)
                            original_image_features_pickle_file_name = os.path.split(k)[-1]
                            with open(os.path.join(saving_folder,original_image_features_pickle_file_name), 'rb') as handle:
                                image_features_i = pickle.load(handle)
                            image_features_set1.append(image_features_i)
                            original_image_features_pickle_file_name = os.path.split(l)[-1]
                            with open(os.path.join(saving_folder,original_image_features_pickle_file_name), 'rb') as handle:
                                image_features_j = pickle.load(handle)
                            image_features_set2.append(image_features_j)
                            permissible_pairs_set1.append(img_a)
                            permissible_pairs_set2.append(img_b)
                            y_true.append(Y["Time"].iloc[i]-Y["Time"].iloc[j])
                            i_j_pairs.append([i,j])
                            count+=1
                    # print(count)
                # if count==1000:
                #     return image_features_set1, image_features_set2, permissible_pairs_set1 , permissible_pairs_set2 , y_true
        return image_features_set1, image_features_set2, permissible_pairs_set1 , permissible_pairs_set2 , y_true, i_j_pairs
    else:
        valid_pairs = []
        for i in range(len(X)):
            for j in range(i+1, len(X)):
                if Y["Occurence"].iloc[i]==True and (Y["Time"].iloc[i]<(Y["Time"].iloc[j]+DAYS_DIFF)):
                    filepaths_i = X_filenames.iloc[i]
                    filepaths_j = X_filenames.iloc[j]
                    for k in filepaths_i:
                        for l in filepaths_j:
                            # print("Working on file pair: ",filepath_i," and ",filepath_j)
                            img_a = tf.cast(X.iloc[i,:],tf.float32) ## retrieveing all the columns except last as it is for filename
                            img_b = tf.cast(X.iloc[j,:],tf.float32)
                            original_image_features_pickle_file_name_k = os.path.split(k)[-1]
                            # with open(os.path.join(saving_folder,original_image_features_pickle_file_name), 'rb') as handle:
                            #     image_features_i = pickle.load(handle)
                            # image_features_set1.append(image_features_i)
                            original_image_features_pickle_file_name_l = os.path.split(l)[-1]
                            # with open(os.path.join(saving_folder,original_image_features_pickle_file_name), 'rb') as handle:
                            #     image_features_j = pickle.load(handle)
                            # image_features_set2.append(image_features_j)
                            valid_pairs.append([original_image_features_pickle_file_name_k,original_image_features_pickle_file_name_l])
                            permissible_pairs_set1.append(img_a)
                            permissible_pairs_set2.append(img_b)
                            y_true.append(Y["Time"].iloc[i]-Y["Time"].iloc[j])
                            i_j_pairs.append([i,j])
                            # print(count)
                        if Y["Occurence"].iloc[j]==True and ((Y["Time"].iloc[i]+DAYS_DIFF)>Y["Time"].iloc[j]):
                            img_a = tf.cast(X.iloc[i,:],tf.float32)
                            img_b = tf.cast(X.iloc[j,:],tf.float32)
                            original_image_features_pickle_file_name_k = os.path.split(k)[-1]
                            # with open(os.path.join(saving_folder,original_image_features_pickle_file_name), 'rb') as handle:
                            #     image_features_i = pickle.load(handle)
                            # image_features_set1.append(image_features_i)
                            original_image_features_pickle_file_name_l = os.path.split(l)[-1]
                            # with open(os.path.join(saving_folder,original_image_features_pickle_file_name), 'rb') as handle:
                            #     image_features_j = pickle.load(handle)
                            # image_features_set2.append(image_features_j)
                            valid_pairs.append([original_image_features_pickle_file_name_k,original_image_features_pickle_file_name_l])
                            permissible_pairs_set1.append(img_a)
                            permissible_pairs_set2.append(img_b)
                            y_true.append(Y["Time"].iloc[i]-Y["Time"].iloc[j])
                            i_j_pairs.append([i,j])
        shuf = np.arange(0,len(valid_pairs))
        random.shuffle(shuf)
        image_features_set1_valid = []
        image_features_set2_valid = []
        permissible_pairs_set1_valid = []
        permissible_pairs_set2_valid= []
        y_true_valid = []
        i_j_pairs_valid = []
        for i in shuf[0:count_i]:
            permissible_pairs_set1_valid.append(permissible_pairs_set1[i])
            permissible_pairs_set2_valid.append(permissible_pairs_set2[i])
            y_true_valid.append(y_true[i])
            i_j_pairs_valid.append(i_j_pairs[i])
            with open(os.path.join(saving_folder,valid_pairs[i][0]), 'rb') as handle:
                image_features_i = pickle.load(handle)
            image_features_set1_valid.append(image_features_i)
            with open(os.path.join(saving_folder,valid_pairs[i][1]), 'rb') as handle:
                image_features_j = pickle.load(handle)
            image_features_set2_valid.append(image_features_j)
        return image_features_set1_valid, image_features_set2_valid, permissible_pairs_set1_valid , permissible_pairs_set2_valid , y_true_valid, i_j_pairs_valid


def model_def(number_channel,no_clinical_features,first_conv_layer_number_filers,second_conv_layer_number_filers,first_layer_neurons,second_layer_neurons):
    image_input = Input(shape=(None,None,number_channel))
    conv1 = tf.keras.layers.Conv2D(first_conv_layer_number_filers, (3,3), activation='relu')(image_input)
    conv2 = tf.keras.layers.Conv2D(second_conv_layer_number_filers, (3,3), activation='relu')(conv1)
    pool = tf.keras.layers.GlobalAveragePooling2D()(conv2)
    # tf.keras.layers.Conv1D(first_conv_layer_number_filers,(1), activation='relu'),
    # tf.keras.layers.Flatten(),
    clinical_input = Input(shape=(no_clinical_features))
    concatenate_layer = tf.keras.layers.concatenate([pool,clinical_input])
    first_dense = tf.keras.layers.Dense(first_layer_neurons, activation='relu',kernel_regularizer='l1_l2')(concatenate_layer)
    dp1 = tf.keras.layers.Dropout(0.2)(first_dense)
    second_dense = tf.keras.layers.Dense(second_layer_neurons, activation='relu',kernel_regularizer='l1_l2')(dp1)
    dp2 = tf.keras.layers.Dropout(0.2)(second_dense)
    output = tf.keras.layers.Dense(1,kernel_regularizer='l1_l2')(dp2)
    return Model(inputs=[image_input,clinical_input], outputs=output)

def build_model(number_channel,no_clinical_features,first_conv_layer_number_filers,second_conv_layer_number_filers,first_layer_neurons,second_layer_neurons):
    model = model_def(number_channel,no_clinical_features,first_conv_layer_number_filers,second_conv_layer_number_filers,first_layer_neurons,second_layer_neurons)
    print(model.summary())
    ########################
    # SIAMESE NEURAL NETWORK
    ########################
    # Complete model is constructed by calling the branch model on each input image,
    # and then the head model on the resulting 512-vectors.
    img_a = Input(shape=(None,None,number_channel))
    img_b = Input(shape=(None,None,number_channel))

    clinical_a = Input(shape=(no_clinical_features))
    clinical_b = Input(shape=(no_clinical_features))

    xa = model([img_a,clinical_a])
    xb = model([img_b,clinical_b])
#     x = Lambda(lambda x: tf.cast(K.exp(-x[1]) -  K.exp(-x[0]), tf.float32))([xa, xb])
#     x = Lambda(lambda x:x[1] - x[0])([xa, xb])
    subtracted = tf.keras.layers.Subtract()([xa, xb])
    # probability_output = tf.keras.activations.sigmoid(subtracted)
    # x = Lambda(lambda x:tf.concat(x,1))([xa, xb])
#     x = tf.cast(xb-xa, tf.float32)
    model_f = Model(inputs=[img_a,img_b,clinical_a,clinical_b], outputs=[subtracted])
    return model_f


def c_index_prediction(y_true, y_pred):
    correct=0
    for i in range(len(y_true)):
        if (y_true[i]*y_pred[i])>0:
            correct+=1
    total = len(y_pred)
    return float(correct)/float(total)

def c_index(y_true, y_pred):
    temp = y_true*y_pred
    valid_idx = tf.math.greater(temp,0.0)
    correct_tensor = tf.where(valid_idx, 1.0, 0.0)
    return tf.reduce_mean(correct_tensor)

def case_wise_soft_voting(predictions, true_values, i_j_pairs):
    i_j_pairs_dict = {}
    for i in range(len(i_j_pairs)):
        this_pair = (i_j_pairs[i][0],i_j_pairs[i][1])
        if this_pair in i_j_pairs_dict:
            prediction,true_value = i_j_pairs_dict[this_pair]
            prediction = prediction+predictions[i]
            i_j_pairs_dict[this_pair] = [prediction,true_value]
        else:
            i_j_pairs_dict[this_pair] = [predictions[i],true_values[i]]
    y_true = []
    y_pred = []
    for k,v in i_j_pairs_dict.items():
        y_pred.append(v[0])
        y_true.append(v[1])
    return c_index_prediction(y_true,y_pred)

def case_wise_voting(predictions, true_values, i_j_pairs):
    i_j_pairs_dict = {}
    for i in range(len(i_j_pairs)):
        this_pair = (i_j_pairs[i][0],i_j_pairs[i][1])
        if this_pair in i_j_pairs_dict:
            votes_1,votes_neg_1,true_value = i_j_pairs_dict[this_pair]
            if predictions[i]>0:
                votes_1+=1
            else:
                votes_neg_1+=1
            i_j_pairs_dict[this_pair] = [votes_1,votes_neg_1,true_value]
        else:
            if predictions[i]>0:
                votes_1=1
                votes_neg_1=0
            else:
                votes_neg_1=1
                votes_1=0
            i_j_pairs_dict[this_pair] =  [votes_1,votes_neg_1,true_values[i]]
    y_true = []
    y_pred = []
    for k,v in i_j_pairs_dict.items():
        if v[0]>v[1]:
            y_pred.append(1)
        else:
            y_pred.append(-1)
        y_true.append(v[2])
    return c_index_prediction(y_true,y_pred)

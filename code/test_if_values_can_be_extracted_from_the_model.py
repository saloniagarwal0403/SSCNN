# conda activate SurvivalAnalysis_January2021
# python Siamese_with_conv_parallel_processing_average_pooling_multiple__magnifications.py>Epoch_4_using_average.txt
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

def get_features_dataset(X, Y, tensors_size,saving_folder):
    X_filenames = X['filenames'].iloc[:]
    X = X.drop(columns= ["filenames"])
    image_features = []
    clinical_features = []
    case_index = []
    y_value = []
    for i in range(len(X)):
        filepaths_i = X_filenames.iloc[i]
        for k in filepaths_i:
            img_a = tf.cast(X.iloc[i,:],tf.float32) ## retrieveing all the columns except last as it is for filename
            original_image_features_pickle_file_name = os.path.split(k)[-1]
            with open(os.path.join(saving_folder,original_image_features_pickle_file_name), 'rb') as handle:
                image_features_i = pickle.load(handle)
            image_features.append(image_features_i)
            clinical_features.append(img_a)
            y_value.append([Y["Time"].iloc[i],Y["Occurence"].iloc[i]])
            case_index.append(i)
    return image_features, clinical_features, y_value, case_index

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

def c_index_calculation(preds, y_true, case_indexs,DAYS_DIFF):
    permisible_true = []
    permissible_predited = []
    i_j_pairs = []
    for i in range(len(preds)):
        for j in range(i+1,len(preds)):
            if (y_true[i][1]==True and (y_true[i][0]<(y_true[j][0]+DAYS_DIFF))) or (y_true[j][1]==True and (y_true[j][0]<(y_true[i][0]+DAYS_DIFF))):
                permisible_true.append(y_true[i][0]-y_true[j][0])
                permissible_predited.append(preds[i]-preds[j])
                i_j_pairs.append([case_indexs[i],case_indexs[j]])
    return [c_index_prediction(permisible_true,permissible_predited), case_wise_voting(permissible_predited,permisible_true,i_j_pairs), case_wise_voting(permissible_predited,permisible_true,i_j_pairs)]

def shuffle(image_features):
    image_features=np.asarray(image_features)
    image_features_shuffled = np.reshape(np.copy(image_features), (image_features.shape[0],image_features.shape[1] * image_features.shape[2], image_features.shape[3]))
    image_features_shuffled_new = []
    for i_case in range(image_features.shape[0]):
        temp = image_features_shuffled[i_case]
        np.random.shuffle(temp)
        temp_reshaped = np.reshape(temp, image_features.shape[1:])
        image_features_shuffled_new.append(temp_reshaped)
    image_features_shuffled_new=np.asarray(image_features_shuffled_new)
    return image_features_shuffled_new

def prediction_by_average_over_wsi(VE_before_AP,VE_after_AP,image_features,clinical_features,y_values, case_indexs,DAYS_DIFF):
    features_before_AP = VE_before_AP.predict(image_features)
    idx_dict = {}
    for i in range(len(case_indexs)):
        if case_indexs[i] in idx_dict:
            this_case_features = idx_dict[case_indexs[i]]
            this_case_current = list(np.reshape(features_before_AP[i], (features_before_AP[i].shape[0]*features_before_AP[i].shape[1],features_before_AP[i].shape[2])))
            for t in this_case_current:
                this_case_features.append(t)
            idx_dict[case_indexs[i]] = this_case_features
        else:
            idx_dict[case_indexs[i]] = list(np.reshape(features_before_AP[i], (features_before_AP[i].shape[0]*features_before_AP[i].shape[1],features_before_AP[i].shape[2])))
    for i in idx_dict.keys():
        this_case_features = idx_dict[i]
        this_case_features_avg = np.mean(np.asarray(this_case_features), axis=0)
        idx_dict[i] = this_case_features_avg
    y_trues = {}
    idx_dic = {}
    for i in range(len(case_indexs)):
        this_case_clinical_features = clinical_features[i]
        if case_indexs[i] not in idx_dic:
            idx_dic[case_indexs[i]] = tf.cast(np.concatenate((np.array(idx_dict[case_indexs[i]]),np.array(this_case_clinical_features))),tf.float32)
            y_trues[case_indexs[i]] = y_values[i]
    y_model_preds = VE_after_AP.predict(np.array(list(idx_dic.values())))
    y_true = []
    y_pred = []
    for i in idx_dic.keys():
        y_true.append(y_trues[list(idx_dic.keys())[i]])
        y_pred.append(y_model_preds[i])
    with open("Values_feature.pickle", 'wb') as handle:
        pickle.dump([y_pred, y_true],handle)
    return c_index_calculation(y_pred, y_true, list(idx_dic.keys()),DAYS_DIFF)[0]

# TODO: Check how to inset the image feature maps in dataframe
os.environ["CUDA_VISIBLE_DEVICES"]='1'
BATCH_SIZE = 16
DAYS_DIFF = 365
output_files_folder = os.path.join(r"/home","sxa171531","images","TCGA-GBM","output_files")
train_pickle_filename = os.path.join(output_files_folder,'train.pickle')
val_pickle_filename = os.path.join(output_files_folder,'val.pickle')
test_pickle_filename = os.path.join(output_files_folder,'test.pickle')
with open(train_pickle_filename, 'rb') as handle:
    df_train = pickle.load(handle)
with open(test_pickle_filename, 'rb') as handle:
    df_test = pickle.load(handle)
with open(val_pickle_filename, 'rb') as handle:
    df_val =pickle.load(handle)
df_train_X, df_train_Y = get_X_Y_columns(df_train)
df_val_X, df_val_Y = get_X_Y_columns(df_val)
df_test_X, df_test_Y = get_X_Y_columns(df_test)
predictions = []
fields = ['no_pca','no_training_pairs', 'conv_f', 'conv_s', 'f', 's', 'lr',
'wsi_train', 'wsi_val', 'wsi_test',
'case_voting_train', 'case_voting_val', 'case_voting_test',
'case_soft-voting_train', 'case_soft-voting_val', 'case_soft-voting_test',
'SHUFFLED',
'wsi_train', 'wsi_val', 'wsi_test',
'case_voting_train', 'case_voting_val', 'case_voting_test',
'case_soft-voting_train', 'case_soft-voting_val', 'case_soft-voting_test',
'AVGMAPS',
'train', 'val', 'test',
'','0 DAYS_DIFF',
'wsi_train_0_days_diff', 'wsi_val_0_days_diff', 'wsi_test_0_days_diff',
'case_voting_train_0_days_diff', 'case_voting_val_0_days_diff', 'case_voting_test_0_days_diff',
'case_soft-voting_train_0_days_diff', 'case_soft-voting_val_0_days_diff', 'case_soft-voting_test_0_days_diff',
'SHUFFLED',
'wsi_train', 'wsi_val', 'wsi_test',
'case_voting_train', 'case_voting_val', 'case_voting_test',
'case_soft-voting_train', 'case_soft-voting_val', 'case_soft-voting_test',
'AVGMAPS',
'train', 'val', 'test'
 ]
csv_filename = os.path.join(output_files_folder,"Results-testing.csv")
with open(csv_filename, 'w') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(fields)

conv_first_layer_filters = [256] # previously 32
conv_second_layer_filters = [256] #previously 32
first_layer_neurons = [8] #previously 32
second_layer_neurons = [4] #previously 32
lrs = [0.0001]
results = []
number_pca_features = [8]
tensors_size = 100
cvs_file_inputs = []
for n_pca_f in number_pca_features:
    ### FOR BUILDING VALID DATASETS
    gc.disable()
    train_folder = os.path.join(output_files_folder,'PCA_features_train_'+str(n_pca_f))
    val_folder = os.path.join(output_files_folder,'PCA_features_val_'+str(n_pca_f))
    test_folder = os.path.join(output_files_folder,'PCA_features_test_'+str(n_pca_f))
    image_features_train, clinical_features_train, y_value_train, case_index_train = get_features_dataset(df_train_X, df_train_Y, tensors_size,train_folder)
    image_features_val, clinical_features_val, y_value_val, case_index_val = get_features_dataset(df_val_X, df_val_Y, tensors_size,val_folder)
    image_features_test, clinical_features_test, y_value_test, case_index_test = get_features_dataset(df_test_X, df_test_Y,  tensors_size,test_folder)

    image_features_train_shuffled = shuffle(image_features_train)
    image_features_val_shuffled = shuffle(image_features_val)
    image_features_test_shuffled = shuffle(image_features_test)

    for count_i in [50000]:
        for conv_f in conv_first_layer_filters:
            for conv_s in conv_second_layer_filters:
                for f in first_layer_neurons:
                    for s in second_layer_neurons:
                        for lr in lrs:
                            this_config_predictions = []
                            this_config_results = [n_pca_f,count_i,conv_f,conv_s,f,s,lr]
                            weights_folder = os.path.join(output_files_folder,"weights")
                            filepath= os.path.join(weights_folder, "Siamese_with_max_pooling_separate_training_first"+str("-"+str(count_i)+"-"+str(n_pca_f)+"-"+"loss4"+"-"+str(conv_f)+"-"+str(conv_s)+"-"+str(f)+"-"+str(s)+"-"+str(lr))+".h5")
                            base_model = build_model(n_pca_f,len(clinical_features_train[0]),conv_f,conv_s,f,s)
                            base_model.compile(optimizer=tf.keras.optimizers.Adam(lr = lr),
                                               loss = loss4,
                                               metrics=[c_index])
                            base_model.load_weights(filepath)
                            value_extractor = base_model.layers[-2]
                            pred_test = value_extractor.predict([np.array(image_features_test), np.array(clinical_features_test)])
                            softvoting_values = {}
                            for i in range(len(case_index_test)):
                                idx = case_index_test[i]
                                if idx in softvoting_values:
                                    temp = softvoting_values[idx]
                                    temp[0] = temp[0]+pred_test[i]
                                    temp[1] = temp[1]+1
                                    softvoting_values[idx] = temp
                                else:
                                    softvoting_values[idx] = [pred_test[i],1,y_value_test[i]]

                            with open("Values"+str(count_i)+"_softvoting.pickle", 'wb') as handle:
                                pickle.dump(list(softvoting_values.values()),handle)

                            wsi_test, case_voting_test, case_soft_voting_test = c_index_calculation(pred_test, y_value_test, case_index_test,DAYS_DIFF)
                            pred_val = value_extractor.predict([np.array(image_features_val), np.array(clinical_features_val)])
                            wsi_val, case_voting_val, case_soft_voting_val = c_index_calculation(pred_val, y_value_val, case_index_val,DAYS_DIFF)
                            pred_train = value_extractor.predict([np.array(image_features_train), np.array(clinical_features_train)])
                            wsi_train, case_voting_train, case_soft_voting_train = c_index_calculation(pred_train, y_value_train, case_index_train,DAYS_DIFF)

                            this_config_results.append(wsi_train)
                            this_config_results.append(wsi_val)
                            this_config_results.append(wsi_test)
                            this_config_results.append(case_voting_train)
                            this_config_results.append(case_voting_val)
                            this_config_results.append(case_voting_test)
                            this_config_results.append(case_soft_voting_train)
                            this_config_results.append(case_soft_voting_val)
                            this_config_results.append(case_soft_voting_test)

                            this_config_results.append(" ")
                            pred_test_shuffled = value_extractor.predict([np.array(image_features_test_shuffled), np.array(clinical_features_test)])
                            wsi_test_shuffled , case_voting_test_shuffled , case_soft_voting_test_shuffled  = c_index_calculation(pred_test_shuffled , y_value_test, case_index_test,DAYS_DIFF)
                            pred_val_shuffled  = value_extractor.predict([np.array(image_features_val_shuffled), np.array(clinical_features_val)])
                            wsi_val_shuffled , case_voting_val_shuffled , case_soft_voting_val_shuffled  = c_index_calculation(pred_val_shuffled , y_value_val, case_index_val,DAYS_DIFF)
                            pred_train_shuffled  = value_extractor.predict([np.array(image_features_train_shuffled), np.array(clinical_features_train)])
                            wsi_train_shuffled , case_voting_train_shuffled , case_soft_voting_train_shuffled  = c_index_calculation(pred_train_shuffled , y_value_train, case_index_train,DAYS_DIFF)

                            this_config_results.append(wsi_train_shuffled )
                            this_config_results.append(wsi_val_shuffled )
                            this_config_results.append(wsi_test_shuffled )
                            this_config_results.append(case_voting_train_shuffled )
                            this_config_results.append(case_voting_val_shuffled )
                            this_config_results.append(case_voting_test_shuffled )
                            this_config_results.append(case_soft_voting_train_shuffled)
                            this_config_results.append(case_soft_voting_val_shuffled)
                            this_config_results.append(case_soft_voting_test_shuffled)

                            this_config_results.append(" ")#Averagemaps
                            inputs = tf.keras.Input(shape=(None,None, n_pca_f))
                            x1 = value_extractor.layers[1](inputs)
                            # x2 = value_extractor.layers[2](x1)
                            outputs = value_extractor.layers[2](x1)
                            VE_before_AP = tf.keras.Model(inputs, outputs)
                            inputs = tf.keras.Input(shape=(conv_s+len(clinical_features_train[0])))
                            x1 = value_extractor.layers[-5](inputs)
                            x2 = value_extractor.layers[-3](x1)
                            outputs = value_extractor.layers[-1](x2)
                            VE_after_AP = Model(inputs=inputs, outputs=outputs)
                            wsi_train_avg = prediction_by_average_over_wsi(VE_before_AP,VE_after_AP,np.array(image_features_train),np.array(clinical_features_train),y_value_train, case_index_train,DAYS_DIFF)
                            wsi_val_avg = prediction_by_average_over_wsi(VE_before_AP,VE_after_AP,np.array(image_features_val),np.array(clinical_features_val),y_value_val, case_index_val,DAYS_DIFF)
                            wsi_test_avg = prediction_by_average_over_wsi(VE_before_AP,VE_after_AP,np.array(image_features_test),np.array(clinical_features_test),y_value_test, case_index_test,DAYS_DIFF)
                            this_config_results.append(wsi_train_avg)
                            this_config_results.append(wsi_val_avg)
                            this_config_results.append(wsi_test_avg)


                            this_config_results.append(" ")
                            this_config_results.append(" ")

                            wsi_test_0_days_diff, case_voting_test_0_days_diff, case_soft_voting_test_0_days_diff = c_index_calculation(pred_test, y_value_test, case_index_test,0)
                            wsi_val_0_days_diff, case_voting_val_0_days_diff, case_soft_voting_val_0_days_diff = c_index_calculation(pred_val, y_value_val, case_index_val,0)
                            wsi_train_0_days_diff, case_voting_train_0_days_diff, case_soft_voting_train_0_days_diff = c_index_calculation(pred_train, y_value_train, case_index_train,0)

                            this_config_results.append(wsi_train_0_days_diff)
                            this_config_results.append(wsi_val_0_days_diff)
                            this_config_results.append(wsi_test_0_days_diff)
                            this_config_results.append(case_voting_train_0_days_diff)
                            this_config_results.append(case_voting_val_0_days_diff)
                            this_config_results.append(case_voting_test_0_days_diff)
                            this_config_results.append(case_soft_voting_train_0_days_diff)
                            this_config_results.append(case_soft_voting_val_0_days_diff)
                            this_config_results.append(case_soft_voting_test_0_days_diff)

                            this_config_results.append(" ")
                            pred_test_shuffled = value_extractor.predict([np.array(image_features_test_shuffled), np.array(clinical_features_test)])
                            wsi_test_shuffled , case_voting_test_shuffled , case_soft_voting_test_shuffled  = c_index_calculation(pred_test_shuffled , y_value_test, case_index_test,0)
                            pred_val_shuffled  = value_extractor.predict([np.array(image_features_val_shuffled), np.array(clinical_features_val)])
                            wsi_val_shuffled , case_voting_val_shuffled , case_soft_voting_val_shuffled  = c_index_calculation(pred_val_shuffled , y_value_val, case_index_val,0)
                            pred_train_shuffled  = value_extractor.predict([np.array(image_features_train_shuffled), np.array(clinical_features_train)])
                            wsi_train_shuffled , case_voting_train_shuffled , case_soft_voting_train_shuffled  = c_index_calculation(pred_train_shuffled , y_value_train, case_index_train,0)

                            this_config_results.append(wsi_train_shuffled )
                            this_config_results.append(wsi_val_shuffled )
                            this_config_results.append(wsi_test_shuffled )
                            this_config_results.append(case_voting_train_shuffled )
                            this_config_results.append(case_voting_val_shuffled )
                            this_config_results.append(case_voting_test_shuffled )
                            this_config_results.append(case_soft_voting_train_shuffled)
                            this_config_results.append(case_soft_voting_val_shuffled)
                            this_config_results.append(case_soft_voting_test_shuffled)

                            this_config_results.append(" ")#Averagemaps
                            inputs = tf.keras.Input(shape=(None,None, n_pca_f))
                            x1 = value_extractor.layers[1](inputs)
                            outputs = value_extractor.layers[2](x1)
                            VE_before_AP = tf.keras.Model(inputs, outputs)
                            inputs = tf.keras.Input(shape=(conv_s+len(clinical_features_train[0])))
                            x1 = value_extractor.layers[-5](inputs)
                            x2 = value_extractor.layers[-3](x1)
                            outputs = value_extractor.layers[-1](x2)
                            VE_after_AP = Model(inputs=inputs, outputs=outputs)
                            wsi_train_avg = prediction_by_average_over_wsi(VE_before_AP,VE_after_AP,np.array(image_features_train),np.array(clinical_features_train),y_value_train, case_index_train,0)
                            wsi_val_avg = prediction_by_average_over_wsi(VE_before_AP,VE_after_AP,np.array(image_features_val),np.array(clinical_features_val),y_value_val, case_index_val,0)
                            wsi_test_avg = prediction_by_average_over_wsi(VE_before_AP,VE_after_AP,np.array(image_features_test),np.array(clinical_features_test),y_value_test, case_index_test,0)
                            this_config_results.append(wsi_train_avg)
                            this_config_results.append(wsi_val_avg)
                            this_config_results.append(wsi_test_avg)

                            wsi_test_avg = prediction_by_average_over_wsi(VE_before_AP,VE_after_AP,np.array(image_features_test_shuffled),np.array(clinical_features_test),y_value_test, case_index_test,0)
                            this_config_results.append(wsi_test_avg)

                            wsi_test_avg = prediction_by_average_over_wsi(VE_before_AP,VE_after_AP,np.zeros(np.shape(image_features_test)),np.array(clinical_features_test),y_value_test, case_index_test,0)
                            this_config_results.append(wsi_test_avg)

                            wsi_test_avg = prediction_by_average_over_wsi(VE_before_AP,VE_after_AP,np.array(image_features_test),np.zeros(np.shape(clinical_features_test)),y_value_test, case_index_test,0)
                            this_config_results.append(wsi_test_avg)

                            pred_test = value_extractor.predict([np.zeros(np.shape(image_features_test)), np.array(clinical_features_test)])
                            wsi_test , case_voting_test , case_soft_voting_test  = c_index_calculation(pred_test , y_value_test, case_index_test,0)
                            this_config_results.append(["No WSI features",wsi_test , case_voting_test , case_soft_voting_test])

                            pred_test = value_extractor.predict([np.array(image_features_test), np.zeros(np.shape(clinical_features_test))])
                            wsi_test , case_voting_test , case_soft_voting_test  = c_index_calculation(pred_test , y_value_test, case_index_test,0)
                            this_config_results.append(["No clinical features",wsi_test , case_voting_test , case_soft_voting_test])

                            print(this_config_results)
                            predictions.append(this_config_results)
                            with open(csv_filename, 'w') as csvfile:
                                csvwriter = csv.writer(csvfile)
                                csvwriter.writerow(fields)
                                csvwriter.writerows(predictions)

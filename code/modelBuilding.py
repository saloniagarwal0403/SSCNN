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
import warnings
warnings.filterwarnings('ignore')
matplotlib.use('Agg')
random.seed(0)
import gc
from modelBuildingHelper import loss4, get_X_Y_columns, get_features, pca_features_extraction, permissible_pairs, model_def, build_model, c_index_prediction, c_index, case_wise_soft_voting, case_wise_voting
os.environ["CUDA_VISIBLE_DEVICES"]='0'
EPOCHS = 10000
PATIENCE = 150
BATCH_SIZE = 32
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
'wsi_train', 'wsi_val',
'case_voting_train', 'case_voting_val',
'case_soft-voting_train', 'case_soft-voting_val'
 ]
csv_filename = os.path.join(output_files_folder,"Results-modelBuilding.csv")
with open(csv_filename, 'w') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(fields)

conv_first_layer_filters = [32]
conv_second_layer_filters = [64]
first_layer_neurons = [64]
second_layer_neurons = [8]
lrs = [0.0001]
results = []
number_pca_features = [8,16,32,64,128,256,0.7,0.8,0.9,0.95]
tensors_size = 100
cvs_file_inputs = []
for n_pca_f in number_pca_features:
    gc.disable()
    train_folder = os.path.join(output_files_folder,'PCA_features_train_'+str(n_pca_f))
    val_folder = os.path.join(output_files_folder,'PCA_features_val_'+str(n_pca_f))
    test_folder = os.path.join(output_files_folder,'PCA_features_test_'+str(n_pca_f))
    image_features_val_X1_pca, image_features_val_X2_pca, permissible_val_X1, permissible_val_X2, permissible_val_Y,i_j_pairs_val = permissible_pairs(df_val_X, df_val_Y, DAYS_DIFF, tensors_size,val_folder,-1)
    # image_features_test_X1_pca, image_features_test_X2_pca, permissible_test_X1, permissible_test_X2, permissible_test_Y,i_j_pairs_test = permissible_pairs(df_test_X, df_test_Y, DAYS_DIFF, tensors_size,test_folder,-1)
    gc.enable()
    for count_i in [1000,2000,3000,4000,5000]:
        gc.disable()
        image_features_train_X1_pca, image_features_train_X2_pca, permissible_train_X1, permissible_train_X2, permissible_train_Y, i_j_pairs_train = permissible_pairs(df_train_X, df_train_Y, DAYS_DIFF, tensors_size,train_folder,count_i)
        gc.enable()
        print("\n\n\nLength of clinical features:",len(permissible_train_X1[0]))
        for conv_f in conv_first_layer_filters:
            for conv_s in conv_second_layer_filters:
                for f in first_layer_neurons:
                    for s in second_layer_neurons:
                        for lr in lrs:
                            this_config_predictions = []
                            this_config_results = [n_pca_f,count_i,conv_f,conv_s,f,s,lr]
                            weights_folder = os.path.join(output_files_folder,"weights")
                            filepath= os.path.join(weights_folder, "Siamese"+str("-"+str(count_i)+"-"+str(n_pca_f)+"-"+"loss4"+"-"+str(conv_f)+"-"+str(conv_s)+"-"+str(f)+"-"+str(s)+"-"+str(lr))+".h5")
                            early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=PATIENCE)
                            checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath = filepath, verbose=0, monitor = 'val_loss', save_best_only = True)
                            base_model = build_model(n_pca_f,len(permissible_train_X1[0]),conv_f,conv_s,f,s)
                            base_model.compile(optimizer=tf.keras.optimizers.Adam(lr = lr),
                                               loss = loss4,
                                               metrics=[c_index])
                            history = base_model.fit(
                                            [np.array(image_features_train_X1_pca), np.array(image_features_train_X2_pca), np.array(permissible_train_X1),np.array(permissible_train_X2)],
                                            np.array(permissible_train_Y).astype('float32'),
                                            epochs=EPOCHS,
                                            callbacks=[early_stopping, checkpoint],
                                            verbose = 0,
                                            batch_size = BATCH_SIZE,
                                            validation_data=([np.array(image_features_val_X1_pca), np.array(image_features_val_X2_pca),np.array(permissible_val_X1),np.array(permissible_val_X2)], np.asarray(permissible_val_Y).astype('float32'))
                                            )
                            base_model.load_weights(filepath)
                            pred_val = base_model.predict([np.array(image_features_val_X1_pca), np.array(image_features_val_X2_pca)])
                            val_c_index = c_index_prediction(permissible_val_Y,pred_val)
                            pred_train = base_model.predict([np.array(image_features_train_X1_pca), np.array(image_features_train_X2_pca)])
                            train_c_index = c_index_prediction(permissible_train_Y,pred_train)
                            this_config_results.append(train_c_index)
                            this_config_results.append(val_c_index)
                            this_config_results.append( case_wise_voting(pred_train, permissible_train_Y, i_j_pairs_train) )
                            this_config_results.append( case_wise_voting(pred_val, permissible_val_Y, i_j_pairs_val) )
                            this_config_results.append( case_wise_soft_voting(pred_train, permissible_train_Y, i_j_pairs_train) )
                            this_config_results.append( case_wise_soft_voting(pred_val, permissible_val_Y, i_j_pairs_val) )
                            print(this_config_results)
                            predictions.append(this_config_results)
                            with open(csv_filename, 'w') as csvfile:
                                csvwriter = csv.writer(csvfile)
                                csvwriter.writerow(fields)
                                csvwriter.writerows(predictions)

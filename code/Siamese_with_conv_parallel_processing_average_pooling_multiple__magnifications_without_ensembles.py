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

    # valid_idx3 = tf.math.greater(365.0,tf.math.abs(y_pred))
    # valid_loss3 = tf.where(valid_idx3, 1/(tf.math.abs(y_pred)+0.00000001), 0.0)
    # loss3 = tf.math.reduce_sum(valid_loss3)
    # return loss1+loss2+loss3
    return loss1+loss2

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
    max_pool_2d = tf.keras.layers.MaxPooling2D(pool_size=(2, 2),strides=(1, 1))(conv1)
    conv2 = tf.keras.layers.Conv2D(second_conv_layer_number_filers, (3,3), activation='relu')(max_pool_2d)
    pool = tf.keras.layers.GlobalMaxPool2D()(conv2)
    # tf.keras.layers.Conv1D(first_conv_layer_number_filers,(1), activation='relu'),
    # tf.keras.layers.Flatten(),
    clinical_input = Input(shape=(no_clinical_features))
    concatenate_layer = tf.keras.layers.concatenate([pool,clinical_input])
    first_dense = tf.keras.layers.Dense(first_layer_neurons, activation='relu')(concatenate_layer)
    dp1 = tf.keras.layers.Dropout(0.5)(first_dense)
    second_dense = tf.keras.layers.Dense(second_layer_neurons, activation='relu')(dp1)
    dp2 = tf.keras.layers.Dropout(0.5)(second_dense)
    output = tf.keras.layers.Dense(1)(dp2)
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

# TODO: Check how to inset the image feature maps in dataframe
os.environ["CUDA_VISIBLE_DEVICES"]='3'

print("Screen Name: Epoch_0_using_average")
EPOCHS = 10000
PATIENCE = 150
BATCH_SIZE = 32
DAYS_DIFF = 365
# # PLOTS_FOLDER = "Siamese_with_conv_PCA_loss4_2_b"
# # os.mkdir(PLOTS_FOLDER)
output_files_folder = os.path.join(r"/home","sxa171531","images","TCGA-GBM","output_files")
# # f = open(MODEL_NAME+'.pickle',"x")
# # f.close()
# print("Loss: loss4")
# print("EPOCHS: ",EPOCHS)
# print("PATIENCE: ",PATIENCE)
# print("BATCH_SIZE: ",BATCH_SIZE)
# print("Permissible pairs difference: ",DAYS_DIFF)
# # print("Plots are stored in: ",PLOTS_FOLDER)
# print("No Last layer neuron activation")
# print("0.2 Dropouts between the FCLs")
# print("Saving weights in files: ",output_files_folder)
# print("Tile features: average pooling")
# print("Tiles at 20x, 5x, 1.25x, total 6144 features")
# print("WSI features: Feature map generation")

# df = pd.read_excel(os.path.join(r"/home","sxa171531","images","TCGA-GBM","clinical.xlsx"))
#
# df['filenames']=None
# image_features_dir = os.path.join(r"/home","sxa171531","images","TCGA-GBM","original_image_features")
# ids_in_dataframe = []
# for index, row in df.iterrows():
#     if row['case_submitter_id'] in ids_in_dataframe:
#         df.drop(index, inplace=True)
#     else:
#         ids_in_dataframe.append(row['case_submitter_id'])
#
# image_features_path_dic={}
# for filename in os.listdir(image_features_dir):
#     case_submitter_id = "-".join(filename.split("-")[0:3])
#     if case_submitter_id in image_features_path_dic:
#         temp = image_features_path_dic[case_submitter_id]
#         temp.append(os.path.join(image_features_dir,filename))
#         image_features_path_dic[case_submitter_id]=temp
#     else:
#         image_features_path_dic[case_submitter_id]=[os.path.join(image_features_dir,filename)]
# # print(len(list(image_features_path_dic.keys())))
#
# for index, row in df.iterrows():
#     if row['case_submitter_id'] in list(image_features_path_dic.keys()):
#         df.at[index,'filenames'] = image_features_path_dic[row['case_submitter_id']]
#
# columns_of_interest = ['age_at_diagnosis','days_to_death','ethnicity','gender','race','days_to_last_follow_up','filenames']
# df = df[columns_of_interest]
#
# df['ethnicity'] = pd.Categorical(df['ethnicity'])
# df['ethnicity'] = df.ethnicity.cat.codes
# y = pd.get_dummies(df.ethnicity,prefix="ethnicity")
# y = y.drop(y.columns[-1],axis=1)
# df = df.drop(columns= ["ethnicity"])
# df = pd.concat([df, y], axis=1)
#
# df['gender'] = pd.Categorical(df['gender'])
# df['gender'] = df.gender.cat.codes
# y = pd.get_dummies(df.gender,prefix="gender")
# y = y.drop(y.columns[-1],axis=1)
# df = df.drop(columns= ["gender"])
# df = pd.concat([df, y], axis=1)
#
# df['race'] = pd.Categorical(df['race'])
# df['race'] = df.race.cat.codes
# y = pd.get_dummies(df.race,prefix="race")
# y = y.drop(y.columns[-1],axis=1)
# df = df.drop(columns= ["race"])
# df = pd.concat([df, y], axis=1)
#
# df_valid = df['filenames'].notnull()
# df = df[df_valid]
#
# df['Time'] =  df['days_to_death'].replace("'--", np.nan, regex=True)
# df['Occurence'] = df['Time'].notnull()
# df['Time'][df['Time'].isnull()] = df['days_to_last_follow_up']
# df['Time'] = df['Time'].astype(np.int64)
# df = df.drop(columns=['days_to_death','days_to_last_follow_up'])
# np.random.seed(0)
#
# df_dev, df_test = train_test_split(df, test_size = 0.3)
# df_train, df_val = train_test_split(df_dev, test_size = 0.2)
train_pickle_filename = os.path.join(output_files_folder,'train.pickle')
val_pickle_filename = os.path.join(output_files_folder,'val.pickle')
test_pickle_filename = os.path.join(output_files_folder,'test.pickle')

# with open(train_pickle_filename, 'wb') as handle:
#     pickle.dump(df_train, handle)
# with open(test_pickle_filename, 'wb') as handle:
#     pickle.dump(df_test, handle)
# with open(val_pickle_filename, 'wb') as handle:
#     pickle.dump(df_val, handle)

with open(train_pickle_filename, 'rb') as handle:
    df_train = pickle.load(handle)
with open(test_pickle_filename, 'rb') as handle:
    df_test = pickle.load(handle)
with open(val_pickle_filename, 'rb') as handle:
    df_val =pickle.load(handle)
df_train_X, df_train_Y = get_X_Y_columns(df_train)
df_val_X, df_val_Y = get_X_Y_columns(df_val)
df_test_X, df_test_Y = get_X_Y_columns(df_test)
# print("Total number of patients:", len(df.index))
# print("Total number of patients in training set:", df_train_X.shape[0])
# print("Total number of patients in validation set:", df_val_X.shape[0])
# print("Total number of patients in test set:", df_test_X.shape[0])
# print("Creating training features")
# gc.disable()
# training_features = get_features(df_train_X)
# gc.enable()
# print("Created training features")


predictions = []
###ENABLE THEM FOR MODEL TRAINING

fields = ['no_pca','no_training_pairs', 'conv_f', 'conv_s', 'f', 's', 'lr',
'wsi_train', 'wsi_val',
'case_voting_train', 'case_voting_val',
'case_soft-voting_train', 'case_soft-voting_val'
 ]
# name of csv file
csv_filename = os.path.join(output_files_folder,"Results.csv")

# writing to csv file
with open(csv_filename, 'w') as csvfile:
    # creating a csv writer object
    csvwriter = csv.writer(csvfile)
    # writing the fields
    csvwriter.writerow(fields)

conv_first_layer_filters = [256] # maybe 1024 filters will be better
conv_second_layer_filters = [256] # maybe 1024 filters will be better
first_layer_neurons = [8] #maybe 256
second_layer_neurons = [4] #maybe 32
lrs = [0.0001]
results = []
number_pca_features = [8,16]
# 8,16,32,64,128,256,0.7,0.8,0.9,0.95,1]
tensors_size = 100
cvs_file_inputs = []
for n_pca_f in number_pca_features:
    # gc.disable()
    # print("number of pca features:",n_pca_f)
    # pca = PCA(n_components=n_pca_f)
    # pca.fit(np.array(training_features))
    # print("Number of features: ",len(pca.explained_variance_))
    # print("explained_variance_.cumsum: ",pca.explained_variance_.cumsum()[-1])
    # print("explained_variance_ratio_.cumsum",pca.explained_variance_ratio_.cumsum()[-1])
    # gc.enable()
    #
    # ## FOR EXTRACTING PCA FEATURES
    # gc.disable()
    # train_folder = os.path.join(output_files_folder,'PCA_features_train_'+str(n_pca_f))
    # os.mkdir(train_folder)
    # val_folder = os.path.join(output_files_folder,'PCA_features_val_'+str(n_pca_f))
    # os.mkdir(val_folder)
    # test_folder = os.path.join(output_files_folder,'PCA_features_test_'+str(n_pca_f))
    # os.mkdir(test_folder)
    # print("Len train df",len(df_train_X))
    # print("Len test df",len(df_test_X))
    # print("Len val df",len(df_val_X))
    # pca_features_extraction(df_train_X, pca, len(pca.explained_variance_), tensors_size, train_folder)
    # pca_features_extraction(df_val_X, pca, len(pca.explained_variance_), tensors_size, val_folder)
    # pca_features_extraction(df_test_X, pca, len(pca.explained_variance_), tensors_size, test_folder)
    # gc.enable()


    ### FOR BUILDING VALID DATASETS
    gc.disable()
    train_folder = os.path.join(output_files_folder,'PCA_features_train_'+str(n_pca_f))
    val_folder = os.path.join(output_files_folder,'PCA_features_val_'+str(n_pca_f))
    test_folder = os.path.join(output_files_folder,'PCA_features_test_'+str(n_pca_f))
    image_features_val_X1_pca, image_features_val_X2_pca, permissible_val_X1, permissible_val_X2, permissible_val_Y,i_j_pairs_val = permissible_pairs(df_val_X, df_val_Y, DAYS_DIFF, tensors_size,val_folder,-1)
    # image_features_test_X1_pca, image_features_test_X2_pca, permissible_test_X1, permissible_test_X2, permissible_test_Y,i_j_pairs_test = permissible_pairs(df_test_X, df_test_Y, DAYS_DIFF, tensors_size,test_folder,-1)
    gc.enable()
    for count_i in [50000]: ### Once more try with 50000
        gc.disable()
        image_features_train_X1_pca, image_features_train_X2_pca, permissible_train_X1, permissible_train_X2, permissible_train_Y, i_j_pairs_train = permissible_pairs(df_train_X, df_train_Y, DAYS_DIFF, tensors_size,train_folder,count_i)
        gc.enable()
        ### FOR BUILDING ENSEMBLES
        # with open(os.path.join(epoch_output_files_folder,'train_'+str(n_pca_f)+'_.pickle'), 'rb') as handle:
        #     temp = pickle.load(handle)
        # image_features_train_X1_pca = temp[0]
        # image_features_train_X2_pca = temp[1]
        # permissible_train_X1 = temp[2]
        # permissible_train_X2 = temp[3]
        # permissible_train_Y = temp[4]
        # with open(os.path.join(epoch_output_files_folder,'val_'+str(n_pca_f)+'_.pickle'), 'rb') as handle:
        #     temp = pickle.load(handle)
        # image_features_val_X1_pca = temp[0]
        # image_features_val_X2_pca = temp[1]
        # permissible_val_X1 = temp[2]
        # permissible_val_X2 = temp[3]
        # permissible_val_Y = temp[4]
        # with open(os.path.join(epoch_output_files_folder,'test_'+str(n_pca_f)+'_.pickle'), 'rb') as handle:
        #     temp = pickle.load(handle)
        # image_features_test_X1_pca = temp[0]
        # image_features_test_X2_pca = temp[1]
        # permissible_test_X1 = temp[2]
        # permissible_test_X2 = temp[3]
        # permissible_test_Y = temp[4]

        # image_features_train_X1_pca=np.asarray(image_features_train_X1_pca)
        # image_features_train_X1_pca_shuffled = np.reshape(np.copy(image_features_train_X1_pca), (image_features_train_X1_pca.shape[0],image_features_train_X1_pca.shape[1] * image_features_train_X1_pca.shape[2], image_features_train_X1_pca.shape[3]))
        # image_features_train_X1_pca_shuffled_new = []
        # for i_case in range(image_features_train_X1_pca.shape[0]):
        #     temp = image_features_train_X1_pca_shuffled[i_case]
        #     np.random.shuffle(temp)
        #     temp_reshaped = np.reshape(temp, image_features_train_X1_pca.shape[1:])
        #     image_features_train_X1_pca_shuffled_new.append(temp_reshaped)
        # image_features_train_X1_pca_shuffled_new=np.asarray(image_features_train_X1_pca_shuffled_new)
        #
        # image_features_val_X1_pca=np.asarray(image_features_val_X1_pca)
        # image_features_val_X1_pca_shuffled = np.reshape(np.copy(image_features_val_X1_pca), (image_features_val_X1_pca.shape[0],image_features_val_X1_pca.shape[1] * image_features_val_X1_pca.shape[2], image_features_val_X1_pca.shape[3]))
        # image_features_val_X1_pca_shuffled_new = []
        # for i_case in range(image_features_val_X1_pca.shape[0]):
        #     temp = image_features_val_X1_pca_shuffled[i_case]
        #     np.random.shuffle(temp)
        #     temp_reshaped = np.reshape(temp, image_features_val_X1_pca.shape[1:])
        #     image_features_val_X1_pca_shuffled_new.append(temp_reshaped)
        # image_features_val_X1_pca_shuffled_new=np.asarray(image_features_val_X1_pca_shuffled_new)
        #
        # image_features_test_X1_pca=np.asarray(image_features_test_X1_pca)
        # image_features_test_X1_pca_shuffled = np.reshape(np.copy(image_features_test_X1_pca), (image_features_test_X1_pca.shape[0],image_features_test_X1_pca.shape[1] * image_features_test_X1_pca.shape[2], image_features_test_X1_pca.shape[3]))
        # image_features_test_X1_pca_shuffled_new = []
        # for i_case in range(image_features_test_X1_pca.shape[0]):
        #     temp = image_features_test_X1_pca_shuffled[i_case]
        #     np.random.shuffle(temp)
        #     temp_reshaped = np.reshape(temp, image_features_test_X1_pca.shape[1:])
        #     image_features_test_X1_pca_shuffled_new.append(temp_reshaped)
        # image_features_test_X1_pca_shuffled_new=np.asarray(image_features_test_X1_pca_shuffled_new)


        # print("Training Set Length:",len(permissible_train_Y))
        # print("Validation Set Length:",len(permissible_val_Y))
        # print("Testing Set Length:",len(permissible_test_Y))
        # print("Input tensors formed!")

        # image_features_train_X1_pca = CreateInputFeatureMaps_average_pooling_multiple_magnifications.pca_transform_tensors(pca, n_pca_f, 42, 42, image_features_train_X1)
        # image_features_train_X2_pca = CreateInputFeatureMaps_average_pooling_multiple_magnifications.pca_transform_tensors(pca, n_pca_f, 42, 42, image_features_train_X2)
        # image_features_val_X1_pca = CreateInputFeatureMaps_average_pooling_multiple_magnifications.pca_transform_tensors(pca, n_pca_f, 42, 42, image_features_val_X1)
        # image_features_val_X2_pca = CreateInputFeatureMaps_average_pooling_multiple_magnifications.pca_transform_tensors(pca, n_pca_f, 42, 42, image_features_val_X2)
        # image_features_test_X1_pca = CreateInputFeatureMaps_average_pooling_multiple_magnifications.pca_transform_tensors(pca, n_pca_f, 42, 42, image_features_test_X1)
        # image_features_test_X2_pca = CreateInputFeatureMaps_average_pooling_multiple_magnifications.pca_transform_tensors(pca, n_pca_f, 42, 42, image_features_test_X2)
        # print("Train features X1 shape:",image_features_train_X1_pca.shape)
        # print("Train features X2 shape:",image_features_train_X2_pca.shape)
        # print("Validation features X1 shape:",image_features_val_X1_pca.shape)
        # print("Validation features X2 shape:",image_features_val_X2_pca.shape)
        # print("Test features X1 shape:",image_features_test_X1_pca.shape)
        # print("Test features X2 shape:",image_features_test_X2_pca.shape)
        # for l in loss_functions: change l to loss_function

        for conv_f in conv_first_layer_filters:
            for conv_s in conv_second_layer_filters:
                # if conv_s>=conv_f:
                for f in first_layer_neurons:
                    # if f>=conv_s:
                    for s in second_layer_neurons:
                        # if f>=s:
                        for lr in lrs:
                            this_config_predictions = []
                            this_config_results = [n_pca_f,count_i,conv_f,conv_s,f,s,lr]
                            weights_folder = os.path.join(output_files_folder,"weights")
                            filepath= os.path.join(weights_folder, "Siamese_with_max_pooling_separate_training_second"+str("-"+str(count_i)+"-"+str(n_pca_f)+"-"+"loss4"+"-"+str(conv_f)+"-"+str(conv_s)+"-"+str(f)+"-"+str(s)+"-"+str(lr))+".h5")
                            # logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
                            # tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=0)
                            early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=PATIENCE)
                            checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath = filepath, verbose=0, monitor = 'val_loss', save_best_only = True)
                            print("\n\n\nLength of clinical features:",len(permissible_train_X1[0]))
                            base_model = build_model(n_pca_f,len(permissible_train_X1[0]),conv_f,conv_s,f,s)
                            base_model.compile(optimizer=tf.keras.optimizers.Adam(lr = lr),
                                               loss = loss4,
                                               metrics=[c_index])
                            history = base_model.fit(
                                            [np.array(image_features_train_X1_pca), np.array(image_features_train_X2_pca), np.zeros(np.shape(permissible_train_X1)),np.zeros(np.shape(permissible_train_X2))],
                                            np.array(permissible_train_Y).astype('float32'),
                                            epochs=100,
                                            # callbacks=[early_stopping, checkpoint],
                                            verbose = 0,
                                            batch_size = BATCH_SIZE,
                                            validation_data=([np.array(image_features_val_X1_pca), np.array(image_features_val_X2_pca),np.zeros(np.shape(permissible_val_X1)),np.zeros(np.shape(permissible_val_X2))], np.asarray(permissible_val_Y).astype('float32'))
                                            )
                            filepath1= os.path.join(weights_folder, "Siamese_with_max_pooling_separate_training_first"+str("-"+str(count_i)+"-"+str(n_pca_f)+"-"+"loss4"+"-"+str(conv_f)+"-"+str(conv_s)+"-"+str(f)+"-"+str(s)+"-"+str(lr))+".h5")
                            base_model.save_weights(filepath1)
                            history = base_model.fit(
                                            [np.array(image_features_train_X1_pca), np.array(image_features_train_X2_pca), np.array(permissible_train_X1),np.array(permissible_train_X2)],
                                            np.array(permissible_train_Y).astype('float32'),
                                            epochs=EPOCHS,
                                            callbacks=[early_stopping, checkpoint],
                                            verbose = 0,
                                            batch_size = BATCH_SIZE,
                                            validation_data=([np.array(image_features_val_X1_pca), np.array(image_features_val_X2_pca),np.array(permissible_val_X1),np.array(permissible_val_X2)], np.asarray(permissible_val_Y).astype('float32'))
                                            )
                            # # plot_training(history, os.path.join(PLOTS_FOLDER,MODEL_NAME+str("-"+str(n_pca_f)+"-"+"loss4"+"-"+str(f)+"-"+str(s)+"-"+str(lr))+".png"))
                            base_model.load_weights(filepath)
                            print("Training completed")
                            # pred_test = base_model.predict([np.array(image_features_test_X1_pca), np.array(image_features_test_X2_pca)])
                            # test_c_index = c_index_prediction(permissible_test_Y,pred_test)
                            # pred_val = base_model.predict([np.array(image_features_val_X1_pca), np.array(image_features_val_X2_pca),np.array(permissible_val_X1),np.array(permissible_val_X2)])
                            # val_c_index = c_index_prediction(permissible_val_Y,pred_val)
                            # pred_train = base_model.predict([np.array(image_features_train_X1_pca), np.array(image_features_train_X2_pca), np.array(permissible_train_X1),np.array(permissible_train_X2)])
                            # train_c_index = c_index_prediction(permissible_train_Y,pred_train)
                            # #
                            # # shuffled_pred_test = base_model.predict([np.array(image_features_test_X1_pca_shuffled_new), np.array(image_features_test_X2_pca)])
                            # # shuffled_test_c_index = c_index_prediction(permissible_test_Y,shuffled_pred_test)
                            # # shuffled_pred_val = base_model.predict([np.array(image_features_val_X1_pca_shuffled_new), np.array(image_features_val_X2_pca)])
                            # # shuffled_val_c_index = c_index_prediction(permissible_val_Y,shuffled_pred_val)
                            # # shuffled_pred_train = base_model.predict([np.array(image_features_train_X1_pca_shuffled_new), np.array(image_features_train_X2_pca)])
                            # # shuffled_train_c_index = c_index_prediction(permissible_train_Y,shuffled_pred_train)
                            # #
                            # this_config_results.append(train_c_index)
                            # this_config_results.append(val_c_index)
                            # # this_config_results.append(test_c_index)
                            # # this_config_results.append(shuffled_train_c_index)
                            # # this_config_results.append(shuffled_val_c_index)
                            # # this_config_results.append(shuffled_test_c_index)
                            # #
                            # this_config_results.append( case_wise_voting(pred_train, permissible_train_Y, i_j_pairs_train) )
                            # this_config_results.append( case_wise_voting(pred_val, permissible_val_Y, i_j_pairs_val) )
                            # # this_config_results.append( case_wise_voting(pred_test, permissible_test_Y, i_j_pairs_test) )
                            # # this_config_results.append( case_wise_voting(shuffled_pred_train, permissible_train_Y, i_j_pairs_train) )
                            # # this_config_results.append( case_wise_voting(shuffled_pred_val, permissible_val_Y, i_j_pairs_val) )
                            # # this_config_results.append( case_wise_voting(shuffled_pred_test, permissible_test_Y, i_j_pairs_test) )
                            # #
                            # this_config_results.append( case_wise_soft_voting(pred_train, permissible_train_Y, i_j_pairs_train) )
                            # this_config_results.append( case_wise_soft_voting(pred_val, permissible_val_Y, i_j_pairs_val) )
                            # # this_config_results.append( case_wise_soft_voting(pred_test, permissible_test_Y, i_j_pairs_test) )
                            # # this_config_results.append( case_wise_soft_voting(shuffled_pred_train, permissible_train_Y, i_j_pairs_train) )
                            # # this_config_results.append( case_wise_soft_voting(shuffled_pred_val, permissible_val_Y, i_j_pairs_val) )
                            # # this_config_results.append( case_wise_soft_voting(shuffled_pred_test, permissible_test_Y, i_j_pairs_test) )
                            # #
                            # print(this_config_results)
                            # predictions.append(this_config_results)
                            # with open(csv_filename, 'w') as csvfile:
                            #     # creating a csv writer object
                            #     csvwriter = csv.writer(csvfile)
                            #     # writing the fields
                            #     csvwriter.writerow(fields)
                            #     # writing the data rows
                            #     csvwriter.writerows(predictions)

# conda activate SurvivalAnalysis_January2021
# python Just_one_parameter.py>Epoch_1_Just_one_parameter.txt
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

def get_features(df_train_X):
    train_features = []
    X_filenames = df_train_X['filename'].iloc[:]
    shuf = np.arange(0,len(df_train_X))
    random.shuffle(shuf)
    for i in shuf[0:100]:
        filepath_i = X_filenames.iloc[i]
        # print("Working with file: ",i," with path ",filepath_i)
        train_features.extend(CreateInputFeatureMaps_average_pooling_multiple_magnifications.get_model_predictions(filepath_i))
    return train_features

def permissible_pairs(X, Y, DAYS_DIFF, pca, n_pca_f, tensors_size, saving_folder):
    permissible_pairs_set1 = []
    permissible_pairs_set2 = []
    image_features_set1 = []
    image_features_set2 = []
    y_true = []
    X_filenames = X['filename'].iloc[:]
    X = X.drop(columns= ["filename"])
    arguments_for_pooling = []
    count = 0
    #For storing the PCA generated maps uncoment the following for loop and comment rest of the code in this fn
    # for i in range(len(X)):
    #     ###In parallel store all the filesFeatureMaps
    #     filepath_i = X_filenames.iloc[i]
    #     CreateInputFeatureMaps_average_pooling_multiple_magnifications.create_tensors(filepath_i, pca, n_pca_f, tensors_size, saving_folder)
    for i in range(len(X)):
        for j in range(i+1, len(X)):
            if Y["Occurence"].iloc[i]==True and (Y["Time"].iloc[i]<(Y["Time"].iloc[j]+DAYS_DIFF)):
                filepath_i = X_filenames.iloc[i]
                filepath_j = X_filenames.iloc[j]
                # print("Working on file pair: ",filepath_i," and ",filepath_j)
                img_a = tf.cast(X.iloc[i,:],tf.float32) ## retrieveing all the columns except last as it is for filename
                img_b = tf.cast(X.iloc[j,:],tf.float32)
                original_image_features_pickle_file_name = os.path.split(filepath_i)[-1]
                with open(os.path.join(saving_folder,original_image_features_pickle_file_name), 'rb') as handle:
                    image_features_i = pickle.load(handle)
                image_features_set1.append(image_features_i)
                original_image_features_pickle_file_name = os.path.split(filepath_j)[-1]
                with open(os.path.join(saving_folder,original_image_features_pickle_file_name), 'rb') as handle:
                    image_features_j = pickle.load(handle)
                image_features_set2.append(image_features_j)
                permissible_pairs_set1.append(img_a)
                permissible_pairs_set2.append(img_b)
                y_true.append(Y["Time"].iloc[i]-Y["Time"].iloc[j])
                count+=1
                # print(count)
            if Y["Occurence"].iloc[j]==True and ((Y["Time"].iloc[i]+DAYS_DIFF)>Y["Time"].iloc[j]):
                filepath_i = X_filenames.iloc[i]
                filepath_j = X_filenames.iloc[j]
                img_a = tf.cast(X.iloc[i,:],tf.float32)
                img_b = tf.cast(X.iloc[j,:],tf.float32)
                original_image_features_pickle_file_name = os.path.split(filepath_i)[-1]
                with open(os.path.join(saving_folder,original_image_features_pickle_file_name), 'rb') as handle:
                    image_features_i = pickle.load(handle)
                image_features_set1.append(image_features_i)
                original_image_features_pickle_file_name = os.path.split(filepath_j)[-1]
                with open(os.path.join(saving_folder,original_image_features_pickle_file_name), 'rb') as handle:
                    image_features_j = pickle.load(handle)
                image_features_set2.append(image_features_j)
                permissible_pairs_set1.append(img_a)
                permissible_pairs_set2.append(img_b)
                y_true.append(Y["Time"].iloc[i]-Y["Time"].iloc[j])
                count+=1
                # print(count)
            # if count==1000:
            #     return image_features_set1, image_features_set2, permissible_pairs_set1 , permissible_pairs_set2 , y_true
    return image_features_set1, image_features_set2, permissible_pairs_set1 , permissible_pairs_set2 , y_true

def model_def(number_channel,first_conv_layer_number_filers,second_conv_layer_number_filers,first_layer_neurons,second_layer_neurons):
    model = tf.keras.Sequential([
    Input(shape=(None,None,number_channel)),
    tf.keras.layers.Conv2D(first_conv_layer_number_filers, (3,3), activation='relu'),
    tf.keras.layers.Conv2D(second_conv_layer_number_filers, (3,3), activation='relu'),
    tf.keras.layers.GlobalAveragePooling2D(),
    # tf.keras.layers.Conv1D(first_conv_layer_number_filers,(1), activation='relu'),
    # tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(first_layer_neurons, activation='relu',kernel_regularizer='l1_l2'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(second_layer_neurons, activation='relu',kernel_regularizer='l1_l2'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1,kernel_regularizer='l1_l2')
    ])
    return Model(inputs=model.inputs, outputs=model.outputs)

def build_model(number_channel,first_conv_layer_number_filers,second_conv_layer_number_filers,first_layer_neurons,second_layer_neurons):
    model = model_def(number_channel,first_conv_layer_number_filers,second_conv_layer_number_filers,first_layer_neurons,second_layer_neurons)
    print(model.summary())
    ########################
    # SIAMESE NEURAL NETWORK
    ########################
    # Complete model is constructed by calling the branch model on each input image,
    # and then the head model on the resulting 512-vectors.
    img_a = Input(shape=(None,None,number_channel))
    img_b = Input(shape=(None,None,number_channel))
    xa = model([img_a])
    xb = model([img_b])
#     x = Lambda(lambda x: tf.cast(K.exp(-x[1]) -  K.exp(-x[0]), tf.float32))([xa, xb])
#     x = Lambda(lambda x:x[1] - x[0])([xa, xb])
    subtracted = tf.keras.layers.Subtract()([xa, xb])
    # probability_output = tf.keras.activations.sigmoid(subtracted)
    # x = Lambda(lambda x:tf.concat(x,1))([xa, xb])
#     x = tf.cast(xb-xa, tf.float32)
    model_f = Model(inputs=[img_a,img_b], outputs=[subtracted])
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

# TODO: Check how to inset the image feature maps in dataframe
os.environ["CUDA_VISIBLE_DEVICES"]='0'

print("Screen Name: Epoch_0_using_average")
EPOCHS = 10000
PATIENCE = 150
BATCH_SIZE = 32
DAYS_DIFF = 365
# PLOTS_FOLDER = "Siamese_with_conv_PCA_loss4_2_b"
# os.mkdir(PLOTS_FOLDER)
MODEL_NAME = "Siamese_with_conv_PCA_loss4_2_b"
# f = open(MODEL_NAME+'.pickle',"x")
# f.close()
print("Loss: loss4")
print("EPOCHS: ",EPOCHS)
print("PATIENCE: ",PATIENCE)
print("BATCH_SIZE: ",BATCH_SIZE)
print("Permissible pairs difference: ",DAYS_DIFF)
# print("Plots are stored in: ",PLOTS_FOLDER)
print("No Last layer neuron activation")
print("0.2 Dropouts between the FCLs")
print("Saving weights in files: ",MODEL_NAME)
print("Tile features: average pooling")
print("Tiles at 5x only, total 2048 features")
print("WSI features: averaged over the generated tile features")

# df = pd.read_excel(os.path.join(r"/home","sxa171531","images","RMS","DataSummaries","SurvivalData.xlsx"))
# df['filename']=None
# image_features_dir = os.path.join(r"/home","sxa171531","images","RMS","DataSummaries","FeatureMapGeneration","original_image_features")
# # image_features_dir = os.path.join(r"/home","sxa171531","images","RMS","DataSummaries","FeatureMapGeneration","trial_features")
# image_features_path_dic = {}
# for filename in os.listdir(image_features_dir):
#     usi = filename.split("-")[0]
#     image_features_path_dic[usi]=os.path.join(image_features_dir,filename)
# for index, row in df.iterrows():
#     if row['USI'] in list(image_features_path_dic.keys()):
#         df.at[index,'filename'] = image_features_path_dic[row['USI']]
# columns_of_interest = ['Race','Ethnicity','Sex','Metastatic at Diagnosis','Age at Enrollment (days)','Histology','Stage','Group','Tumor Size (cm)', 'filename','Time to Death (days)', 'Time to Follow Up days)']
# df = df[columns_of_interest]
# df_temp_valid = df['Tumor Size (cm)']!='.'
# df = df[df_temp_valid]
# df['Race'] = pd.Categorical(df['Race'])
# df['Race'] = df.Race.cat.codes
# y = pd.get_dummies(df.Race,prefix="Race")
# y = y.drop(y.columns[-1],axis=1)
# df = df.drop(columns= ["Race"])
# df = pd.concat([df, y], axis=1)
# df['Ethnicity'] = pd.Categorical(df['Ethnicity'])
# df['Ethnicity'] = df.Ethnicity.cat.codes
# y = pd.get_dummies(df.Ethnicity,prefix="Ethnicity")
# y = y.drop(y.columns[-1],axis=1)
# df = df.drop(columns= ["Ethnicity"])
# df = pd.concat([df, y], axis=1)
# df['Sex'] = pd.Categorical(df['Sex'])
# df['Sex'] = df.Sex.cat.codes
# y = pd.get_dummies(df.Sex,prefix="Sex")
# y = y.drop(y.columns[-1],axis=1)
# df = df.drop(columns= ["Sex"])
# df = pd.concat([df, y], axis=1)
#
# df_valid = df['Metastatic at Diagnosis'].notnull()
# df = df[df_valid]
#
# df_valid = df['filename'].notnull()
# df = df[df_valid]
#
#
# MD = pd.Series(pd.Categorical(df['Metastatic at Diagnosis']))
# df['MD'] = MD.cat.codes
# y = pd.get_dummies(df.MD,prefix="MD")
# y = y.drop(y.columns[-1],axis=1)
# df = df.drop(columns= ["MD","Metastatic at Diagnosis"])
# df = pd.concat([df, y], axis=1)
#
# df['Histology'] = pd.Categorical(df['Histology'])
# df['Histology'] = df.Histology.cat.codes
# y = pd.get_dummies(df.Histology,prefix="Histology")
# y = y.drop(y.columns[-1],axis=1)
# df = df.drop(columns= ["Histology"])
# df = pd.concat([df, y], axis=1)
#
# df['Stage'] = pd.Categorical(df['Stage'])
# df['Stage'] = df.Stage.cat.codes
# y = pd.get_dummies(df.Stage,prefix="Stage")
# y = y.drop(y.columns[-1],axis=1)
# df = df.drop(columns= ["Stage"])
# df = pd.concat([df, y], axis=1)
#
# df['Group'] = pd.Categorical(df['Group'])
# df['Group'] = df.Group.cat.codes
# y = pd.get_dummies(df.Group,prefix="Group")
# y = y.drop(y.columns[-1],axis=1)
# df = df.drop(columns= ["Group"])
# df = pd.concat([df, y], axis=1)
#
# df['Tumor Size (cm)'] = df['Tumor Size (cm)'].astype(np.float64)
#
# df['Time'] =  df['Time to Death (days)'].replace(r'^\s*$', np.nan, regex=True)
# df['Occurence'] = df['Time'].notnull()
# df['Time'][df['Time'].isnull()] = df['Time to Follow Up days)']
# df['Time'] = df['Time'].astype(np.int64)
# df = df.drop(columns=['Time to Death (days)','Time to Follow Up days)'])
# np.random.seed(0)
# shuffled = df.sample(frac=1)
# result = np.array_split(shuffled, 5)
# for part in result:
#     print(part,'\n')
output_files_folder = os.path.join(r"/home","sxa171531","images","RMS","DataSummaries","FeatureMapGeneration","SSCNN","output_files_average_pooling_multiple_magnifications")
for i in range(0,1): ###Folds
    predictions = []
    epoch_output_files_folder = os.path.join(output_files_folder,"epoch_"+str(i))
    # os.mkdir(epoch_output_files_folder)
    print("This is Epoch:", i)

    ###ENABLE THEM FOR MODEL TRAINING

    fields = ['no_pca', 'conv_f', 'conv_s', 'f', 's', 'lr',
    'Ensemble-1_train', 'Ensemble-1_val', 'Ensemble-1_test','Ensemble-1_shuffled_train', 'Ensemble-1_shuffled_val', 'Ensemble-1_shuffled_test',
    'Ensemble-2_train', 'Ensemble-2_val', 'Ensemble-2_test','Ensemble-2_shuffled_train', 'Ensemble-2_shuffled_val', 'Ensemble-2_shuffled_test',
    'Ensemble-3_train', 'Ensemble-3_val', 'Ensemble-3_test','Ensemble-3_shuffled_train', 'Ensemble-3_shuffled_val', 'Ensemble-3_shuffled_test',
    'Ensemble-4_train', 'Ensemble-4_val', 'Ensemble-4_test','Ensemble-4_shuffled_train', 'Ensemble-4_shuffled_val', 'Ensemble-4_shuffled_test',
    'Ensemble-5_train', 'Ensemble-5_val', 'Ensemble-5_test','Ensemble-5_shuffled_train', 'Ensemble-5_shuffled_val', 'Ensemble-5_shuffled_test',
    'Ensemble-6_train', 'Ensemble-6_val', 'Ensemble-6_test','Ensemble-6_shuffled_train', 'Ensemble-6_shuffled_val', 'Ensemble-6_shuffled_test',
    'Ensemble-7_train', 'Ensemble-7_val', 'Ensemble-7_test','Ensemble-7_shuffled_train', 'Ensemble-7_shuffled_val', 'Ensemble-7_shuffled_test',
    'Ensemble-8_train', 'Ensemble-8_val', 'Ensemble-8_test','Ensemble-8_shuffled_train', 'Ensemble-8_shuffled_val', 'Ensemble-8_shuffled_test',
    'Ensemble-9_train', 'Ensemble-9_val', 'Ensemble-9_test','Ensemble-9_shuffled_train', 'Ensemble-9_shuffled_val', 'Ensemble-9_shuffled_test',
    'Ensemble-10_train', 'Ensemble-10_val', 'Ensemble-10_test','Ensemble-10_shuffled_train', 'Ensemble-10_shuffled_val', 'Ensemble-10_shuffled_test',
    'voting_train','soft-voting_train',
    'voting_val','soft-voting_val',
    'voting_test','soft-voting_test',
    'shuffled_voting_train','shuffled_soft-voting_train',
    'shuffled_voting_val','shuffled_soft-voting_val',
    'shuffled_voting_test','shuffled_soft-voting_test',
     ]
    # name of csv file
    csv_filename = "Epoch-using-average-Just_one_parameter-"+str(i)+".csv"

    # writing to csv file
    with open(csv_filename, 'w') as csvfile:
        # creating a csv writer object
        csvwriter = csv.writer(csvfile)
        # writing the fields
        csvwriter.writerow(fields)



    # df_dev = None
    # for j in range(i):
    #     if df_dev is not None:
    #         df_dev = pd.concat([df_dev,result[j]])
    #     else:
    #         df_dev = result[j]
    # for j in range(i+1,5):
    #     if df_dev is not None:
    #         df_dev = pd.concat([df_dev,result[j]])
    #     else:
    #         df_dev = result[j]
    # df_test = result[i]
    # print("Training dataframe:",df_dev)
    # print("Testing dataframe:", df_test)
    pickle_file_location = os.path.join(r"/home","sxa171531","images","RMS","DataSummaries","FeatureMapGeneration","SSCNN","output_files","epoch_"+str(i))
    train_pickle_filename = os.path.join(pickle_file_location,'train.pickle')
    test_pickle_filename = os.path.join(pickle_file_location,'test.pickle')
    # with open(train_pickle_filename, 'wb') as handle:
    #     pickle.dump(df_dev, handle)
    # with open(test_pickle_filename, 'wb') as handle:
    #     pickle.dump(df_test, handle)
    with open(train_pickle_filename, 'rb') as handle:
        df_dev = pickle.load(handle)
    with open(test_pickle_filename, 'rb') as handle:
        df_test = pickle.load(handle)

    df_train, df_val = train_test_split(df_dev, test_size = 0.25)
    # print("Total number of patients:", df.shape[0])
    print("Total number of patients in training set:", df_train.shape[0])
    print("Total number of patients in validation set:", df_val.shape[0])
    print("Total number of patients in test set:", df_test.shape[0])
    df_train_X, df_train_Y = get_X_Y_columns(df_train)
    df_val_X, df_val_Y = get_X_Y_columns(df_val)
    df_test_X, df_test_Y = get_X_Y_columns(df_test)
    # print("Creating training features")
    # gc.disable()
    # training_features = get_features(df_train_X)
    # gc.enable()
    # print("Created training features")
    conv_first_layer_filters = [32]
    conv_second_layer_filters = [32]
    first_layer_neurons = [32]
    second_layer_neurons = [32]
    lrs = [0.0001]
    results = []
    number_pca_features = [.95]
    tensors_size = 42
    cvs_file_inputs = []
    for n_pca_f in number_pca_features:
        # gc.disable()
        print("number of pca features:",n_pca_f)
        # pca = PCA(n_components=n_pca_f)
        # pca.fit(np.array(training_features))
        # print("Number of features: ",len(pca.explained_variance_))
        # # print("explained_variance_: ",pca.explained_variance_)
        # # print("explained_variance_ratio_",pca.explained_variance_ratio_)
        # print("explained_variance_ratio_.cumsum",pca.explained_variance_ratio_.cumsum()[-1])
        # gc.enable()

        ## FOR EXTRACTING PCA FEATURES
        gc.disable()
        train_folder = os.path.join(epoch_output_files_folder,'train_'+str(n_pca_f))
        # os.mkdir(train_folder)
        val_folder = os.path.join(epoch_output_files_folder,'train_'+str(n_pca_f))
        # os.mkdir(val_folder)
        test_folder = os.path.join(epoch_output_files_folder,'test_'+str(n_pca_f))
        # os.mkdir(test_folder)
        # permissible_pairs(df_train_X, df_train_Y, DAYS_DIFF, pca, len(pca.explained_variance_), tensors_size, train_folder)
        # permissible_pairs(df_val_X, df_val_Y, DAYS_DIFF, pca, len(pca.explained_variance_), tensors_size, val_folder)
        # permissible_pairs(df_test_X, df_test_Y, DAYS_DIFF, pca, len(pca.explained_variance_), tensors_size, test_folder)
        gc.enable()


        ### FOR BUILDING VALID DATASETS
        gc.disable()
        train_folder = os.path.join(epoch_output_files_folder,'train_'+str(n_pca_f))
        val_folder = os.path.join(epoch_output_files_folder,'train_'+str(n_pca_f))
        test_folder = os.path.join(epoch_output_files_folder,'test_'+str(n_pca_f))
        image_features_train_X1_pca, image_features_train_X2_pca, permissible_train_X1, permissible_train_X2, permissible_train_Y = permissible_pairs(df_train_X, df_train_Y, DAYS_DIFF, None, 0, tensors_size, train_folder)
        # with open(os.path.join(epoch_output_files_folder,'train_'+str(n_pca_f)+'_.pickle'), 'wb') as handle:
        #     pickle.dump([image_features_train_X1_pca, image_features_train_X2_pca, permissible_train_X1, permissible_train_X2, permissible_train_Y], handle)
        image_features_val_X1_pca, image_features_val_X2_pca, permissible_val_X1, permissible_val_X2, permissible_val_Y = permissible_pairs(df_val_X, df_val_Y, DAYS_DIFF, None, 0, tensors_size, val_folder)
        # with open(os.path.join(epoch_output_files_folder,'val_'+str(n_pca_f)+'_.pickle'), 'wb') as handle:
        #     pickle.dump([image_features_val_X1_pca, image_features_val_X2_pca, permissible_val_X1, permissible_val_X2, permissible_val_Y], handle)
        image_features_test_X1_pca, image_features_test_X2_pca, permissible_test_X1, permissible_test_X2, permissible_test_Y = permissible_pairs(df_test_X, df_test_Y, DAYS_DIFF, None, 0, tensors_size, test_folder)
        # with open(os.path.join(epoch_output_files_folder,'test_'+str(n_pca_f)+'_.pickle'), 'wb') as handle:
        #     pickle.dump([image_features_test_X1_pca, image_features_test_X2_pca, permissible_test_X1, permissible_test_X2, permissible_test_Y], handle)
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

        image_features_train_X1_pca=np.asarray(image_features_train_X1_pca)
        image_features_train_X1_pca_shuffled = np.reshape(np.copy(image_features_train_X1_pca), (image_features_train_X1_pca.shape[0],image_features_train_X1_pca.shape[1] * image_features_train_X1_pca.shape[2], image_features_train_X1_pca.shape[3]))
        image_features_train_X1_pca_shuffled_new = []
        for i_case in range(image_features_train_X1_pca.shape[0]):
            temp = image_features_train_X1_pca_shuffled[i_case]
            np.random.shuffle(temp)
            temp_reshaped = np.reshape(temp, image_features_train_X1_pca.shape[1:])
            image_features_train_X1_pca_shuffled_new.append(temp_reshaped)
        image_features_train_X1_pca_shuffled_new=np.asarray(image_features_train_X1_pca_shuffled_new)

        image_features_val_X1_pca=np.asarray(image_features_val_X1_pca)
        image_features_val_X1_pca_shuffled = np.reshape(np.copy(image_features_val_X1_pca), (image_features_val_X1_pca.shape[0],image_features_val_X1_pca.shape[1] * image_features_val_X1_pca.shape[2], image_features_val_X1_pca.shape[3]))
        image_features_val_X1_pca_shuffled_new = []
        for i_case in range(image_features_val_X1_pca.shape[0]):
            temp = image_features_val_X1_pca_shuffled[i_case]
            np.random.shuffle(temp)
            temp_reshaped = np.reshape(temp, image_features_val_X1_pca.shape[1:])
            image_features_val_X1_pca_shuffled_new.append(temp_reshaped)
        image_features_val_X1_pca_shuffled_new=np.asarray(image_features_val_X1_pca_shuffled_new)

        image_features_test_X1_pca=np.asarray(image_features_test_X1_pca)
        image_features_test_X1_pca_shuffled = np.reshape(np.copy(image_features_test_X1_pca), (image_features_test_X1_pca.shape[0],image_features_test_X1_pca.shape[1] * image_features_test_X1_pca.shape[2], image_features_test_X1_pca.shape[3]))
        image_features_test_X1_pca_shuffled_new = []
        for i_case in range(image_features_test_X1_pca.shape[0]):
            temp = image_features_test_X1_pca_shuffled[i_case]
            np.random.shuffle(temp)
            temp_reshaped = np.reshape(temp, image_features_test_X1_pca.shape[1:])
            image_features_test_X1_pca_shuffled_new.append(temp_reshaped)
        image_features_test_X1_pca_shuffled_new=np.asarray(image_features_test_X1_pca_shuffled_new)


        # print("Training Set Length:",len(permissible_train_Y))
        # print("Validation Set Length:",len(permissible_val_Y))
        # print("Testing Set Length:",len(permissible_test_Y))
        print("Input tensors formed!")

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
                if conv_s>=conv_f:
                    for f in first_layer_neurons:
                        if f>=conv_s:
                            for s in second_layer_neurons:
                                if f>=s:
                                    for lr in lrs:
                                        if f>=s:
                                            this_config_predictions = []
                                            this_config_results = [n_pca_f,conv_f,conv_s,f,s,lr]
                                            for ensemble in range(10):
                                                weights_folder = os.path.join(epoch_output_files_folder,"weights")
                                                filepath= os.path.join(weights_folder, MODEL_NAME+str("-Just_one_parameter-"+"-"+str(ensemble)+"-"+str(n_pca_f)+"-"+"loss4"+"-"+str(conv_f)+"-"+str(conv_s)+"-"+str(f)+"-"+str(s)+"-"+str(lr))+".h5")
                                                # logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
                                                # tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=0)
                                                early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=PATIENCE)
                                                checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath = filepath, verbose=0, monitor = 'val_loss', save_best_only = True)
                                                base_model = build_model(n_pca_f,conv_f,conv_s,f,s)
                                                base_model.compile(optimizer=tf.keras.optimizers.Adam(lr = lr),
                                                                   loss = loss4,
                                                                   metrics=[c_index])
                                                history = base_model.fit(
                                                                [np.array(image_features_train_X1_pca), np.array(image_features_train_X2_pca)],
                                                                np.array(permissible_train_Y).astype('float32'),
                                                                epochs=EPOCHS,
                                                                callbacks=[early_stopping, checkpoint],
                                                                verbose = 0,
                                                                batch_size = BATCH_SIZE,
                                                                validation_data=([np.array(image_features_val_X1_pca), np.array(image_features_val_X2_pca)], np.asarray(permissible_val_Y).astype('float32'))
                                                                )
                                                # plot_training(history, os.path.join(PLOTS_FOLDER,MODEL_NAME+str("-"+str(n_pca_f)+"-"+"loss4"+"-"+str(f)+"-"+str(s)+"-"+str(lr))+".png"))
                                                if os.path.exists(filepath):
                                                    base_model.load_weights(filepath)
                                                    pred_test = base_model.predict([np.array(image_features_test_X1_pca), np.array(image_features_test_X2_pca)])
                                                    test_c_index = c_index_prediction(permissible_test_Y,pred_test)
                                                    pred_val = base_model.predict([np.array(image_features_val_X1_pca), np.array(image_features_val_X2_pca)])
                                                    val_c_index = c_index_prediction(permissible_val_Y,pred_val)
                                                    pred_train = base_model.predict([np.array(image_features_train_X1_pca), np.array(image_features_train_X2_pca)])
                                                    train_c_index = c_index_prediction(permissible_train_Y,pred_train)


                                                    shuffled_pred_test = base_model.predict([np.array(image_features_test_X1_pca_shuffled_new), np.array(image_features_test_X2_pca)])
                                                    shuffled_test_c_index = c_index_prediction(permissible_test_Y,shuffled_pred_test)
                                                    shuffled_pred_val = base_model.predict([np.array(image_features_val_X1_pca_shuffled_new), np.array(image_features_val_X2_pca)])
                                                    shuffled_val_c_index = c_index_prediction(permissible_val_Y,shuffled_pred_val)
                                                    shuffled_pred_train = base_model.predict([np.array(image_features_train_X1_pca_shuffled_new), np.array(image_features_train_X2_pca)])
                                                    shuffled_train_c_index = c_index_prediction(permissible_train_Y,shuffled_pred_train)

                                                    this_config_predictions.append([pred_train,pred_val,pred_test,shuffled_pred_train,shuffled_pred_val,shuffled_pred_test])
                                                    this_config_results.append(train_c_index)
                                                    this_config_results.append(val_c_index)
                                                    this_config_results.append(test_c_index)
                                                    this_config_results.append(shuffled_train_c_index)
                                                    this_config_results.append(shuffled_val_c_index)
                                                    this_config_results.append(shuffled_test_c_index)

                                                    parameters = {
                                                    'Ensemble':ensemble,
                                                    'number_pca_features':n_pca_f,
                                                    'conv_first_layer_filters': conv_f,
                                                    'conv_second_layer_filters':conv_s,
                                                    'loss_function':"loss4",
                                                    'first_layer_neurons':f,
                                                    'second_layer_neurons':s,
                                                    'lr':lr}
                                                    c_index_values = {'train':train_c_index,'validation':val_c_index, 'test':test_c_index}
                                                    print(parameters)
                                                    print(c_index_values)
                                            true_values = [permissible_train_Y,permissible_val_Y,permissible_test_Y,permissible_train_Y,permissible_val_Y,permissible_test_Y]
                                            temp_arr = ["Training","Validation","Testing","Shuffled Training","Shuffled Validation","Shuffled Testing"]
                                            for t_i in range(6):
                                                voting = np.zeros(len(true_values[t_i]))
                                                soft_voting = np.zeros(len(true_values[t_i]))
                                                for e in range(10):
                                                    for t_x in range(len(voting)):
                                                        if this_config_predictions[e][t_i][t_x]>0:
                                                            voting[t_x] = voting[t_x]+1
                                                    for t_x in range(len(soft_voting)):
                                                        soft_voting[t_x] = soft_voting[t_x]+this_config_predictions[e][t_i][t_x]
                                                for t_x in range(len(voting)):
                                                    if voting[t_x]<5:
                                                        voting[t_x]=-1
                                                voting_result = c_index_prediction(true_values[t_i],voting)
                                                this_config_results.append(voting_result)
                                                print("Voting result:", i , voting_result)
                                                soft_voting_result = c_index_prediction(true_values[t_i],soft_voting)
                                                this_config_results.append(soft_voting_result)
                                                print("Soft voting result:", i , soft_voting_result)
                                            predictions.append(this_config_results)
                                            with open(csv_filename, 'w') as csvfile:
                                                # creating a csv writer object
                                                csvwriter = csv.writer(csvfile)
                                                # writing the fields
                                                csvwriter.writerow(fields)
                                                # writing the data rows
                                                csvwriter.writerows(predictions)

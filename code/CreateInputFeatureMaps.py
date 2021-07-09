import pickle
import os
import numpy as np
import tensorflow as tf

def get_model_predictions(original_image_features_pickle_file):
    filenames_and_features = []
    with open(original_image_features_pickle_file, "rb") as input_file:
        filenames_and_features = pickle.load(input_file)
    filenames = filenames_and_features[0]
    features = filenames_and_features[1]
    max_x = -1
    max_y = -1
    index_level_x_y = []
    for filename in filenames:
        tile_name = os.path.split(filename)[-1]
        tile_name = tile_name.split(".")[0]
        splited_tile_name = tile_name.split("_")
        tile_level = int(splited_tile_name[-3])
        tile_x = int(splited_tile_name[-2])
        tile_y = int(splited_tile_name[-1])
        index_level_x_y.append([tile_level,tile_x,tile_y])
        if max_x<tile_x:
            max_x = tile_x
        if max_y<tile_y:
            max_y = tile_y
    num_feature = np.array(features[0]).shape[-1]
    feature_map = []
    for i in range(len(features)):
        feature_i = features[i]
        for ii in range(len(features[0])):
            for jj in range(len(features[0][0])):
                feature_i_ii_jj = np.reshape(feature_i[ii][jj],(num_feature))
                if index_level_x_y[i][0]==1:
                    feature_map.append(feature_i_ii_jj)
    return feature_map


def pca_transform_tensors(pca, num_feature, tensor_width, tensor_height, tensor_3D):
    number_of_samples = len(tensor_3D)
    resized_tensor = []
    for sample in range(number_of_samples):
        w = len(tensor_3D[sample])
        h = len(tensor_3D[sample][0])
        feature_map = np.zeros((w,h,num_feature))
        for i in range(w):
            for j in range(h):
                feature_i_j = tensor_3D[sample][i][j]
                transformed_feature_i_j = pca.transform([feature_i_j])[0]
                feature_map[i,j,:] = transformed_feature_i_j
        resized_tensor.append(tf.image.resize_with_crop_or_pad(feature_map, tensor_width, tensor_height))
    resized_tensor = np.asarray(resized_tensor)
    return resized_tensor

def create_tensors(original_image_features_pickle_file, pca, pca_num_feature, tensors_size, saving_folder):
    filenames_and_features = []
    with open(original_image_features_pickle_file, "rb") as input_file:
        filenames_and_features = pickle.load(input_file)
    filenames = filenames_and_features[0]
    features = filenames_and_features[1]
    ### find the height and width of input tensor using filenames_and_features
    ### filename pattern tile_path = os.path.join(tile_folder,image_name+"_"+str(i)+"_"+str(x)+"_"+str(y)+".png")
    max_x = -1
    max_y = -1
    index_level_x_y = []
    for filename in filenames:
        tile_name = os.path.split(filename)[-1]
        tile_name = tile_name.split(".")[0]
        splited_tile_name = tile_name.split("_")
        tile_level = int(splited_tile_name[-3])
        tile_x = int(splited_tile_name[-2])
        tile_y = int(splited_tile_name[-1])
        index_level_x_y.append([tile_level,tile_x,tile_y])
        if max_x<tile_x:
            max_x = tile_x
        if max_y<tile_y:
            max_y = tile_y
    w,h,num_feature = np.array(features[0]).shape
    feature_map = np.zeros(((max_x+1)*w,(max_y+1)*h,num_feature))
    for i in range(len(features)):
        feature_i = features[i]
        feature_i = np.reshape(feature_i,(w,h,num_feature))
        if index_level_x_y[i][0]==1:
            x = index_level_x_y[i][1]
            y = index_level_x_y[i][2]
            feature_map[x*w:(x+1)*w,y*h:(y+1)*h,:] = feature_i

    w = len(feature_map)
    h = len(feature_map[0])
    resized_tensor = np.zeros((w,h,pca_num_feature))
    for i in range(w):
        for j in range(h):
            feature_i_j = feature_map[i][j]
            transformed_feature_i_j = pca.transform([feature_i_j])[0]
            resized_tensor[i,j,:] = transformed_feature_i_j
    resized_tensor = tf.image.resize_with_crop_or_pad(resized_tensor, tensors_size, tensors_size)
    original_image_features_pickle_file_name = os.path.split(original_image_features_pickle_file)[-1]
    print(original_image_features_pickle_file_name)
    with open(os.path.join(saving_folder,original_image_features_pickle_file_name), 'wb') as handle:
        pickle.dump(resized_tensor, handle)
    # return resized_tensor
    # for i in range(len(features)):
    #     feature_i = np.mean(np.mean(features[i],axis=0),axis=0)
    #     feature_i = np.reshape(feature_i,(num_feature))
    #     start_channel = num_feature*index_level_x_y[i][0]
    #     end_channel = num_feature*(index_level_x_y[i][0]+1)
    #     x = index_level_x_y[i][1]
    #     y = index_level_x_y[i][2]
    #     feature_map[x,y,start_channel:end_channel] = feature_i
    # return np.amax(np.amax(feature_map,axis=0),axis=0)

    # num_feature = np.array(features[0]).shape[-1]
    # feature_map = np.zeros(num_feature)
    # for i in range(len(features)):
    #     feature_i = np.mean(np.mean(features[i],axis=0),axis=0)
    #     feature_i = np.reshape(feature_i,(num_feature))
    #     if index_level_x_y[i][0]==1:
    #         feature_map = feature_map+feature_i
    # return feature_map/(len(features)/3.0)

    # feature_map = np.zeros((42,42,num_feature))
    # for i in range(len(features)):
    #     if index_level_x_y[i][0]==0:
    #         feature_i = np.mean(np.mean(features[i],axis=0),axis=0)
    #         feature_i = np.reshape(feature_i,(num_feature))
    #         x = index_level_x_y[i][1]
    #         y = index_level_x_y[i][2]
    #         feature_map[x,y,:] = feature_i
    # return np.amax(np.amax(feature_map,axis=0),axis=0)



# def create_tensors(original_image_features_pickle_file):
#     filenames_and_features = []
#     with open(original_image_features_pickle_file, "rb") as input_file:
#         filenames_and_features = pickle.load(input_file)
#     filenames = filenames_and_features[0]
#     features = filenames_and_features[1]
#     ### find the height and width of input tensor using filenames_and_features
#     ### filename pattern tile_path = os.path.join(tile_folder,image_name+"_"+str(i)+"_"+str(x)+"_"+str(y)+".png")
#     max_x = -1
#     max_y = -1
#     index_level_x_y = []
#     for filename in filenames:
#         tile_name = os.path.split(filename)[-1]
#         tile_name = tile_name.split(".")[0]
#         splited_tile_name = tile_name.split("_")
#         tile_level = int(splited_tile_name[-3])
#         tile_x = int(splited_tile_name[-2])
#         tile_y = int(splited_tile_name[-1])
#         index_level_x_y.append([tile_level,tile_x,tile_y])
#         if max_x<tile_x:
#             max_x = tile_x
#         if max_y<tile_y:
#             max_y = tile_y
#     # for i in range(len(features)):
#     #     feature_i = np.mean(np.mean(features[i],axis=0),axis=0)
#     #     feature_i = np.reshape(feature_i,(num_feature))
#     #     start_channel = num_feature*index_level_x_y[i][0]
#     #     end_channel = num_feature*(index_level_x_y[i][0]+1)
#     #     x = index_level_x_y[i][1]
#     #     y = index_level_x_y[i][2]
#     #     feature_map[x,y,start_channel:end_channel] = feature_i
#     # return np.amax(np.amax(feature_map,axis=0),axis=0)
#
#     num_feature = np.array(features[0]).shape[-1]
#     feature_map = np.zeros(num_feature)
#     for i in range(len(features)):
#         feature_i = np.mean(np.mean(features[i],axis=0),axis=0)
#         feature_i = np.reshape(feature_i,(num_feature))
#         if index_level_x_y[i][0]==1:
#             feature_map = feature_map+feature_i
#     return feature_map/(len(features)/3.0)
#
#     # feature_map = np.zeros((42,42,num_feature))
#     # for i in range(len(features)):
#     #     if index_level_x_y[i][0]==0:
#     #         feature_i = np.mean(np.mean(features[i],axis=0),axis=0)
#     #         feature_i = np.reshape(feature_i,(num_feature))
#     #         x = index_level_x_y[i][1]
#     #         y = index_level_x_y[i][2]
#     #         feature_map[x,y,:] = feature_i
#     # return np.amax(np.amax(feature_map,axis=0),axis=0)

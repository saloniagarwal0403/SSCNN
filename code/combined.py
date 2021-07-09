import os
import helper_parallel
import pickle
# import tensorflow as tf
import gc
from multiprocessing import Pool
# import multiprocessing as mp4
import math
import numpy as np
import openslide
from PIL import Image# os.environ["CUDA_VISIBLE_DEVICES"]='7'
slides_folder = os.path.join(r"/home","sxa171531","images","TCGA-GBM","Slides")
features = {}
 ### This is the model for extracting features
original_image_features_dir = os.listdir(os.path.join(r"/home","sxa171531","images","TCGA-GBM","original_image_features"))
print(len(os.listdir(slides_folder)))
for slide in os.listdir(slides_folder)[20:100]:
    for image in os.listdir(os.path.join(slides_folder,slide)):
        if image.split(".")[-1]=="svs":
            if str(image.split(".")[0]+".pickle") not in original_image_features_dir:
                image_path = os.path.join(os.path.join(slides_folder,slide),image)
                image = os.path.split(image_path)[-1]
                temp = os.path.join("Slides1",image)
                print(image_path)
                try:
                    shutil.rmtree(temp)
                except Exception as e:
                    print(e)
                image_name = os.path.split(image_path)[-1]
                image_name = str(image_name).split(".")[0]
                os.mkdir(temp)
                tile_folder = os.path.join(temp,image_name)
                os.mkdir(tile_folder)
                tile_size = [2048,512,128]
                reference_image=['reference_0.png','reference_1.png','reference_2.png']
                img = openslide.open_slide(image_path)
                img_dims = []
                if img.level_count>=3:
                    for i in range(3):
                        print(img.level_count)
                        print(img.level_downsamples[i],img.level_dimensions[i])
                        img_dims.append(img.level_dimensions[i])
                    for i in range(3):
                        print("Number of tiles at resulution ",i," : ",math.ceil(img_dims[i][0]/tile_size[i]),math.ceil(img_dims[i][1]/tile_size[i]))
                        for x in range(math.ceil(img_dims[i][0]/tile_size[i])):
                            for y in range(math.ceil(img_dims[i][1]/tile_size[i])):
                                tile = img.read_region((x*tile_size[i]*(4**i),y*tile_size[i]*(4**i)),i,(tile_size[i], tile_size[i]))
                                tile_path = os.path.join(tile_folder,image_name+"_"+str(i)+"_"+str(x)+"_"+str(y)+".png")
                                # data = np.asarray(tile)
                                # avg = np.average(np.average(data, axis = 0), axis = 0)
                                # if not(avg[0]>220 and avg[1]>220 and avg[2]>220):
                                tile.save(tile_path)
                else:
                    for i in range(img.level_count):
                        print(img.level_downsamples[i],img.level_dimensions[i])
                        img_dims.append(img.level_dimensions[i])
                    print("Number of tiles at resulution ",0," : ",math.ceil(img_dims[0][0]/tile_size[0]),math.ceil(img_dims[0][1]/tile_size[0]))
                    for x in range(math.ceil(img_dims[0][0]/tile_size[0])):
                        for y in range(math.ceil(img_dims[0][1]/tile_size[0])):
                            tile = img.read_region((x*tile_size[0]*(4**0),y*tile_size[0]*(4**0)),0,(tile_size[0], tile_size[0]))
                            tile_path = os.path.join(tile_folder,image_name+"_"+str(0)+"_"+str(x)+"_"+str(y)+".png")
                            # data = np.asarray(tile)
                            # avg = np.average(np.average(data, axis = 0), axis = 0)
                            # if not(avg[0]>220 and avg[1]>220 and avg[2]>220):
                            tile.save(tile_path)


                            tile_path = os.path.join(tile_folder,image_name+"_"+str(1)+"_"+str(x)+"_"+str(y)+".png")
                            resized_tile = tile.resize((round(tile.size[0]*0.25), round(tile.size[1]*0.25)))
                            # data = np.asarray(resized_tile)
                            # avg = np.average(np.average(data, axis = 0), axis = 0)
                            # if not(avg[0]>220 and avg[1]>220 and avg[2]>220):
                            resized_tile.save(tile_path)

                            tile_path = os.path.join(tile_folder,image_name+"_"+str(2)+"_"+str(x)+"_"+str(y)+".png")
                            resized_tile = tile.resize((round(tile.size[0]*0.0625), round(tile.size[1]*0.0625)))
                            # avg = np.average(np.average(data, axis = 0), axis = 0)
                            # data = np.asarray(resized_tile)
                            # if not(avg[0]>220 and avg[1]>220 and avg[2]>220):
                            resized_tile.save(tile_path)
                img.close()

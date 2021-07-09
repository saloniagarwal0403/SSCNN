import os
import openslide
from PIL import Image
# import tensorflow as tf
import shutil
import math
import gc
import numpy as np

def extract_tile_features(image_path):
    """
    Input: image path, DL model to extract the features from
    Task:   1) Generate tiles from the given svs image
            2) Normalize the generated tiles
            3) Process the normalized tile through InceptionV3
    Output: DL features for all the tiles
    """
    # gc.disable()
    image = os.path.split(image_path)[-1]
    temp = os.path.join("Slides",image)

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
                tile.save(tile_path)


                tile_path = os.path.join(tile_folder,image_name+"_"+str(1)+"_"+str(x)+"_"+str(y)+".png")
                resized_tile = tile.resize((round(tile.size[0]*0.25), round(tile.size[1]*0.25)))
                resized_tile.save(tile_path)

                tile_path = os.path.join(tile_folder,image_name+"_"+str(2)+"_"+str(x)+"_"+str(y)+".png")
                resized_tile = tile.resize((round(tile.size[0]*0.0625), round(tile.size[1]*0.0625)))
                resized_tile.save(tile_path)
    print("Done: ",image_path)
    img.close()

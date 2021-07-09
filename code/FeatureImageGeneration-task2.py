import os
# import helper_parallel
import pickle
import tensorflow as tf
import gc
import shutil
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.inception_v3 import preprocess_input
os.environ["CUDA_VISIBLE_DEVICES"]='3'
base_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')
new_input = base_model.input
hidden_layer = base_model.layers[-1].output
feature_extractor = tf.keras.Model(new_input, hidden_layer)

slides_folder = os.path.join(r"/home","sxa171531","images","TCGA-GBM","Slides")
features = {}
 ### This is the model for extracting features
original_image_features_dir = os.listdir(os.path.join(r"/home","sxa171531","images","TCGA-GBM","original_image_features"))
for slide in os.listdir(slides_folder):
    for image in os.listdir(os.path.join(slides_folder,slide)):
        if image.split(".")[-1]=="svs":
            if str(image.split(".")[0]+".pickle") not in original_image_features_dir:
                image_path = os.path.join(os.path.join(slides_folder,slide),image)
                image = os.path.split(image_path)[-1]
                if image in os.listdir("Slides"):
                    temp = os.path.join("Slides",image)
                    image_name = os.path.split(image_path)[-1]
                    image_name = str(image_name).split(".")[0]
                    tile_folder = os.path.join(temp,image_name)
                    if len(os.listdir(os.path.join(tile_folder)))>0:
                        dataGenerator = ImageDataGenerator(
                            preprocessing_function=preprocess_input,
                        )
                        gen= dataGenerator.flow_from_directory(temp)
                        predictions = feature_extractor.predict_generator(gen,use_multiprocessing=True, workers=256)
                        output = [gen.filenames,predictions]
                        try:
                            shutil.rmtree(temp)
                        except Exception as e:
                            print(e)
                        if output!=None:
                            with open(os.path.join(r"/home","sxa171531","images","TCGA-GBM","original_image_features",str(image.split(".")[0]+".pickle")), "wb") as output_file:
                                pickle.dump(output, output_file)

# PCA_features_transformation MAYBE NOT NEEDED
# save_transformed_featuremaps MAYBE NOT NEEDED
### TO DO
### Build SurvivalPredictionModel that takes in these features and surival data to give the output

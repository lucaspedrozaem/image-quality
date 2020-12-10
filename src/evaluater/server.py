#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
from flask import Flask, request, jsonify
from evaluater.predict import image_file_to_json, image_dir_to_json, predict
from utils.utils import calc_mean_score, save_json
import urllib
import shutil
import argparse
from tensorflow.python.keras import backend as K
from PIL import ImageFile, Image
from handlers.model_builder import Nima
from handlers.data_generator import TestDataGenerator

app = Flask('server')

def load_model(config):
    global model
    model = Nima(config.base_model_name)
    model.build()
    model.nima_model.load_weights(config.weights_file)
    model.nima_model._make_predict_function()  # https://github.com/keras-team/keras/issues/6462
    model.nima_model.summary()

def main(image_source, predictions_file, img_format='jpg'):
    # load samples
    if os.path.isfile(image_source):
        image_dir, samples = image_file_to_json(image_source)
    else:
        image_dir = image_source
        samples = image_dir_to_json(image_dir, img_type='jpg')

    ImageFile.LOAD_TRUNCATED_IMAGES = True
    Image.MAX_IMAGE_PIXELS = None

    # initialize data generator
    data_generator = TestDataGenerator(samples, image_dir, 64, 10, model.preprocessing_function(),
                                       img_format=img_format)

    # get predictions
    predictions = predict(model.nima_model, data_generator)
    K.clear_session()

    # calc mean scores and add to samples
    for i, sample in enumerate(samples):
        sample['mean_score_prediction'] = calc_mean_score(predictions[i])

    if predictions_file is not None:
        save_json(samples, predictions_file)

    return samples
@app.route('/')
def home():
    return "Welcome to Tripsee API"

@app.route('/prediction', methods=['POST'])
def prediction():

    global images

    if request.method == 'POST':
        images = request.json

        if images:
            try:
                shutil.rmtree('tempcurate')
            except:
                print("folder didn't exist")

            os.mkdir('tempcurate')
            for image in images:
                filename_w_ext = os.path.basename(image)
                print(filename_w_ext)
                try:
                    urllib.request.urlretrieve(image, 'tempcurate/'+ filename_w_ext)
                except:
                    print('An exception occurred :' + image)

            result = main('tempcurate', None)

            return jsonify(result)

        return jsonify({'error': 'Image is not available'})

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--base-model-name', help='CNN base model name', required=True)
    parser.add_argument('-w', '--weights-file', help='path of weights file', required=True)
    args = parser.parse_args()

    load_model(args)
    app.run(host='0.0.0.0')

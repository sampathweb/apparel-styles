import os
import uuid
import json
from glob import glob
import numpy as np
import requests
import pandas as pd

import tornado.web
from tornado import concurrent
from tornado import gen
from concurrent.futures import ThreadPoolExecutor

from app.base_handler import BaseApiHandler
from app.settings import MAX_MODEL_THREAD_POOL

from ml_src.preprocessing import get_attribute_dims, load_label_values
from ml_src.classifiers import get_pretrained_model, create_attributes_model, AttributeFCN
from ml_src.classifiers import predict_attributes
from ml_src.utils import is_gpu_available
from ml_src.classifiers import evaluate_model, test_models

# Train and Validation Images
TRAIN_IMAGES_FOLDER = "ml_src/data/ClothingAttributeDataset/train/"
VALID_IMAGES_FOLDER = "ml_src/data/ClothingAttributeDataset/valid/"
labels_file = "ml_src/data/labels.csv"
label_values_file = "ml_src/data/label_values.json"

use_gpu = is_gpu_available()
target_dims = get_attribute_dims(label_values_file)
label_values = load_label_values(label_values_file)

pretrained_conv_model, _, _ = get_pretrained_model("vgg16", pop_last_pool_layer=True, use_gpu=use_gpu)
target_dims = get_attribute_dims(label_values_file)
attribute_models = create_attributes_model(AttributeFCN, 512, pretrained_conv_model,
                                target_dims, 
                                # dict(list(target_dims.items())[:3]),
                                "ml_src/weights/vgg16-fcn-266-2/",
                                labels_file, 
                                TRAIN_IMAGES_FOLDER, 
                                VALID_IMAGES_FOLDER, 
                                num_epochs=1, 
                                is_train=False,
                                use_gpu=use_gpu)

SELECT_FILES = np.random.permutation(glob("app/static/select-img/*.jpg"))

class IndexHandler(tornado.web.RequestHandler):
    """APP is live"""

    def _get_rand_images(self, n=8):
        filepaths = SELECT_FILES[:n]
        return [filepath.split("/")[-1] for filepath in filepaths]

    def get(self):
        """Return Index Page"""
        select_images = self._get_rand_images()
        self.render("templates/index.html", image=None, image_url=None,
            select_images=select_images, predictions_df=None)

    def head(self):
        """Verify that App is live"""
        self.finish()

    def post(self):
        image_url = self.get_argument("image-url")
        image_files = self.request.files.get("image-file")
        image_select = self.get_argument("image-select")
        print(image_url, image_select)
        image_location = "temp-img"
        image_filename = None
        if image_files:
            image_file = image_files[0]
            fname = image_file["filename"]
            extn = os.path.splitext(fname)[1]
            image_filename = str(uuid.uuid4()) + extn
            image = "app/static/temp-img/" + image_filename
            with open(image, "wb") as fh:
                fh.write(image_file["body"])

        # upload_recs = recommend_pets(cname)
        # print(upload_recs)
        # return self.render("templates/recommended-pets.html", images=[upload_recs])
        elif image_select:
            image_filename = image_select.split("/")[-1]
            image_location = "select-img"
            image = "app/static/select-img/" + image_filename
        elif image_url:
            image_filename = str(uuid.uuid4()) + ".jpg"
            image = "app/static/temp-img/" + image_filename
            with open(image, "wb") as fh:
                fh.write(requests.get(image_url).content)
        else:
            return self.redirect("/")

        print(image)
        # results = predict_attributes(image, pretrained_conv_model, attribute_models,
        #                     attribute_idx_map=label_values["idx_to_names"],
        #                    flatten_pretrained_out=True,
        #                    use_gpu=use_gpu)
        #results = [{k: (v1, str(round(v2, 1)) + "%") for k, (v1, v2) in results.items()}]
        #df = pd.DataFrame(results[0]).T
        results = test_models(attribute_models, pretrained_conv_model, image,
            attribute_idx_map=label_values["idx_to_names"])
        df = pd.DataFrame(results).T
        df.columns = ["Prediction", "Pred_Index", "Confidence"]
        df["Confidence"] = df["Confidence"].astype(float)
        df.index = df.index.str.replace("_GT", "").str.capitalize()
        print(results)
        select_images = self._get_rand_images()
        return self.render("templates/index.html",
            image=image_filename,
            image_location=image_location,
            image_url=image_url,
            select_images=select_images,
            predictions_df=df[["Prediction", "Confidence"]].to_html(formatters={"Confidence": '{:,.2%}'.format },classes="table table-striped"))


class PredictionHandler(BaseApiHandler):
    """Main Prediction Handler"""

    _thread_pool = ThreadPoolExecutor(max_workers=MAX_MODEL_THREAD_POOL)

    def initialize(self, model, *args, **kwargs):
        """Initiailze the models"""
        self.model = model
        super().initialize(*args, **kwargs)

    @concurrent.run_on_executor(executor='_thread_pool')
    def _blocking_predict(self, x):
        """Blocking Call to call Predict Function"""
        # target_values = self.model.predict(X)
        # target_names = ['setosa', 'versicolor', 'virginica']
        # results = [target_names[pred] for pred in target_values]
        # return results
        return {
            "sleeve_length": {
                "prediction": "Full Sleve",
                "confidence": 0.58
            }
        }

    @gen.coroutine
    def predict(self, data):
        """Return prediction result"""
        print(data)
        results = yield self._blocking_predict(data)
        self.respond(results)

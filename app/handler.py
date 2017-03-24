import os
import uuid
import json

import tornado.web
from tornado import concurrent
from tornado import gen
from concurrent.futures import ThreadPoolExecutor

from app.base_handler import BaseApiHandler
from app.settings import MAX_MODEL_THREAD_POOL


class IndexHandler(tornado.web.RequestHandler):
    """APP is live"""

    def get(self):
        """Return Index Page"""
        self.render("templates/index.html", image=None, image_url=None, results=json.dumps(""))

    def head(self):
        """Verify that App is live"""
        self.finish()

    def post(self):
        image_url = self.get_argument("image-url")
        image_files = self.request.files.get("image-file")
        image_filename = None
        if image_files:
            image_file = image_files[0]
            fname = image_file["filename"]
            extn = os.path.splitext(fname)[1]
            image_filename = str(uuid.uuid4()) + extn
            with open("app/static/temp-img/" + image_filename, "wb") as fh:
                fh.write(image_file["body"])

        # upload_recs = recommend_pets(cname)
        # print(upload_recs)
        # return self.render("templates/recommended-pets.html", images=[upload_recs])
        results = [{
            "name": "sleeve_length",
            "prediction": "Full Sleve",
            "confidence": 0.58
        }]
        print(json.dumps(results))
        return self.render("templates/index.html",
            image=image_filename,
            image_url=image_url,
            results=json.dumps(results))


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

"""
Base Handler
"""

import json
import traceback
import logging
import tornado.web
import tornado.escape

from app.exceptions import ApplicationError, RouteNotFound, ServerError

logger = logging.getLogger("app")

class BaseApiHandler(tornado.web.RequestHandler):

    @tornado.web.asynchronous
    def post(self, action):
        try:
            # Fetch appropriate handler
            if not hasattr(self, str(action)):
                raise RouteNotFound(action)

            # Pass along the data and get a result
            handler = getattr(self, str(action))
            data = tornado.escape.json_decode(self.request.body)
            handler(data)
        except ApplicationError as e:
            logger.warning(e.message, e.code)
            self.respond(e.message, e.code)
        except Exception as e:
            logger.error(traceback.format_exc())
            error = ServerError()
            self.respond(error.message, error.code)


    def respond(self, data, code=200):
        self.set_status(code)
        self.write(json.JSONEncoder().encode({
            "status": code,
            "data": data
        }))
        self.finish()

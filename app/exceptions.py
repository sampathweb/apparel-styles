"""
Application Errors
"""

class ApplicationError(Exception):
    def __init__(self, message, code):
        self.message = message
        self.code = code
        super(Exception, self).__init__(message)


class InvalidJSON(ApplicationError):
    def __init__(self):
        ApplicationError.__init__(self,
            "No JSON object could be decoded.",
            400
        )


class AuthError(ApplicationError):
    def __init__(self):
        ApplicationError.__init__(self,
            "User not authenticated",
            401
        )


class RouteNotFound(ApplicationError):
    def __init__(self, action):
        ApplicationError.__init__(self,
            "%s route could not be found" % action,
            404
        )


class ServerError(ApplicationError):
    def __init__(self):
        ApplicationError.__init__(self,
            "we screwed up and have some debugging to do",
            500
        )

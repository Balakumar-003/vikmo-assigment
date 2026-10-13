from rest_framework.views import exception_handler
from rest_framework.response import Response
from rest_framework.exceptions import APIException, NotFound, ValidationError
from django.http import Http404


class APIError(Exception):
    def __init__(self, code: str, message: str, status_code: int = 400, details: list = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.details = details or []


def custom_exception_handler(exc, context):
    if isinstance(exc, APIError):
        return Response(
            {
                "error": {
                    "code": exc.code,
                    "message": exc.message,
                    "details": exc.details,
                }
            },
            status=exc.status_code,
        )

    if isinstance(exc, (Http404, NotFound)):
        return Response(
            {
                "error": {
                    "code": "NOT_FOUND",
                    "message": str(exc) if str(exc) else "Not found",
                    "details": [],
                }
            },
            status=404,
        )

    if isinstance(exc, ValidationError):
        message = "Validation error"
        details = []
        if isinstance(exc.detail, dict):
            details = [{"field": k, "error": str(v)} for k, v in exc.detail.items()]
            message = "; ".join(f"{k}: {v}" for k, v in exc.detail.items())
        elif isinstance(exc.detail, list):
            details = [str(x) for x in exc.detail]
            message = "; ".join(str(x) for x in exc.detail)
        else:
            message = str(exc.detail)

        return Response(
            {
                "error": {
                    "code": "VALIDATION_ERROR",
                    "message": message,
                    "details": details,
                }
            },
            status=400,
        )

    response = exception_handler(exc, context)
    if response is not None:
        return Response(
            {
                "error": {
                    "code": "ERROR",
                    "message": str(response.data),
                    "details": [],
                }
            },
            status=response.status_code,
        )

    return None

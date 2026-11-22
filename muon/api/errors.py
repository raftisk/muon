"""The error envelope `{"error": {"code", "message"}}` for every non-2xx response."""

import logging
from collections.abc import Mapping, Sequence
from http import HTTPStatus
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.types import ASGIApp, Message, Receive, Scope, Send

logger = logging.getLogger(__name__)

INTERNAL_ERROR_MESSAGE = "Internal server error"
VALIDATION_ERROR_CODE = "validation_error"
STATUS_CODE_OVERRIDES: dict[int, str] = {HTTPStatus.UNPROCESSABLE_ENTITY: VALIDATION_ERROR_CODE}
RESPONSE_START_TYPE = "http.response.start"


class ErrorBody(BaseModel):
    code: str
    message: str


class ErrorEnvelope(BaseModel):
    error: ErrorBody


def derive_error_code(status_code: int) -> str:
    if status_code in STATUS_CODE_OVERRIDES:
        return STATUS_CODE_OVERRIDES[status_code]
    return HTTPStatus(status_code).phrase.lower().replace(" ", "_").replace("-", "_")


def build_error_response(
    status_code: int, message: str, headers: Mapping[str, str] | None = None
) -> JSONResponse:
    envelope = ErrorEnvelope(error=ErrorBody(code=derive_error_code(status_code), message=message))
    return JSONResponse(envelope.model_dump(), status_code=status_code, headers=dict(headers or {}))


def format_validation_errors(errors: Sequence[Mapping[str, Any]]) -> str:
    return "; ".join(
        f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}" for error in errors
    )


async def handle_http_exception(request: Request, exception: Exception) -> JSONResponse:
    if not isinstance(exception, StarletteHTTPException):
        raise TypeError(f"handle_http_exception got {type(exception).__name__}")
    return build_error_response(
        exception.status_code, str(exception.detail), headers=exception.headers
    )


async def handle_validation_error(request: Request, exception: Exception) -> JSONResponse:
    if not isinstance(exception, RequestValidationError):
        raise TypeError(f"handle_validation_error got {type(exception).__name__}")
    return build_error_response(
        HTTPStatus.UNPROCESSABLE_ENTITY, format_validation_errors(exception.errors())
    )


class ErrorEnvelopeMiddleware:
    """Turns an unhandled exception into a 500 envelope.

    Pure ASGI, so streaming responses pass through untouched. When the response has already
    started, the exception is re-raised because the status line is gone.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        has_response_started = False

        async def track_send(message: Message) -> None:
            nonlocal has_response_started
            if message["type"] == RESPONSE_START_TYPE:
                has_response_started = True
            await send(message)

        try:
            await self.app(scope, receive, track_send)
        except Exception:
            logger.exception("Unhandled error on %s %s", scope["method"], scope["path"])
            if has_response_started:
                raise
            response = build_error_response(
                HTTPStatus.INTERNAL_SERVER_ERROR, INTERNAL_ERROR_MESSAGE
            )
            await response(scope, receive, send)


def install_error_handling(app: FastAPI) -> None:
    """Registers the envelope handlers.

    Call this before adding CORSMiddleware. Starlette runs the last middleware added as the
    outermost, and CORS must wrap this one so a 500 still carries the CORS headers.
    """
    app.add_exception_handler(StarletteHTTPException, handle_http_exception)
    app.add_exception_handler(RequestValidationError, handle_validation_error)
    app.add_middleware(ErrorEnvelopeMiddleware)

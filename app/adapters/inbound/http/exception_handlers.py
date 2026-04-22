"""
Centralised HTTP exception handlers for FastAPI.

Converts database-level and unhandled exceptions into well-structured
JSON error responses so clients never receive a bare "Internal Server Error".
"""

import structlog
from asyncpg.exceptions import (
    ForeignKeyViolationError,
    NotNullViolationError,
    StringDataRightTruncationError,
    UniqueViolationError,
)
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

logger = structlog.get_logger(__name__)


def _error(status: int, code: str, detail: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": code, "detail": detail})


def register_exception_handlers(app: FastAPI) -> None:
    """Register all exception handlers on the given FastAPI application."""

    @app.exception_handler(UniqueViolationError)
    async def unique_violation(_: Request, exc: UniqueViolationError) -> JSONResponse:
        logger.warning("db.unique_violation", detail=str(exc.detail))
        return _error(
            409, "conflict", f"A record with these values already exists: {exc.detail}"
        )

    @app.exception_handler(ForeignKeyViolationError)
    async def fk_violation(_: Request, exc: ForeignKeyViolationError) -> JSONResponse:
        logger.warning("db.foreign_key_violation", detail=str(exc.detail))
        return _error(
            422,
            "foreign_key_violation",
            f"Referenced record does not exist: {exc.detail}",
        )

    @app.exception_handler(NotNullViolationError)
    async def not_null(_: Request, exc: NotNullViolationError) -> JSONResponse:
        logger.warning("db.not_null_violation", detail=str(exc.detail))
        return _error(
            422, "missing_required_field", f"A required field is missing: {exc.detail}"
        )

    @app.exception_handler(StringDataRightTruncationError)
    async def truncation(
        _: Request, exc: StringDataRightTruncationError
    ) -> JSONResponse:
        logger.warning("db.truncation", detail=str(exc.detail))
        return _error(
            422,
            "value_too_long",
            f"A value exceeds the maximum allowed length: {exc.detail}",
        )

    @app.exception_handler(Exception)
    async def unhandled(_: Request, exc: Exception) -> JSONResponse:
        logger.exception("http.unhandled_error", exc_info=exc)
        return _error(
            500,
            "internal_server_error",
            "An unexpected error occurred. Please try again later.",
        )

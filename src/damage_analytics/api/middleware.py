"""Bounded upload admission before multipart parsing, request IDs and safe JSON logging."""

import asyncio
import logging
from time import perf_counter
from uuid import uuid4

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from damage_analytics.api.config import UploadLimits

logger = logging.getLogger("damage_analytics.api")


class RequestMiddleware:
    """Allow one upload/inference per worker; health/retrieval requests remain independent."""

    def __init__(self, app: ASGIApp, settings: UploadLimits) -> None:
        self.app, self.settings = app, settings
        self.upload_active = False

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        request_id = str(uuid4())
        scope.setdefault("state", {})["request_id"] = request_id
        started, status = perf_counter(), 500

        async def observed(message: Message) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
                message["headers"] = list(message.get("headers", [])) + [
                    (b"x-request-id", request_id.encode()),
                    (b"x-content-type-options", b"nosniff"),
                    (b"cache-control", b"no-store"),
                ]
            await send(message)

        async def reject(code: int, name: str, message: str) -> None:
            response = JSONResponse(
                {"error": {"code": name, "message": message, "request_id": request_id}},
                status_code=code,
                headers={"Retry-After": "1"} if code == 503 else None,
            )
            await response(scope, receive, observed)

        acquired = False
        try:
            if scope["method"] == "POST":
                if self.upload_active:
                    await reject(503, "inference_busy", "This worker is busy; retry shortly")
                    return
                self.upload_active, acquired = True, True  # No await between check and admission.
                limit = self.settings.max_upload_bytes + 64 * 1024  # Multipart envelope allowance.
                length = dict(scope.get("headers", [])).get(b"content-length")
                if length is not None:
                    try:
                        claimed = int(length)
                        if claimed < 0:
                            raise ValueError
                    except ValueError:
                        await reject(400, "invalid_content_length", "Invalid Content-Length")
                        return
                    if claimed > limit:
                        await reject(413, "upload_too_large", "Request body exceeds the limit")
                        return
                body = bytearray()
                try:
                    async with asyncio.timeout(self.settings.upload_timeout_seconds):
                        while True:
                            message = await receive()
                            if message["type"] == "http.disconnect":
                                return
                            chunk = message.get("body", b"")
                            if len(body) + len(chunk) > limit:
                                await reject(
                                    413, "upload_too_large", "Request body exceeds the limit"
                                )
                                return
                            body.extend(chunk)
                            if not message.get("more_body", False):
                                break
                except TimeoutError:
                    await reject(408, "upload_timeout", "Upload did not finish in time")
                    return
                delivered = False

                async def buffered_receive() -> Message:
                    nonlocal delivered
                    if not delivered:
                        delivered = True
                        return {"type": "http.request", "body": bytes(body), "more_body": False}
                    return await receive()

                await self.app(scope, buffered_receive, observed)
            else:
                await self.app(scope, receive, observed)
        finally:
            if acquired:
                self.upload_active = False
            logger.info(
                "request_complete",
                extra={
                    "request_id": request_id,
                    "method": scope["method"],
                    "path": scope["path"],
                    "status": status,
                    "duration_ms": round((perf_counter() - started) * 1000, 2),
                },
            )

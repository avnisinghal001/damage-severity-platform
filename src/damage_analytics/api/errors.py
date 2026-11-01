"""Safe domain errors, independent of HTTP/framework imports."""


class ApiError(Exception):
    """Expose only a deliberately written public message, never a database/decoder exception."""

    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status, self.code, self.message = status, code, message

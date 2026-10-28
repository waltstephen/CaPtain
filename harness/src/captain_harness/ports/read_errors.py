"""Shared public-reader failure contract for model and Python callers."""


class ToolInputError(ValueError):
    def __init__(self, message, *, code="invalid_tool_arguments", **details):
        super().__init__(message)
        self.code=code
        self.details=details


class NativeToolInternalError(RuntimeError):
    scope="native_tool"
    def __init__(self, tool_name, cause):
        self.tool_name=tool_name
        self.cause_type=type(cause).__name__
        super().__init__(f"{tool_name}: {self.cause_type}: {cause}")


def tool_error_payload(exc, **guidance):
    return {**guidance, **getattr(exc,"details",{}),
            "success":False,"recoverable":True,
            "error_code":getattr(exc,"code","invalid_tool_arguments"),
            "error":str(exc)[:600]}

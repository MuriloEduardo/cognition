from pydantic import BaseModel


class GenerateRequest(BaseModel):
    request_id: str
    prompt: str
    model: str = "default"
    max_tokens: int = 1024


class GenerateResponse(BaseModel):
    request_id: str
    content: str
    model: str
    error: str | None = None

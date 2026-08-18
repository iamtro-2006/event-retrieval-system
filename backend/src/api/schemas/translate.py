from pydantic import BaseModel, Field


class TranslateRequest(BaseModel):
    text: str = Field(..., min_length=1)
    source: str = "vi"
    target: str = "en"
    provider: str = "google"
    api_key: str | None = None


class TranslateResponse(BaseModel):
    text: str
    translated_text: str
    provider: str
    fallback: bool = False

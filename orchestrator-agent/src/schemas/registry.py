from typing import List
from pydantic import BaseModel, Field, HttpUrl


class ToolExample(BaseModel):
    prompt: str
    reasoning: str


class ToolParameters(BaseModel):
    required: List[str]
    optional: List[str] = Field(default_factory=list)


class ToolDefinition(BaseModel):
    name: str
    description: str
    endpoint: HttpUrl
    capabilities: List[str]
    parameters: ToolParameters
    examples: List[ToolExample] = Field(default_factory=list)
from pydantic import BaseModel, Field
from typing import Optional

class PaperSummary(BaseModel):
    title: str
    url: str
    year: Optional[int] = None
    main_claim: str = Field(description="Makalenin ana iddiası, 1-2 cümle")
    method: str = Field(description="Kullanılan yöntem veya yaklaşım")
    key_findings: list[str] = Field(description="3-5 madde halinde bulgular")
    confidence: float = Field(ge=0.0, le=1.0, description="Özetin güvenilirliği")

class ContradictionReport(BaseModel):
    contradictions: list[str] = Field(description="Çelişen bulgular")
    consensus_points: list[str] = Field(description="Tüm çalışmaların hemfikir olduğu noktalar")
    open_questions: list[str] = Field(description="Hâlâ yanıtsız sorular")

class SynthesisReport(BaseModel):
    question: str
    summary: str = Field(description="Genel konsensüs, 3-4 paragraf")
    contradictions: list[str]
    open_questions: list[str]
    sources: list[str]
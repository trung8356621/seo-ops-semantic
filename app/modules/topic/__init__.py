"""Topic semantic analysis module (proposal only — not Laravel Topic authority)."""

from app.modules.topic.analyzer import TopicAnalyzer
from app.modules.topic.contracts import TopicAnalysisRequest, TopicAnalysisResponse

__all__ = [
    "TopicAnalyzer",
    "TopicAnalysisRequest",
    "TopicAnalysisResponse",
]

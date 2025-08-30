from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum

class ProcessingStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class PageInfo(BaseModel):
    page_number: int
    original_position: int
    new_position: int
    confidence_score: float = Field(ge=0.0, le=1.0)
    content_preview: str
    reasoning: str

class DocumentAnalysis(BaseModel):
    document_type: Optional[str] = None
    total_pages: int
    detected_sections: List[str] = []
    missing_pages: List[int] = []
    duplicate_pages: List[int] = []

class ProcessingLog(BaseModel):
    timestamp: str
    level: str  # INFO, WARNING, ERROR
    message: str
    details: Optional[Dict[str, Any]] = None

class JobResponse(BaseModel):
    job_id: str
    status: ProcessingStatus
    progress: int = Field(ge=0, le=100)
    message: str
    
class JobResult(BaseModel):
    job_id: str
    status: ProcessingStatus
    original_filename: str
    pages_reordered: List[PageInfo]
    document_analysis: DocumentAnalysis
    processing_logs: List[ProcessingLog]
    output_filename: Optional[str] = None
    total_processing_time: Optional[float] = None

class UploadResponse(BaseModel):
    job_id: str
    filename: str
    message: str
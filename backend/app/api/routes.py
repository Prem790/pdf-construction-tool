# backend\app\api\routes.py

from fastapi import APIRouter, UploadFile, File, HTTPException, Request, BackgroundTasks
from fastapi.responses import FileResponse
import uuid
import os
from pathlib import Path
from datetime import datetime
import asyncio
from typing import Dict

from app.core.config import settings
from app.models.schemas import *
from app.services.pdf_service import PDFService
from app.services.ocr_service import OCRService
from app.services.ordering_service import AdaptiveOrderingService

router = APIRouter()

# Global processing status storage
processing_jobs: Dict[str, JobResult] = {}

@router.post("/upload", response_model=UploadResponse)
async def upload_pdf(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Upload PDF file and start processing"""
    
    # Validate file
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    if file.size and file.size > settings.MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File too large")
    
    # Generate job ID
    job_id = str(uuid.uuid4())
    print(f"Created job ID: {job_id}")  # Debug log
    
    # Save uploaded file
    upload_path = os.path.join(settings.UPLOAD_DIR, f"{job_id}_{file.filename}")
    
    try:
        content = await file.read()
        with open(upload_path, "wb") as f:
            f.write(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")
    
    # Initialize job status BEFORE starting background task
    processing_jobs[job_id] = JobResult(
        job_id=job_id,
        status=ProcessingStatus.PENDING,
        original_filename=file.filename,
        pages_reordered=[],
        document_analysis=DocumentAnalysis(total_pages=0),
        processing_logs=[]
    )
    
    print(f"Initialized job {job_id} in storage. Current jobs: {list(processing_jobs.keys())}")
    
    # Start background processing
    background_tasks.add_task(process_pdf_background, job_id, upload_path, file.filename)
    
    return UploadResponse(
        job_id=job_id,
        filename=file.filename,
        message="File uploaded successfully. Processing started."
    )

@router.get("/status/{job_id}", response_model=JobResult)
async def get_job_status(job_id: str):
    """Get processing status for a job"""
    print(f"Looking for job {job_id}. Available jobs: {list(processing_jobs.keys())}")
    
    if job_id not in processing_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return processing_jobs[job_id]

@router.get("/download/{job_id}")
async def download_result(job_id: str):
    """Download the reordered PDF"""
    if job_id not in processing_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = processing_jobs[job_id]
    if job.status != ProcessingStatus.COMPLETED or not job.output_filename:
        raise HTTPException(status_code=400, detail="Job not completed or no output file")
    
    output_path = os.path.join(settings.OUTPUT_DIR, job.output_filename)
    if not os.path.exists(output_path):
        raise HTTPException(status_code=404, detail="Output file not found")
    
    return FileResponse(
        output_path,
        media_type='application/pdf',
        filename=f"reordered_{job.original_filename}"
    )

@router.get("/logs/{job_id}")
async def get_processing_logs(job_id: str):
    """Get detailed processing logs for a job"""
    if job_id not in processing_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = processing_jobs[job_id]
    return {
        "job_id": job_id,
        "logs": job.processing_logs,
        "page_details": job.pages_reordered
    }

@router.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
    """Delete job and associated files"""
    if job_id not in processing_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Delete output file if exists
    job = processing_jobs[job_id]
    if job.output_filename:
        output_path = os.path.join(settings.OUTPUT_DIR, job.output_filename)
        try:
            if os.path.exists(output_path):
                os.remove(output_path)
        except:
            pass
    
    # Remove from processing jobs
    del processing_jobs[job_id]
    
    return {"message": "Job deleted successfully"}

@router.post("/feedback/{job_id}")
async def submit_ordering_feedback(job_id: str, feedback: dict):
    """Submit feedback on page ordering for learning"""
    if job_id not in processing_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    try:
        job = processing_jobs[job_id]
        original_order = [page.original_position for page in job.pages_reordered]
        corrected_order = feedback.get("corrected_order", [])
        document_type = feedback.get("document_type", "unknown")
        
        # Initialize a learning-enabled ordering service for feedback
        learning_service = AdaptiveOrderingService()
        learning_service.learn_from_feedback(original_order, corrected_order, document_type)
        
        return {"message": "Feedback recorded successfully", "job_id": job_id}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to record feedback: {str(e)}")

@router.get("/strategy-performance")
async def get_strategy_performance():
    """Get performance statistics for debugging"""
    service = AdaptiveOrderingService()
    return service.get_strategy_performance()

@router.get("/jobs")
async def list_jobs():
    """List all processing jobs"""
    return {
        "jobs": [
            {
                "job_id": job_id,
                "status": job.status,
                "filename": job.original_filename,
                "created": job.processing_logs[0].timestamp if job.processing_logs else None
            }
            for job_id, job in processing_jobs.items()
        ]
    }

async def process_pdf_background(job_id: str, pdf_path: str, original_filename: str):
    """Background task to process PDF"""
    print(f"Background processing started for job {job_id}")
    start_time = datetime.now()
    
    try:
        # Ensure the job exists in storage
        if job_id not in processing_jobs:
            print(f"ERROR: Job {job_id} not found in storage during background processing")
            return
        
        # Update status to processing
        processing_jobs[job_id].status = ProcessingStatus.PROCESSING
        processing_jobs[job_id].processing_logs.append(
            ProcessingLog(
                timestamp=start_time.isoformat(),
                level="INFO",
                message="Starting PDF processing"
            )
        )
        
        print(f"Updated job {job_id} status to PROCESSING")
        
        # Initialize services
        pdf_service = PDFService()
        ocr_service = OCRService()
        ordering_service = AdaptiveOrderingService()
        
        # Step 1: Extract pages
        processing_jobs[job_id].processing_logs.append(
            ProcessingLog(
                timestamp=datetime.now().isoformat(),
                level="INFO",
                message="Extracting pages from PDF"
            )
        )
        
        pages_data = pdf_service.extract_pages(pdf_path)
        processing_jobs[job_id].document_analysis.total_pages = len(pages_data)
        
        # Step 2: OCR Processing
        processing_jobs[job_id].processing_logs.append(
            ProcessingLog(
                timestamp=datetime.now().isoformat(),
                level="INFO",
                message=f"Processing {len(pages_data)} pages with OCR"
            )
        )
        
        for i, page_data in enumerate(pages_data):
            # Only do OCR if we have an image path
            if page_data.get('image_path'):
                try:
                    ocr_result = ocr_service.extract_text_from_image(page_data['image_path'])
                    page_data['ocr_result'] = ocr_result
                    
                    # Analyze structure
                    if ocr_result.get('text'):
                        structure = ocr_service.analyze_document_structure(ocr_result['text'])
                        page_data['structure_analysis'] = structure
                    else:
                        page_data['structure_analysis'] = {}
                except Exception as e:
                    print(f"OCR failed for page {i}: {e}")
                    page_data['ocr_result'] = {'text': '', 'confidence': 0.0, 'method': 'failed'}
                    page_data['structure_analysis'] = {}
            else:
                # For pages without images, create empty OCR result
                page_data['ocr_result'] = {'text': '', 'confidence': 0.0, 'method': 'none'}
                page_data['structure_analysis'] = {}
        
        # Step 2.5: Remove blank and duplicate pages BEFORE ordering
        processing_jobs[job_id].processing_logs.append(
            ProcessingLog(
                timestamp=datetime.now().isoformat(),
                level="INFO",
                message="Removing blank and duplicate pages from document"
            )
        )
        
        try:
            cleaned_pages, removal_log = ordering_service.remove_blank_and_duplicate_pages(pages_data)
            
            # Log the cleaning results
            processing_jobs[job_id].processing_logs.append(
                ProcessingLog(
                    timestamp=datetime.now().isoformat(),
                    level="INFO",
                    message=f"Page cleaning completed: {removal_log['original_count']} → {removal_log['final_count']} pages"
                )
            )
            
            if removal_log['blank_pages_removed']:
                processing_jobs[job_id].processing_logs.append(
                    ProcessingLog(
                        timestamp=datetime.now().isoformat(),
                        level="INFO",
                        message=f"Removed {len(removal_log['blank_pages_removed'])} blank pages: {[p+1 for p in removal_log['blank_pages_removed']]}"
                    )
                )
            
            if removal_log['duplicate_pages_removed']:
                duplicate_info = [f"{dup[0]+1}→{dup[1]+1}" for dup in removal_log['duplicate_pages_removed']]
                processing_jobs[job_id].processing_logs.append(
                    ProcessingLog(
                        timestamp=datetime.now().isoformat(),
                        level="INFO",
                        message=f"Removed {len(removal_log['duplicate_pages_removed'])} duplicate pages: {duplicate_info}"
                    )
                )
            
            # Update pages_data to use cleaned pages
            pages_data = cleaned_pages
            
            # Update document analysis with new page count
            processing_jobs[job_id].document_analysis.total_pages = len(pages_data)
            
        except Exception as e:
            print(f"Error during page cleaning: {e}")
            processing_jobs[job_id].processing_logs.append(
                ProcessingLog(
                    timestamp=datetime.now().isoformat(),
                    level="WARNING",
                    message=f"Page cleaning failed: {str(e)}. Proceeding with original pages."
                )
            )
        
        # Step 3: Determine page order (using cleaned pages)
        processing_jobs[job_id].processing_logs.append(
            ProcessingLog(
                timestamp=datetime.now().isoformat(),
                level="INFO",
                message=f"Analyzing document structure and determining optimal page order for {len(pages_data)} pages"
            )
        )
        
        try:
            page_infos, ordering_logs = ordering_service.order_pages(pages_data)
            processing_jobs[job_id].pages_reordered = page_infos
            processing_jobs[job_id].processing_logs.extend(ordering_logs)
        except Exception as e:
            print(f"Page ordering failed: {e}")
            # Create fallback ordering
            fallback_infos = []
            for i in range(len(pages_data)):
                fallback_infos.append(PageInfo(
                    page_number=i + 1,
                    original_position=i,
                    new_position=i,
                    confidence_score=0.3,
                    content_preview="Fallback ordering",
                    reasoning="Ordering service failed, keeping original sequence"
                ))
            processing_jobs[job_id].pages_reordered = fallback_infos
            processing_jobs[job_id].processing_logs.append(
                ProcessingLog(
                    timestamp=datetime.now().isoformat(),
                    level="WARNING",
                    message=f"Page ordering failed: {str(e)}. Using fallback."
                )
            )
        
        # Step 4: Create reordered PDF
        new_order = [page.original_position for page in processing_jobs[job_id].pages_reordered]
        output_filename = f"{job_id}_reordered.pdf"
        output_path = os.path.join(settings.OUTPUT_DIR, output_filename)
        
        processing_jobs[job_id].processing_logs.append(
            ProcessingLog(
                timestamp=datetime.now().isoformat(),
                level="INFO",
                message="Creating reordered PDF"
            )
        )
        
        pdf_service.create_reordered_pdf(pdf_path, new_order, output_path)
        
        # Step 5: Document analysis (simplified to avoid errors)
        missing_pages = []
        duplicate_pages = []
        
        try:
            # Create safe page contents for analysis
            safe_page_contents = []
            for page in pages_data:
                direct_text = page.get('direct_text', '')
                ocr_text = ''
                if page.get('ocr_result') and isinstance(page['ocr_result'], dict):
                    ocr_text = page['ocr_result'].get('text', '')
                
                safe_content = {
                    'content': direct_text + ' ' + ocr_text,
                    'page_number': page['page_number']
                }
                safe_page_contents.append(safe_content)
            
            missing_pages = ordering_service.detect_missing_pages(safe_page_contents)
            duplicate_pages = ordering_service.detect_duplicate_pages(safe_page_contents)
        except Exception as e:
            print(f"Warning: Document analysis failed: {e}")
        
        # Determine document type from first page
        document_type = "unknown"
        try:
            if pages_data and pages_data[0].get('structure_analysis'):
                document_type = pages_data[0]['structure_analysis'].get('document_type', 'unknown')
        except:
            pass
        
        processing_jobs[job_id].document_analysis = DocumentAnalysis(
            document_type=document_type,
            total_pages=len(pages_data),
            detected_sections=[],
            missing_pages=missing_pages,
            duplicate_pages=duplicate_pages
        )
        
        # Complete processing
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        processing_jobs[job_id].status = ProcessingStatus.COMPLETED
        processing_jobs[job_id].output_filename = output_filename
        processing_jobs[job_id].total_processing_time = processing_time
        processing_jobs[job_id].processing_logs.append(
            ProcessingLog(
                timestamp=end_time.isoformat(),
                level="INFO",
                message=f"Processing completed successfully in {processing_time:.2f} seconds"
            )
        )
        
        # Cleanup
        pdf_service.cleanup()
        
        print(f"Background processing completed successfully for job {job_id}")
        
    except Exception as e:
        print(f"Background processing failed for job {job_id}: {e}")
        # Handle errors - but keep the job in storage
        if job_id in processing_jobs:
            processing_jobs[job_id].status = ProcessingStatus.FAILED
            processing_jobs[job_id].processing_logs.append(
                ProcessingLog(
                    timestamp=datetime.now().isoformat(),
                    level="ERROR",
                    message=f"Processing failed: {str(e)}"
                )
            )
    
    finally:
        # Clean up uploaded file
        try:
            if os.path.exists(pdf_path):
                os.remove(pdf_path)
                print(f"Cleaned up uploaded file: {pdf_path}")
        except Exception as e:
            print(f"Warning: Could not clean up uploaded file: {e}")
        
        # DO NOT DELETE THE JOB FROM STORAGE HERE
        # The job should remain available for status checking and download
        print(f"Background processing task finished for job {job_id}")
# backend/app/services/ocr_service.py

import os
import re
import time
from typing import Dict, Optional, List
from PIL import Image
import google.generativeai as genai
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from msrest.authentication import CognitiveServicesCredentials

from app.core.config import settings

class OCRService:
    def __init__(self):
        self.gemini_client = None
        self.azure_client = None
        self._setup_clients()
    
    def _setup_clients(self):
        """Initialize OCR clients with proper error handling"""
        # Setup Gemini
        if hasattr(settings, 'GEMINI_API_KEY') and settings.GEMINI_API_KEY:
            try:
                genai.configure(api_key=settings.GEMINI_API_KEY)
                self.gemini_client = genai.GenerativeModel('gemini-1.5-flash')
                print("Gemini OCR client initialized successfully")
            except Exception as e:
                print(f"Failed to initialize Gemini OCR: {e}")
        
        # Setup Azure Computer Vision
        if (hasattr(settings, 'AZURE_VISION_KEY') and settings.AZURE_VISION_KEY and 
            hasattr(settings, 'AZURE_VISION_ENDPOINT') and settings.AZURE_VISION_ENDPOINT):
            try:
                credential = CognitiveServicesCredentials(settings.AZURE_VISION_KEY)
                self.azure_client = ComputerVisionClient(settings.AZURE_VISION_ENDPOINT, credential)
                print("Azure Computer Vision client initialized successfully")
            except Exception as e:
                print(f"Failed to initialize Azure OCR: {e}")
    
    def extract_text_from_image(self, image_path: str) -> Dict:
        """Extract text from image using available OCR services with fallback"""
        
        if not os.path.exists(image_path):
            return {
                'text': '',
                'confidence': 0.0,
                'method': 'file_not_found',
                'error': f'Image file not found: {image_path}'
            }
        
        # Try Gemini first (if available and not rate limited)
        if self.gemini_client:
            try:
                result = self._extract_with_gemini(image_path)
                if result['text'].strip() and result['confidence'] > 0.3:
                    return result
            except Exception as e:
                print(f"Gemini OCR failed for {image_path}: {e}")
                # Continue to Azure fallback
        
        # Try Azure as fallback
        if self.azure_client:
            try:
                result = self._extract_with_azure(image_path)
                if result['text'].strip():
                    return result
            except Exception as e:
                print(f"Azure OCR failed for {image_path}: {e}")
        
        # Ultimate fallback - return empty result
        print(f"All OCR methods failed for {image_path}")
        return {
            'text': '',
            'confidence': 0.0,
            'method': 'all_failed',
            'error': 'All OCR services failed'
        }
    
    def _extract_with_gemini(self, image_path: str) -> Dict:
        """Extract text using Gemini Vision API"""
        try:
            # Load image
            image = Image.open(image_path)
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Create prompt for text extraction
            prompt = """
            Extract ALL text from this document page. Return the text exactly as it appears, maintaining structure and formatting.
            Include headers, body text, numbers, signatures, and any other visible text content.
            """
            
            # Generate content with image
            response = self.gemini_client.generate_content([prompt, image])
            
            if response.text:
                extracted_text = response.text.strip()
                
                # Calculate confidence based on text quality
                confidence = self._calculate_text_confidence(extracted_text)
                
                return {
                    'text': extracted_text,
                    'confidence': confidence,
                    'method': 'gemini',
                    'word_count': len(extracted_text.split())
                }
            else:
                return {
                    'text': '',
                    'confidence': 0.0,
                    'method': 'gemini_no_response',
                    'error': 'Gemini returned empty response'
                }
                
        except Exception as e:
            if "429" in str(e) or "quota" in str(e).lower():
                raise Exception(f"Gemini rate limit exceeded: {e}")
            else:
                raise Exception(f"Gemini OCR processing error: {e}")
    
    def _extract_with_azure(self, image_path: str) -> Dict:
        """Extract text using Azure Computer Vision API"""
        try:
            with open(image_path, 'rb') as image_stream:
                # Start read operation
                read_response = self.azure_client.read_in_stream(image_stream, raw=True)
            
            # Get operation ID from response headers
            read_operation_location = read_response.headers["Operation-Location"]
            operation_id = read_operation_location.split("/")[-1]
            
            # Poll for completion
            max_attempts = 20
            for attempt in range(max_attempts):
                read_result = self.azure_client.get_read_result(operation_id)
                
                if read_result.status == OperationStatusCodes.succeeded:
                    break
                elif read_result.status == OperationStatusCodes.failed:
                    raise Exception("Azure OCR operation failed")
                
                time.sleep(0.5)  # Wait before next check
            
            if read_result.status != OperationStatusCodes.succeeded:
                raise Exception("Azure OCR operation timed out")
            
            # Extract text from results
            extracted_lines = []
            confidence_scores = []
            
            for read_result_page in read_result.analyze_result.read_results:
                for line in read_result_page.lines:
                    extracted_lines.append(line.text)
                    # Azure doesn't provide line confidence, use default
                    confidence_scores.append(0.85)
            
            extracted_text = '\n'.join(extracted_lines)
            avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
            
            return {
                'text': extracted_text,
                'confidence': avg_confidence,
                'method': 'azure',
                'line_count': len(extracted_lines)
            }
            
        except Exception as e:
            raise Exception(f"Azure OCR processing error: {e}")
    
    def _calculate_text_confidence(self, text: str) -> float:
        """Calculate confidence score based on extracted text quality"""
        if not text:
            return 0.0
        
        score = 0.5  # Base score
        
        # Length factor
        if len(text) > 100:
            score += 0.2
        elif len(text) > 50:
            score += 0.1
        
        # Word count factor
        word_count = len(text.split())
        if word_count > 20:
            score += 0.2
        elif word_count > 10:
            score += 0.1
        
        # Structure indicators
        if re.search(r'\d+', text):  # Has numbers
            score += 0.1
        if re.search(r'[A-Z][a-z]+', text):  # Has proper case words
            score += 0.1
        if re.search(r'[.!?]', text):  # Has sentence endings
            score += 0.1
        
        return min(0.95, score)
    
    def analyze_document_structure(self, text: str) -> Dict:
        """Analyze document structure and classify content"""
        if not text:
            return {
                'document_type': 'unknown',
                'sections': [],
                'has_page_numbers': False,
                'has_dates': False,
                'structure_confidence': 0.0
            }
        
        text_lower = text.lower()
        
        # Document type detection
        document_type = 'unknown'
        type_confidence = 0.0
        
        # Loan agreement detection
        loan_indicators = ['loan agreement', 'ireda', 'borrower', 'lender', 'article']
        loan_score = sum(1 for indicator in loan_indicators if indicator in text_lower)
        
        # Mortgage detection  
        mortgage_indicators = ['mortgage', 'property address', 'appraised value', 'borrower information']
        mortgage_score = sum(1 for indicator in mortgage_indicators if indicator in text_lower)
        
        if loan_score >= 2:
            document_type = 'loan_agreement'
            type_confidence = min(0.9, loan_score / len(loan_indicators))
        elif mortgage_score >= 2:
            document_type = 'mortgage'
            type_confidence = min(0.9, mortgage_score / len(mortgage_indicators))
        
        # Section detection
        detected_sections = []
        section_mapping = {
            'article - i': 'definitions',
            'article - ii': 'loan_terms',
            'article - iii': 'security',
            'article - iv': 'appointment',
            'article - v': 'special_conditions',
            'article - vi': 'effective_date',
            'schedule': 'schedules',
            'witness whereof': 'signatures',
            'borrower information': 'borrower_info',
            'employment': 'employment',
            'property': 'property'
        }
        
        for pattern, section in section_mapping.items():
            if pattern in text_lower:
                detected_sections.append(section)
        
        # Feature detection
        has_page_numbers = bool(re.search(r'-\d+-|page\s+\d+|\d+\s+of\s+\d+', text))
        has_dates = bool(re.search(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\w+\s+\d{1,2},?\s+\d{4}', text))
        
        # Calculate structure confidence
        structure_indicators = len(detected_sections) + (1 if has_page_numbers else 0) + (1 if has_dates else 0)
        structure_confidence = min(0.9, structure_indicators / 5.0)
        
        return {
            'document_type': document_type,
            'sections': detected_sections,
            'has_page_numbers': has_page_numbers,
            'has_dates': has_dates,
            'structure_confidence': structure_confidence,
            'word_count': len(text.split()),
            'line_count': len(text.split('\n')),
            'type_confidence': type_confidence
        }
    
    def get_available_services(self) -> Dict[str, bool]:
        """Check which OCR services are available"""
        return {
            'gemini_available': self.gemini_client is not None,
            'azure_available': self.azure_client is not None,
            'has_any_service': self.gemini_client is not None or self.azure_client is not None
        }
    
    def test_service_connectivity(self) -> Dict[str, str]:
        """Test connectivity to OCR services"""
        results = {}
        
        if self.gemini_client:
            try:
                # Simple test
                test_response = self.gemini_client.generate_content("Hello")
                results['gemini'] = 'connected' if test_response else 'failed'
            except Exception as e:
                results['gemini'] = f'failed: {str(e)[:100]}'
        else:
            results['gemini'] = 'not_configured'
        
        if self.azure_client:
            try:
                # Azure doesn't have a simple test, so we just check if client exists
                results['azure'] = 'configured'
            except Exception as e:
                results['azure'] = f'failed: {str(e)[:100]}'
        else:
            results['azure'] = 'not_configured'
        
        return results
import PyPDF2
from pdf2image import convert_from_path
import io
from pathlib import Path
from typing import List, Tuple, Dict
from PIL import Image
import tempfile
import os

class PDFService:
    def __init__(self):
        self.temp_dir = tempfile.mkdtemp()
    
    def extract_pages(self, pdf_path: str) -> List[Dict]:
        """Extract pages from PDF and convert to images for OCR"""
        pages_data = []
        
        try:
            # First, try to extract any existing text using PyPDF2
            direct_text_pages = []
            try:
                with open(pdf_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    for page_num in range(len(pdf_reader.pages)):
                        page = pdf_reader.pages[page_num]
                        text = page.extract_text()
                        direct_text_pages.append(text)
            except Exception as e:
                print(f"Warning: Could not extract direct text: {e}")
                # If PyPDF2 fails, create empty text for each page
                # We'll determine page count from images
                direct_text_pages = []
            
            # Convert PDF pages to images using pdf2image
            try:
                # Convert PDF to images (300 DPI for good OCR quality)
                images = convert_from_path(pdf_path, dpi=300, first_page=None, last_page=None)
            except Exception as e:
                raise Exception(f"Error converting PDF to images: {str(e)}")
            
            # Ensure we have text entries for all image pages
            while len(direct_text_pages) < len(images):
                direct_text_pages.append("")
            
            for page_num, image in enumerate(images):
                # Save temporary image file for OCR
                temp_image_path = os.path.join(self.temp_dir, f"page_{page_num}.png")
                image.save(temp_image_path, "PNG")
                
                page_data = {
                    'page_number': page_num,
                    'original_position': page_num,
                    'image_path': temp_image_path,
                    'direct_text': direct_text_pages[page_num] if page_num < len(direct_text_pages) else "",
                    'image_size': image.size,
                    'image': image  # Keep reference for potential use
                }
                
                pages_data.append(page_data)
                
            return pages_data
            
        except Exception as e:
            raise Exception(f"Error extracting pages: {str(e)}")
    
    def create_reordered_pdf(self, original_pdf_path: str, new_order: List[int], output_path: str) -> str:
        """Create a new PDF with pages in the specified order using PyPDF2"""
        try:
            # Open the source PDF
            with open(original_pdf_path, 'rb') as source_file:
                pdf_reader = PyPDF2.PdfReader(source_file)
                pdf_writer = PyPDF2.PdfWriter()
                
                # Add pages in the new order
                for page_index in new_order:
                    if 0 <= page_index < len(pdf_reader.pages):
                        page = pdf_reader.pages[page_index]
                        pdf_writer.add_page(page)
                
                # Write the reordered PDF
                with open(output_path, 'wb') as output_file:
                    pdf_writer.write(output_file)
            
            return output_path
            
        except Exception as e:
            # Fallback: create PDF from images if PyPDF2 fails
            try:
                return self._create_pdf_from_images(original_pdf_path, new_order, output_path)
            except Exception as fallback_error:
                raise Exception(f"Error creating reordered PDF: {str(e)}. Fallback also failed: {str(fallback_error)}")
    
    def _create_pdf_from_images(self, original_pdf_path: str, new_order: List[int], output_path: str) -> str:
        """Fallback method: create PDF from images when PyPDF2 fails"""
        try:
            # Convert original PDF to images
            images = convert_from_path(original_pdf_path, dpi=200)  # Lower DPI for output
            
            # Reorder images
            reordered_images = [images[i] for i in new_order if 0 <= i < len(images)]
            
            if reordered_images:
                # Convert images to PDF using PIL
                reordered_images[0].save(
                    output_path, 
                    "PDF",
                    resolution=200.0,
                    save_all=True,
                    append_images=reordered_images[1:] if len(reordered_images) > 1 else []
                )
            
            return output_path
            
        except Exception as e:
            raise Exception(f"Error creating PDF from images: {str(e)}")
    
    def get_pdf_info(self, pdf_path: str) -> Dict:
        """Get basic information about the PDF"""
        try:
            info = {
                'file_size': Path(pdf_path).stat().st_size,
                'page_count': 0,
                'title': '',
                'author': '',
                'creation_date': ''
            }
            
            # Try to get metadata using PyPDF2
            try:
                with open(pdf_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    info['page_count'] = len(pdf_reader.pages)
                    
                    # Get metadata if available
                    if pdf_reader.metadata:
                        info['title'] = pdf_reader.metadata.get('/Title', '') or ''
                        info['author'] = pdf_reader.metadata.get('/Author', '') or ''
                        info['creation_date'] = pdf_reader.metadata.get('/CreationDate', '') or ''
            except Exception as e:
                print(f"Warning: Could not extract PDF metadata: {e}")
                # Fallback: get page count from images
                try:
                    images = convert_from_path(pdf_path, dpi=72, first_page=1, last_page=1)
                    # Convert just first page to get total count estimate
                    all_images = convert_from_path(pdf_path, dpi=72)
                    info['page_count'] = len(all_images)
                except:
                    info['page_count'] = 0
            
            return info
        except Exception as e:
            raise Exception(f"Error getting PDF info: {str(e)}")
    
    def cleanup(self):
        """Clean up temporary files"""
        try:
            import shutil
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        except Exception as e:
            print(f"Warning: Could not clean up temp directory: {e}")
            pass
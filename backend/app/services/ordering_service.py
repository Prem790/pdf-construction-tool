# backend\app\services\ordering_service.py

from typing import List, Dict, Tuple, Optional, Any
import re
import json
from datetime import datetime
from pathlib import Path
import numpy as np
from collections import Counter, defaultdict
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from dateutil import parser

from app.core.config import settings
from app.models.schemas import PageInfo, ProcessingLog
from app.services.ordering_strategies import (
    PageNumberStrategy, ConfigurableBusinessLogicStrategy, 
    StructuralPatternStrategy, SemanticSimilarityStrategy,
    DateSequenceStrategy, LLMReasoningStrategy, OrderingResult
)

class AdaptiveOrderingService:
    """Enhanced ordering service that combines multiple strategies with domain expertise"""
    
    def __init__(self):
        self.embedding_model = None
        self.gemini_client = None
        self.strategies = []
        self.performance_stats = defaultdict(lambda: {'attempts': 0, 'successes': 0})
        self._setup_models()
        self._initialize_strategies()
    
    def _setup_models(self):
        """Initialize ML models and AI services"""
        try:
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            print("Sentence transformer loaded successfully")
        except Exception as e:
            print(f"Failed to load sentence transformer: {e}")
        
        if settings.GEMINI_API_KEY:
            try:
                genai.configure(api_key=settings.GEMINI_API_KEY)
                self.gemini_client = genai.GenerativeModel('gemini-pro')
                print("Gemini reasoning client initialized")
            except Exception as e:
                print(f"Failed to initialize Gemini reasoning: {e}")
    
    def _initialize_strategies(self):
        """Initialize all ordering strategies in priority order"""
        self.strategies = [
            PageNumberStrategy(),
            ConfigurableBusinessLogicStrategy(),
            StructuralPatternStrategy(),
            DateSequenceStrategy(),
            SemanticSimilarityStrategy(self.embedding_model),
            LLMReasoningStrategy(self.gemini_client)
        ]
    
    def order_pages(self, pages_data: List[Dict]) -> Tuple[List[PageInfo], List[ProcessingLog]]:
        """Enhanced page ordering with multiple strategies and domain heuristics"""
        logs = []
        start_time = datetime.now()
        
        # Prepare page contents with enhanced preprocessing
        page_contents = self._prepare_page_contents(pages_data)
        
        # DEBUG: Print what text is being extracted
        print("=== DEBUG: PAGE CONTENT EXTRACTION ===")
        for i, page in enumerate(page_contents):
            content = page.get('content', '')
            print(f"PAGE {i+1} (original pos {page.get('original_position', i)}):")
            print(f"Content length: {len(content)}")
            print(f"First 200 chars: {content[:200]}")
            print(f"Section type: {page.get('section_type', 'unknown')}")
            
            # Look for page numbers specifically
            page_nums = re.findall(r'-(\d+)-', content)
            if page_nums:
                print(f"FOUND PAGE NUMBERS: {page_nums}")
            print("-" * 50)
        
        logs.append(ProcessingLog(
            timestamp=start_time.isoformat(),
            level="INFO",
            message=f"Starting adaptive page ordering for {len(page_contents)} pages"
        ))
        
        # Apply strategies in order of reliability
        strategy_results = []
        
        for strategy in self.strategies:
            self.performance_stats[strategy.name]['attempts'] += 1
            
            try:
                if strategy.can_handle(page_contents):
                    result = strategy.attempt_ordering(page_contents)
                    strategy_results.append(result)
                    
                    logs.append(ProcessingLog(
                        timestamp=datetime.now().isoformat(),
                        level="INFO",
                        message=f"{strategy.name}: confidence {result.confidence:.2f}"
                    ))
                    
                    # Early exit for high-confidence results
                    if result.confidence >= 0.85:
                        self.performance_stats[strategy.name]['successes'] += 1
                        logs.append(ProcessingLog(
                            timestamp=datetime.now().isoformat(),
                            level="INFO",
                            message=f"High confidence result from {strategy.name}, using this ordering"
                        ))
                        return self._create_enhanced_page_info_list(result, page_contents, logs)
                
            except Exception as e:
                logs.append(ProcessingLog(
                    timestamp=datetime.now().isoformat(),
                    level="WARNING",
                    message=f"{strategy.name} failed: {str(e)}"
                ))
        
        # Combine results using weighted voting
        if strategy_results:
            best_result = self._combine_strategy_results(strategy_results, page_contents)
            logs.append(ProcessingLog(
                timestamp=datetime.now().isoformat(),
                level="INFO",
                message=f"Combined strategy result with confidence {best_result.confidence:.2f}"
            ))
        else:
            # Ultimate fallback
            best_result = OrderingResult(
                order=list(range(len(page_contents))),
                confidence=0.3,
                reasoning=['All strategies failed', 'Using original order'],
                method="fallback_original"
            )
            logs.append(ProcessingLog(
                timestamp=datetime.now().isoformat(),
                level="WARNING",
                message="All strategies failed, keeping original order"
            ))
        
        return self._create_enhanced_page_info_list(best_result, page_contents, logs)
    
    def _prepare_page_contents(self, pages_data: List[Dict]) -> List[Dict]:
        """Enhanced preprocessing of page contents"""
        page_contents = []
        
        for i, page in enumerate(pages_data):
            # Safely extract all text sources
            direct_text = page.get('direct_text', '')
            ocr_result = page.get('ocr_result', {})
            ocr_text = ocr_result.get('text', '') if isinstance(ocr_result, dict) else ''
            
            # Combine and clean text
            combined_text = self._clean_and_normalize_text(direct_text + ' ' + ocr_text)
            
            # Enhanced content structure
            content_dict = {
                'page_number': page.get('page_number', i + 1),
                'original_position': i,
                'content': combined_text,
                'structure': page.get('structure_analysis', {}),
                'ocr_confidence': ocr_result.get('confidence', 0.0) if isinstance(ocr_result, dict) else 0.0,
                'word_count': len(combined_text.split()),
                'has_numbers': bool(re.search(r'\d', combined_text)),
                'has_dates': bool(re.search(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', combined_text)),
                'has_signatures': self._detect_signature_indicators(combined_text),
                'section_type': self._classify_section_type(combined_text)
            }
            
            page_contents.append(content_dict)
        
        return page_contents
    
    def _clean_and_normalize_text(self, text: str) -> str:
        """Clean and normalize extracted text"""
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Fix common OCR errors
        text = re.sub(r'(\w)1(\w)', r'\1l\2', text)  # 1 -> l in words
        text = re.sub(r'(\w)0(\w)', r'\1o\2', text)  # 0 -> o in words
        
        # Normalize common abbreviations
        text = re.sub(r'\bSSN\b', 'social security', text, flags=re.IGNORECASE)
        text = re.sub(r'\bDOB\b', 'date of birth', text, flags=re.IGNORECASE)
        
        return text.strip()
    
    def _detect_signature_indicators(self, text: str) -> bool:
        """Detect if page likely contains signatures"""
        signature_patterns = [
            r'signature.*date',
            r'borrower.*signature',
            r'_+\s*date',
            r'sign.*here',
            r'acknowledgment',
            r'I/we.*acknowledge',
            r'witnessed.*by'
        ]
        
        for pattern in signature_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False
    
    def _classify_section_type(self, text: str) -> str:
        """Classify the likely section type of a page with enhanced multi-page detection"""
        text_lower = text.lower()
        
        # Priority-based classification with multi-page awareness
        if any(term in text_lower for term in ['signature', 'acknowledge', 'witnessed', 'borrower signature', 'lender signature']):
            return 'signature'
        elif any(term in text_lower for term in ['borrower information', 'applicant', 'ssn', 'social security', 'name:', 'address:']):
            return 'borrower_info'
        elif any(term in text_lower for term in ['employment', 'employer', 'income', 'salary', 'position:']):
            return 'employment'
        elif any(term in text_lower for term in ['property', 'address', 'subject property', 'appraised value']):
            return 'property'
        elif any(term in text_lower for term in ['loan terms', 'interest rate', 'monthly payment', 'loan amount']):
            return 'loan_terms'
        elif any(term in text_lower for term in ['disclosure', 'notice', 'privacy', 'respa', 'tila', 'legal disclosure']):
            return 'disclosure'
        elif any(term in text_lower for term in ['application', 'mortgage loan', 'loan request', 'loan application']):
            return 'application_header'
        else:
            return 'general'
    
    def _combine_strategy_results(self, results: List[OrderingResult], page_contents: List[Dict]) -> OrderingResult:
        """Intelligently combine results from multiple strategies"""
        if not results:
            return OrderingResult(
                order=list(range(len(page_contents))),
                confidence=0.1,
                reasoning=['No strategy results available'],
                method="no_results"
            )
        
        # Weight strategies by their confidence and reliability
        weighted_results = []
        for result in results:
            weight = self._calculate_strategy_weight(result)
            weighted_results.append((result, weight))
        
        # Sort by weighted confidence
        weighted_results.sort(key=lambda x: x[0].confidence * x[1], reverse=True)
        
        # Use highest weighted result as base
        best_result = weighted_results[0][0]
        
        # Apply domain-specific post-processing
        enhanced_order = self._apply_domain_heuristics(best_result.order, page_contents)
        
        # Adjust confidence based on domain compliance
        domain_confidence = self._evaluate_domain_compliance(enhanced_order, page_contents)
        final_confidence = (best_result.confidence * 0.7) + (domain_confidence * 0.3)
        
        combined_reasoning = best_result.reasoning + [
            f"Combined from {len(results)} strategies",
            f"Applied domain-specific heuristics"
        ]
        
        return OrderingResult(
            order=enhanced_order,
            confidence=min(0.95, final_confidence),
            reasoning=combined_reasoning,
            method=f"combined_{best_result.method}"
        )
    
    def _calculate_strategy_weight(self, result: OrderingResult) -> float:
        """Calculate reliability weight for a strategy"""
        base_weight = 1.0
        
        # Boost reliable strategies
        if result.method in ['explicit_page_numbers', 'configurable_business_logic']:
            base_weight *= 1.5
        elif result.method in ['structural_patterns', 'date_sequence']:
            base_weight *= 1.2
        
        # Consider past performance
        strategy_stats = self.performance_stats.get(result.method, {'attempts': 1, 'successes': 0})
        if strategy_stats['attempts'] > 0:
            success_rate = strategy_stats['successes'] / strategy_stats['attempts']
            base_weight *= (0.5 + success_rate)
        
        return base_weight
    
    def _apply_domain_heuristics(self, order: List[int], page_contents: List[Dict]) -> List[int]:
        """Apply mortgage industry domain knowledge to refine ordering"""
        enhanced_order = order.copy()
        
        # Step 1: Remove duplicates first
        enhanced_order = self._remove_duplicates(enhanced_order, page_contents)
        
        # Step 2: Group and order multi-page sections
        enhanced_order = self._order_multi_page_sections(enhanced_order, page_contents)
        
        # Step 3: Apply business logic ordering
        enhanced_order = self._apply_business_priority_ordering(enhanced_order, page_contents)
        
        return enhanced_order
    
    def _remove_duplicates(self, order: List[int], page_contents: List[Dict]) -> List[int]:
        """Remove duplicate pages from the ordering"""
        if not self.embedding_model or len(order) < 2:
            return order
        
        try:
            # Get texts for pages in current order
            texts = []
            valid_indices = []
            for idx in order:
                if idx < len(page_contents):
                    content = page_contents[idx].get('content', '').strip()
                    if content and len(content) > 50:  # Only consider substantial content
                        texts.append(content)
                        valid_indices.append(idx)
            
            if len(texts) < 2:
                return order
            
            # Calculate similarities
            embeddings = self.embedding_model.encode(texts)
            similarity_matrix = cosine_similarity(embeddings)
            
            # Find duplicates (high similarity pairs)
            to_remove = set()
            duplicate_threshold = 0.95
            
            for i in range(len(similarity_matrix)):
                for j in range(i + 1, len(similarity_matrix)):
                    if similarity_matrix[i][j] > duplicate_threshold:
                        # Keep the first occurrence, remove the second
                        idx_to_remove = valid_indices[j]
                        to_remove.add(idx_to_remove)
            
            # Remove duplicates from order
            cleaned_order = [idx for idx in order if idx not in to_remove]
            
            if len(to_remove) > 0:
                print(f"Removed {len(to_remove)} duplicate pages: {list(to_remove)}")
            
            return cleaned_order
            
        except Exception as e:
            print(f"Duplicate removal failed: {e}")
            return order
    
    def _order_multi_page_sections(self, order: List[int], page_contents: List[Dict]) -> List[int]:
        """Properly order multi-page sections like 'Page 1', 'Page 2', etc."""
        
        # Group pages by section type and identify multi-page sections
        section_groups = defaultdict(list)
        
        for idx in order:
            if idx < len(page_contents):
                page = page_contents[idx]
                content = page.get('content', '').lower()
                section_type = page.get('section_type', 'general')
                
                # Look for page numbering within sections
                page_number_in_section = self._extract_section_page_number(content)
                
                section_groups[section_type].append({
                    'idx': idx,
                    'page_number': page_number_in_section,
                    'content': content
                })
        
        # Sort each multi-page section internally
        reordered = []
        processed_indices = set()
        
        for idx in order:
            if idx in processed_indices:
                continue
                
            if idx < len(page_contents):
                page = page_contents[idx]
                section_type = page.get('section_type', 'general')
                
                if len(section_groups[section_type]) > 1:
                    # This is a multi-page section - process all pages at once
                    section_pages = section_groups[section_type]
                    
                    # Sort by page number within section
                    section_pages.sort(key=lambda x: x['page_number'] if x['page_number'] is not None else 999)
                    
                    for section_page in section_pages:
                        if section_page['idx'] not in processed_indices:
                            reordered.append(section_page['idx'])
                            processed_indices.add(section_page['idx'])
                else:
                    # Single page section
                    reordered.append(idx)
                    processed_indices.add(idx)
        
        return reordered
    
    def _extract_section_page_number(self, content: str) -> Optional[int]:
        """Extract page number from within a section (e.g., 'Legal Disclosures (Page 2)')"""
        
        # Look for patterns like "(Page 2)", "(Page 1)", etc.
        page_patterns = [
            r'\(page\s+(\d+)\)',
            r'page\s+(\d+)\s*of\s*\d+',
            r'section\s+(\d+):',
            r'part\s+(\d+)',
        ]
        
        for pattern in page_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                try:
                    return int(matches[0])
                except (ValueError, IndexError):
                    continue
        
        return None
    
    def _apply_business_priority_ordering(self, order: List[int], page_contents: List[Dict]) -> List[int]:
        """Apply business logic priority ordering"""
        
        # Mortgage document priority rules
        priority_rules = {
            'application_header': 1,
            'borrower_info': 2,
            'employment': 3,
            'property': 4,
            'loan_terms': 5,
            'disclosure': 8,
            'signature': 10
        }
        
        # Create sections with their priorities
        sections = []
        for idx in order:
            if idx < len(page_contents):
                page = page_contents[idx]
                section_type = page.get('section_type', 'general')
                priority = priority_rules.get(section_type, 6)
                
                sections.append({
                    'original_idx': idx,
                    'section_type': section_type,
                    'priority': priority,
                    'has_signatures': page.get('has_signatures', False),
                    'word_count': page.get('word_count', 0)
                })
        
        # Sort by priority, then by word count (longer content first within same priority)
        sections.sort(key=lambda x: (x['priority'], -x['word_count']))
        
        # Force signature pages to the end
        signature_pages = [s for s in sections if s['has_signatures']]
        non_signature_pages = [s for s in sections if not s['has_signatures']]
        
        # Recombine
        final_sections = non_signature_pages + signature_pages
        enhanced_order = [s['original_idx'] for s in final_sections]
        
        return enhanced_order
    
    def _evaluate_domain_compliance(self, order: List[int], page_contents: List[Dict]) -> float:
        """Evaluate how well the order follows mortgage document conventions"""
        if not order or not page_contents:
            return 0.0
        
        score = 0.0
        max_score = 4.0
        
        # Check if application/header comes first
        first_page = page_contents[order[0]] if order[0] < len(page_contents) else {}
        if first_page.get('section_type') in ['application_header', 'borrower_info']:
            score += 1.0
        
        # Check if signature pages come last
        last_two_indices = order[-2:] if len(order) >= 2 else order[-1:]
        signature_at_end = 0
        for idx in last_two_indices:
            if idx < len(page_contents) and page_contents[idx].get('has_signatures'):
                signature_at_end += 1
        score += min(1.0, signature_at_end / len(last_two_indices))
        
        # Check for logical section progression
        section_sequence = []
        for idx in order:
            if idx < len(page_contents):
                section_sequence.append(page_contents[idx].get('section_type', 'general'))
        
        # Reward proper sequence: header -> borrower -> employment -> property -> terms -> disclosure -> signature
        expected_sequence = ['application_header', 'borrower_info', 'employment', 'property', 'loan_terms', 'disclosure', 'signature']
        sequence_score = self._calculate_sequence_similarity(section_sequence, expected_sequence)
        score += sequence_score * 2.0  # Weight this heavily
        
        return min(1.0, score / max_score)
    
    def _calculate_sequence_similarity(self, actual: List[str], expected: List[str]) -> float:
        """Calculate how similar the actual sequence is to the expected sequence"""
        if not actual or not expected:
            return 0.0
        
        # Use longest common subsequence approach
        m, n = len(actual), len(expected)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if actual[i-1] == expected[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n] / max(m, n)
    
    def _create_enhanced_page_info_list(self, result: OrderingResult, page_contents: List[Dict], logs: List[ProcessingLog]) -> Tuple[List[PageInfo], List[ProcessingLog]]:
        """Create enhanced PageInfo objects with detailed analysis"""
        page_infos = []
        
        for new_position, original_idx in enumerate(result.order):
            if original_idx < len(page_contents):
                page = page_contents[original_idx]
                
                # Create meaningful content preview
                content = page.get('content', '')
                content_preview = self._create_content_preview(content, page.get('section_type', 'general'))
                
                # Generate detailed reasoning
                detailed_reasoning = self._generate_page_reasoning(page, new_position, result)
                
                page_info = PageInfo(
                    page_number=original_idx + 1,
                    original_position=original_idx,
                    new_position=new_position,
                    confidence_score=result.confidence,
                    content_preview=content_preview,
                    reasoning=detailed_reasoning
                )
                page_infos.append(page_info)
        
        # Add summary log
        logs.append(ProcessingLog(
            timestamp=datetime.now().isoformat(),
            level="INFO",
            message=f"Generated {len(page_infos)} page ordering recommendations"
        ))
        
        return page_infos, logs
    
    def _create_content_preview(self, content: str, section_type: str) -> str:
        """Create an intelligent content preview"""
        if not content:
            return "Empty page"
        
        # Extract key information based on section type
        if section_type == 'borrower_info':
            # Look for names, SSN patterns
            names = re.findall(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b', content)
            if names:
                return f"Borrower Information: {names[0]}"
        
        elif section_type == 'signature':
            return "Signature page with acknowledgments"
        
        elif section_type == 'employment':
            # Look for employer names
            employer_match = re.search(r'employer[:\s]+([^,\n]+)', content, re.IGNORECASE)
            if employer_match:
                return f"Employment: {employer_match.group(1).strip()}"
        
        # Default preview - first meaningful sentence
        sentences = re.split(r'[.!?]+', content)
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 20 and not re.match(r'^\s*\d+\s*$', sentence):
                return sentence[:100] + "..." if len(sentence) > 100 else sentence
        
        return content[:100] + "..." if len(content) > 100 else content
    
    def _generate_page_reasoning(self, page: Dict, position: int, result: OrderingResult) -> str:
        """Generate detailed reasoning for page placement"""
        reasons = []
        
        section_type = page.get('section_type', 'general')
        if section_type != 'general':
            reasons.append(f"Identified as {section_type.replace('_', ' ')} section")
        
        if page.get('has_signatures'):
            reasons.append("Contains signature fields (placed toward end)")
        
        if position == 0:
            reasons.append("Placed first as document header/opening")
        elif position == len(result.order) - 1:
            reasons.append("Placed last as concluding/signature page")
        
        if page.get('ocr_confidence', 0) > 0.8:
            reasons.append("High OCR confidence")
        elif page.get('ocr_confidence', 0) < 0.5:
            reasons.append("Low OCR confidence - placement uncertain")
        
        method_description = {
            'explicit_page_numbers': 'Found clear page numbers',
            'configurable_business_logic': 'Business logic rules applied',
            'semantic_similarity': 'Content flow analysis',
            'llm_reasoning': 'AI reasoning applied'
        }
        
        if result.method in method_description:
            reasons.append(method_description[result.method])
        
        return "; ".join(reasons) if reasons else "Standard document flow"
    
    def detect_missing_pages(self, page_contents: List[Dict]) -> List[int]:
        """Enhanced missing page detection"""
        missing_pages = []
        
        # Method 1: Page number gap analysis
        page_numbers = []
        total_pages_mentioned = None
        
        for page in page_contents:
            content = page.get('content', '').lower()
            
            # Look for "X of Y" patterns
            of_pattern = re.findall(r'(\d+)\s+of\s+(\d+)', content)
            if of_pattern:
                page_num, total = map(int, of_pattern[0])
                page_numbers.append(page_num)
                if total_pages_mentioned is None or total > total_pages_mentioned:
                    total_pages_mentioned = total
            
            # Look for simple page numbers
            page_pattern = re.findall(r'page\s+(\d+)', content)
            if page_pattern:
                page_numbers.extend(map(int, page_pattern))
        
        if page_numbers and total_pages_mentioned:
            page_numbers = list(set(page_numbers))  # Remove duplicates
            for i in range(1, total_pages_mentioned + 1):
                if i not in page_numbers:
                    missing_pages.append(i)
        
        return missing_pages
    
    def detect_duplicate_pages(self, page_contents: List[Dict]) -> List[int]:
        """Enhanced duplicate page detection"""
        duplicates = []
        
        if not self.embedding_model or len(page_contents) < 2:
            return duplicates
        
        try:
            texts = [page.get('content', '') for page in page_contents]
            
            # Skip empty pages
            valid_indices = [i for i, text in enumerate(texts) if text.strip()]
            if len(valid_indices) < 2:
                return duplicates
            
            valid_texts = [texts[i] for i in valid_indices]
            embeddings = self.embedding_model.encode(valid_texts)
            similarity_matrix = cosine_similarity(embeddings)
            
            # Find high-similarity pairs
            threshold = 0.95
            found_duplicates = set()
            
            for i in range(len(similarity_matrix)):
                for j in range(i + 1, len(similarity_matrix)):
                    if similarity_matrix[i][j] > threshold:
                        # Map back to original indices
                        orig_i = valid_indices[i]
                        orig_j = valid_indices[j]
                        
                        # Additional verification - check if content is actually similar
                        text1 = texts[orig_i].strip()
                        text2 = texts[orig_j].strip()
                        
                        if len(text1) > 50 and len(text2) > 50:  # Only flag substantial content
                            # Additional check: ensure they're not just similar section headers
                            # but actual duplicate content
                            word_overlap = self._calculate_word_overlap(text1, text2)
                            if word_overlap > 0.8:  # 80% word overlap
                                found_duplicates.add(orig_j)  # Keep the first occurrence
            
            duplicates = list(found_duplicates)
            
        except Exception as e:
            print(f"Duplicate detection failed: {e}")
        
        return duplicates
    
    def _calculate_word_overlap(self, text1: str, text2: str) -> float:
        """Calculate word-level overlap between two texts"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def get_strategy_performance(self) -> Dict[str, Any]:
        """Get performance statistics for debugging"""
        stats = {}
        for strategy_name, perf in self.performance_stats.items():
            if perf['attempts'] > 0:
                stats[strategy_name] = {
                    'attempts': perf['attempts'],
                    'successes': perf['successes'],
                    'success_rate': perf['successes'] / perf['attempts']
                }
        return stats
    
    def learn_from_feedback(self, original_order: List[int], corrected_order: List[int], document_type: str):
        """Learn from user feedback to improve future ordering"""
        # This is a placeholder for learning functionality
        # In a production system, you might store feedback and retrain models
        print(f"Feedback received for {document_type}: {original_order} -> {corrected_order}")
        
        # Update strategy performance based on feedback
        # This is simplified - in practice you'd want more sophisticated learning
        feedback_quality = self._evaluate_feedback_quality(original_order, corrected_order)
        
        # Log feedback for future analysis
        feedback_log = {
            'timestamp': datetime.now().isoformat(),
            'document_type': document_type,
            'original_order': original_order,
            'corrected_order': corrected_order,
            'quality_score': feedback_quality
        }
        
        # In production, you'd save this to a database
        print(f"Logged feedback with quality score: {feedback_quality}")
    
    def _evaluate_feedback_quality(self, original: List[int], corrected: List[int]) -> float:
        """Evaluate the quality of user feedback"""
        if len(original) != len(corrected):
            return 0.0
        
        if set(original) != set(corrected):
            return 0.0
        
        # Calculate how different the orders are
        differences = sum(1 for i, j in zip(original, corrected) if i != j)
        similarity = 1.0 - (differences / len(original))
        
        return similarity
# backend\app\services\ordering_strategies.py

from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Optional
import re
import json
import os
from pathlib import Path
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from datetime import datetime
from dateutil import parser
import google.generativeai as genai

from app.core.config import settings

class OrderingResult:
    def __init__(self, order: List[int], confidence: float, reasoning: List[str], 
                 method: str, classifications: Optional[List[Dict]] = None):
        self.order = order
        self.confidence = confidence
        self.reasoning = reasoning
        self.method = method
        self.classifications = classifications or []

class BaseOrderingStrategy(ABC):
    def __init__(self, threshold: float = 0.6):
        self.threshold = threshold
        self.name = self.__class__.__name__
    
    @abstractmethod
    def attempt_ordering(self, page_contents: List[Dict]) -> OrderingResult:
        pass
    
    def can_handle(self, page_contents: List[Dict]) -> bool:
        """Check if this strategy can handle the given pages"""
        return True

class PageNumberStrategy(BaseOrderingStrategy):
    def __init__(self):
        super().__init__(threshold=0.8)
        # Load patterns from config if available
        self.config = self._load_page_number_patterns()
    
    def _load_page_number_patterns(self) -> List[str]:
        """Load page number patterns from config file"""
        try:
            config_path = Path(__file__).parent.parent / "config" / "document_rules.json"
            with open(config_path, 'r') as f:
                config = json.load(f)
                return config.get("general_patterns", {}).get("page_number_patterns", [])
        except:
            # Fallback patterns if config is not available
            return [
                r'-(\d+)-',  # -7-, -20- pattern
                r'page\s+(\d+)',
                r'(\d+)\s+of\s+\d+',
                r'^\s*(\d+)\s*$',
                r'p\.?\s*(\d+)'
            ]
    
    def attempt_ordering(self, page_contents: List[Dict]) -> OrderingResult:
        page_numbers = {}
        pattern_matches = {}  # Track which pattern worked for each page
        
        print(f"\nTesting page number detection with {len(self.config)} patterns...")
        
        for i, page in enumerate(page_contents):
            content = page['content']
            
            # Try patterns from config first, then fallback patterns
            all_patterns = self.config + [
                r'-(\d+)-',           # IREDA specific: -7-, -20-
                r'—(\d+)—',           # Em dash variant
                r'–(\d+)–',           # En dash variant  
                r'- (\d+) -',         # With spaces
                r'^\s*-(\d+)-\s*$',   # Line with just page number
                r'page\s+(\d+)',      # Standard page notation
                r'(\d+)\s+of\s+\d+',  # X of Y format
                r'^\s*(\d+)\s*$',     # Just a number on its line
                r'p\.?\s*(\d+)',      # p.7 or p 7
            ]
            
            for pattern_idx, pattern in enumerate(all_patterns):
                try:
                    matches = re.findall(pattern, content, re.MULTILINE | re.IGNORECASE)
                    if matches:
                        for match in matches:
                            try:
                                page_num = int(match)
                                # Reasonable range check (allow for some missing pages)
                                if 1 <= page_num <= len(page_contents) * 2:
                                    if i not in page_numbers or page_num < page_numbers[i]:
                                        page_numbers[i] = page_num
                                        pattern_matches[i] = {
                                            'pattern': pattern,
                                            'match': match,
                                            'full_text_sample': content[:100]
                                        }
                                        print(f"Page {i+1}: Found page number {page_num} using pattern '{pattern}'")
                                        break
                            except ValueError:
                                continue
                    if i in page_numbers:
                        break  # Found a match, move to next page
                except re.error as e:
                    print(f"Invalid regex pattern '{pattern}': {e}")
                    continue
        
        print(f"Successfully detected page numbers for {len(page_numbers)}/{len(page_contents)} pages")
        
        # Analyze the quality of detection
        coverage = len(page_numbers) / len(page_contents)
        
        # Check for sequence continuity (good sign)
        if len(page_numbers) >= 2:
            detected_numbers = sorted(page_numbers.values())
            sequence_gaps = sum(1 for i in range(1, len(detected_numbers)) 
                              if detected_numbers[i] - detected_numbers[i-1] > 1)
            sequence_quality = 1.0 - (sequence_gaps / len(detected_numbers))
        else:
            sequence_quality = 0.5
        
        # Need good coverage for high confidence
        min_coverage = 0.6  # At least 60% of pages should have detectable numbers
        
        if coverage >= min_coverage:
            # Sort by detected page numbers
            sorted_pages = sorted(page_numbers.items(), key=lambda x: x[1])
            order = [item[0] for item in sorted_pages]
            
            # Add pages without detected numbers at the end (in original order)
            undetected_pages = [i for i in range(len(page_contents)) if i not in page_numbers]
            order.extend(sorted(undetected_pages))
            
            # Calculate confidence based on coverage and sequence quality
            base_confidence = min(coverage, 0.9)
            sequence_bonus = sequence_quality * 0.1
            final_confidence = min(0.95, base_confidence + sequence_bonus)
            
            reasoning = [
                f'Found page numbers in {len(page_numbers)} out of {len(page_contents)} pages',
                f'Coverage: {coverage:.1%}, Sequence quality: {sequence_quality:.1%}'
            ]
            
            # Add examples of successful detections
            for i, (page_idx, page_num) in enumerate(sorted_pages[:3]):
                if page_idx in pattern_matches:
                    pattern_info = pattern_matches[page_idx]
                    reasoning.append(f'Page {page_idx+1} → {page_num} (pattern: {pattern_info["pattern"]})')
            
            return OrderingResult(
                order=order,
                confidence=final_confidence,
                reasoning=reasoning,
                method="explicit_page_numbers"
            )
        
        return OrderingResult(
            order=list(range(len(page_contents))),
            confidence=0.2,
            reasoning=[f'Only found page numbers in {len(page_numbers)} pages (need {min_coverage:.0%} coverage)'],
            method="page_numbers_insufficient"
        )

class ConfigurableBusinessLogicStrategy(BaseOrderingStrategy):
    def __init__(self):
        super().__init__(threshold=0.6)
        self.document_config = self._load_document_config()
    
    def _load_document_config(self) -> Dict:
        """Load document configuration from JSON file"""
        config_path = Path(__file__).parent.parent / "config" / "document_rules.json"
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Warning: Could not load document config: {e}")
            return {"document_types": {}, "general_patterns": {}}
    
    def _detect_document_type(self, page_contents: List[Dict]) -> str:
        """Automatically detect document type based on content analysis"""
        combined_text = " ".join([page['content'].lower() for page in page_contents])
        
        doc_scores = {}
        for doc_type, config in self.document_config.get("document_types", {}).items():
            score = 0
            total_possible_score = 0
            
            for section in config["sections"]:
                section_weight = section.get("weight", 1.0)
                total_possible_score += section_weight
                
                # Check indicators
                for indicator in section["indicators"]:
                    if indicator.lower() in combined_text:
                        score += section_weight * 0.5
                
                # Check required_any patterns
                for required in section.get("required_any", []):
                    if required.lower() in combined_text:
                        score += section_weight * 0.3
                
                # Check boost patterns (regex)
                for pattern in section.get("boost_patterns", []):
                    try:
                        if re.search(pattern.lower(), combined_text):
                            score += section_weight * 1.0
                    except re.error:
                        continue
            
            # Normalize score
            if total_possible_score > 0:
                normalized_score = score / total_possible_score
                doc_scores[doc_type] = normalized_score
                print(f"Document type '{doc_type}' scored: {normalized_score:.3f}")
        
        if doc_scores:
            best_match = max(doc_scores.items(), key=lambda x: x[1])
            if best_match[1] > 0.3:  # Minimum threshold
                print(f"Detected document type: {best_match[0]} (confidence: {best_match[1]:.3f})")
                return best_match[0]
        
        print("No document type matched, using 'loan_agreement' as fallback")
        return "loan_agreement"  # Fallback for loan documents
    
    def attempt_ordering(self, page_contents: List[Dict]) -> OrderingResult:
        """Order pages using configurable business logic from JSON"""
        
        # Detect document type dynamically
        doc_type = self._detect_document_type(page_contents)
        
        if doc_type not in self.document_config.get("document_types", {}):
            return OrderingResult(
                order=list(range(len(page_contents))),
                confidence=0.1,
                reasoning=[f'Unknown document type: {doc_type}'],
                method="business_logic_no_config"
            )
        
        doc_config = self.document_config["document_types"][doc_type]
        page_classifications = []
        
        print(f"\nClassifying pages using {doc_config['name']} rules...")
        
        for i, page in enumerate(page_contents):
            content_lower = page['content'].lower()
            best_match = None
            best_score = 0
            matched_indicators = []
            
            for section in doc_config["sections"]:
                score = 0
                section_matched_indicators = []
                
                # Check basic indicators
                for indicator in section["indicators"]:
                    if indicator.lower() in content_lower:
                        score += 1 * section.get("weight", 1.0)
                        section_matched_indicators.append(indicator)
                
                # Check required_any patterns
                for required in section.get("required_any", []):
                    if required.lower() in content_lower:
                        score += 2 * section.get("weight", 1.0)
                        section_matched_indicators.append(f"required: {required}")
                
                # Check boost patterns (regex patterns for stronger matches)
                for pattern in section.get("boost_patterns", []):
                    try:
                        if re.search(pattern.lower(), content_lower):
                            score += 3 * section.get("weight", 1.0)
                            section_matched_indicators.append(f"pattern: {pattern}")
                    except re.error:
                        print(f"Invalid regex pattern: {pattern}")
                        continue
                
                # Update best match if this section scores higher
                if score > best_score:
                    best_score = score
                    best_match = section
                    matched_indicators = section_matched_indicators
            
            # Store classification results
            classification = {
                'page_index': i,
                'section': best_match,
                'priority': best_match['priority'] if best_match else 99,
                'confidence': min(1.0, best_score / 5.0) if best_match else 0.0,
                'score': best_score,
                'matched_indicators': matched_indicators
            }
            
            page_classifications.append(classification)
            
            section_name = best_match['name'] if best_match else 'unknown'
            print(f"Page {i+1}: {section_name} (score: {best_score:.1f}, priority: {classification['priority']})")
            if matched_indicators:
                print(f"  Matched: {matched_indicators[:3]}")  # Show first 3 matches
        
        # Sort by priority, then by confidence, then by original position
        sorted_classifications = sorted(
            page_classifications, 
            key=lambda x: (x['priority'], -x['confidence'], x['page_index'])
        )
        
        new_order = [item['page_index'] for item in sorted_classifications]
        
        # Calculate overall confidence based on classification success
        classified_pages = sum(1 for item in page_classifications if item['section'] is not None)
        classification_rate = classified_pages / len(page_contents)
        
        # Boost confidence if we have good coverage and high individual scores
        avg_individual_confidence = sum(item['confidence'] for item in page_classifications) / len(page_classifications)
        overall_confidence = min(0.95, (classification_rate * 0.7) + (avg_individual_confidence * 0.3))
        
        # Generate reasoning
        reasoning_parts = [f"Document classified as: {doc_config['name']}"]
        reasoning_parts.append(f"Successfully classified {classified_pages}/{len(page_contents)} pages")
        
        # Add details for top classifications
        for item in sorted_classifications[:3]:
            if item['section']:
                section_name = item['section']['name'].replace('_', ' ').title()
                reasoning_parts.append(f"Page {item['page_index'] + 1}: {section_name}")
        
        return OrderingResult(
            order=new_order,
            confidence=overall_confidence,
            reasoning=reasoning_parts,
            method="configurable_business_logic",
            classifications=sorted_classifications
        )

class StructuralPatternStrategy(BaseOrderingStrategy):
    def __init__(self):
        super().__init__(threshold=0.5)
    
    def attempt_ordering(self, page_contents: List[Dict]) -> OrderingResult:
        """Detect structural patterns like headers, footers, and layout"""
        structural_scores = []
        
        for i, page in enumerate(page_contents):
            content = page['content']
            lines = content.split('\n')
            
            score_factors = {
                'header_like': 0,
                'footer_like': 0,
                'content_density': len(content.split()) / max(1, len(lines)),
                'has_title': 0,
                'has_signature_space': 0
            }
            
            # Check for header-like patterns
            if lines:
                first_lines = lines[:3]
                for line in first_lines:
                    if re.match(r'^[A-Z][A-Z\s]+$', line.strip()):
                        score_factors['header_like'] += 1
                    if any(word in line.lower() for word in ['application', 'report', 'document']):
                        score_factors['has_title'] = 1
            
            # Check for footer-like patterns
            if lines:
                last_lines = lines[-3:]
                for line in last_lines:
                    if re.search(r'page\s+\d+|signature|date.*signed', line.lower()):
                        score_factors['footer_like'] += 1
                    if re.search(r'_+\s*(date|sign)', line.lower()):
                        score_factors['has_signature_space'] = 1
            
            structural_scores.append({
                'page_index': i,
                'scores': score_factors,
                'priority': self._calculate_structural_priority(score_factors)
            })
        
        # Sort by structural priority
        structural_scores.sort(key=lambda x: x['priority'])
        order = [item['page_index'] for item in structural_scores]
        
        # Calculate confidence based on how distinct the structural patterns are
        priorities = [item['priority'] for item in structural_scores]
        confidence = min(0.8, (max(priorities) - min(priorities)) / max(1, max(priorities)))
        
        return OrderingResult(
            order=order,
            confidence=confidence,
            reasoning=['Ordered by structural patterns (headers, footers, layout)'],
            method="structural_patterns"
        )
    
    def _calculate_structural_priority(self, score_factors: Dict) -> float:
        """Calculate priority based on structural indicators"""
        priority = 50  # Base priority
        
        # Headers should come first
        if score_factors['has_title']:
            priority -= 30
        if score_factors['header_like'] > 0:
            priority -= 20
        
        # Signatures should come last
        if score_factors['has_signature_space']:
            priority += 40
        if score_factors['footer_like'] > 0:
            priority += 20
        
        # Content density affects ordering
        if score_factors['content_density'] > 20:  # Rich content
            priority -= 5
        elif score_factors['content_density'] < 5:  # Sparse content (maybe signature page)
            priority += 10
        
        return priority

class SemanticSimilarityStrategy(BaseOrderingStrategy):
    def __init__(self, embedding_model):
        super().__init__(threshold=0.6)
        self.embedding_model = embedding_model
    
    def attempt_ordering(self, page_contents: List[Dict]) -> OrderingResult:
        if not self.embedding_model:
            return OrderingResult(
                order=list(range(len(page_contents))),
                confidence=0.1,
                reasoning=['No embedding model available'],
                method="semantic_no_model"
            )
        
        try:
            texts = [page['content'] for page in page_contents]
            embeddings = self.embedding_model.encode(texts)
            similarity_matrix = cosine_similarity(embeddings)
            
            # Greedy ordering based on similarity
            order = []
            used = set()
            
            # Start with page that has highest average similarity
            avg_similarities = np.mean(similarity_matrix, axis=1)
            current = np.argmax(avg_similarities)
            order.append(current)
            used.add(current)
            
            # Add most similar unused pages
            while len(order) < len(page_contents):
                best_similarity = -1
                best_page = -1
                
                for candidate in range(len(page_contents)):
                    if candidate in used:
                        continue
                    
                    sim = similarity_matrix[current][candidate]
                    if sim > best_similarity:
                        best_similarity = sim
                        best_page = candidate
                
                if best_page != -1:
                    order.append(best_page)
                    used.add(best_page)
                    current = best_page
                else:
                    # Add remaining pages
                    for i in range(len(page_contents)):
                        if i not in used:
                            order.append(i)
                            used.add(i)
                    break
            
            # Calculate confidence
            consecutive_similarities = []
            for i in range(len(order) - 1):
                sim = similarity_matrix[order[i]][order[i + 1]]
                consecutive_similarities.append(sim)
            
            avg_consecutive_sim = np.mean(consecutive_similarities) if consecutive_similarities else 0
            confidence = min(0.9, max(0.3, avg_consecutive_sim))
            
            return OrderingResult(
                order=order,
                confidence=confidence,
                reasoning=[f'Semantic flow analysis with {avg_consecutive_sim:.2f} average similarity'],
                method="semantic_similarity"
            )
            
        except Exception as e:
            return OrderingResult(
                order=list(range(len(page_contents))),
                confidence=0.1,
                reasoning=[f'Semantic analysis failed: {str(e)}'],
                method="semantic_failed"
            )

class DateSequenceStrategy(BaseOrderingStrategy):
    def __init__(self):
        super().__init__(threshold=0.7)
    
    def attempt_ordering(self, page_contents: List[Dict]) -> OrderingResult:
        page_dates = {}
        
        for i, page in enumerate(page_contents):
            content = page['content']
            
            date_patterns = [
                r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',
                r'\d{4}[/-]\d{1,2}[/-]\d{1,2}',
                r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},?\s+\d{4}',
            ]
            
            dates_found = []
            for pattern in date_patterns:
                matches = re.findall(pattern, content)
                for match in matches:
                    try:
                        parsed_date = parser.parse(match)
                        dates_found.append(parsed_date)
                    except:
                        continue
            
            if dates_found:
                page_dates[i] = min(dates_found)
        
        if len(page_dates) >= len(page_contents) * 0.5:
            sorted_pages = sorted(page_dates.items(), key=lambda x: x[1])
            order = [item[0] for item in sorted_pages]
            
            # Add pages without dates
            for i in range(len(page_contents)):
                if i not in order:
                    order.append(i)
            
            return OrderingResult(
                order=order,
                confidence=0.7,
                reasoning=[f'Found chronological sequence in {len(page_dates)} pages'],
                method="date_sequence"
            )
        
        return OrderingResult(
            order=list(range(len(page_contents))),
            confidence=0.2,
            reasoning=['No clear date sequence found'],
            method="date_sequence_failed"
        )

class LLMReasoningStrategy(BaseOrderingStrategy):
    def __init__(self, gemini_client=None):
        super().__init__(threshold=0.5)
        self.gemini_client = gemini_client
    
    def attempt_ordering(self, page_contents: List[Dict]) -> OrderingResult:
        if not self.gemini_client:
            return OrderingResult(
                order=list(range(len(page_contents))),
                confidence=0.1,
                reasoning=['No AI reasoning available'],
                method="llm_no_client"
            )
        
        try:
            page_summaries = []
            for i, page in enumerate(page_contents):
                content = page['content']
                summary = content[:500] + "..." if len(content) > 500 else content
                page_summaries.append(f"Page {i}: {summary}")
            
            prompt = f"""
            Analyze these {len(page_summaries)} pages from a document and determine the correct logical order.
            
            DOCUMENT ORDERING PRINCIPLES:
            1. Title/Header pages typically come first
            2. Personal information usually follows headers
            3. Supporting documents follow main content
            4. Legal disclosures often appear before signatures
            5. Signature pages typically come last
            6. Multi-page sections should be kept together
            
            Pages:
            {chr(10).join(page_summaries)}
            
            Return the optimal order as a comma-separated list of page indices (0-based).
            Also provide a confidence score (0.0-1.0) and brief reasoning.
            
            Format:
            ORDER: 0,2,1,3
            CONFIDENCE: 0.85
            REASONING: Page 0 has header content, page 2 contains main information, etc.
            """
            
            response = self.gemini_client.generate_content(prompt)
            response_text = response.text
            
            # Parse response
            order_match = re.search(r'ORDER:\s*([0-9,\s]+)', response_text)
            confidence_match = re.search(r'CONFIDENCE:\s*([0-9.]+)', response_text)
            reasoning_match = re.search(r'REASONING:\s*(.+)', response_text, re.DOTALL)
            
            if order_match:
                order_str = order_match.group(1).strip()
                order = [int(x.strip()) for x in order_str.split(',')]
                
                if len(order) == len(page_contents) and all(0 <= x < len(page_contents) for x in order):
                    confidence = float(confidence_match.group(1)) if confidence_match else 0.6
                    reasoning = [reasoning_match.group(1).strip()] if reasoning_match else ['AI reasoning applied']
                    
                    return OrderingResult(
                        order=order,
                        confidence=min(0.9, max(0.4, confidence)),
                        reasoning=reasoning,
                        method="llm_reasoning"
                    )
            
            return OrderingResult(
                order=list(range(len(page_contents))),
                confidence=0.3,
                reasoning=['AI analysis inconclusive'],
                method="llm_reasoning_failed"
            )
            
        except Exception as e:
            return OrderingResult(
                order=list(range(len(page_contents))),
                confidence=0.1,
                reasoning=[f'AI reasoning failed: {str(e)}'],
                method="llm_reasoning_error"
            )
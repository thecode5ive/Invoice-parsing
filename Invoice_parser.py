import os
import torch
from PIL import Image
try:
    import fitz  # PyMuPDF
except ImportError:
    print("❌ PyMuPDF not installed. Installing...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'PyMuPDF'])
    import fitz

import openpyxl
from transformers import DonutProcessor, VisionEncoderDecoderModel
from typing import Dict, List, Optional, Tuple
import re
import json
from dataclasses import dataclass
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

@dataclass
class InvoiceData:
    """Structure for invoice data"""
    party_name: str = "Not Found"
    invoice_number: str = "Not Found"
    invoice_date: str = "Not Found"
    invoice_amount: str = "Not Found"

class DonutInvoiceParser:
    def __init__(self):
        """
        Initialize Donut model for invoice parsing
        Loads model directly from HuggingFace
        """
        print("🍩 Initializing Enhanced Donut Invoice Parser...")
        print("📥 Loading model from HuggingFace (this may take a minute)...")

        # Check if CUDA is available in Colab
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🖥️ Using device: {self.device}")

        try:
            # Load Donut model and processor
            self.processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa")
            self.model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa")

            # Move model to GPU if available
            self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode

            print("✅ Donut model loaded successfully!")
            print(f"📊 Model size: {sum(p.numel() for p in self.model.parameters()) / 1e6:.1f}M parameters")

            # Invalid company name keywords (things that are NOT company names)
            self.invalid_keywords = {
                'invoice', 'bill', 'receipt', 'tax', 'sales', 'purchase', 'date',
                'number', 'no', 'amount', 'total', 'subtotal', 'grand', 'gst',
                'item', 'description', 'quantity', 'rate', 'price', 'customer',
                'vendor', 'buyer', 'seller', 'from', 'to', 'attention', 'subject',
                'page', 'original', 'copy', 'duplicate', 'triplicate', 'challan',
                'delivery', 'payment', 'terms', 'conditions', 'signature', 'stamp',
                'authorized', 'for', 'the', 'and', 'or', 'of', 'in', 'on', 'at'
            }

        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise

    def pdf_to_images(self, pdf_path: str, dpi: int = 300) -> List[Image.Image]:
        """Convert PDF pages to PIL images with higher DPI for better OCR"""
        try:
            # Open PDF document
            try:
                pdf_document = fitz.open(pdf_path)
            except AttributeError:
                # If fitz.open doesn't work, try importing PyMuPDF directly
                import pymupdf
                pdf_document = pymupdf.open(pdf_path)

            images = []

            for page_num in range(len(pdf_document)):
                page = pdf_document[page_num]
                mat = fitz.Matrix(dpi / 72, dpi / 72)
                pix = page.get_pixmap(matrix=mat)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

                # Convert to RGB if necessary
                if img.mode != 'RGB':
                    img = img.convert('RGB')

                images.append(img)

            pdf_document.close()
            return images

        except Exception as e:
            print(f"❌ Error converting PDF: {e}")
            print("\n💡 Troubleshooting steps:")
            print("   1. Install PyMuPDF: pip install PyMuPDF")
            print("   2. Or try: pip install --upgrade PyMuPDF")
            print("   3. Check if there's a naming conflict with 'fitz' module")
            raise

    def _clean_donut_output(self, text: str) -> str:
        """Clean Donut model output by removing all formatting tags"""
        if not text:
            return ""

        # Remove all Donut special tokens and tags
        text = re.sub(r'<s_docvqa>', '', text)
        text = re.sub(r'<s_question>.*?</s_question>', '', text)
        text = re.sub(r'<s_answer>', '', text)
        text = re.sub(r'</s_answer>', '', text)
        text = re.sub(r'</s>', '', text)
        text = re.sub(r'<s>', '', text)

        # Remove any remaining angle bracket tags
        text = re.sub(r'<[^>]+>', '', text)

        # Clean up whitespace
        text = " ".join(text.split())
        text = text.strip()

        return text

    def query_donut(self, image: Image.Image, question: str) -> str:
        """Query Donut model with an image and question"""
        try:
            # Prepare the prompt in Donut format
            task_prompt = f"<s_docvqa><s_question>{question}</s_question><s_answer>"

            # Process image and prompt
            pixel_values = self.processor(image, return_tensors="pt").pixel_values
            decoder_input_ids = self.processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids

            # Move to device
            pixel_values = pixel_values.to(self.device)
            decoder_input_ids = decoder_input_ids.to(self.device)

            best_answer = ""

            # Try with different beam sizes for better results
            for num_beams in [3, 1]:
                with torch.no_grad():
                    outputs = self.model.generate(
                        pixel_values,
                        decoder_input_ids=decoder_input_ids,
                        max_length=self.model.decoder.config.max_position_embeddings,
                        early_stopping=True,
                        pad_token_id=self.processor.tokenizer.pad_token_id,
                        eos_token_id=self.processor.tokenizer.eos_token_id,
                        use_cache=True,
                        num_beams=num_beams,
                        bad_words_ids=[[self.processor.tokenizer.unk_token_id]],
                        return_dict_in_generate=True,
                    )

                # Decode and clean the answer
                answer = self.processor.batch_decode(outputs.sequences)[0]
                answer = self._clean_donut_output(answer)

                if len(answer) > len(best_answer) and answer.lower() not in ["none", "no", "n/a", ""]:
                    best_answer = answer
                    break

            return best_answer

        except Exception as e:
            print(f"⚠️ Error querying Donut: {e}")
            return ""

    def _is_valid_company_name(self, text: str) -> Tuple[bool, float]:
        """
        Validate if text is likely a company name
        Returns: (is_valid, confidence_score)
        """
        if not text or len(text) < 2:
            return False, 0.0

        text_lower = text.lower()
        confidence = 100.0

        # Rule 1: Must not be too short or too long
        if len(text) < 3:
            return False, 0.0
        if len(text) > 100:
            confidence -= 30

        # Rule 2: Check for invalid keywords (invoice-related terms)
        words = set(text_lower.split())
        invalid_count = len(words.intersection(self.invalid_keywords))
        if invalid_count > 0:
            confidence -= (invalid_count * 20)
            # If it's ONLY invoice keywords, reject it
            if invalid_count == len(words):
                return False, 0.0

        # Rule 3: Company names usually have capital letters or are short abbreviations
        if text.isupper() and len(text) > 2:
            confidence += 20  # All caps like "A.R SONS" or "LEX ENTERPRISES"
        elif text[0].isupper():
            confidence += 10  # Proper capitalization

        # Rule 4: Check for company suffixes (positive indicators)
        company_suffixes = [
            'ltd', 'limited', 'pvt', 'private', 'llc', 'inc', 'incorporated',
            'corp', 'corporation', 'co', 'company', 'enterprises', 'industries',
            'mills', 'textile', 'group', 'international', 'trading', 'sons'
        ]
        for suffix in company_suffixes:
            if suffix in text_lower:
                confidence += 15
                break

        # Rule 5: Should have at least one letter
        if not re.search(r'[a-zA-Z]', text):
            return False, 0.0

        # Rule 6: Penalize if it's too generic
        generic_terms = ['company', 'business', 'store', 'shop']
        if text_lower in generic_terms:
            confidence -= 40

        # Rule 7: Check for numbers/special chars (company names can have them)
        if re.search(r'[\d\.]', text):
            # Numbers are OK in company names like "A.R SONS"
            confidence += 5

        # Rule 8: Must not be just numbers or special characters
        if re.match(r'^[\d\s\.\-\/]+$', text):
            return False, 0.0

        # Final decision
        is_valid = confidence >= 40
        return is_valid, max(0.0, min(100.0, confidence))

    def _extract_company_name_multi_strategy(self, image: Image.Image) -> str:
        """
        Extract company name using multiple strategies and validation
        """
        print("  🔍 Strategy 1: Direct company name query...")
        candidates = []

        # Strategy 1: Ask directly for company name
        direct_questions = [
            "What is the name of the company at the top of this document?",
            "What is the vendor name or seller name?",
            "What company issued this invoice?",
            "What is the business name on this invoice?",
        ]

        for question in direct_questions[:2]:
            answer = self.query_donut(image, question)
            if answer:
                is_valid, confidence = self._is_valid_company_name(answer)
                if is_valid:
                    candidates.append((answer, confidence, "Direct Query"))
                    print(f"    • Found: '{answer}' (confidence: {confidence:.1f}%)")

        # Strategy 2: Ask for text at the very top
        print("  🔍 Strategy 2: Top text extraction...")
        top_question = "What is the first text or heading at the top of this page?"
        top_text = self.query_donut(image, top_question)

        if top_text:
            # Split by lines and check each
            lines = top_text.split('\n')
            for line in lines[:3]:  # Check first 3 lines
                line = line.strip()
                is_valid, confidence = self._is_valid_company_name(line)
                if is_valid:
                    candidates.append((line, confidence, "Top Text"))
                    print(f"    • Found: '{line}' (confidence: {confidence:.1f}%)")

        # Strategy 3: Ask for largest/bold text
        print("  🔍 Strategy 3: Bold/prominent text...")
        bold_questions = [
            "What is the largest or most prominent text on this document?",
            "What text appears in the largest font?",
        ]

        for question in bold_questions[:1]:
            answer = self.query_donut(image, question)
            if answer:
                is_valid, confidence = self._is_valid_company_name(answer)
                if is_valid:
                    candidates.append((answer, confidence, "Bold Text"))
                    print(f"    • Found: '{answer}' (confidence: {confidence:.1f}%)")

        # Strategy 4: Extract from PDF metadata if available
        print("  🔍 Strategy 4: Header region analysis...")
        header_question = "What company name appears in the header or letterhead?"
        header_text = self.query_donut(image, header_question)

        if header_text:
            is_valid, confidence = self._is_valid_company_name(header_text)
            if is_valid:
                candidates.append((header_text, confidence, "Header"))
                print(f"    • Found: '{header_text}' (confidence: {confidence:.1f}%)")

        # Strategy 5: Look for text before "invoice" keyword
        print("  🔍 Strategy 5: Context-based extraction...")
        context_question = "What text appears above or before the word 'invoice'?"
        context_text = self.query_donut(image, context_question)

        if context_text:
            # Clean up and validate
            context_text = re.sub(r'\b(sales|tax|invoice)\b', '', context_text, flags=re.IGNORECASE).strip()
            if context_text:
                is_valid, confidence = self._is_valid_company_name(context_text)
                if is_valid:
                    candidates.append((context_text, confidence, "Context"))
                    print(f"    • Found: '{context_text}' (confidence: {confidence:.1f}%)")

        # Analyze and select best candidate
        if not candidates:
            print("    ⚠️ No valid company names found")
            return "Not Found"

        # Sort by confidence score
        candidates.sort(key=lambda x: x[1], reverse=True)

        print(f"\n  📊 Found {len(candidates)} candidate(s)")
        print("  🏆 Best candidates:")
        for idx, (name, conf, source) in enumerate(candidates[:3], 1):
            print(f"     {idx}. '{name}' - {conf:.1f}% ({source})")

        # Return the highest confidence candidate
        best_candidate = candidates[0]
        final_name = self._clean_company_name(best_candidate[0])

        print(f"\n  ✅ Selected: '{final_name}' (confidence: {best_candidate[1]:.1f}%)")

        return final_name

    def _extract_grand_total_multipage(self, images: List[Image.Image]) -> str:
        """
        Extract grand total from multi-page invoice by checking last pages first
        """
        print("\n  💰 Multi-page total extraction...")
        amount_candidates = []

        # Check last 2 pages first (most invoices have totals at the end)
        pages_to_check = []
        if len(images) > 1:
            pages_to_check.extend([(len(images)-1, "Last Page"), (len(images)-2, "Second Last")])
        pages_to_check.append((0, "First Page"))  # Also check first page

        for page_idx, page_label in pages_to_check:
            if page_idx < 0 or page_idx >= len(images):
                continue

            print(f"    📄 Checking {page_label} (Page {page_idx + 1}/{len(images)})...")
            image = images[page_idx]

            # Questions targeting grand total
            total_questions = [
                "What is the grand total amount?",
                "What is the final total amount at the bottom?",
                "What is the total amount payable?",
                "Find the sum total or net amount",
                "What is the largest amount on this page?",
            ]

            for question in total_questions:
                answer = self.query_donut(image, question)
                if answer:
                    # Extract all numbers from the answer
                    extracted_amounts = self._extract_all_amounts(answer)
                    for amount_str, numeric_value in extracted_amounts:
                        # Only consider amounts above a threshold (likely to be totals, not line items)
                        if numeric_value > 1000:  # Adjust threshold as needed
                            amount_candidates.append((amount_str, numeric_value, page_idx + 1, question))
                            print(f"      • Found: {amount_str} (numeric: {numeric_value:,.2f})")

        if not amount_candidates:
            print("    ⚠️ No significant amounts found")
            return "Not Found"

        # Sort by numeric value (largest first)
        amount_candidates.sort(key=lambda x: x[1], reverse=True)

        print(f"\n  📊 Found {len(amount_candidates)} amount candidate(s)")
        print("  🏆 Top candidates:")
        for idx, (amount, value, page, question) in enumerate(amount_candidates[:5], 1):
            print(f"     {idx}. {amount} ({value:,.2f}) - Page {page}")

        # Return the largest amount
        best_amount = amount_candidates[0]
        print(f"\n  ✅ Selected Grand Total: {best_amount[0]} from Page {best_amount[2]}")

        return best_amount[0]

    def _extract_all_amounts(self, text: str) -> List[Tuple[str, float]]:
        """Extract all monetary amounts from text and return with numeric values"""
        if not text:
            return []

        text = self._clean_donut_output(text)
        text = re.sub(r'(Rs\.?|PKR|Rupees|USD|\$|€|Paisa)', '', text, flags=re.IGNORECASE)

        amounts = []

        # Patterns for different number formats
        patterns = [
            r'[\d,]+\.\d{2}',      # With decimals: 6,308,772.00
            r'[\d,]{4,}',          # Large numbers with commas: 6,308,772
            r'\d{4,}',             # Large numbers without commas: 6308772
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                num_str = match.replace(',', '')
                try:
                    numeric_value = float(num_str)
                    amounts.append((match, numeric_value))
                except:
                    continue

        # Sort by value and remove duplicates
        amounts = list(set(amounts))
        amounts.sort(key=lambda x: x[1], reverse=True)

        return amounts

    def extract_invoice_fields_enhanced(self, images: List[Image.Image]) -> InvoiceData:
        """Enhanced extraction with multi-page support"""
        invoice_data = InvoiceData()

        # Always use first page for company name, invoice number, and date
        first_page = images[0]

        # Extract party name with validation
        print("\n  📝 Extracting Party Name (with validation)...")
        invoice_data.party_name = self._extract_company_name_multi_strategy(first_page)

        # Extract invoice number with multiple strategies
        print("\n  📝 Extracting Invoice Number...")
        invoice_data.invoice_number = self._extract_invoice_number_enhanced(first_page)

        # Extract invoice date with multiple strategies
        print("\n  📝 Extracting Invoice Date...")
        invoice_data.invoice_date = self._extract_invoice_date_enhanced(first_page)

        # Extract total amount from ALL pages (multi-page support)
        print("\n  📝 Extracting Invoice Amount (Multi-page)...")
        invoice_data.invoice_amount = self._extract_grand_total_multipage(images)

        return invoice_data

    def _clean_company_name(self, text: str) -> str:
        """Clean and format company name"""
        if not text or len(text) < 2:
            return "Not Found"

        text = self._clean_donut_output(text)

        # Remove common invoice-related words
        text = re.sub(r'\b(invoice|bill|receipt|tax|sales|date|number|no|#)\b', '', text, flags=re.IGNORECASE)

        # Clean up whitespace
        text = " ".join(text.split())
        text = text.strip()

        # Proper capitalization
        if text and len(text) > 2:
            # If all caps, keep it (like "A.R SONS")
            if text.isupper():
                return text
            # Otherwise, title case
            text = ' '.join(word.capitalize() for word in text.split())
            return text

        return "Not Found"

    def _extract_invoice_number_enhanced(self, image: Image.Image) -> str:
        """Enhanced invoice number extraction with multiple strategies"""
        candidates = []

        print("  🔍 Multiple strategies for invoice number...")

        # Strategy 1: Direct questions
        questions = [
            "What is the invoice number or invoice no?",
            "Find the reference number or bill number",
            "What text appears next to 'Invoice No' or 'Invoice number'?",
            "What is the document number in the top right corner?",
            "Find the alphanumeric code that looks like an invoice reference",
        ]

        for question in questions:
            answer = self.query_donut(image, question)
            if answer:
                cleaned = self._clean_invoice_number(answer)
                if cleaned != "Not Found" and len(cleaned) > 3:
                    # Validate it looks like an invoice number
                    if re.search(r'[A-Z]{2,}[\-\/]\d{4}[\-\/]\d{4}', cleaned, re.IGNORECASE):
                        # Format like TPL-2324-0094
                        candidates.append((cleaned, 100, "Pattern Match"))
                        print(f"    • Found: '{cleaned}' (high confidence)")
                    elif re.search(r'[A-Z0-9\-\/]{5,}', cleaned):
                        # Generic alphanumeric pattern
                        candidates.append((cleaned, 70, "Generic"))
                        print(f"    • Found: '{cleaned}' (medium confidence)")

        # Strategy 2: Look for common invoice number patterns in raw text
        raw_questions = [
            "What text is in the top right section of the document?",
            "List all numbers and codes visible in the header area",
        ]

        for question in raw_questions:
            answer = self.query_donut(image, question)
            if answer:
                # Look for patterns like: TPL-2324-0094, SPL-2324-0094
                pattern_matches = re.findall(r'[A-Z]{2,}[\-\/]\d{4}[\-\/]\d{4}', answer, re.IGNORECASE)
                for match in pattern_matches:
                    candidates.append((match, 95, "Pattern Extract"))
                    print(f"    • Extracted: '{match}' (pattern)")

        if not candidates:
            print("    ⚠️ No invoice number found")
            return "Not Found"

        # Sort by confidence
        candidates.sort(key=lambda x: x[1], reverse=True)

        print(f"  ✅ Selected: '{candidates[0][0]}' (confidence: {candidates[0][1]}%)")
        return candidates[0][0]

    def _extract_invoice_date_enhanced(self, image: Image.Image) -> str:
        """Enhanced invoice date extraction with multiple strategies"""
        candidates = []

        print("  🔍 Multiple strategies for invoice date...")

        # Strategy 1: Direct date questions
        questions = [
            "What is the invoice date or date?",
            "What date appears next to 'Date' or 'Invoice Date'?",
            "Find the date in DD-MM-YY or DD-MM-YYYY format",
            "What is the date in the top right corner?",
            "When was this invoice issued?",
        ]

        for question in questions:
            answer = self.query_donut(image, question)
            if answer:
                cleaned = self._clean_date(answer)
                if cleaned != "Not Found":
                    # Validate it looks like a date
                    confidence = self._validate_date_format(cleaned)
                    if confidence > 50:
                        candidates.append((cleaned, confidence, "Direct Query"))
                        print(f"    • Found: '{cleaned}' (confidence: {confidence}%)")

        # Strategy 2: Look for date patterns in raw text
        raw_questions = [
            "What text appears in the right side header area?",
            "List all dates visible on this page",
        ]

        for question in raw_questions:
            answer = self.query_donut(image, question)
            if answer:
                # Extract dates using patterns
                date_patterns = [
                    r'\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}',  # DD-MM-YY or DD-MM-YYYY
                    r'\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{2,4}',
                ]

                for pattern in date_patterns:
                    matches = re.findall(pattern, answer, re.IGNORECASE)
                    for match in matches:
                        confidence = self._validate_date_format(match)
                        if confidence > 50:
                            candidates.append((match, confidence, "Pattern Extract"))
                            print(f"    • Extracted: '{match}' (pattern)")

        if not candidates:
            print("    ⚠️ No date found")
            return "Not Found"

        # Sort by confidence
        candidates.sort(key=lambda x: x[1], reverse=True)

        print(f"  ✅ Selected: '{candidates[0][0]}' (confidence: {candidates[0][1]}%)")
        return candidates[0][0]

    def _validate_date_format(self, date_str: str) -> int:
        """Validate if a string looks like a date and return confidence score"""
        if not date_str:
            return 0

        confidence = 60  # Base confidence

        # Check for common date patterns
        if re.match(r'\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}', date_str):
            confidence += 30  # DD-MM-YY format

        if re.search(r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)', date_str, re.IGNORECASE):
            confidence += 20  # Month name

        # Check if numbers are in reasonable ranges
        parts = re.findall(r'\d+', date_str)
        if len(parts) >= 2:
            try:
                first_num = int(parts[0])
                second_num = int(parts[1])

                # Day should be 1-31
                if 1 <= first_num <= 31:
                    confidence += 10

                # Month should be 1-12
                if 1 <= second_num <= 12:
                    confidence += 10
            except:
                pass

        return min(confidence, 100)

    def _clean_invoice_number(self, text: str) -> str:
        """Clean and extract invoice number with pattern recognition"""
        if not text or len(text) < 1:
            return "Not Found"

        text = self._clean_donut_output(text)

        # First, try to find specific patterns like TPL-2324-0094
        pattern_matches = re.findall(r'[A-Z]{2,}[\-\/]\d{4}[\-\/]\d{4}', text, re.IGNORECASE)
        if pattern_matches:
            return pattern_matches[0]

        # Remove common prefixes
        text = re.sub(r'^(Invoice|Inv|Bill|No|Number|#|:|[\s])+', '', text, flags=re.IGNORECASE)

        # Look for alphanumeric codes
        matches = re.findall(r'[A-Z0-9][\w\-\/]+', text, re.IGNORECASE)
        if matches:
            for match in matches:
                # Prefer longer matches (more likely to be complete invoice numbers)
                if len(match) > 5:
                    return match

        text = " ".join(text.split()).strip()
        return text if len(text) > 0 else "Not Found"

    def _clean_date(self, text: str) -> str:
        """Clean and extract date"""
        if not text or len(text) < 3:
            return "Not Found"

        text = self._clean_donut_output(text)

        date_patterns = [
            r'\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}',
            r'\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{2,4}',
            r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{2,4}',
            r'\d{4}[-/]\d{1,2}[-/]\d{1,2}',
        ]

        for pattern in date_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(0).strip()

        text = " ".join(text.split()).strip()
        if re.search(r'\d', text):
            return text

        return "Not Found"

    def process_invoice(self, pdf_path: str, output_excel: str = "invoices_validated.xlsx") -> Dict:
        """Main processing function with validated extraction"""
        if not os.path.exists(pdf_path):
            print(f"❌ PDF file not found: {pdf_path}")
            return None

        print(f"\n{'='*60}")
        print(f"📄 Processing: {os.path.basename(pdf_path)}")
        print(f"{'='*60}")

        try:
            print("🖼️ Converting PDF to images...")
            images = self.pdf_to_images(pdf_path, dpi=300)
            print(f"  ✓ Converted {len(images)} page(s)")

            print("\n🍩 Using Donut with multi-page validation...")
            invoice_data = self.extract_invoice_fields_enhanced(images)

            # Display results
            print(f"\n{'='*60}")
            print("✨ EXTRACTED DATA:")
            print(f"{'='*60}")
            print(f"  Party Name:      {invoice_data.party_name}")
            print(f"  Invoice Number:  {invoice_data.invoice_number}")
            print(f"  Invoice Date:    {invoice_data.invoice_date}")
            print(f"  Invoice Amount:  {invoice_data.invoice_amount}")
            print(f"{'='*60}")

            # Calculate success rate
            found_fields = sum([
                invoice_data.party_name != "Not Found",
                invoice_data.invoice_number != "Not Found",
                invoice_data.invoice_date != "Not Found",
                invoice_data.invoice_amount != "Not Found"
            ])
            print(f"\n✅ Extraction Success: {found_fields}/4 fields ({found_fields*25}%)")

            # Save to Excel
            self.save_to_excel(invoice_data, output_excel, pdf_path)
            print(f"💾 Data saved to: {output_excel}")

            return {
                "extracted_data": invoice_data.__dict__,
                "excel_path": output_excel,
                "pdf_file": os.path.basename(pdf_path),
                "success_rate": f"{found_fields}/4"
            }

        except Exception as e:
            print(f"\n❌ Error processing invoice: {e}")
            import traceback
            traceback.print_exc()
            return None

    def save_to_excel(self, data: InvoiceData, output_path: str, pdf_path: str = ""):
        """Save extracted data to Excel"""
        try:
            if os.path.exists(output_path):
                wb = openpyxl.load_workbook(output_path)
                ws = wb.active
            else:
                wb = openpyxl.Workbook()
                ws = wb.active
                ws.append(["Source PDF", "Party Name", "Invoice Date", "Invoice Number", "Invoice Amount"])

            ws.append([
                os.path.basename(pdf_path) if pdf_path else "",
                data.party_name,
                data.invoice_date,
                data.invoice_number,
                data.invoice_amount
            ])

            for column in ws.columns:
                max_length = 0
                column = [cell for cell in column]
                for cell in column:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                    except:
                        pass
                adjusted_width = min(max_length + 2, 50)
                ws.column_dimensions[column[0].column_letter].width = adjusted_width

            wb.save(output_path)

        except Exception as e:
            print(f"❌ Error saving to Excel: {e}")

    def batch_process_invoices(self, folder_path: str, output_excel: str = "all_invoices_validated.xlsx") -> List[Dict]:
        """Batch process with validation"""
        if not os.path.exists(folder_path):
            print(f"❌ Folder not found: {folder_path}")
            return []

        pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.pdf')]

        if not pdf_files:
            print(f"❌ No PDF files found in {folder_path}")
            return []

        print(f"\n{'='*60}")
        print(f"📁 BATCH PROCESSING WITH VALIDATION")
        print(f"{'='*60}")
        print(f"📊 Found {len(pdf_files)} PDF files")

        results = []
        success_count = 0

        for idx, pdf_file in enumerate(pdf_files, 1):
            print(f"\n[{idx}/{len(pdf_files)}] {pdf_file}")
            print("-" * 40)

            pdf_path = os.path.join(folder_path, pdf_file)

            try:
                result = self.process_invoice(pdf_path, output_excel)
                if result:
                    results.append(result)
                    fields_found = int(result['success_rate'].split('/')[0])
                    if fields_found >= 3:
                        success_count += 1
            except Exception as e:
                print(f"❌ Error: {str(e)}")
                continue

        print(f"\n{'='*60}")
        print(f"📊 BATCH SUMMARY")
        print(f"{'='*60}")
        print(f"✅ Successfully processed: {success_count}/{len(pdf_files)}")
        print(f"💾 Saved to: {output_excel}")
        print(f"{'='*60}\n")

        return results

# Usage
if __name__ == "__main__":
    parser = DonutInvoiceParser()

    # Test single invoice
    result = parser.process_invoice("ADAMJEE INS 180.pdf")

    # Or batch process
    #results = parser.batch_process_invoices("./invoices")
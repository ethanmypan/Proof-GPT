#!/usr/bin/env python3
"""
PDF Exam Solution Parser using Anthropic Claude API
Extracts questions and answers from PDF exam solutions into JSON format
Supports recursive directory processing for nested folder structures
"""

import os
import json
import base64
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import re

# Required libraries - install with:
# pip install anthropic PyPDF2 pillow pdf2image python-dotenv
import anthropic
from PyPDF2 import PdfReader
from pdf2image import convert_from_path
from PIL import Image
import io
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class PDFExamParser:
    def __init__(self, api_key: str, model: str = "claude-3-5-sonnet-latest"):
        """
        Initialize the parser with Anthropic API credentials
        
        Args:
            api_key: Your Anthropic API key
            model: Claude model to use (default: claude-3-5-sonnet-20241022)
        """
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.processed_count = 0
        self.error_count = 0
        
    def pdf_to_text(self, pdf_path: str) -> str:
        """
        Extract text from PDF using PyPDF2
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Extracted text as string
        """
        try:
            reader = PdfReader(pdf_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            print(f"Error extracting text from {pdf_path}: {e}")
            return ""
    
    def pdf_to_images(self, pdf_path: str, dpi: int = 150) -> List[bytes]:
        """
        Convert PDF pages to images for better OCR with Claude
        
        Args:
            pdf_path: Path to the PDF file
            dpi: Resolution for image conversion
            
        Returns:
            List of image bytes
        """
        try:
            images = convert_from_path(pdf_path, dpi=dpi)
            image_bytes_list = []
            
            for img in images:
                # Convert to bytes
                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format='PNG')
                img_byte_arr.seek(0)
                image_bytes_list.append(img_byte_arr.read())
                
            return image_bytes_list
        except Exception as e:
            print(f"Error converting PDF to images: {e}")
            return []
    
    def parse_with_claude(self, content: str, use_vision: bool = False, image_data: Optional[List[bytes]] = None, retry_count: int = 3) -> List[Dict]:
        """
        Send content to Claude API for parsing
        
        Args:
            content: Text content or description
            use_vision: Whether to use vision API for images
            image_data: Image bytes if using vision
            retry_count: Number of retries for API calls
            
        Returns:
            List of parsed question-answer dictionaries
        """
        system_prompt = """You are an expert at parsing exam solutions from PDFs. 
        Extract all questions and their corresponding answers from the provided content.
        
        CRITICAL: Return ONLY a valid JSON array - no other text before or after.
        Do not include phrases like "Here's the extracted questions" or "I'll format this".
        Start directly with [ and end with ]
        
        Format:
        [
          {
            "question": "The complete question text including question number and all details",
            "answer": "The complete answer/solution text"
          }
        ]
        
        Guidelines:
        - Include the question number in the question field
        - Preserve all mathematical notation, symbols, and formatting
        - Keep answers complete and detailed
        - If there are sub-parts (a, b, c), include them as separate entries
        - Ensure the JSON is valid and properly escaped"""
        
        for attempt in range(retry_count):
            try:
                if use_vision and image_data:
                    # Use vision API for better accuracy with complex formatting
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "Extract all questions and answers from these exam solution pages. Return ONLY the JSON array, no other text."
                                }
                            ]
                        }
                    ]
                    
                    # Add images to the message
                    for img_bytes in image_data:
                        base64_image = base64.b64encode(img_bytes).decode('utf-8')
                        messages[0]["content"].append({
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": base64_image
                            }
                        })
                else:
                    # Use text-based parsing
                    messages = [
                        {
                            "role": "user",
                            "content": f"Extract all questions and answers from this exam solution text. Return ONLY the JSON array, no other text.\n\n{content}"
                        }
                    ]
                
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=8192,
                    temperature=0,
                    system=system_prompt,
                    messages=messages
                )
                
                # Parse the response
                result_text = response.content[0].text.strip()
                
                # Aggressive cleaning to extract JSON
                # Remove markdown code blocks
                if "```json" in result_text:
                    start = result_text.find("```json") + 7
                    end = result_text.rfind("```")
                    if end > start:
                        result_text = result_text[start:end].strip()
                elif "```" in result_text:
                    start = result_text.find("```") + 3
                    end = result_text.rfind("```")
                    if end > start:
                        result_text = result_text[start:end].strip()
                
                # Find the JSON array in the text
                # Look for the first [ and last ]
                if "[" in result_text and "]" in result_text:
                    start_idx = result_text.find("[")
                    end_idx = result_text.rfind("]") + 1
                    if start_idx < end_idx:
                        result_text = result_text[start_idx:end_idx]
                
                # Remove any remaining non-JSON text before or after
                lines = result_text.split('\n')
                json_lines = []
                in_json = False
                for line in lines:
                    if line.strip().startswith('['):
                        in_json = True
                    if in_json:
                        json_lines.append(line)
                    if line.strip().endswith(']') and in_json:
                        break
                
                if json_lines:
                    result_text = '\n'.join(json_lines)
                
                # Parse JSON
                parsed_data = json.loads(result_text)
                return parsed_data
                
            except json.JSONDecodeError as e:
                print(f"Attempt {attempt + 1}: Error parsing JSON response: {e}")
                
                # Try to extract JSON from response
                json_match = re.search(r'\[\s*\{.*?\}\s*\]', result_text, re.DOTALL)
                if json_match:
                    try:
                        parsed_data = json.loads(json_match.group())
                        print(f"Successfully extracted {len(parsed_data)} questions after cleaning")
                        return parsed_data
                    except:
                        pass
                
                if attempt < retry_count - 1:
                    print(f"Retrying in {2 ** attempt} seconds...")
                    time.sleep(2 ** attempt)
                else:
                    print(f"Failed to extract JSON after {retry_count} attempts. Raw response preview: {result_text[:200]}...")
                    
            except Exception as e:
                if attempt < retry_count - 1:
                    print(f"Attempt {attempt + 1}: Error calling Claude API: {e}")
                    print(f"Retrying in {2 ** attempt} seconds...")
                    time.sleep(2 ** attempt)
                else:
                    print(f"Failed after {retry_count} attempts: {e}")
                    
        return []
    
    def process_pdf(self, pdf_path: str, use_vision: bool = True) -> Tuple[List[Dict], Dict]:
        """
        Process a single PDF file
        
        Args:
            pdf_path: Path to the PDF file
            use_vision: Whether to use vision API (recommended for complex PDFs)
            
        Returns:
            Tuple of (List of question-answer dictionaries, metadata dict)
        """
        print(f"Processing: {pdf_path}")
        
        # Extract metadata
        pdf_path_obj = Path(pdf_path)
        metadata = {
            "source_file": pdf_path_obj.name,
            "source_path": str(pdf_path),
            "semester": pdf_path_obj.parent.name,
            "processed_at": datetime.now().isoformat()
        }
        
        questions = []
        
        if use_vision:
            # Convert PDF to images for better accuracy
            image_data = self.pdf_to_images(pdf_path)
            if image_data:
                questions = self.parse_with_claude("", use_vision=True, image_data=image_data)
            else:
                print(f"Failed to convert PDF to images, falling back to text extraction")
        
        # Fall back to text extraction if vision fails or is disabled
        if not questions:
            text_content = self.pdf_to_text(pdf_path)
            if text_content:
                questions = self.parse_with_claude(text_content, use_vision=False)
            else:
                print(f"Failed to extract text from {pdf_path}")
        
        # Add metadata to each question
        for q in questions:
            q["metadata"] = metadata
        
        return questions, metadata
    
    def find_all_pdfs(self, directory: str) -> List[Path]:
        """
        Recursively find all PDF files in directory and subdirectories
        
        Args:
            directory: Root directory to search
            
        Returns:
            List of Path objects for all PDF files
        """
        root_path = Path(directory)
        pdf_files = list(root_path.rglob("*.pdf"))
        
        # Sort for consistent processing order
        pdf_files.sort()
        
        return pdf_files
    
    def process_directory(self, directory: str, output_file: str = "exam_questions.json", 
                         use_vision: bool = True, max_workers: int = 3):
        """
        Process all PDFs in a directory and its subdirectories
        
        Args:
            directory: Root directory containing PDF files
            output_file: Output JSON file path
            use_vision: Whether to use vision API
            max_workers: Number of parallel workers
        """
        # Find all PDFs
        pdf_files = self.find_all_pdfs(directory)
        
        if not pdf_files:
            print(f"No PDF files found in {directory}")
            return
        
        print(f"Found {len(pdf_files)} PDF files to process")
        print("Directory structure:")
        
        # Show directory structure
        semesters = {}
        for pdf in pdf_files:
            semester = pdf.parent.name
            if semester not in semesters:
                semesters[semester] = []
            semesters[semester].append(pdf.name)
        
        for semester, files in semesters.items():
            print(f"  {semester}: {len(files)} files")
            for file in files[:3]:  # Show first 3 files
                print(f"    - {file}")
            if len(files) > 3:
                print(f"    ... and {len(files) - 3} more")
        
        print("\nStarting processing...")
        
        all_questions = []
        semester_counts = {}
        
        # Process PDFs with rate limiting
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_pdf = {executor.submit(self.process_pdf, str(pdf), use_vision): pdf 
                           for pdf in pdf_files}
            
            for future in as_completed(future_to_pdf):
                pdf = future_to_pdf[future]
                try:
                    questions, metadata = future.result()
                    semester = metadata["semester"]
                    
                    if questions:
                        all_questions.extend(questions)
                        
                        # Track counts by semester for summary
                        if semester not in semester_counts:
                            semester_counts[semester] = 0
                        semester_counts[semester] += len(questions)
                        
                        self.processed_count += 1
                        print(f"✓ Extracted {len(questions)} questions from {semester}/{pdf.name}")
                    else:
                        self.error_count += 1
                        print(f"✗ No questions extracted from {semester}/{pdf.name}")
                    
                    # Rate limiting to avoid API throttling
                    time.sleep(1)
                    
                except Exception as e:
                    self.error_count += 1
                    print(f"✗ Error processing {pdf.name}: {e}")
        
        # Save all questions to single JSON file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_questions, f, indent=2, ensure_ascii=False)
        
        # Print summary
        print("\n" + "="*50)
        print("PROCESSING COMPLETE")
        print("="*50)
        print(f"Total PDFs processed: {self.processed_count + self.error_count}")
        print(f"Successful: {self.processed_count}")
        print(f"Failed: {self.error_count}")
        print(f"Total questions extracted: {len(all_questions)}")
        print(f"\nResults saved to: {output_file}")
        
        # Print breakdown by semester
        if semester_counts:
            print("\nQuestions by semester:")
            for semester, count in sorted(semester_counts.items()):
                print(f"  {semester}: {count} questions")
        
        # Save summary report
        summary = {
            "processing_date": datetime.now().isoformat(),
            "total_pdfs": len(pdf_files),
            "successful": self.processed_count,
            "failed": self.error_count,
            "total_questions": len(all_questions),
            "by_semester": semester_counts,
            "output_file": output_file
        }
        
        summary_file = Path(output_file).parent / "processing_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        print(f"Summary report saved to: {summary_file}")

def main():
    parser = argparse.ArgumentParser(description='Parse exam PDFs and extract questions/answers to JSON')
    parser.add_argument('input', help='Input PDF file or directory containing PDFs')
    parser.add_argument('-o', '--output', default='exam_questions.json',
                       help='Output JSON file (default: exam_questions.json)')
    parser.add_argument('-k', '--api-key', help='Anthropic API key (overrides .env file)')
    parser.add_argument('--model', default='claude-3-5-sonnet-latest',
                       help='Claude model to use')
    parser.add_argument('--no-vision', action='store_true',
                       help='Disable vision API and use text extraction only')
    parser.add_argument('--workers', type=int, default=3,
                       help='Number of parallel workers for batch processing')
    
    args = parser.parse_args()
    
    # Get API key - prioritize command line, then .env file, then environment variable
    api_key = args.api_key or os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        print("Error: Anthropic API key required.")
        print("Please set it in one of the following ways:")
        print("  1. Create a .env file with: ANTHROPIC_API_KEY=your-key-here")
        print("  2. Set environment variable: export ANTHROPIC_API_KEY=your-key-here")
        print("  3. Pass via command line: --api-key your-key-here")
        return
    
    # Initialize parser
    parser = PDFExamParser(api_key, model=args.model)
    
    # Process input
    input_path = Path(args.input)
    
    if input_path.is_file() and input_path.suffix.lower() == '.pdf':
        # Single PDF file
        questions, metadata = parser.process_pdf(str(input_path), use_vision=not args.no_vision)
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(questions, f, indent=2, ensure_ascii=False)
        print(f"Extracted {len(questions)} questions to {args.output}")
        
    elif input_path.is_dir():
        # Directory of PDFs (recursive)
        parser.process_directory(str(input_path), 
                               output_file=args.output,
                               use_vision=not args.no_vision, 
                               max_workers=args.workers)
    else:
        print(f"Error: {args.input} is not a valid PDF file or directory")

if __name__ == "__main__":
    main()
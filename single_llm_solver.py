"""
single_llm_solver.py
--------------------
Research-focused baseline implementation: Tests 3 latest LLMs (GPT-4o, Gemini 2.0 Flash, Claude-3.5-Sonnet)
on proof-based problems with systematic comparison and enhanced identification.

Usage:
  python single_llm_solver.py --input exam_questions.json --limit 10 --output results/baseline_research/
"""

import argparse
import json
import os
import sys
import time
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

try:
    from openai import OpenAI
    from anthropic import Anthropic
    import google.generativeai as genai
    from dotenv import load_dotenv
except ImportError as e:
    raise RuntimeError(
        "Required libraries not found. Please run: "
        "pip install --upgrade openai anthropic google-generativeai python-dotenv"
    ) from e

# Load environment variables
load_dotenv()

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kwargs: x


class ResearchBaselineSolver:
    """Research-focused solver testing latest LLMs systematically"""
    
    # Latest model configurations
    RESEARCH_MODELS = {
        "gpt-4o": {
            "provider": "openai",
            "model_id": "gpt-4o",
            "pricing": {"input": 0.0025, "output": 0.01}  # per 1K tokens
        },
        "gemini-2.0-flash": {
            "provider": "google",
            "model_id": "gemini-2.0-flash-exp",
            "pricing": {"input": 0.00075, "output": 0.003}  # per 1K tokens
        },
        "claude-3.5-sonnet": {
            "provider": "anthropic", 
            "model_id": "claude-3-5-sonnet-20241022",
            "pricing": {"input": 0.003, "output": 0.015}  # per 1K tokens
        }
    }
    
    def __init__(self):
        """Initialize all API clients"""
        self.clients = self._initialize_all_clients()
        self.session_id = f"baseline_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}"
    
    def _initialize_all_clients(self) -> Dict:
        """Initialize all API clients for the research models"""
        clients = {}
        
        # OpenAI client
        if os.getenv("OPENAI_API_KEY"):
            clients["openai"] = OpenAI()
        else:
            print("Warning: OPENAI_API_KEY not found")
            
        # Anthropic client  
        if os.getenv("ANTHROPIC_API_KEY"):
            clients["anthropic"] = Anthropic()
        else:
            print("Warning: ANTHROPIC_API_KEY not found")
            
        # Google AI client
        if os.getenv("GOOGLE_API_KEY"):
            genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
            clients["google"] = genai
        else:
            print("Warning: GOOGLE_API_KEY not found")
            
        return clients
    
    def generate_question_id(self, qa_item: Dict) -> str:
        """Generate enhanced question ID from metadata"""
        metadata = qa_item.get('metadata', {})
        question_text = qa_item.get('question', '')
        
        # Extract source file and semester
        source_file = metadata.get('source_file', 'unknown')
        semester = metadata.get('semester', 'unknown')
        
        # Clean source file name
        source_clean = source_file.lower().replace('.pdf', '').replace(' ', '_')
        
        # Extract question number from text
        question_num = self._extract_question_number(question_text)
        
        return f"{source_clean}_{semester}_{question_num}"
    
    def _extract_question_number(self, question_text: str) -> str:
        """Extract question number from question text"""
        # Look for patterns like "Question 1", "Problem 3", etc.
        patterns = [
            r'Question\s+(\d+)',
            r'Problem\s+(\d+)',
            r'Q(\d+)',
            r'^(\d+)\.',  # Number at start
        ]
        
        for pattern in patterns:
            match = re.search(pattern, question_text, re.IGNORECASE)
            if match:
                return f"q{match.group(1)}"
        
        return "q_unknown"
    
    def solve_with_model(self, model_name: str, question: str, question_id: str) -> Dict[str, Any]:
        """Solve question with specific model"""
        if model_name not in self.RESEARCH_MODELS:
            raise ValueError(f"Unknown model: {model_name}")
            
        model_config = self.RESEARCH_MODELS[model_name]
        provider = model_config["provider"]
        model_id = model_config["model_id"]
        
        start_time = time.time()
        
        # Standardized system prompt for all models
        system_prompt = (
            "You are an expert in theoretical computer science and mathematics. "
            "Solve the following proof-based problem with rigorous mathematical reasoning. "
            "Provide a complete, step-by-step solution with clear justification for each step. "
            "Use formal mathematical notation where appropriate."
        )
        
        try:
            if provider == "openai" and "openai" in self.clients:
                response = self.clients["openai"].chat.completions.create(
                    model=model_id,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": question}
                    ],
                    temperature=0.1,
                    max_tokens=3000
                )
                answer = response.choices[0].message.content.strip()
                prompt_tokens = response.usage.prompt_tokens
                completion_tokens = response.usage.completion_tokens
                total_tokens = response.usage.total_tokens
                
            elif provider == "anthropic" and "anthropic" in self.clients:
                response = self.clients["anthropic"].messages.create(
                    model=model_id,
                    system=system_prompt,
                    messages=[{"role": "user", "content": question}],
                    max_tokens=3000,
                    temperature=0.1
                )
                answer = response.content[0].text.strip()
                prompt_tokens = response.usage.input_tokens
                completion_tokens = response.usage.output_tokens
                total_tokens = prompt_tokens + completion_tokens
                
            elif provider == "google" and "google" in self.clients:
                model = self.clients["google"].GenerativeModel(model_id)
                prompt_with_system = f"{system_prompt}\n\nQuestion: {question}"
                
                response = model.generate_content(
                    prompt_with_system,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.1,
                        max_output_tokens=3000,
                    )
                )
                answer = response.text.strip()
                # Gemini doesn't provide token counts in the same way
                prompt_tokens = len(prompt_with_system.split()) * 1.3  # Rough estimate
                completion_tokens = len(answer.split()) * 1.3
                total_tokens = int(prompt_tokens + completion_tokens)
                
            else:
                raise ValueError(f"No client available for {provider}")
            
            execution_time = (time.time() - start_time) * 1000  # ms
            cost = self._calculate_model_cost(model_name, prompt_tokens, completion_tokens)
            
            return {
                "question_id": question_id,
                "model": model_name,
                "model_id": model_id,
                "provider": provider,
                "answer": answer,
                "tokens": {
                    "prompt": int(prompt_tokens),
                    "completion": int(completion_tokens),
                    "total": int(total_tokens)
                },
                "execution_time_ms": execution_time,
                "cost": cost,
                "timestamp": datetime.now().isoformat(),
                "success": True,
                "error": None
            }
            
        except Exception as e:
            return {
                "question_id": question_id,
                "model": model_name,
                "model_id": model_id,
                "provider": provider,
                "answer": None,
                "tokens": {"prompt": 0, "completion": 0, "total": 0},
                "execution_time_ms": (time.time() - start_time) * 1000,
                "cost": 0,
                "timestamp": datetime.now().isoformat(),
                "success": False,
                "error": str(e)
            }
    
    def _calculate_model_cost(self, model_name: str, prompt_tokens: int, completion_tokens: int) -> float:
        """Calculate cost for specific model and token usage"""
        if model_name not in self.RESEARCH_MODELS:
            return 0.0
            
        pricing = self.RESEARCH_MODELS[model_name]["pricing"]
        
        prompt_cost = (prompt_tokens / 1000) * pricing["input"]
        completion_cost = (completion_tokens / 1000) * pricing["output"]
        
        return prompt_cost + completion_cost
    
    def run_baseline_research(self, questions: List[Dict], limit: Optional[int] = None) -> Dict[str, Any]:
        """Run baseline research testing all 3 models on the same questions"""
        if limit:
            questions = questions[:limit]
            
        print(f"\nStarting Baseline Research Session: {self.session_id}")
        print(f"Testing {len(questions)} questions across {len(self.RESEARCH_MODELS)} models")
        print(f"Models: {', '.join(self.RESEARCH_MODELS.keys())}")
        
        # Results storage
        all_results = {model: [] for model in self.RESEARCH_MODELS.keys()}
        session_metadata = {
            "session_id": self.session_id,
            "timestamp": datetime.now().isoformat(),
            "models_tested": list(self.RESEARCH_MODELS.keys()),
            "num_questions": len(questions),
            "standardized_prompt": True
        }
        
        # Process each question with all models
        for i, qa in enumerate(tqdm(questions, desc="Processing questions"), 1):
            question = qa.get('question', '')
            correct_answer = qa.get('answer', '')
            question_id = self.generate_question_id(qa)
            
            print(f"\nQuestion {i}/{len(questions)}: {question_id}")
            
            # Test each model on this question
            for model_name in self.RESEARCH_MODELS.keys():
                print(f"Testing {model_name}...")
                
                result = self.solve_with_model(model_name, question, question_id)
                result["question"] = question
                result["correct_answer"] = correct_answer
                result["question_metadata"] = qa.get('metadata', {})
                
                all_results[model_name].append(result)
                
                if result["success"]:
                    print(f"Success - {result['tokens']['total']} tokens, ${result['cost']:.4f}")
                else:
                    print(f"Error: {result['error']}")
        
        # Calculate summary statistics
        summary = self._calculate_summary_stats(all_results)
        
        return {
            "session_metadata": session_metadata,
            "results": all_results,
            "summary": summary
        }
    
    def _calculate_summary_stats(self, all_results: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """Calculate summary statistics for all models"""
        summary = {}
        
        for model_name, results in all_results.items():
            if not results:
                summary[model_name] = {"error": "No results"}
                continue
                
            successful_results = [r for r in results if r["success"]]
            
            if successful_results:
                total_cost = sum(r["cost"] for r in successful_results)
                avg_time = sum(r["execution_time_ms"] for r in successful_results) / len(successful_results)
                avg_tokens = sum(r["tokens"]["total"] for r in successful_results) / len(successful_results)
                
                summary[model_name] = {
                    "questions_attempted": len(results),
                    "questions_successful": len(successful_results),
                    "success_rate": len(successful_results) / len(results),
                    "total_cost": total_cost,
                    "avg_cost_per_question": total_cost / len(successful_results),
                    "avg_execution_time_ms": avg_time,
                    "avg_tokens": avg_tokens,
                    "total_tokens": sum(r["tokens"]["total"] for r in successful_results)
                }
            else:
                summary[model_name] = {
                    "questions_attempted": len(results),
                    "questions_successful": 0,
                    "success_rate": 0,
                    "error": "All requests failed"
                }
        
        return summary


def save_research_results(results: Dict[str, Any], output_dir: Path) -> Dict[str, Path]:
    """Save research results to organized file structure"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    responses_dir = output_dir / "responses"
    responses_dir.mkdir(exist_ok=True)
    
    session_id = results["session_metadata"]["session_id"]
    saved_files = {}
    
    # Save individual model responses for grading
    for model_name, model_results in results["results"].items():
        model_file = responses_dir / f"{model_name.replace('-', '_')}_responses.json"
        
        # Format for grading system
        grading_format = {
            "metadata": {
                "model": model_name,
                "session_id": session_id,
                "timestamp": results["session_metadata"]["timestamp"],
                "num_questions": len(model_results)
            },
            "responses": model_results
        }
        
        with open(model_file, 'w', encoding='utf-8') as f:
            json.dump(grading_format, f, indent=2, ensure_ascii=False)
            
        saved_files[f"{model_name}_responses"] = model_file
    
    # Save complete research session
    session_file = output_dir / f"{session_id}_complete_results.json"
    with open(session_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    saved_files["complete_session"] = session_file
    
    return saved_files


def print_research_summary(results: Dict[str, Any]) -> None:
    """Print formatted research summary"""
    print(f"\n{'='*80}")
    print(f"BASELINE RESEARCH SUMMARY")
    print(f"{'='*80}")
    
    session = results["session_metadata"]
    print(f"Session: {session['session_id']}")
    print(f"Timestamp: {session['timestamp']}")
    print(f"Questions tested: {session['num_questions']}")
    
    print(f"\n{'Model Performance':>20} | {'Success':>8} | {'Avg Cost':>10} | {'Avg Time':>10} | {'Avg Tokens':>12}")
    print(f"{'-'*20} | {'-'*8} | {'-'*10} | {'-'*10} | {'-'*12}")
    
    summary = results["summary"]
    total_cost = 0
    
    for model_name in session["models_tested"]:
        stats = summary.get(model_name, {})
        if "error" not in stats:
            success_rate = f"{stats['success_rate']:.1%}"
            avg_cost = f"${stats['avg_cost_per_question']:.4f}"
            avg_time = f"{stats['avg_execution_time_ms']:.0f}ms"
            avg_tokens = f"{stats['avg_tokens']:.0f}"
            total_cost += stats['total_cost']
            
            print(f"{model_name:>20} | {success_rate:>8} | {avg_cost:>10} | {avg_time:>10} | {avg_tokens:>12}")
        else:
            print(f"{model_name:>20} | {'ERROR':>8} | {'-':>10} | {'-':>10} | {'-':>12}")
    
    print(f"{'-'*80}")
    print(f" Total session cost: ${total_cost:.4f}")
    print(f"Results ready for grading system")


def main():
    parser = argparse.ArgumentParser(
        description="Research-focused baseline testing of latest LLMs on proof problems"
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        default=Path("exam_questions.json"),
        help="Input JSON file with questions and metadata"
    )
    parser.add_argument(
        "--output", "-o", 
        type=Path,
        default=Path("results/baseline_research"),
        help="Output directory for research results"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Number of questions to test (default: 10)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show configuration without running"
    )
    
    args = parser.parse_args()
    
    # Validate input
    if not args.input.exists():
        print(f"ERROR: Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)
    
    if args.dry_run:
        print(f"Configuration:")
        print(f"   Input: {args.input}")
        print(f"   Output: {args.output}")
        print(f"   Question limit: {args.limit}")
        print(f"   Models to test: {', '.join(ResearchBaselineSolver.RESEARCH_MODELS.keys())}")
        return
    
    # Load questions
    print(f"Loading questions from {args.input}")
    with open(args.input, 'r', encoding='utf-8') as f:
        questions = json.load(f)
    
    print(f"Loaded {len(questions)} questions")
    
    # Initialize research solver
    try:
        solver = ResearchBaselineSolver()
    except Exception as e:
        print(f"Failed to initialize solver: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Run baseline research
    results = solver.run_baseline_research(questions, limit=args.limit)
    
    # Save results
    print(f"\nSaving results to {args.output}")
    saved_files = save_research_results(results, args.output)
    
    # Print summary
    print_research_summary(results)
    
    print(f"\nFiles saved:")
    for desc, path in saved_files.items():
        print(f"   {desc}: {path}")
    
    print(f"\nBaseline research complete! Results ready for grading system.")


if __name__ == "__main__":
    main()
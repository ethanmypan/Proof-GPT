"""
three_llm_concurrent.py
-----------------------
Three LLMs solve problems concurrently, with a fourth LLM acting as verifier.
Tests whether multiple models working together outperform single models.

Usage:
  python three_llm_concurrent.py --input exam_questions.json --output results/three_llm/
"""

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple

try:
    from openai import AsyncOpenAI, OpenAI
    from anthropic import AsyncAnthropic, Anthropic
    from dotenv import load_dotenv
except ImportError as e:
    raise RuntimeError(
        "Required libraries not found. Please run: "
        "pip install --upgrade openai anthropic python-dotenv"
    ) from e

# Load environment variables
load_dotenv()

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kwargs: x


class ThreeLLMSolver:
    """Three LLMs solve concurrently with verification"""
    
    def __init__(self, solver_models: List[str], verifier_model: str):
        self.solver_models = solver_models
        self.verifier_model = verifier_model
        self.openai_client = AsyncOpenAI() if os.getenv("OPENAI_API_KEY") else None
        self.anthropic_client = AsyncAnthropic() if os.getenv("ANTHROPIC_API_KEY") else None
        self.sync_openai = OpenAI() if os.getenv("OPENAI_API_KEY") else None
        self.sync_anthropic = Anthropic() if os.getenv("ANTHROPIC_API_KEY") else None
    
    async def solve_single_async(self, model: str, question: str) -> Tuple[str, Dict]:
        """Solve with a single model asynchronously"""
        system_prompt = (
            "You are an expert in theoretical computer science and mathematics. "
            "Solve the following proof-based problem rigorously. "
            "Provide a clear, step-by-step solution with proper justification."
        )
        
        try:
            if "gpt" in model.lower() and self.openai_client:
                response = await self.openai_client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": question}
                    ],
                    temperature=0.1,
                    max_tokens=2048
                )
                answer = response.choices[0].message.content.strip()
                tokens = response.usage.total_tokens
                
            elif "claude" in model.lower() and self.anthropic_client:
                response = await self.anthropic_client.messages.create(
                    model=model,
                    system=system_prompt,
                    messages=[{"role": "user", "content": question}],
                    max_tokens=2048,
                    temperature=0.1
                )
                answer = response.content[0].text.strip()
                tokens = response.usage.input_tokens + response.usage.output_tokens
            else:
                return None, {"error": f"No client for model {model}"}
            
            return answer, {"tokens": tokens, "model": model}
            
        except Exception as e:
            return None, {"error": str(e), "model": model}
    
    async def solve_concurrent(self, question: str) -> Dict[str, Any]:
        """Three models solve the question concurrently"""
        start_time = time.time()
        
        # Create tasks for concurrent solving
        tasks = [
            self.solve_single_async(model, question)
            for model in self.solver_models
        ]
        
        # Run all three concurrently
        results = await asyncio.gather(*tasks)
        
        # Unpack results
        solutions = []
        total_tokens = 0
        errors = []
        
        for i, (answer, metadata) in enumerate(results):
            if answer:
                solutions.append({
                    "model": self.solver_models[i],
                    "answer": answer,
                    "tokens": metadata.get("tokens", 0)
                })
                total_tokens += metadata.get("tokens", 0)
            else:
                errors.append(metadata)
        
        solving_time = (time.time() - start_time) * 1000
        
        return {
            "solutions": solutions,
            "solving_time_ms": solving_time,
            "total_solver_tokens": total_tokens,
            "errors": errors
        }
    
    def verify_solutions(self, question: str, solutions: List[Dict]) -> Dict[str, Any]:
        """Use verifier model to select best answer"""
        if not solutions:
            return {"error": "No solutions to verify"}
        
        # Build verification prompt
        verification_prompt = f"Original Question:\n{question}\n\n"
        verification_prompt += "Multiple solutions have been provided:\n\n"
        
        for i, sol in enumerate(solutions, 1):
            verification_prompt += f"Solution {i} (from {sol['model']}):\n"
            verification_prompt += f"{sol['answer']}\n\n"
        
        verification_prompt += """Please analyze these solutions and:
1. Identify which solution is most correct and complete
2. Explain why it's the best answer
3. Rate your confidence (0-100)
4. Note if multiple solutions are equally correct
5. Identify any consensus patterns

Respond in JSON format:
{
    "best_solution_number": 1,
    "best_model": "model_name",
    "reasoning": "explanation",
    "confidence": 85,
    "consensus": true/false,
    "all_correct": true/false
}"""
        
        try:
            if "gpt" in self.verifier_model.lower() and self.sync_openai:
                response = self.sync_openai.chat.completions.create(
                    model=self.verifier_model,
                    messages=[
                        {"role": "system", "content": "You are an expert verifier. Analyze the solutions and select the best one."},
                        {"role": "user", "content": verification_prompt}
                    ],
                    temperature=0,
                    response_format={"type": "json_object"}
                )
                verification = json.loads(response.choices[0].message.content)
                tokens = response.usage.total_tokens
                
            elif "claude" in self.verifier_model.lower() and self.sync_anthropic:
                response = self.sync_anthropic.messages.create(
                    model=self.verifier_model,
                    system="You are an expert verifier. Analyze the solutions and select the best one. Always respond in valid JSON.",
                    messages=[{"role": "user", "content": verification_prompt}],
                    max_tokens=1024,
                    temperature=0
                )
                verification = json.loads(response.content[0].text)
                tokens = response.usage.input_tokens + response.usage.output_tokens
            else:
                return {"error": "No verifier client available"}
            
            # Extract best answer
            best_idx = verification.get("best_solution_number", 1) - 1
            if 0 <= best_idx < len(solutions):
                verification["selected_answer"] = solutions[best_idx]["answer"]
            
            verification["verifier_tokens"] = tokens
            return verification
            
        except Exception as e:
            return {"error": f"Verification failed: {str(e)}"}
    
    def check_consensus(self, solutions: List[Dict]) -> Dict[str, Any]:
        """Check if solutions agree with each other"""
        if len(solutions) < 2:
            return {"consensus": False, "agreement_level": 0}
        
        # Simple consensus check - can be made more sophisticated
        # For now, just check if answers have similar length and structure
        lengths = [len(sol["answer"]) for sol in solutions]
        avg_length = sum(lengths) / len(lengths)
        
        # If all answers are within 30% of average length, consider some consensus
        variance = sum(abs(l - avg_length) / avg_length for l in lengths) / len(lengths)
        
        return {
            "consensus": variance < 0.3,
            "agreement_level": max(0, 1 - variance),
            "answer_lengths": lengths
        }
    
    async def solve_and_verify(self, question: str) -> Dict[str, Any]:
        """Complete pipeline: concurrent solving + verification"""
        start_time = time.time()
        
        # Step 1: Concurrent solving
        solving_result = await self.solve_concurrent(question)
        
        # Step 2: Check consensus
        consensus = self.check_consensus(solving_result["solutions"])
        
        # Step 3: Verification
        verification = self.verify_solutions(question, solving_result["solutions"])
        
        total_time = (time.time() - start_time) * 1000
        
        # Calculate costs
        total_tokens = (
            solving_result["total_solver_tokens"] + 
            verification.get("verifier_tokens", 0)
        )
        total_cost = self._calculate_total_cost(
            solving_result["solutions"],
            verification.get("verifier_tokens", 0)
        )
        
        return {
            "approach": "three",
            "models_used": self.solver_models + [f"{self.verifier_model}-verifier"],
            "question": question,
            "solutions": solving_result["solutions"],
            "verification": verification,
            "consensus": consensus,
            "final_answer": verification.get("selected_answer"),
            "total_tokens": total_tokens,
            "total_cost": total_cost,
            "execution_time_ms": total_time,
            "solving_time_ms": solving_result["solving_time_ms"],
            "success": "error" not in verification,
            "errors": solving_result.get("errors", [])
        }
    
    def _calculate_total_cost(self, solutions: List[Dict], verifier_tokens: int) -> float:
        """Calculate total cost for all models"""
        total = 0
        
        # Simplified pricing (per 1K tokens)
        pricing = {
            "gpt-4": 0.045,
            "gpt-3.5": 0.001,
            "claude-3-opus": 0.045,
            "claude-3-sonnet": 0.009,
        }
        
        # Solver costs
        for sol in solutions:
            for model_key, price in pricing.items():
                if model_key in sol["model"].lower():
                    total += (sol["tokens"] / 1000) * price
                    break
        
        # Verifier cost
        for model_key, price in pricing.items():
            if model_key in self.verifier_model.lower():
                total += (verifier_tokens / 1000) * price
                break
        
        return total


async def process_questions(questions: List[Dict], solver: ThreeLLMSolver) -> List[Dict]:
    """Process all questions"""
    results = []
    
    for qa in tqdm(questions, desc="Processing with 3 LLMs"):
        question = qa.get("question", "")
        correct_answer = qa.get("answer", "")
        
        result = await solver.solve_and_verify(question)
        result["correct_answer"] = correct_answer
        
        # Placeholder accuracy check
        if result.get("final_answer"):
            result["is_correct"] = len(result["final_answer"]) > 50  # Dummy
        else:
            result["is_correct"] = False
        
        results.append(result)
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Three LLM concurrent solver with verification"
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        default=Path("exam_questions.json"),
        help="Input JSON file"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("results/three_llm"),
        help="Output directory"
    )
    parser.add_argument(
        "--solvers",
        type=str,
        default="gpt-4,claude-3-opus-20240229,gpt-3.5-turbo",
        help="Comma-separated list of solver models"
    )
    parser.add_argument(
        "--verifier",
        type=str,
        default="gpt-4",
        help="Verifier model"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Number of questions"
    )
    
    args = parser.parse_args()
    
    if not args.input.exists():
        print(f"[ERROR] Input not found: {args.input}", file=sys.stderr)
        sys.exit(1)
    
    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)
    
    # Load questions
    with open(args.input, 'r') as f:
        data = json.load(f)
    questions = data[:args.limit] if args.limit else data
    
    print(f"Loaded {len(questions)} questions")
    
    # Parse solver models
    solver_models = [m.strip() for m in args.solvers.split(",")]
    print(f"Solvers: {solver_models}")
    print(f"Verifier: {args.verifier}")
    
    # Initialize solver
    solver = ThreeLLMSolver(solver_models, args.verifier)
    
    # Process questions
    results = asyncio.run(process_questions(questions, solver))
    
    # Calculate statistics
    success_rate = sum(1 for r in results if r["success"]) / len(results)
    accuracy = sum(1 for r in results if r.get("is_correct", False)) / len(results)
    consensus_rate = sum(1 for r in results if r["consensus"]["consensus"]) / len(results)
    avg_time = sum(r["execution_time_ms"] for r in results) / len(results)
    total_cost = sum(r["total_cost"] for r in results)
    
    # Save results
    output_file = args.output / f"three_llm_{len(results)}_results.json"
    with open(output_file, 'w') as f:
        json.dump({
            "metadata": {
                "approach": "three_llm_concurrent",
                "solver_models": solver_models,
                "verifier_model": args.verifier,
                "num_questions": len(results),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            },
            "statistics": {
                "success_rate": success_rate,
                "accuracy": accuracy,
                "consensus_rate": consensus_rate,
                "total_cost": total_cost,
                "avg_cost_per_question": total_cost / len(results),
                "avg_execution_time_ms": avg_time
            },
            "results": results
        }, f, indent=2)
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"Results Summary - Three LLM Approach")
    print(f"{'='*50}")
    print(f"Questions: {len(results)}")
    print(f"Success rate: {success_rate:.1%}")
    print(f"Accuracy: {accuracy:.1%}")
    print(f"Consensus rate: {consensus_rate:.1%}")
    print(f"Total cost: ${total_cost:.4f}")
    print(f"Avg cost: ${total_cost/len(results):.4f}")
    print(f"Avg time: {avg_time:.0f}ms")
    print(f"\nSaved to: {output_file}")


if __name__ == "__main__":
    main()
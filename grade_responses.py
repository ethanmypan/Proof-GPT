"""
grade_responses.py
------------------
3-LLM grading system for evaluating baseline responses.
Each response is graded by 3 different LLMs, with final score being the mean.

Usage:
  python grade_responses.py --input results/baseline_research/responses/ --output results/baseline_research/grading/
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from statistics import mean, stdev

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


class ThreeLLMGrader:
    """3-LLM grading system for consistent evaluation"""
    
    # Grader model configurations
    GRADER_MODELS = {
        "gpt-4o": {
            "provider": "openai",
            "model_id": "gpt-4o",
            "grader_id": "grader_1"
        },
        "claude-3.5-sonnet": {
            "provider": "anthropic", 
            "model_id": "claude-3-5-sonnet-20241022",
            "grader_id": "grader_2"
        },
        "gemini-2.0-flash": {
            "provider": "google",
            "model_id": "gemini-2.0-flash-exp",
            "grader_id": "grader_3"
        }
    }
    
    def __init__(self):
        """Initialize all grader clients"""
        self.clients = self._initialize_all_clients()
        self.grading_session_id = f"grade_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}"
    
    def _initialize_all_clients(self) -> Dict:
        """Initialize all API clients for grading"""
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
    
    def create_grading_prompt(self, question: str, model_response: str, correct_answer: str) -> str:
        """Create standardized grading prompt"""
        return f"""You are an expert grader for theoretical computer science and mathematics problems. 

Your task is to evaluate the correctness and quality of a student's response to a proof-based problem.

QUESTION:
{question}

STUDENT'S RESPONSE:
{model_response}

CORRECT ANSWER (for reference):
{correct_answer}

GRADING CRITERIA:
- Correctness: Is the solution mathematically/algorithmically correct?
- Completeness: Are all necessary steps and justifications provided?
- Clarity: Is the reasoning clear and well-explained?
- Rigor: Does it meet the standards expected for proof-based problems?

INSTRUCTIONS:
1. Score the response from 0-10 (integers only)
2. Provide detailed reasoning for your score
3. Focus on mathematical/algorithmic correctness above all else
4. Consider partial credit for correct approaches with minor errors

Respond ONLY in this JSON format:
{{
    "score": [integer 0-10],
    "reasoning": "[detailed explanation of your grading decision]"
}}"""

    def grade_with_model(self, grader_name: str, question: str, model_response: str, correct_answer: str) -> Dict[str, Any]:
        """Grade a response with a specific grader model"""
        if grader_name not in self.GRADER_MODELS:
            raise ValueError(f"Unknown grader model: {grader_name}")
            
        grader_config = self.GRADER_MODELS[grader_name]
        provider = grader_config["provider"]
        model_id = grader_config["model_id"]
        grader_id = grader_config["grader_id"]
        
        grading_prompt = self.create_grading_prompt(question, model_response, correct_answer)
        
        try:
            if provider == "openai" and "openai" in self.clients:
                response = self.clients["openai"].chat.completions.create(
                    model=model_id,
                    messages=[
                        {"role": "system", "content": "You are an expert grader. Always respond in valid JSON format."},
                        {"role": "user", "content": grading_prompt}
                    ],
                    temperature=0,
                    response_format={"type": "json_object"}
                )
                grading_text = response.choices[0].message.content.strip()
                
            elif provider == "anthropic" and "anthropic" in self.clients:
                response = self.clients["anthropic"].messages.create(
                    model=model_id,
                    system="You are an expert grader. Always respond in valid JSON format.",
                    messages=[{"role": "user", "content": grading_prompt}],
                    max_tokens=1024,
                    temperature=0
                )
                grading_text = response.content[0].text.strip()
                
            elif provider == "google" and "google" in self.clients:
                model = self.clients["google"].GenerativeModel(model_id)
                system_prompt = "You are an expert grader. Always respond in valid JSON format."
                full_prompt = f"{system_prompt}\n\n{grading_prompt}"
                
                response = model.generate_content(
                    full_prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0,
                        max_output_tokens=1024,
                    )
                )
                grading_text = response.text.strip()
                
            else:
                return {
                    "grader_model": grader_name,
                    "grader_id": grader_id,
                    "score": 0,
                    "max_score": 10,
                    "reasoning": f"No client available for {provider}",
                    "grading_timestamp": datetime.now().isoformat(),
                    "success": False,
                    "error": f"No client for {provider}"
                }
            
            # Parse JSON response
            try:
                grading_result = json.loads(grading_text)
                score = int(grading_result.get("score", 0))
                reasoning = grading_result.get("reasoning", "No reasoning provided")
                
                # Validate score range
                if not 0 <= score <= 10:
                    score = max(0, min(10, score))  # Clamp to valid range
                
            except (json.JSONDecodeError, ValueError) as e:
                return {
                    "grader_model": grader_name,
                    "grader_id": grader_id,
                    "score": 0,
                    "max_score": 10,
                    "reasoning": f"Failed to parse grading response: {str(e)}",
                    "grading_timestamp": datetime.now().isoformat(),
                    "success": False,
                    "error": f"JSON parsing failed: {str(e)}"
                }
            
            return {
                "grader_model": grader_name,
                "grader_id": grader_id,
                "score": score,
                "max_score": 10,
                "reasoning": reasoning,
                "grading_timestamp": datetime.now().isoformat(),
                "success": True,
                "error": None
            }
            
        except Exception as e:
            return {
                "grader_model": grader_name,
                "grader_id": grader_id,
                "score": 0,
                "max_score": 10,
                "reasoning": f"Grading failed: {str(e)}",
                "grading_timestamp": datetime.now().isoformat(),
                "success": False,
                "error": str(e)
            }
    
    def grade_response(self, question: str, model_response: str, correct_answer: str, 
                      question_id: str, model_being_evaluated: str) -> Dict[str, Any]:
        """Grade a single response with all 3 graders"""
        print(f"      🎯 Grading response for {question_id}")
        
        individual_grades = []
        successful_grades = []
        
        # Get grades from all 3 graders
        for grader_name in self.GRADER_MODELS.keys():
            print(f"         👨‍🏫 {grader_name} grading...")
            
            grade = self.grade_with_model(grader_name, question, model_response, correct_answer)
            individual_grades.append(grade)
            
            if grade["success"]:
                successful_grades.append(grade["score"])
                print(f"            ✅ Score: {grade['score']}/10")
            else:
                print(f"            ❌ Failed: {grade['error']}")
        
        # Calculate final score and statistics
        if successful_grades:
            final_score = mean(successful_grades)
            score_std = stdev(successful_grades) if len(successful_grades) > 1 else 0
            grader_agreement = self._assess_agreement(successful_grades)
        else:
            final_score = 0
            score_std = 0
            grader_agreement = "no_grades"
        
        return {
            "grading_session_id": self.grading_session_id,
            "question_id": question_id,
            "question_text": question,
            "correct_answer": correct_answer,
            "model_being_evaluated": model_being_evaluated,
            "model_response": model_response,
            "grading_details": individual_grades,
            "final_score": round(final_score, 2),
            "score_breakdown": {
                "individual_scores": [g["score"] for g in individual_grades if g["success"]],
                "mean_score": round(final_score, 2),
                "std_deviation": round(score_std, 2),
                "grader_agreement": grader_agreement,
                "successful_grades": len(successful_grades),
                "total_graders": len(individual_grades)
            },
            "metadata": {
                "graded_at": datetime.now().isoformat(),
                "grading_model_versions": {
                    config["grader_id"]: config["model_id"] 
                    for config in self.GRADER_MODELS.values()
                }
            }
        }
    
    def _assess_agreement(self, scores: List[int]) -> str:
        """Assess agreement level between graders"""
        if len(scores) < 2:
            return "insufficient_data"
        
        score_range = max(scores) - min(scores)
        
        if score_range == 0:
            return "perfect"
        elif score_range <= 1:
            return "high"
        elif score_range <= 2:
            return "moderate"
        elif score_range <= 3:
            return "low"
        else:
            return "very_low"
    
    def grade_model_responses(self, responses_file: Path) -> Dict[str, Any]:
        """Grade all responses from a single model"""
        with open(responses_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        metadata = data["metadata"]
        responses = data["responses"]
        model_name = metadata["model"]
        
        print(f"\n📊 Grading {len(responses)} responses from {model_name}")
        
        graded_responses = []
        total_score = 0
        successful_gradings = 0
        
        for response in tqdm(responses, desc=f"Grading {model_name}"):
            if not response["success"]:
                # Skip failed responses
                continue
                
            graded_response = self.grade_response(
                question=response["question"],
                model_response=response["answer"],
                correct_answer=response["correct_answer"],
                question_id=response["question_id"],
                model_being_evaluated=model_name
            )
            
            graded_responses.append(graded_response)
            
            if graded_response["score_breakdown"]["successful_grades"] > 0:
                total_score += graded_response["final_score"]
                successful_gradings += 1
        
        # Calculate summary statistics
        avg_score = total_score / successful_gradings if successful_gradings > 0 else 0
        
        return {
            "metadata": {
                "model_evaluated": model_name,
                "grading_session_id": self.grading_session_id,
                "graded_at": datetime.now().isoformat(),
                "total_responses": len(responses),
                "successful_responses": len([r for r in responses if r["success"]]),
                "graded_responses": len(graded_responses),
                "successful_gradings": successful_gradings
            },
            "summary": {
                "average_score": round(avg_score, 2),
                "total_graded": successful_gradings,
                "score_distribution": self._calculate_score_distribution(graded_responses)
            },
            "graded_responses": graded_responses
        }
    
    def _calculate_score_distribution(self, graded_responses: List[Dict]) -> Dict[str, int]:
        """Calculate distribution of scores"""
        distribution = {f"{i}-{i+1}": 0 for i in range(0, 10)}
        distribution["10"] = 0
        
        for response in graded_responses:
            score = response["final_score"]
            if score == 10:
                distribution["10"] += 1
            else:
                bucket = f"{int(score)}-{int(score)+1}"
                if bucket in distribution:
                    distribution[bucket] += 1
        
        return distribution


def main():
    parser = argparse.ArgumentParser(
        description="3-LLM grading system for baseline evaluation"
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        default=Path("results/baseline_research/responses"),
        help="Directory containing model response files"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("results/baseline_research/grading"),
        help="Output directory for graded results"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Grade only responses from specific model (optional)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show configuration without running"
    )
    
    args = parser.parse_args()
    
    # Validate input directory
    if not args.input.exists():
        print(f"❌ ERROR: Input directory not found: {args.input}", file=sys.stderr)
        sys.exit(1)
    
    # Find response files
    response_files = list(args.input.glob("*_responses.json"))
    if not response_files:
        print(f"❌ ERROR: No *_responses.json files found in {args.input}", file=sys.stderr)
        sys.exit(1)
    
    # Filter by model if specified
    if args.model:
        model_file = f"{args.model.replace('-', '_')}_responses.json"
        response_files = [f for f in response_files if f.name == model_file]
        if not response_files:
            print(f"❌ ERROR: No responses found for model {args.model}", file=sys.stderr)
            sys.exit(1)
    
    if args.dry_run:
        print(f"🔧 Configuration:")
        print(f"   Input directory: {args.input}")
        print(f"   Output directory: {args.output}")
        print(f"   Response files found: {[f.name for f in response_files]}")
        print(f"   Grader models: {', '.join(ThreeLLMGrader.GRADER_MODELS.keys())}")
        return
    
    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)
    
    # Initialize grader
    grader = ThreeLLMGrader()
    
    print(f"🎯 Starting 3-LLM grading session: {grader.grading_session_id}")
    print(f"📁 Found {len(response_files)} response files to grade")
    
    # Grade each model's responses
    all_graded_results = {}
    
    for response_file in response_files:
        print(f"\n📊 Processing {response_file.name}")
        
        try:
            graded_results = grader.grade_model_responses(response_file)
            all_graded_results[response_file.stem] = graded_results
            
            # Save individual model results
            model_name = graded_results["metadata"]["model_evaluated"]
            output_file = args.output / f"{model_name.replace('-', '_')}_graded.json"
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(graded_results, f, indent=2, ensure_ascii=False)
            
            print(f"✅ Saved graded results to {output_file}")
            
            # Print summary
            summary = graded_results["summary"]
            print(f"   📈 Average score: {summary['average_score']:.2f}/10")
            print(f"   📊 Successfully graded: {summary['total_graded']} responses")
            
        except Exception as e:
            print(f"❌ Failed to grade {response_file.name}: {e}")
            continue
    
    # Save combined results
    combined_file = args.output / f"{grader.grading_session_id}_all_graded.json"
    with open(combined_file, 'w', encoding='utf-8') as f:
        json.dump(all_graded_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Grading complete!")
    print(f"📁 Combined results saved to: {combined_file}")
    print(f"🎯 Session ID: {grader.grading_session_id}")


if __name__ == "__main__":
    main()
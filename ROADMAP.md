# Proof-GPT: AI-Powered Proof Grading Assistant

## Project Evolution
**Original Research Goal**: Compare how different numbers of LLMs (1, 2, and 3) perform when solving proof-based mathematical questions  
**New Product Goal**: Build an AI-powered grading assistant tool for mathematics professors to grade student proof submissions efficiently and consistently

---

## ✅ PHASE 0: COMPLETED FOUNDATION (What We've Already Built)

### Core Infrastructure (100% Complete)

#### 1. PDF Parsing System (`pdf_exam_parser.py`) ✅
- Extracts questions and answers from academic exam PDFs
- Handles complex mathematical notation via Claude Vision API
- Processes entire directories recursively
- Generates structured JSON with metadata

#### 2. Multi-LLM Grading System (`grade_responses.py`) ✅
**Already Implemented:**
- 3-LLM consensus grading (GPT-4o, Claude-3.5-Sonnet, Gemini-2.0-Flash)
- 0-10 scoring scale with detailed reasoning
- Grader agreement analysis and reliability metrics
- Individual scorer tracking and deviation analysis
- JSON output with complete grading details

**Current Grading Output Structure:**
```json
{
  "question_id": "exam1_fall19_q1",
  "model_being_evaluated": "student_response",
  "final_score": 8.0,
  "individual_grades": [
    {"grader": "gpt-4o", "score": 8, "reasoning": "..."},
    {"grader": "claude-3.5-sonnet", "score": 7, "reasoning": "..."},
    {"grader": "gemini-2.0-flash", "score": 9, "reasoning": "..."}
  ],
  "score_breakdown": {
    "mean_score": 8.0,
    "std_deviation": 0.82,
    "grader_agreement": "high"
  }
}
```

#### 3. Baseline Testing Framework (`single_llm_solver.py`) ✅
- Systematic evaluation of GPT-4o, Claude-3.5-Sonnet, Gemini-2.0-Flash
- Cost tracking: $0.003-0.013 per response
- Performance metrics: 8-16 second response times
- Standardized prompting system across all models

#### 4. Analysis & Visualization (`visualize_baseline_results.ipynb`) ✅
- Performance dashboards with matplotlib/seaborn
- Cost-effectiveness analysis
- Accuracy heatmaps by question and model
- Grader agreement visualizations
- Export to JSON summary reports

#### 5. Supporting Infrastructure ✅
- **Question Processing** (`main.py`): Clean question extraction
- **Environment Setup**: API keys for OpenAI, Anthropic, Google
- **Data Storage**: Structured results in `results/baseline_research/`
- **Cost Analysis**: Detailed token and pricing tracking

---

## 🎯 NEW DIRECTION: Transform into Professor Grading Assistant

### Why This Pivot Makes Sense:
- ✅ We already have working multi-LLM grading with consensus
- ✅ Our system provides detailed reasoning for scores
- ✅ Cost is reasonable (<$0.50 per submission with 3 LLMs)
- ✅ 8-16 second grading time is acceptable for async grading
- ❌ Professors need: rubrics, educational feedback, batch processing, grade exports

---

## 📅 PHASE 1: Grading Tool Enhancement (Week 1 - Current Sprint)

### 1.1 Enhanced Grading Engine (`professor_grader.py`) 🆕
**Purpose**: Main grading interface designed for professor workflows

**Features to Implement:**
```python
class ProfessorGrader:
    def __init__(self, rubric_path):
        # Load customizable rubric
        # Initialize 3-LLM grading system
        
    def grade_submission(self, student_proof):
        # Apply rubric-based scoring
        # Generate educational feedback
        # Return structured grade report
        
    def export_grades(self, format='csv'):
        # Export to CSV/Canvas/Blackboard
```

**Rubric Structure (JSON):**
```json
{
  "rubric_name": "Calculus Midterm - Proof Questions",
  "total_points": 100,
  "categories": {
    "correctness": {"weight": 0.4, "description": "Mathematical accuracy"},
    "clarity": {"weight": 0.2, "description": "Clear logical flow"},
    "completeness": {"weight": 0.2, "description": "All steps shown"},
    "rigor": {"weight": 0.2, "description": "Formal proof structure"}
  },
  "partial_credit_rules": [...]
}
```

### 1.2 Enhance Current Grading System (`grade_responses.py`) 🔧
**Current State**: Generic 0-10 scoring with reasoning  
**Enhancements Needed**:
- Add rubric category breakdowns
- Implement partial credit logic
- Detect common mistakes patterns
- Generate specific feedback per rubric category

### 1.3 Professor CLI Tool (`grade_submissions.py`) 🆕
**Command-Line Interface for Batch Grading:**
```bash
# Basic usage
python grade_submissions.py --input student_proofs/ --rubric midterm_rubric.json

# With options
python grade_submissions.py \
  --input student_proofs/ \
  --rubric midterm_rubric.json \
  --output grades.csv \
  --feedback individual \
  --export-format canvas
```

### 1.4 Sample Rubrics Library 🆕
Create standard rubrics for common proof types:
- `rubrics/calculus_limits.json`
- `rubrics/linear_algebra_proofs.json`
- `rubrics/discrete_math_induction.json`
- `rubrics/real_analysis_convergence.json`

---

## 📅 PHASE 2: Educational Feedback System (Week 2)

### 2.1 Feedback Generator Enhancement 🔧
**Transform Current Output → Educational Feedback**

Current grading prompt:
```
"Grade this proof from 0-10 with reasoning"
```

Enhanced educational prompt:
```
"As a mathematics professor, grade this proof using the rubric provided.
For each error, explain why it's wrong and suggest the correct approach.
Highlight what the student did well.
Provide specific suggestions for improvement."
```

### 2.2 Class Analytics Tool (`class_analyzer.py`) 🆕
```python
Features:
- Aggregate performance by question
- Identify common mistakes across class
- Generate distribution curves
- Suggest topics needing review
- Export teaching insights report
```

### 2.3 Student Feedback Reports 🆕
Generate individual PDF/HTML reports with:
- Score breakdown by rubric category
- Specific errors and corrections
- Step-by-step guidance
- Similar practice problems
- Progress tracking over semester

---

## 📅 PHASE 3: Integration & Professor Dashboard (Week 3)

### 3.1 Web Interface (Optional - Streamlit) 🆕
Simple dashboard for professors:
```python
# streamlit_app.py
- Upload student submissions
- Select/customize rubric
- Review AI grades with ability to override
- Bulk approve or request manual review
- Export to LMS
```

### 3.2 Grade Validation System 🆕
- Compare AI grades with historical professor grades
- Confidence scoring (flag low-confidence for review)
- Calibration tools to match professor's grading style
- A/B testing framework for accuracy measurement

### 3.3 LMS Integration 🆕
Export formats for:
- Canvas (CSV with specific columns)
- Blackboard (XML format)
- Gradescope (JSON API)
- Generic CSV for Excel

---

## 📊 Metrics & Success Criteria

### Current Performance Baselines (Already Measured):
- **Grading Time**: 8-16 seconds per proof
- **Cost**: $0.10-0.50 per submission (3 LLMs)
- **Grader Agreement**: 80%+ consensus rate
- **Reliability**: std deviation < 1.0 on 0-10 scale

### Target Metrics for Grading Tool:
- **Professor Time Saved**: 70% reduction (30 min → 9 min for 30 proofs)
- **Grading Consistency**: <5% variance on similar solutions
- **Student Satisfaction**: Detailed feedback within 24 hours
- **Accuracy**: 90%+ agreement with professor grades
- **Adoption**: 3 professors using by end of semester

---

## 🚀 Immediate Action Items (This Week)

### Day 1-2: Foundation
1. ✅ Update ROADMAP.md (THIS DOCUMENT)
2. ⏳ Create `professor_grader.py` with rubric system
3. ⏳ Design JSON rubric format and create 2 samples

### Day 3-4: Enhancement  
4. ⏳ Modify `grade_responses.py` prompts for educational feedback
5. ⏳ Build `grade_submissions.py` CLI tool
6. ⏳ Create sample student submissions for testing

### Day 5: Demo Preparation
7. ⏳ Grade 5 sample proofs with full feedback
8. ⏳ Generate class performance summary
9. ⏳ Prepare presentation for mentor showing grading workflow

---

## 💡 Technical Architecture

```
┌──────────────────┐     ┌──────────────────┐
│ Student          │     │ Professor        │
│ Submissions      │     │ Rubric          │
│ (PDF/Text)       │     │ (JSON)          │
└────────┬─────────┘     └────────┬─────────┘
         │                         │
         ↓                         ↓
    ┌────────────────────────────────┐
    │     professor_grader.py        │
    │   (Main Grading Interface)     │
    └────────────┬───────────────────┘
                 ↓
    ┌────────────────────────────────┐
    │    grade_responses.py          │
    │  (3-LLM Consensus Grading)     │
    └────────────┬───────────────────┘
                 ↓
    ┌────────────────────────────────┐
    │    GPT-4o + Claude + Gemini    │
    │    (Parallel Grading)          │
    └────────────┬───────────────────┘
                 ↓
    ┌────────────────────────────────┐
    │   Consensus Score + Feedback   │
    └────────────┬───────────────────┘
                 ↓
         ┌───────┴───────┐
         ↓               ↓
    ┌──────────┐   ┌──────────┐
    │Individual│   │  Class   │
    │ Reports  │   │Analytics │
    └──────────┘   └──────────┘
```

---

## 📁 Repository Structure

```
Proof-GPT/
├── 📄 Core Grading System (Existing)
│   ├── grade_responses.py          ✅ 3-LLM consensus grading
│   ├── single_llm_solver.py        ✅ Individual model testing
│   └── pdf_exam_parser.py          ✅ PDF question extraction
│
├── 📄 Professor Tools (New)
│   ├── professor_grader.py         🆕 Main grading interface
│   ├── grade_submissions.py        🆕 CLI batch grading tool
│   └── class_analyzer.py           🆕 Class performance analytics
│
├── 📁 rubrics/                      🆕 Grading rubrics library
│   ├── calculus_limits.json
│   └── discrete_math_induction.json
│
├── 📁 results/
│   ├── baseline_research/          ✅ Existing research data
│   └── grading_outputs/            🆕 Professor grading results
│
├── 📓 visualize_baseline_results.ipynb  ✅ Analysis notebook
├── 📄 ROADMAP.md                    ✅ This document
└── 📄 .env                          ✅ API keys configured
```

---

## 🎯 Why This Pivot Will Succeed

### Leveraging Existing Work:
- **90% of code remains useful** - grading system is core functionality
- **Multi-LLM consensus** provides reliable, unbiased grading
- **Cost analysis** helps departments budget for AI grading
- **PDF parsing** handles real exam formats professors use

### Addressing Real Needs:
- **Professors**: Save hours on grading while maintaining quality
- **Students**: Get detailed feedback quickly
- **Departments**: Consistent grading across sections
- **TAs**: Focus on helping students rather than grading

### Competitive Advantages:
1. **3-LLM consensus** → More reliable than single AI graders
2. **Customizable rubrics** → Professors maintain control
3. **Educational feedback** → Students learn from mistakes
4. **Cost-effective** → <$0.50 per submission

---

## 📝 Next Meeting with Mentor

### Demo Script:
1. Show current 3-LLM grading system working
2. Present new rubric-based scoring
3. Grade 3 sample student proofs live
4. Show generated feedback quality
5. Display class analytics
6. Export grades to CSV

### Discussion Points:
- Validate rubric categories with teaching experience
- Get sample proofs from real courses
- Discuss integration with university systems
- Plan pilot testing with 1-2 professors

---

## 🚦 Risk Mitigation

### Technical Risks:
- **AI Hallucination**: Mitigated by 3-LLM consensus
- **Grading Errors**: Flagging system for manual review
- **Cost Overruns**: Configurable LLM usage limits

### Adoption Risks:
- **Professor Trust**: Start with low-stakes assignments
- **Student Acceptance**: Transparent about AI usage
- **Department Approval**: Emphasize human oversight

---

## 📅 Long-term Vision (Next Semester)

### Phase 4: Advanced Features
- Real-time grading during exams
- Plagiarism detection across submissions
- Adaptive feedback based on student level
- Integration with online homework systems

### Phase 5: Research Applications
- Publish paper on AI grading effectiveness
- Dataset of graded proofs for ML research
- Open-source components for educational use

---

## ✅ Summary

**What We Have**: A working multi-LLM system that can grade proofs with consensus scoring

**What We're Building**: A professor-friendly tool that saves time while providing educational value

**Timeline**: 3 weeks to working prototype, pilot testing next semester

**Success Metric**: 3 professors actively using the tool by end of term

---

*Last Updated: [Today's Date]*  
*Project Lead: [Your Name]*  
*Mentor: [Mentor's Name]*
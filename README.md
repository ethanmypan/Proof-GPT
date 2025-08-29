# Proof-GPT

A PDF exam solution parser that extracts questions and answers from academic exam PDFs using the Anthropic Claude API.

## Prerequisites

### System Dependencies

**macOS:**
```bash
brew install poppler
```

**Ubuntu/Debian:**
```bash
sudo apt-get install poppler-utils
```

**Windows:**
Download and install poppler from: https://github.com/oschwartz10612/poppler-windows/releases/
Add the `bin` folder to your PATH.

### Python Dependencies

```bash
pip install -r requirements.txt
```

## Setup

1. Create a `.env` file with your Anthropic API key:
```
ANTHROPIC_API_KEY=your-key-here
```

2. Verify installation:
```bash
python verify_setup.py
```

## Usage

Process a single PDF:
```bash
python pdf_exam_parser.py path/to/exam.pdf -o output.json
```

Process entire directory recursively:
```bash
python pdf_exam_parser.py solutions/ -o exam_questions.json
```

### Options

- `--no-vision` - Disable vision API, use text extraction only
- `--workers N` - Set number of parallel workers (default: 3)
- `--model MODEL` - Specify Claude model
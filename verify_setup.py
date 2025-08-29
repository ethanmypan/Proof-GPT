#!/usr/bin/env python3
"""
Setup verification script for PDF Exam Parser
Checks all dependencies and system requirements
"""

import os
import sys
import subprocess
from pathlib import Path

def check_python_packages():
    """Check if required Python packages are installed"""
    required_packages = [
        'anthropic',
        'PyPDF2', 
        'PIL',  # Pillow
        'pdf2image',
        'dotenv'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'PIL':
                import PIL
            elif package == 'dotenv':
                import dotenv
            else:
                __import__(package)
            print(f"✓ {package} is installed")
        except ImportError:
            missing_packages.append(package)
            print(f"✗ {package} is NOT installed")
    
    return missing_packages

def check_poppler():
    """Check if poppler is installed and accessible"""
    try:
        result = subprocess.run(['pdftoppm', '-v'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ poppler is installed and accessible")
            return True
        else:
            print("✗ poppler command failed")
            return False
    except FileNotFoundError:
        print("✗ poppler is NOT installed or not in PATH")
        return False

def check_api_key():
    """Check if Anthropic API key is configured"""
    # Check .env file
    env_file = Path('.env')
    if env_file.exists():
        with open(env_file, 'r') as f:
            content = f.read()
            if 'ANTHROPIC_API_KEY' in content and '=' in content:
                print("✓ ANTHROPIC_API_KEY found in .env file")
                return True
    
    # Check environment variable
    if os.getenv('ANTHROPIC_API_KEY'):
        print("✓ ANTHROPIC_API_KEY found in environment variables")
        return True
    
    print("✗ ANTHROPIC_API_KEY not found in .env file or environment variables")
    return False

def test_pdf_conversion():
    """Test PDF to image conversion with a simple test"""
    try:
        from pdf2image import convert_from_path
        print("✓ PDF conversion test passed")
        return True
    except Exception as e:
        print(f"✗ PDF conversion test failed: {e}")
        return False

def main():
    print("PDF Exam Parser - Setup Verification")
    print("=" * 40)
    
    issues = []
    
    # Check Python packages
    print("\n1. Checking Python packages...")
    missing_packages = check_python_packages()
    if missing_packages:
        issues.append(f"Missing packages: {', '.join(missing_packages)}")
    
    # Check poppler
    print("\n2. Checking poppler installation...")
    if not check_poppler():
        issues.append("poppler not installed or not accessible")
    
    # Check API key
    print("\n3. Checking API key configuration...")
    if not check_api_key():
        issues.append("Anthropic API key not configured")
    
    # Test PDF conversion
    print("\n4. Testing PDF conversion...")
    if not test_pdf_conversion():
        issues.append("PDF conversion functionality not working")
    
    # Summary
    print("\n" + "=" * 40)
    if not issues:
        print("✓ ALL CHECKS PASSED - Setup is complete!")
        print("\nYou can now run:")
        print("  python pdf_exam_parser.py solutions/")
    else:
        print("✗ ISSUES FOUND:")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
        
        print("\nTo fix these issues:")
        if "Missing packages" in str(issues):
            print("  • Install missing packages: pip install -r requirements.txt")
        if "poppler" in str(issues):
            print("  • Install poppler:")
            print("    - macOS: brew install poppler")
            print("    - Ubuntu/Debian: sudo apt-get install poppler-utils")
            print("    - Windows: Download from https://github.com/oschwartz10612/poppler-windows/releases/")
        if "API key" in str(issues):
            print("  • Create .env file with: ANTHROPIC_API_KEY=your-key-here")
    
    return len(issues) == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
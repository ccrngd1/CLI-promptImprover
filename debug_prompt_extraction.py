#!/usr/bin/env python3
"""
Debug script to test prompt extraction from LLM responses.
"""

import re
import sys
import os

# Add the current directory to the path so we can import modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_prompt_extraction():
    """Test prompt extraction with various LLM response formats."""
    
    print("🔍 Testing Prompt Extraction from LLM Responses")
    print("=" * 50)
    
    # Test case 1: Multi-line prompt with proper markers
    test_response_1 = """
SYNTHESIS REASONING:
Based on the agent analysis, I've improved the prompt structure and clarity.

ORCHESTRATION DECISIONS:
1. Added clear structure with sections
2. Included specific examples
3. Enhanced clarity for target audience

OPTIMIZED PROMPT:
# Academic Paper Summarization Request

Please provide a comprehensive summary of the following academic paper. Your summary should include:

## Key Components:
1. **Main Research Question**: What problem does this paper address?
2. **Methodology**: How did the researchers approach the problem?
3. **Key Findings**: What are the most important results?
4. **Implications**: What do these findings mean for the field?

## Format Requirements:
- Use clear, accessible language
- Limit to 300-500 words
- Include relevant citations if mentioned
- Highlight any limitations or future research directions

CONFIDENCE: 0.85
This synthesis incorporates structural improvements and clarity enhancements.
"""

    # Test case 2: Single line prompt (problematic case)
    test_response_2 = """
SYNTHESIS REASONING:
Improved the prompt for better clarity.

OPTIMIZED PROMPT:
# Academic Paper Summarization Request

CONFIDENCE: 0.80
"""

    # Test case 3: No clear markers
    test_response_3 = """
Based on the analysis, here's an improved version:

Academic Paper Summarization Request

Please provide a comprehensive summary including methodology, findings, and implications.

This should work better for the intended use case.
"""

    def extract_optimized_prompt(llm_response):
        """Extract optimized prompt using the same logic as orchestration engine."""
        prompt_match = re.search(
            r'OPTIMIZED PROMPT[:\s]*\n(.*?)(?=\nCONFIDENCE|\Z)',
            llm_response,
            re.DOTALL | re.IGNORECASE
        )
        if prompt_match:
            optimized_prompt = prompt_match.group(1).strip()
            if optimized_prompt and len(optimized_prompt) > 10:
                return optimized_prompt
        return None

    def extract_refined_prompt_refiner_style(raw_response):
        """Extract using refiner agent logic."""
        # Look for refined prompt section
        refined_prompt_match = re.search(
            r'---\s*REFINED PROMPT\s*---\s*\n(.*?)\n\s*---\s*END REFINED PROMPT\s*---',
            raw_response,
            re.DOTALL | re.IGNORECASE
        )
        
        if refined_prompt_match:
            return refined_prompt_match.group(1).strip()
        
        # Fallback: look for other prompt markers
        fallback_patterns = [
            r'improved prompt[:\s]*\n(.*?)(?=\nIMPROVEMENTS|\nBEST PRACTICES|\Z)',
            r'refined version[:\s]*\n(.*?)(?=\nIMPROVEMENTS|\nBEST PRACTICES|\Z)',
            r'optimized prompt[:\s]*\n(.*?)(?=\nIMPROVEMENTS|\nBEST PRACTICES|\Z)'
        ]
        
        for pattern in fallback_patterns:
            match = re.search(pattern, raw_response, re.DOTALL | re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return None

    # Test all cases
    test_cases = [
        ("Multi-line prompt with proper markers", test_response_1),
        ("Single line prompt", test_response_2),
        ("No clear markers", test_response_3)
    ]

    for test_name, response in test_cases:
        print(f"\n📝 Testing: {test_name}")
        print("-" * 30)
        
        # Test orchestration extraction
        orchestration_result = extract_optimized_prompt(response)
        print(f"Orchestration extraction result:")
        if orchestration_result:
            print(f"  Length: {len(orchestration_result)} characters")
            print(f"  Lines: {len(orchestration_result.split(chr(10)))} lines")
            print(f"  Preview: {orchestration_result[:100]}...")
            print(f"  Full result:")
            for i, line in enumerate(orchestration_result.split('\n'), 1):
                print(f"    {i:2d}: {line}")
        else:
            print("  ❌ No prompt extracted")
        
        # Test refiner extraction
        refiner_result = extract_refined_prompt_refiner_style(response)
        print(f"\nRefiner extraction result:")
        if refiner_result:
            print(f"  Length: {len(refiner_result)} characters")
            print(f"  Lines: {len(refiner_result.split(chr(10)))} lines")
            print(f"  Preview: {refiner_result[:100]}...")
        else:
            print("  ❌ No prompt extracted")

def test_real_llm_response_format():
    """Test with a realistic LLM response that might cause issues."""
    
    print("\n\n🧪 Testing Real LLM Response Format")
    print("=" * 40)
    
    # Simulate what an actual LLM might return
    realistic_response = """I'll help you improve this prompt by analyzing the requirements and applying prompt engineering best practices.

SYNTHESIS REASONING:
The original prompt lacks structure and specific guidance. I've enhanced it by adding clear sections, specific requirements, and formatting guidelines to make it more effective for academic paper summarization tasks.

ORCHESTRATION DECISIONS:
1. Added structured format with clear sections
2. Specified word count and formatting requirements  
3. Included guidance on what to focus on
4. Made the language more precise and actionable

OPTIMIZED PROMPT:
# Academic Paper Summarization Request

Please provide a comprehensive summary of the following academic paper. Your summary should include:

## Key Components:
1. **Main Research Question**: What problem does this paper address?
2. **Methodology**: How did the researchers approach the problem?  
3. **Key Findings**: What are the most important results?
4. **Implications**: What do these findings mean for the field?

## Format Requirements:
- Use clear, accessible language
- Limit to 300-500 words
- Include relevant citations if mentioned
- Highlight any limitations or future research directions

## Quality Standards:
- Focus on the most significant contributions
- Maintain objectivity and accuracy
- Use appropriate academic tone
- Ensure logical flow between sections

CONFIDENCE: 0.88
This optimized version provides clear structure, specific guidance, and quality standards that will help generate more consistent and useful academic paper summaries."""

    def extract_with_debug(response):
        """Extract with detailed debugging."""
        print("🔍 Debugging extraction process:")
        
        # Show the raw response structure
        lines = response.split('\n')
        print(f"  Total lines: {len(lines)}")
        
        # Find the OPTIMIZED PROMPT section
        optimized_start = None
        for i, line in enumerate(lines):
            if re.match(r'OPTIMIZED PROMPT[:\s]*', line, re.IGNORECASE):
                optimized_start = i
                print(f"  Found 'OPTIMIZED PROMPT' at line {i+1}: '{line.strip()}'")
                break
        
        if optimized_start is None:
            print("  ❌ No 'OPTIMIZED PROMPT' marker found")
            return None
        
        # Find the end marker
        confidence_start = None
        for i in range(optimized_start + 1, len(lines)):
            if re.match(r'CONFIDENCE[:\s]*', lines[i], re.IGNORECASE):
                confidence_start = i
                print(f"  Found 'CONFIDENCE' at line {i+1}: '{lines[i].strip()}'")
                break
        
        if confidence_start is None:
            print("  No 'CONFIDENCE' marker found, using end of response")
            confidence_start = len(lines)
        
        # Extract the content between markers
        prompt_lines = lines[optimized_start + 1:confidence_start]
        
        # Remove empty lines at start and end
        while prompt_lines and not prompt_lines[0].strip():
            prompt_lines.pop(0)
        while prompt_lines and not prompt_lines[-1].strip():
            prompt_lines.pop()
        
        if not prompt_lines:
            print("  ❌ No content found between markers")
            return None
        
        extracted_prompt = '\n'.join(prompt_lines)
        print(f"  ✅ Extracted {len(prompt_lines)} lines, {len(extracted_prompt)} characters")
        
        return extracted_prompt

    result = extract_with_debug(realistic_response)
    
    if result:
        print(f"\n📋 Final extracted prompt:")
        print(f"Length: {len(result)} characters")
        print(f"Lines: {len(result.split(chr(10)))} lines")
        print("\nFull content:")
        for i, line in enumerate(result.split('\n'), 1):
            print(f"  {i:2d}: {line}")
    else:
        print("\n❌ Extraction failed")

if __name__ == "__main__":
    test_prompt_extraction()
    test_real_llm_response_format()
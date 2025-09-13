#!/usr/bin/env python3
"""
Test script to verify that Rich markup escaping fixes the parsing error.
"""

import sys
import os

# Add the current directory to the path so we can import modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_rich_markup_escaping():
    """Test that Rich markup escaping prevents parsing errors."""
    
    print("🧪 Testing Rich Markup Escaping Fix")
    print("=" * 40)
    
    # Import the CLI formatter
    try:
        from cli.main import CLIFormatter
    except ImportError as e:
        print(f"❌ Failed to import CLIFormatter: {e}")
        return False
    
    # Create formatter instance
    formatter = CLIFormatter()
    
    # Test cases that would cause the original error
    test_cases = [
        "[PROMPT] This is a test prompt [/PROMPT]",
        "[The final synthesized and optimized prompt]",
        "Analysis: [bold]Important[/bold] findings",
        "Results: [dim]Some dim text[/dim]",
        "Mixed content: [PROMPT]test[/PROMPT] and [bold]formatting[/bold]",
        "Normal text without brackets",
        "",
        None
    ]
    
    print("Testing escape_rich_markup function:")
    for i, test_case in enumerate(test_cases, 1):
        try:
            result = formatter.escape_rich_markup(test_case)
            print(f"  {i}. Input:  {repr(test_case)}")
            print(f"     Output: {repr(result)}")
            print(f"     ✅ Success")
        except Exception as e:
            print(f"  {i}. Input:  {repr(test_case)}")
            print(f"     ❌ Error: {e}")
            return False
        print()
    
    # Test that the escaping actually prevents Rich parsing errors
    print("Testing Rich Panel with escaped content:")
    try:
        if formatter.use_rich and formatter.console:
            # This would have caused the original error
            problematic_content = "[PROMPT] Test content [/PROMPT]"
            escaped_content = formatter.escape_rich_markup(problematic_content)
            
            # Try to create a panel with the escaped content
            from rich.panel import Panel
            panel = Panel(escaped_content, title="Test Panel")
            print("  ✅ Panel created successfully with escaped content")
            
            # Also test the formatter method
            formatter.print_panel(problematic_content, "Test Panel Title [PROMPT]")
            print("  ✅ print_panel method works with problematic content")
        else:
            print("  ℹ️  Rich not available, testing fallback mode")
            formatter.print_panel("[PROMPT] Test [/PROMPT]", "Test Title")
            print("  ✅ Fallback mode works correctly")
    except Exception as e:
        print(f"  ❌ Rich Panel test failed: {e}")
        return False
    
    print("\n🎉 All tests passed! Rich markup escaping fix is working correctly.")
    return True

def test_orchestration_tree_escaping():
    """Test that orchestration tree display handles escaped content."""
    
    print("\n🌳 Testing Orchestration Tree Escaping")
    print("=" * 40)
    
    try:
        from cli.main import CLIFormatter
    except ImportError as e:
        print(f"❌ Failed to import CLIFormatter: {e}")
        return False
    
    formatter = CLIFormatter()
    
    # Mock orchestration result with problematic content
    mock_result = {
        'agent_results': {
            'analyzer[test]': {
                'success': True,
                'confidence_score': 0.85,
                'suggestions': [
                    'Improve [PROMPT] structure',
                    'Add [bold] formatting [/bold]',
                    'Consider [dim] subtle changes [/dim]'
                ],
                'analysis': {
                    'llm_analysis': {
                        'raw_response': 'Analysis result with [PROMPT] tags and [/PROMPT] closing tags',
                        'model_used': 'claude-3-sonnet',
                        'tokens_used': 150
                    }
                }
            }
        },
        'orchestration_decisions': [
            'Decision with [PROMPT] content',
            'Another decision with [bold] formatting [/bold]'
        ],
        'conflict_resolutions': [
            {
                'description': 'Resolved conflict with [PROMPT] tags [/PROMPT]'
            }
        ],
        'processing_time': 2.5,
        'llm_orchestrator_confidence': 0.9
    }
    
    try:
        # This should not raise any Rich parsing errors
        formatter.print_orchestration_tree(mock_result)
        print("✅ Orchestration tree displayed successfully with escaped content")
        return True
    except Exception as e:
        print(f"❌ Orchestration tree test failed: {e}")
        return False

if __name__ == "__main__":
    success1 = test_rich_markup_escaping()
    success2 = test_orchestration_tree_escaping()
    
    if success1 and success2:
        print("\n🎊 All Rich markup escaping tests passed!")
        print("The CLI should no longer have the '[/PROMPT] closing tag' error.")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed. Please check the implementation.")
        sys.exit(1)
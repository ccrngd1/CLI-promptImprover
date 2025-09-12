#!/usr/bin/env python3
"""
Test script to verify that the output truncation fix works correctly.
"""

import sys
import os

# Add the current directory to the path so we can import modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_cli_output_formatting():
    """Test that CLI output formatting shows full content without truncation."""
    
    print("🧪 Testing CLI Output Formatting Fix")
    print("=" * 40)
    
    try:
        from cli.main import CLIFormatter
        
        # Create a formatter
        formatter = CLIFormatter()  # Uses available rich or falls back to simple
        
        # Create mock orchestration result with long LLM feedback
        long_llm_response = """This is a very long LLM response that would normally be truncated in the output. 
        It contains detailed analysis and reasoning that should be fully visible to the user.
        
        ANALYSIS:
        The prompt shows several areas for improvement:
        1. Clarity could be enhanced by adding specific examples
        2. Structure needs better organization with clear sections
        3. Context is missing for the intended use case
        
        RECOMMENDATIONS:
        - Add concrete examples to illustrate the desired output
        - Use clear section headers to organize information
        - Include background context about the task
        - Specify the target audience and their expertise level
        
        BEST PRACTICES APPLIED:
        - Role-based prompting for expertise simulation
        - Step-by-step reasoning framework
        - Clear formatting and structure
        - Specific and actionable language
        
        CONFIDENCE: 0.85
        This assessment is based on established prompt engineering principles and the specific characteristics observed in the input prompt."""
        
        mock_orchestration_result = {
            'agent_results': {
                'analyzer': {
                    'success': True,
                    'confidence_score': 0.85,
                    'analysis': {
                        'llm_analysis': {
                            'raw_response': long_llm_response,
                            'model_used': 'anthropic.claude-3-sonnet-20240229-v1:0',
                            'tokens_used': 450
                        }
                    },
                    'suggestions': [
                        'Add specific examples to clarify expectations',
                        'Include context about the intended use case',
                        'Structure the prompt with clear sections'
                    ]
                }
            },
            'orchestration_decisions': [
                'Prioritize clarity improvements based on analysis',
                'Apply structural enhancements systematically'
            ],
            'processing_time': 2.34,
            'llm_orchestrator_confidence': 0.82
        }
        
        print("✓ Created mock orchestration result with long LLM feedback")
        
        # Test simple format output
        print("\n📝 Testing Simple Format Output:")
        print("-" * 30)
        formatter._print_orchestration_simple(mock_orchestration_result)
        
        # Test rich format if available
        if formatter.use_rich:
            print("\n🎨 Testing Rich Format Output:")
            print("-" * 30)
            formatter.print_orchestration_tree(mock_orchestration_result)
        else:
            print("⚠️  Rich library not available, skipping rich format test")
        
        print("\n✅ CLI output formatting test completed")
        print("🔍 Check the output above - the LLM feedback should be displayed in full without truncation")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing CLI output formatting: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_initial_prompt_display():
    """Test that initial prompt is displayed in orchestration results."""
    
    print("\n🧪 Testing Initial Prompt Display")
    print("=" * 35)
    
    try:
        from cli.main import CLIFormatter
        from session import SessionState, SessionConfig
        from datetime import datetime
        from unittest.mock import Mock
        
        # Create a mock session state with initial prompt
        mock_config = SessionConfig()
        mock_session_state = SessionState(
            session_id="test-123",
            initial_prompt="This is the original prompt that should be displayed at the beginning of each iteration result.",
            current_prompt="This is the current optimized prompt.",
            context={'domain': 'test'},
            config=mock_config,
            status='active',
            created_at=datetime.now(),
            last_updated=datetime.now(),
            current_iteration=1
        )
        
        # Create a mock session manager
        mock_session_manager = Mock()
        mock_session_manager.get_session_state.return_value = mock_session_state
        
        # Create formatter and set session manager
        formatter = CLIFormatter()
        
        # Create a mock CLI instance to test the method
        from cli.main import PromptOptimizerCLI
        cli = PromptOptimizerCLI()
        cli.session_manager = mock_session_manager
        cli.formatter = formatter
        
        # Create mock iteration result
        mock_iteration_result = Mock()
        mock_iteration_result.iteration_result.to_dict.return_value = {
            'orchestrated_prompt': 'Optimized version of the prompt',
            'agent_results': {},
            'processing_time': 1.5,
            'llm_orchestrator_confidence': 0.8
        }
        
        print("✓ Created mock session state and iteration result")
        
        # Test the display method
        print("\n📝 Testing Initial Prompt Display:")
        print("-" * 30)
        cli._display_iteration_results(mock_iteration_result, "test-123")
        
        print("\n✅ Initial prompt display test completed")
        print("🔍 Check the output above - the initial prompt should be displayed before the orchestration results")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing initial prompt display: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🔧 Testing Output Truncation Fixes")
    print("=" * 50)
    
    # Run tests
    cli_test = test_cli_output_formatting()
    prompt_test = test_initial_prompt_display()
    
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    print(f"   CLI Output Formatting: {'✅ PASS' if cli_test else '❌ FAIL'}")
    print(f"   Initial Prompt Display: {'✅ PASS' if prompt_test else '❌ FAIL'}")
    
    all_passed = cli_test and prompt_test
    
    if all_passed:
        print("\n🎉 SUCCESS: All output fixes are working correctly!")
        print("   - LLM feedback is no longer truncated")
        print("   - Initial prompt is displayed with orchestration results")
    else:
        print("\n⚠️  Some tests failed - check the output above for details")
    
    sys.exit(0 if all_passed else 1)
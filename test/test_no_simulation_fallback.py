#!/usr/bin/env python3
"""
Test script to verify that all simulation methods have been removed and 
that the system properly throws errors instead of falling back to mock responses.
"""

import sys
import os
import inspect

# Add the current directory to the path so we can import modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_simulation_methods_removed():
    """Test that all simulation methods have been completely removed."""
    
    print("🚫 Testing Simulation Methods Removal")
    print("=" * 45)
    
    try:
        from agents.llm_agent import LLMAgent
        
        print("✓ Successfully imported LLMAgent")
        
        # Check that simulation methods no longer exist
        simulation_methods = [
            '_simulate_llm_response',
            '_simulate_analyzer_response', 
            '_simulate_refiner_response',
            '_simulate_validator_response'
        ]
        
        all_removed = True
        for method_name in simulation_methods:
            if hasattr(LLMAgent, method_name):
                print(f"❌ {method_name} still exists (should be removed)")
                all_removed = False
            else:
                print(f"✅ {method_name} successfully removed")
        
        # Check that _should_use_fallback method is removed
        if hasattr(LLMAgent, '_should_use_fallback'):
            print(f"❌ _should_use_fallback still exists (should be removed)")
            all_removed = False
        else:
            print(f"✅ _should_use_fallback successfully removed")
        
        # Check that _process_with_fallback_agent method is removed
        if hasattr(LLMAgent, '_process_with_fallback_agent'):
            print(f"❌ _process_with_fallback_agent still exists (should be removed)")
            all_removed = False
        else:
            print(f"✅ _process_with_fallback_agent successfully removed")
        
        # Check that new error handling method exists
        if hasattr(LLMAgent, '_handle_llm_failure'):
            print(f"✅ _handle_llm_failure method exists (good)")
        else:
            print(f"❌ _handle_llm_failure method missing")
            all_removed = False
        
        return all_removed
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_llm_agent_source_code():
    """Test that the source code doesn't contain any simulation references."""
    
    print("\n📝 Testing Source Code for Simulation References")
    print("=" * 50)
    
    try:
        import agents.llm_agent as llm_agent_module
        
        # Get the source code
        source_code = inspect.getsource(llm_agent_module)
        
        # Check for simulation patterns that shouldn't exist
        forbidden_patterns = [
            'def _simulate_',
            '_simulate_llm_response',
            '_simulate_analyzer_response',
            '_simulate_refiner_response', 
            '_simulate_validator_response',
            'Simulate an LLM response',
            'Analysis Results:',
            'Structure Assessment:',
            'Clarity Evaluation:',
            'Best Practices Applied:',
            'Refined Prompt Suggestion:',
            'Validation Results:',
            'Syntax Check: PASS'
        ]
        
        found_patterns = []
        for pattern in forbidden_patterns:
            if pattern in source_code:
                found_patterns.append(pattern)
        
        if found_patterns:
            print("❌ Found simulation patterns in source code:")
            for pattern in found_patterns:
                print(f"   - {pattern}")
            return False
        else:
            print("✅ No simulation patterns found in source code")
            return True
        
    except Exception as e:
        print(f"❌ Error checking source code: {e}")
        return False

def test_enhanced_agents_no_fallback():
    """Test that enhanced agents don't have fallback logic."""
    
    print("\n🔄 Testing Enhanced Agents for Fallback Removal")
    print("=" * 50)
    
    agent_files = [
        'agents.llm_enhanced_analyzer',
        'agents.llm_enhanced_refiner', 
        'agents.llm_enhanced_validator'
    ]
    
    all_clean = True
    
    for agent_module_name in agent_files:
        try:
            agent_module = __import__(agent_module_name, fromlist=[''])
            source_code = inspect.getsource(agent_module)
            
            # Check for fallback patterns
            fallback_patterns = [
                '_should_use_fallback',
                '_process_with_fallback_agent',
                'fallback_enabled',
                'Try fallback if enabled',
                'HeuristicAnalyzerAgent',
                'HeuristicRefinerAgent',
                'HeuristicValidatorAgent'
            ]
            
            found_patterns = []
            for pattern in fallback_patterns:
                if pattern in source_code:
                    found_patterns.append(pattern)
            
            if found_patterns:
                print(f"❌ {agent_module_name} still has fallback patterns:")
                for pattern in found_patterns:
                    print(f"   - {pattern}")
                all_clean = False
            else:
                print(f"✅ {agent_module_name} is clean of fallback logic")
                
        except Exception as e:
            print(f"❌ Error checking {agent_module_name}: {e}")
            all_clean = False
    
    return all_clean

def test_error_handling_behavior():
    """Test that the system properly raises errors instead of using fallbacks."""
    
    print("\n⚠️  Testing Error Handling Behavior")
    print("=" * 40)
    
    try:
        from agents.llm_enhanced_analyzer import LLMAnalyzerAgent
        from unittest.mock import Mock, patch
        
        # Create a mock config
        mock_config = {
            'llm_model': 'test-model',
            'llm_temperature': 0.3,
            'llm_max_tokens': 1000,
            'bedrock': {'region': 'us-east-1'}
        }
        
        # Mock the Bedrock executor to simulate failure
        with patch('agents.llm_agent.BedrockExecutor') as mock_bedrock:
            mock_executor = Mock()
            mock_executor.execute_prompt.side_effect = Exception("Simulated Bedrock failure")
            mock_bedrock.return_value = mock_executor
            
            # Create agent
            agent = LLMAnalyzerAgent(mock_config)
            
            # Test that it raises an exception instead of falling back
            test_prompt = "Test prompt for error handling"
            
            try:
                result = agent.process(test_prompt)
                print("❌ Agent should have raised an exception but didn't")
                return False
            except Exception as e:
                if "LLM service failure" in str(e) or "Bedrock" in str(e):
                    print("✅ Agent properly raises exception on LLM failure")
                    print(f"   Exception: {str(e)[:100]}...")
                    return True
                else:
                    print(f"❌ Unexpected exception type: {e}")
                    return False
        
    except Exception as e:
        print(f"❌ Error testing error handling: {e}")
        return False

if __name__ == "__main__":
    print("🚫 No Simulation Fallback Verification")
    print("=" * 60)
    
    # Run all tests
    methods_test = test_simulation_methods_removed()
    source_test = test_llm_agent_source_code()
    agents_test = test_enhanced_agents_no_fallback()
    error_test = test_error_handling_behavior()
    
    print("\n" + "=" * 60)
    print("📊 Test Results Summary:")
    print(f"   Simulation Methods Removed: {'✅ PASS' if methods_test else '❌ FAIL'}")
    print(f"   Source Code Clean: {'✅ PASS' if source_test else '❌ FAIL'}")
    print(f"   Enhanced Agents Clean: {'✅ PASS' if agents_test else '❌ FAIL'}")
    print(f"   Error Handling: {'✅ PASS' if error_test else '❌ FAIL'}")
    
    all_passed = methods_test and source_test and agents_test and error_test
    
    if all_passed:
        print("\n🎉 SUCCESS: All simulation methods removed!")
        print("   - No more mock LLM responses")
        print("   - System properly throws errors on LLM failure")
        print("   - No fallback to heuristic agents")
        print("   - Clean error handling with detailed messages")
    else:
        print("\n❌ Some simulation code still exists - check the issues above")
    
    sys.exit(0 if all_passed else 1)
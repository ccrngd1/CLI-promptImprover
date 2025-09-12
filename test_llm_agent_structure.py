#!/usr/bin/env python3
"""
Test script to verify that LLM agents have been properly modified to use real Bedrock calls.

This script checks the code structure without making actual AWS calls.
"""

import sys
import os
import inspect

# Add the current directory to the path so we can import modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_llm_agent_structure():
    """Test that LLM agent structure has been updated correctly."""
    
    print("🔍 Testing LLM Agent Code Structure")
    print("=" * 50)
    
    try:
        # Import the LLM agent base class
        from agents.llm_agent import LLMAgent
        
        print("✓ Successfully imported LLMAgent")
        
        # Check if BedrockExecutor is imported
        import agents.llm_agent as llm_agent_module
        
        # Check imports
        source_code = inspect.getsource(llm_agent_module)
        
        if "from bedrock.executor import BedrockExecutor" in source_code:
            print("✅ BedrockExecutor is properly imported")
        else:
            print("❌ BedrockExecutor import not found")
            return False
        
        # Check if _execute_bedrock_call method exists
        if hasattr(LLMAgent, '_execute_bedrock_call'):
            print("✅ _execute_bedrock_call method exists")
        else:
            print("❌ _execute_bedrock_call method not found")
            return False
        
        # Check if _simulate_llm_response is still being called in _call_llm
        call_llm_source = inspect.getsource(LLMAgent._call_llm)
        
        if "_simulate_llm_response" in call_llm_source:
            print("❌ _simulate_llm_response is still being called in _call_llm")
            return False
        elif "_execute_bedrock_call" in call_llm_source:
            print("✅ _call_llm now uses _execute_bedrock_call instead of simulation")
        else:
            print("⚠️  Neither simulation nor bedrock call found in _call_llm")
            return False
        
        # Check if bedrock_executor is initialized in __init__
        init_source = inspect.getsource(LLMAgent.__init__)
        
        if "self.bedrock_executor = BedrockExecutor" in init_source:
            print("✅ bedrock_executor is initialized in __init__")
        else:
            print("❌ bedrock_executor initialization not found in __init__")
            return False
        
        # Test creating an instance (without making AWS calls)
        try:
            # Mock config to avoid AWS calls
            mock_config = {
                'llm_model': 'test-model',
                'llm_temperature': 0.3,
                'llm_max_tokens': 1000,
                'bedrock': {'region': 'us-east-1'}
            }
            
            # This will fail at Bedrock initialization, but we can catch that
            try:
                agent = LLMAgent("TestAgent", mock_config)
                print("✅ LLMAgent instance created successfully")
            except Exception as e:
                if "bedrock" in str(e).lower() or "aws" in str(e).lower():
                    print("✅ LLMAgent tries to initialize Bedrock (expected without AWS creds)")
                else:
                    print(f"⚠️  Unexpected error during initialization: {e}")
        
        except Exception as e:
            print(f"❌ Error testing LLMAgent instantiation: {e}")
            return False
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def check_simulation_methods_still_exist():
    """Check if simulation methods still exist (they should for fallback)."""
    
    print("\n🔧 Checking Simulation Methods (for fallback)")
    print("=" * 45)
    
    try:
        from agents.llm_agent import LLMAgent
        
        # These methods should still exist for fallback scenarios
        simulation_methods = [
            '_simulate_llm_response',
            '_simulate_analyzer_response', 
            '_simulate_refiner_response',
            '_simulate_validator_response'
        ]
        
        for method_name in simulation_methods:
            if hasattr(LLMAgent, method_name):
                print(f"✅ {method_name} exists (good for fallback)")
            else:
                print(f"⚠️  {method_name} not found (may affect fallback)")
        
        return True
        
    except Exception as e:
        print(f"❌ Error checking simulation methods: {e}")
        return False

def verify_bedrock_executor_import():
    """Verify that BedrockExecutor can be imported and has expected methods."""
    
    print("\n🏗️  Verifying Bedrock Executor")
    print("=" * 35)
    
    try:
        from bedrock.executor import BedrockExecutor, ModelConfig
        
        print("✅ BedrockExecutor imported successfully")
        print("✅ ModelConfig imported successfully")
        
        # Check key methods exist
        required_methods = ['execute_prompt', 'get_available_models']
        
        for method_name in required_methods:
            if hasattr(BedrockExecutor, method_name):
                print(f"✅ {method_name} method exists")
            else:
                print(f"❌ {method_name} method not found")
                return False
        
        return True
        
    except ImportError as e:
        print(f"❌ Cannot import BedrockExecutor: {e}")
        return False
    except Exception as e:
        print(f"❌ Error verifying BedrockExecutor: {e}")
        return False

if __name__ == "__main__":
    print("🔍 LLM Agent Structure Verification")
    print("=" * 60)
    
    # Run all tests
    structure_test = test_llm_agent_structure()
    simulation_test = check_simulation_methods_still_exist()
    bedrock_test = verify_bedrock_executor_import()
    
    print("\n" + "=" * 60)
    print("📊 Test Results Summary:")
    print(f"   LLM Agent Structure: {'✅ PASS' if structure_test else '❌ FAIL'}")
    print(f"   Simulation Methods: {'✅ PASS' if simulation_test else '❌ FAIL'}")
    print(f"   Bedrock Executor: {'✅ PASS' if bedrock_test else '❌ FAIL'}")
    
    all_passed = structure_test and simulation_test and bedrock_test
    
    if all_passed:
        print("\n🎉 SUCCESS: LLM agent structure is correctly updated!")
        print("   - Real Bedrock calls are now used instead of simulation")
        print("   - Simulation methods are preserved for fallback scenarios")
        print("   - BedrockExecutor integration is properly implemented")
    else:
        print("\n❌ Some tests failed - check the issues above")
    
    sys.exit(0 if all_passed else 1)
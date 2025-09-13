#!/usr/bin/env python3
"""
Test script to verify that LLM agents are making real Bedrock calls instead of simulations.

This script tests the LLMAnalyzerAgent to ensure it's using actual Bedrock API calls
rather than the previous simulation methods.
"""

import sys
import os
from typing import Dict, Any

# Add the current directory to the path so we can import modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_llm_agent_real_calls():
    """Test that LLM agents are making real Bedrock calls."""
    
    print("🧪 Testing LLM Agent Real Bedrock Calls")
    print("=" * 50)
    
    try:
        # Import the LLM enhanced analyzer
        from agents.llm_enhanced_analyzer import LLMAnalyzerAgent
        from config_loader import load_config
        
        print("✓ Successfully imported LLMAnalyzerAgent")
        
        # Load configuration
        config = load_config()
        print("✓ Configuration loaded successfully")
        
        # Create agent instance
        agent = LLMAnalyzerAgent(config)
        print("✓ LLMAnalyzerAgent instance created")
        
        # Check if Bedrock executor is initialized
        if hasattr(agent, 'bedrock_executor'):
            print("✓ Bedrock executor is initialized")
        else:
            print("✗ Bedrock executor is NOT initialized")
            return False
        
        # Test prompt for analysis
        test_prompt = """
        Analyze the following data and provide insights.
        Make sure to be thorough in your analysis.
        """
        
        print(f"\n📝 Testing with prompt: {test_prompt.strip()}")
        print("-" * 50)
        
        # Test context
        context = {
            'intended_use': 'data analysis',
            'target_audience': 'data scientists',
            'domain': 'analytics'
        }
        
        # Process the prompt
        print("🚀 Processing prompt with LLM agent...")
        result = agent.process(test_prompt, context)
        
        # Check results
        if result.success:
            print("✅ LLM agent processing successful!")
            print(f"   Agent: {result.agent_name}")
            print(f"   Confidence: {result.confidence_score:.2f}")
            print(f"   Suggestions count: {len(result.suggestions)}")
            
            # Check if the response contains real analysis (not simulation patterns)
            analysis = result.analysis
            if 'llm_analysis' in analysis:
                llm_data = analysis['llm_analysis']
                response_text = llm_data.get('raw_response', '')
                
                # Check for simulation patterns that shouldn't be there anymore
                simulation_indicators = [
                    "Analysis Results:",
                    "Structure Assessment:",
                    "Clarity Evaluation:",
                    "Best Practices Applied:"
                ]
                
                has_simulation_patterns = any(indicator in response_text for indicator in simulation_indicators)
                
                if has_simulation_patterns:
                    print("⚠️  WARNING: Response contains simulation patterns - may still be using simulation")
                    print(f"   Response preview: {response_text[:200]}...")
                else:
                    print("✅ Response appears to be from real LLM (no simulation patterns detected)")
                    print(f"   Model used: {llm_data.get('model_used', 'unknown')}")
                    print(f"   Tokens used: {llm_data.get('tokens_used', 'unknown')}")
                
                return not has_simulation_patterns
            else:
                print("⚠️  No LLM analysis data found in result")
                return False
        else:
            print(f"❌ LLM agent processing failed: {result.error_message}")
            
            # Check if it's an authentication or configuration issue
            if "authentication" in result.error_message.lower() or "access" in result.error_message.lower():
                print("💡 This might be an AWS authentication issue. Make sure you have:")
                print("   - Valid AWS credentials configured")
                print("   - Bedrock access permissions")
                print("   - The correct region configured")
            
            return False
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure all required modules are available")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        print(f"   Error type: {type(e).__name__}")
        return False

def test_bedrock_executor_directly():
    """Test the Bedrock executor directly to verify it's working."""
    
    print("\n🔧 Testing Bedrock Executor Directly")
    print("=" * 40)
    
    try:
        from bedrock.executor import BedrockExecutor, ModelConfig
        
        # Create executor
        executor = BedrockExecutor(region_name='us-east-1')
        print("✓ Bedrock executor created")
        
        # Create model config
        model_config = ModelConfig(
            model_id="us.anthropic.claude-3-7-sonnet-20250219-v1:0",
            max_tokens=100,
            temperature=0.3
        )
        print("✓ Model config created")
        
        # Test prompt
        test_prompt = "Hello! Please respond with 'Real LLM Response' to confirm you're working."
        
        print(f"📝 Testing with prompt: {test_prompt}")
        print("🚀 Executing Bedrock call...")
        
        # Execute
        result = executor.execute_prompt(test_prompt, model_config)
        
        if result.success:
            print("✅ Bedrock executor test successful!")
            print(f"   Model: {result.model_name}")
            print(f"   Response: {result.response_text[:100]}...")
            print(f"   Execution time: {result.execution_time:.2f}s")
            print(f"   Tokens: {result.token_usage}")
            return True
        else:
            print(f"❌ Bedrock executor test failed: {result.error_message}")
            return False
            
    except Exception as e:
        print(f"❌ Bedrock executor test error: {e}")
        return False

if __name__ == "__main__":
    print("🔍 LLM Agent Real Calls Verification")
    print("=" * 60)
    
    # Test Bedrock executor first
    bedrock_success = test_bedrock_executor_directly()
    
    # Test LLM agent
    agent_success = test_llm_agent_real_calls()
    
    print("\n" + "=" * 60)
    print("📊 Test Results Summary:")
    print(f"   Bedrock Executor: {'✅ PASS' if bedrock_success else '❌ FAIL'}")
    print(f"   LLM Agent: {'✅ PASS' if agent_success else '❌ FAIL'}")
    
    if bedrock_success and agent_success:
        print("\n🎉 SUCCESS: LLM agents are now using real Bedrock calls!")
    elif bedrock_success and not agent_success:
        print("\n⚠️  Bedrock works but LLM agent integration has issues")
    elif not bedrock_success:
        print("\n❌ Bedrock executor is not working - check AWS configuration")
    
    sys.exit(0 if (bedrock_success and agent_success) else 1)
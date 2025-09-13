#!/usr/bin/env python3
"""
Test script to verify Bedrock model access with updated configuration.
"""

import json
import sys
from bedrock.executor import BedrockExecutor, ModelConfig

def test_model_access():
    """Test access to the configured Bedrock models."""
    
    # Load config
    try:
        with open('config.json', 'r') as f:
            config = json.load(f)
    except Exception as e:
        print(f"Error loading config: {e}")
        return False
    
    # Initialize executor
    try:
        executor = BedrockExecutor(region_name=config['bedrock']['region'])
        print("✓ Bedrock executor initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize Bedrock executor: {e}")
        return False
    
    # Test model access
    test_model = config['bedrock']['default_model']
    print(f"Testing model: {test_model}")
    
    # Also test Claude 3.5 Sonnet as fallback
    fallback_model = "anthropic.claude-3-5-sonnet-20241022-v2:0"
    
    try:
        model_config = ModelConfig(
            model_id=test_model,
            max_tokens=100,
            temperature=0.1
        )
        
        result = executor.execute_prompt(
            "Hello, this is a test prompt. Please respond with 'Test successful'.",
            model_config
        )
        
        if result.success:
            print("✓ Model access test successful!")
            print(f"Response: {result.response_text[:100]}...")
            return True
        else:
            print(f"✗ Model access test failed: {result.error_message}")
            
            # Try fallback model
            print(f"\nTrying fallback model: {fallback_model}")
            model_config.model_id = fallback_model
            
            result = executor.execute_prompt(
                "Hello, this is a test prompt. Please respond with 'Test successful'.",
                model_config
            )
            
            if result.success:
                print("✓ Fallback model access test successful!")
                print(f"Response: {result.response_text[:100]}...")
                print(f"\nRecommendation: Update your config to use {fallback_model}")
                return True
            else:
                print(f"✗ Fallback model also failed: {result.error_message}")
                return False
            
    except Exception as e:
        print(f"✗ Unexpected error during model test: {e}")
        return False

if __name__ == "__main__":
    success = test_model_access()
    sys.exit(0 if success else 1)
#!/usr/bin/env python3
"""
Test multiple Claude models to find one that works.
"""

from bedrock.executor import BedrockExecutor, ModelConfig

def test_models():
    """Test multiple Claude models to find working ones."""
    
    # Models to test (in order of preference)
    models_to_test = [
        "anthropic.claude-3-5-haiku-20241022-v1:0",
        "anthropic.claude-3-haiku-20240307-v1:0", 
        "anthropic.claude-3-sonnet-20240229-v1:0",
        "anthropic.claude-instant-v1:2:100k",
        "anthropic.claude-v2:1:200k"
    ]
    
    executor = BedrockExecutor(region_name="us-east-1")
    print("✓ Bedrock executor initialized successfully\n")
    
    working_models = []
    
    for model_id in models_to_test:
        print(f"Testing: {model_id}")
        
        try:
            model_config = ModelConfig(
                model_id=model_id,
                max_tokens=50,
                temperature=0.1
            )
            
            result = executor.execute_prompt(
                "Hello, respond with just 'Working'",
                model_config
            )
            
            if result.success:
                print(f"  ✓ SUCCESS - Response: {result.response_text.strip()}")
                working_models.append(model_id)
            else:
                print(f"  ✗ FAILED - {result.error_message}")
                
        except Exception as e:
            print(f"  ✗ ERROR - {e}")
        
        print()
    
    if working_models:
        print("=" * 60)
        print("WORKING MODELS:")
        for model in working_models:
            print(f"  {model}")
        print(f"\nRecommendation: Use {working_models[0]} in your config")
    else:
        print("No working models found!")

if __name__ == "__main__":
    test_models()
#!/usr/bin/env python3
"""
Simple integration test to verify the BedrockExecutor can be imported and instantiated.
"""

def test_basic_import():
    """Test that we can import and create basic objects."""
    from bedrock.executor import BedrockExecutor, ModelConfig
    from models import ExecutionResult
    
    # Test ModelConfig creation
    config = ModelConfig(
        model_id="anthropic.claude-3-sonnet-20240229-v1:0",
        max_tokens=100,
        temperature=0.7
    )
    
    assert config.model_id == "anthropic.claude-3-sonnet-20240229-v1:0"
    assert config.max_tokens == 100
    assert config.temperature == 0.7
    
    # Test ExecutionResult creation
    result = ExecutionResult(
        model_name="test-model",
        response_text="test response",
        execution_time=1.0,
        token_usage={'input_tokens': 10, 'output_tokens': 5},
        success=True
    )
    
    assert result.model_name == "test-model"
    assert result.success is True
    assert result.validate() is True
    
    print("✅ All basic imports and object creation tests passed!")

if __name__ == "__main__":
    test_basic_import()
#!/usr/bin/env python3
"""
Simple test script for Bedrock Converse API.

This script demonstrates basic usage of the Bedrock Converse API
with a simple conversation test.
"""

import boto3
import json
from botocore.exceptions import ClientError


def test_bedrock_converse():
    """Test the Bedrock Converse API with a simple prompt."""
    
    # Initialize Bedrock client
    try:
        bedrock_client = boto3.client('bedrock-runtime', region_name='us-east-1')
        print("✓ Bedrock client initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize Bedrock client: {e}")
        return False
    
    # Test configuration
    model_id = "us.anthropic.claude-sonnet-4-20250514-v1:0"
    test_prompt = "Hello! Can you tell me a short joke about programming?"
    
    # Prepare the conversation
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "text": test_prompt
                }
            ]
        }
    ]
    
    # Inference configuration
    inference_config = {
        "maxTokens": 500,
        "temperature": 0.7,
        "topP": 0.9
    }
    
    print(f"\nTesting with model: {model_id}")
    print(f"Prompt: {test_prompt}")
    print("-" * 60)
    
    try:
        # Call the Converse API
        response = bedrock_client.converse(
            modelId=model_id,
            messages=messages,
            inferenceConfig=inference_config
        )
        
        # Extract the response
        output_message = response['output']['message']
        response_text = output_message['content'][0]['text']
        
        # Display results
        print("✓ API call successful!")
        print(f"\nResponse:")
        print(response_text)
        
        # Show usage statistics
        usage = response.get('usage', {})
        if usage:
            print(f"\nToken Usage:")
            print(f"  Input tokens: {usage.get('inputTokens', 'N/A')}")
            print(f"  Output tokens: {usage.get('outputTokens', 'N/A')}")
            print(f"  Total tokens: {usage.get('totalTokens', 'N/A')}")
        
        # Show metadata
        metadata = response.get('metrics', {})
        if metadata:
            print(f"\nMetrics:")
            print(f"  Latency: {metadata.get('latencyMs', 'N/A')} ms")
        
        return True
        
    except ClientError as e:
        error_code = e.response['Error']['Code']
        error_message = e.response['Error']['Message']
        print(f"✗ Bedrock API error: {error_code}")
        print(f"  Message: {error_message}")
        
        # Provide helpful error guidance
        if error_code == 'AccessDeniedException':
            print("\n💡 Tip: Make sure you have proper AWS credentials and Bedrock permissions")
        elif error_code == 'ValidationException':
            print("\n💡 Tip: Check if the model ID is correct and available in your region")
        elif error_code == 'ThrottlingException':
            print("\n💡 Tip: You're being rate limited. Try again in a moment")
        
        return False
        
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Bedrock Converse API Test")
    print("="*40)
    
    # Run basic test
    success = test_bedrock_converse()
     
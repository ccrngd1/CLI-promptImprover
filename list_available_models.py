#!/usr/bin/env python3
"""
List available Bedrock models to see what we can actually use.
"""

import json
from bedrock.executor import BedrockExecutor

def list_models():
    """List all available Bedrock models."""
    
    try:
        executor = BedrockExecutor(region_name="us-east-1")
        print("✓ Bedrock executor initialized successfully")
        
        models = executor.get_available_models()
        
        if not models:
            print("No models returned - this might indicate permission issues")
            return
        
        print(f"\nFound {len(models)} available models:")
        print("-" * 80)
        
        claude_models = []
        other_models = []
        
        for model in models:
            model_id = model['model_id']
            provider = model['provider_name']
            name = model['model_name']
            
            if 'claude' in model_id.lower():
                claude_models.append(f"  {model_id} ({provider} - {name})")
            else:
                other_models.append(f"  {model_id} ({provider} - {name})")
        
        if claude_models:
            print("\nClaude Models:")
            for model in sorted(claude_models):
                print(model)
        
        if other_models:
            print("\nOther Models:")
            for model in sorted(other_models):
                print(model)
                
    except Exception as e:
        print(f"Error listing models: {e}")

if __name__ == "__main__":
    list_models()
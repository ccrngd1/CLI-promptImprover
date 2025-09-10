#!/usr/bin/env python3
"""
Integration test for configuration loading and runtime changes.

This script tests the complete configuration system including:
- Configuration loading
- Runtime configuration changes
- Orchestration engine configuration handling
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock

from config_loader import ConfigurationLoader, update_config_runtime, is_llm_only_mode
from orchestration.engine import LLMOrchestrationEngine
from bedrock.executor import BedrockExecutor
from evaluation.evaluator import Evaluator


def test_config_integration():
    """Test configuration integration with orchestration engine."""
    print("🧪 Testing Configuration Integration")
    print("=" * 50)
    
    # Create temporary config file
    temp_dir = tempfile.mkdtemp()
    config_file = Path(temp_dir) / 'test_config.json'
    
    # Sample configuration
    sample_config = {
        'bedrock': {
            'region': 'us-east-1',
            'default_model': 'anthropic.claude-3-sonnet-20240229-v1:0',
            'timeout': 30,
            'max_retries': 3
        },
        'orchestration': {
            'orchestrator_model': 'anthropic.claude-3-sonnet-20240229-v1:0',
            'orchestrator_temperature': 0.3,
            'orchestrator_max_tokens': 2000,
            'min_iterations': 3,
            'max_iterations': 10,
            'score_improvement_threshold': 0.02,
            'stability_window': 3,
            'convergence_confidence_threshold': 0.8
        },
        'optimization': {
            'llm_only_mode': False,
            'fallback_to_heuristic': True
        },
        'agents': {
            'analyzer': {
                'enabled': True,
                'model': 'anthropic.claude-3-sonnet-20240229-v1:0',
                'temperature': 0.2,
                'max_tokens': 1500
            },
            'refiner': {
                'enabled': True,
                'model': 'anthropic.claude-3-sonnet-20240229-v1:0',
                'temperature': 0.4,
                'max_tokens': 2000
            },
            'validator': {
                'enabled': True,
                'model': 'anthropic.claude-3-sonnet-20240229-v1:0',
                'temperature': 0.1,
                'max_tokens': 1000
            }
        }
    }
    
    # Write sample config to file
    with open(config_file, 'w') as f:
        json.dump(sample_config, f, indent=2)
    
    try:
        # Test 1: Configuration loading
        print("\n📋 Test 1: Configuration Loading")
        config_loader = ConfigurationLoader(str(config_file))
        config = config_loader.get_config()
        
        print(f"✅ Configuration loaded successfully")
        print(f"   - LLM-only mode: {config_loader.is_llm_only_mode()}")
        print(f"   - Fallback enabled: {config_loader.is_fallback_enabled()}")
        print(f"   - Max iterations: {config.get('orchestration', {}).get('max_iterations')}")
        
        # Test 2: Runtime configuration changes
        print("\n🔄 Test 2: Runtime Configuration Changes")
        
        # Enable LLM-only mode
        result = config_loader.update_config({'optimization.llm_only_mode': True})
        print(f"✅ LLM-only mode enabled: {result['success']}")
        print(f"   - New mode: {config_loader.is_llm_only_mode()}")
        
        # Update orchestration settings
        result = config_loader.update_config({
            'orchestration.max_iterations': 15,
            'orchestration.orchestrator_temperature': 0.5
        })
        print(f"✅ Orchestration settings updated: {result['success']}")
        print(f"   - Max iterations: {config_loader.get_config_value('orchestration.max_iterations')}")
        print(f"   - Temperature: {config_loader.get_config_value('orchestration.orchestrator_temperature')}")
        
        # Test 3: Configuration validation
        print("\n✅ Test 3: Configuration Validation")
        validation = config_loader.validate_current_config()
        print(f"✅ Configuration valid: {validation['valid']}")
        if validation['warnings']:
            print(f"   - Warnings: {len(validation['warnings'])}")
        if validation['errors']:
            print(f"   - Errors: {len(validation['errors'])}")
        
        # Test 4: File reload
        print("\n🔄 Test 4: Configuration File Reload")
        
        # Modify config file
        modified_config = sample_config.copy()
        modified_config['optimization']['llm_only_mode'] = False
        modified_config['orchestration']['max_iterations'] = 20
        
        with open(config_file, 'w') as f:
            json.dump(modified_config, f, indent=2)
        
        # Reload from file
        reload_result = config_loader.reload_from_file()
        print(f"✅ Configuration reloaded: {reload_result['success']}")
        print(f"   - Changes detected: {len(reload_result.get('changes_detected', []))}")
        print(f"   - LLM-only mode: {config_loader.is_llm_only_mode()}")
        print(f"   - Max iterations: {config_loader.get_config_value('orchestration.max_iterations')}")
        
        # Test 5: Orchestration engine integration
        print("\n🎭 Test 5: Orchestration Engine Integration")
        
        # Create mock components
        mock_bedrock_executor = Mock(spec=BedrockExecutor)
        mock_evaluator = Mock(spec=Evaluator)
        
        # Initialize orchestration engine
        engine = LLMOrchestrationEngine(
            bedrock_executor=mock_bedrock_executor,
            evaluator=mock_evaluator,
            config=config_loader.get_config()
        )
        
        print(f"✅ Orchestration engine initialized")
        print(f"   - LLM-only mode: {engine.llm_only_mode}")
        print(f"   - Available agents: {list(engine.agents.keys())}")
        print(f"   - Max iterations: {engine.convergence_config['max_iterations']}")
        
        # Test runtime configuration update through engine
        print("\n🔧 Test 6: Runtime Updates Through Engine")
        
        update_result = engine.update_configuration({
            'optimization.llm_only_mode': True,
            'orchestration.max_iterations': 12
        })
        
        print(f"✅ Configuration updated through engine: {update_result['success']}")
        print(f"   - Applied changes: {len(update_result.get('applied_changes', []))}")
        print(f"   - Engine LLM-only mode: {engine.llm_only_mode}")
        print(f"   - Engine max iterations: {engine.convergence_config['max_iterations']}")
        
        print("\n🎉 All tests passed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Clean up
        if config_file.exists():
            config_file.unlink()
        Path(temp_dir).rmdir()
    
    return True


if __name__ == '__main__':
    success = test_config_integration()
    exit(0 if success else 1)
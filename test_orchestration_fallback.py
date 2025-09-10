#!/usr/bin/env python3
"""
Test script for orchestration engine fallback mechanism.

This script tests the fallback functionality in the orchestration engine
when LLM agents fail during prompt optimization.
"""

import sys
import json
from typing import Dict, Any

# Add the project root to the path
sys.path.insert(0, '.')

from orchestration.engine import LLMOrchestrationEngine
from agents.factory import AgentFactory
from bedrock.executor import BedrockExecutor, ModelConfig
from evaluation.evaluator import Evaluator


class MockBedrockExecutor:
    """Mock Bedrock executor for testing."""
    
    def __init__(self):
        self.call_count = 0
    
    def execute_prompt(self, prompt: str, model_config: ModelConfig):
        """Mock prompt execution."""
        self.call_count += 1
        
        # Simulate a simple response
        from models import ExecutionResult
        return ExecutionResult(
            success=True,
            response_text=f"Mock response for: {prompt[:50]}...",
            model_name=model_config.model_id,
            token_usage={'input_tokens': 50, 'output_tokens': 50},
            execution_time=0.5
        )


class MockEvaluator:
    """Mock evaluator for testing."""
    
    def evaluate_response(self, prompt: str, response: str, context: Dict[str, Any] = None):
        """Mock response evaluation."""
        from models import EvaluationResult
        return EvaluationResult(
            overall_score=0.8,
            relevance_score=0.8,
            clarity_score=0.8,
            completeness_score=0.8,
            custom_metrics={'coherence': 0.8},
            qualitative_feedback="Mock evaluation feedback",
            improvement_suggestions=["Mock suggestion 1", "Mock suggestion 2"]
        )


def test_orchestration_with_fallback():
    """Test orchestration engine with fallback enabled."""
    print("Testing orchestration engine with fallback...")
    
    # Configuration with LLM-only mode and fallback enabled
    config = {
        'optimization': {
            'llm_only_mode': True,
            'fallback_to_heuristic': True
        },
        'orchestration': {
            'orchestrator_model': 'mock-model',
            'orchestrator_temperature': 0.3,
            'orchestrator_max_tokens': 2000,
            'min_iterations': 1,
            'max_iterations': 3,
            'score_improvement_threshold': 0.02,
            'stability_window': 2,
            'convergence_confidence_threshold': 0.8
        },
        'agents': {
            'analyzer': {'enabled': True, 'model': 'mock-analyzer-model'},
            'refiner': {'enabled': True, 'model': 'mock-refiner-model'},
            'validator': {'enabled': True, 'model': 'mock-validator-model'}
        }
    }
    
    # Create mock components
    bedrock_executor = MockBedrockExecutor()
    evaluator = MockEvaluator()
    
    # Create orchestration engine
    engine = LLMOrchestrationEngine(bedrock_executor, evaluator, config)
    
    # Verify agents were created
    assert len(engine.agents) > 0
    print(f"✓ Created {len(engine.agents)} agents")
    
    # Test prompt optimization
    test_prompt = "Write a short story"
    context = {'intended_use': 'creative writing', 'session_id': 'test_session'}
    
    model_config = ModelConfig(
        model_id='mock-model',
        temperature=0.7,
        max_tokens=1000
    )
    
    # Run orchestration iteration
    result = engine.run_llm_orchestrated_iteration(
        prompt=test_prompt,
        context=context,
        model_config=model_config
    )
    
    assert result.success == True
    assert result.orchestrated_prompt is not None
    assert len(result.agent_results) > 0
    print("✓ Orchestration iteration completed successfully")
    
    return True


def test_orchestration_fallback_on_failure():
    """Test orchestration engine fallback when agents fail."""
    print("\nTesting orchestration fallback on agent failure...")
    
    # Configuration with fallback enabled
    config = {
        'optimization': {
            'llm_only_mode': True,
            'fallback_to_heuristic': True
        },
        'orchestration': {
            'orchestrator_model': 'mock-model',
            'min_iterations': 1,
            'max_iterations': 2
        },
        'agents': {
            'analyzer': {'enabled': True},
            'refiner': {'enabled': True},
            'validator': {'enabled': True}
        }
    }
    
    bedrock_executor = MockBedrockExecutor()
    evaluator = MockEvaluator()
    
    engine = LLMOrchestrationEngine(bedrock_executor, evaluator, config)
    
    # Test error recovery methods
    error_context = {
        'agent_name': 'analyzer',
        'error': 'LLM service unavailable',
        'agent_type': 'LLMAnalyzerAgent'
    }
    
    recovery_result = engine._handle_agent_failure(error_context)
    print(f"✓ Agent failure recovery result: {recovery_result.get('success', False)}")
    
    # Test LLM service unavailable handling
    llm_unavailable_result = engine._handle_llm_service_unavailable({'error': 'Service timeout'})
    print(f"✓ LLM service unavailable handling: {llm_unavailable_result.get('success', False)}")
    
    return True


def test_orchestration_without_fallback():
    """Test orchestration engine without fallback."""
    print("\nTesting orchestration without fallback...")
    
    # Configuration without fallback
    config = {
        'optimization': {
            'llm_only_mode': True,
            'fallback_to_heuristic': False
        },
        'orchestration': {
            'orchestrator_model': 'mock-model'
        },
        'agents': {
            'analyzer': {'enabled': True},
            'refiner': {'enabled': True},
            'validator': {'enabled': True}
        }
    }
    
    bedrock_executor = MockBedrockExecutor()
    evaluator = MockEvaluator()
    
    engine = LLMOrchestrationEngine(bedrock_executor, evaluator, config)
    
    # Test that fallback is disabled
    assert engine.agent_factory.fallback_to_heuristic == False
    print("✓ Fallback correctly disabled")
    
    # Test error handling without fallback
    error_context = {'agent_name': 'analyzer', 'error': 'LLM failed'}
    recovery_result = engine._handle_llm_service_unavailable(error_context)
    
    assert recovery_result.get('success', True) == False
    print("✓ Error handling correctly fails when fallback disabled")
    
    return True


def test_agent_factory_integration():
    """Test agent factory integration with orchestration engine."""
    print("\nTesting agent factory integration...")
    
    config = {
        'optimization': {
            'llm_only_mode': True,
            'fallback_to_heuristic': True
        },
        'agents': {
            'analyzer': {'enabled': True},
            'refiner': {'enabled': True},
            'validator': {'enabled': True}
        }
    }
    
    bedrock_executor = MockBedrockExecutor()
    evaluator = MockEvaluator()
    
    engine = LLMOrchestrationEngine(bedrock_executor, evaluator, config)
    
    # Test agent factory properties
    assert engine.agent_factory.llm_only_mode == True
    assert engine.agent_factory.fallback_to_heuristic == True
    print("✓ Agent factory configuration correct")
    
    # Test available agents
    available_agents = engine.agent_factory.get_available_agents()
    assert len(available_agents) > 0
    print(f"✓ Available agents: {available_agents}")
    
    # Test bypassed agents
    bypassed_agents = engine.agent_factory.get_bypassed_agents()
    print(f"✓ Bypassed agents: {bypassed_agents}")
    
    # Test mode description
    mode_desc = engine.agent_factory.get_mode_description()
    assert 'fallback' in mode_desc.lower()
    print(f"✓ Mode description: {mode_desc}")
    
    return True


def test_configuration_update():
    """Test configuration updates with fallback settings."""
    print("\nTesting configuration updates...")
    
    initial_config = {
        'optimization': {
            'llm_only_mode': False,
            'fallback_to_heuristic': False
        }
    }
    
    bedrock_executor = MockBedrockExecutor()
    evaluator = MockEvaluator()
    
    engine = LLMOrchestrationEngine(bedrock_executor, evaluator, initial_config)
    
    # Verify initial state
    assert engine.agent_factory.llm_only_mode == False
    assert engine.agent_factory.fallback_to_heuristic == False
    print("✓ Initial configuration loaded")
    
    # Update configuration to enable LLM-only mode with fallback
    new_config = {
        'optimization': {
            'llm_only_mode': True,
            'fallback_to_heuristic': True
        }
    }
    
    engine.agent_factory.update_config(new_config)
    
    # Verify updated state
    assert engine.agent_factory.llm_only_mode == True
    assert engine.agent_factory.fallback_to_heuristic == True
    print("✓ Configuration updated successfully")
    
    return True


def run_all_tests():
    """Run all orchestration fallback tests."""
    print("Running Orchestration Fallback Tests")
    print("=" * 50)
    
    try:
        test_orchestration_with_fallback()
        test_orchestration_fallback_on_failure()
        test_orchestration_without_fallback()
        test_agent_factory_integration()
        test_configuration_update()
        
        print("\n" + "=" * 50)
        print("✅ All orchestration fallback tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
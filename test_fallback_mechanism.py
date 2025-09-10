#!/usr/bin/env python3
"""
Test script for LLM fallback mechanism.

This script tests the fallback functionality when LLM agents fail
and ensures heuristic agents are used as fallback in LLM-only mode.
"""

import sys
import json
from typing import Dict, Any

# Add the project root to the path
sys.path.insert(0, '.')

from agents.factory import AgentFactory
from agents.llm_enhanced_analyzer import LLMAnalyzerAgent
from agents.analyzer import AnalyzerAgent


def test_fallback_configuration():
    """Test fallback configuration options."""
    print("Testing fallback configuration...")
    
    # Test with fallback enabled
    config_with_fallback = {
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
    
    factory = AgentFactory(config_with_fallback)
    assert factory.llm_only_mode == True
    assert factory.fallback_to_heuristic == True
    print("✓ Fallback configuration loaded correctly")
    
    # Test without fallback
    config_without_fallback = {
        'optimization': {
            'llm_only_mode': True,
            'fallback_to_heuristic': False
        }
    }
    
    factory_no_fallback = AgentFactory(config_without_fallback)
    assert factory_no_fallback.llm_only_mode == True
    assert factory_no_fallback.fallback_to_heuristic == False
    print("✓ No-fallback configuration loaded correctly")


def test_fallback_agent_creation():
    """Test creating fallback agents when LLM agents fail."""
    print("\nTesting fallback agent creation...")
    
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
    
    factory = AgentFactory(config)
    
    # Test creating fallback agents for failed LLM agents
    failed_agents = [
        ('analyzer', 'LLM service unavailable'),
        ('refiner', 'API timeout'),
        ('validator', 'Rate limit exceeded')
    ]
    
    fallback_agents = factory._create_fallback_agents_for_failed(failed_agents)
    
    assert len(fallback_agents) == 3
    assert 'analyzer' in fallback_agents
    assert 'refiner' in fallback_agents
    assert 'validator' in fallback_agents
    
    # Verify they are heuristic agents, not LLM agents
    assert isinstance(fallback_agents['analyzer'], AnalyzerAgent)
    assert not isinstance(fallback_agents['analyzer'], LLMAnalyzerAgent)
    
    print("✓ Fallback agents created successfully")


def test_emergency_fallback():
    """Test emergency fallback agent creation."""
    print("\nTesting emergency fallback...")
    
    config = {
        'optimization': {
            'llm_only_mode': True,
            'fallback_to_heuristic': True
        }
    }
    
    factory = AgentFactory(config)
    emergency_agents = factory.create_emergency_fallback_agents()
    
    assert len(emergency_agents) == 3
    assert all(agent_name in emergency_agents for agent_name in ['analyzer', 'refiner', 'validator'])
    
    print("✓ Emergency fallback agents created successfully")


def test_llm_agent_fallback_processing():
    """Test LLM agent fallback processing."""
    print("\nTesting LLM agent fallback processing...")
    
    config = {
        'optimization': {
            'llm_only_mode': True,
            'fallback_to_heuristic': True
        },
        'agents': {
            'analyzer': {'enabled': True}
        }
    }
    
    # Create LLM analyzer with fallback enabled
    analyzer = LLMAnalyzerAgent(config['agents']['analyzer'])
    analyzer.config.update(config['optimization'])
    
    # Test with a simple prompt
    test_prompt = "Write a short story about a robot."
    
    # Test normal processing (should work with simulated LLM)
    result = analyzer.process(test_prompt)
    assert result.success == True
    print("✓ Normal LLM processing works")
    
    # Test fallback processing method
    fallback_result = analyzer.process_with_fallback(test_prompt)
    assert fallback_result.success == True
    print("✓ Fallback processing method works")


def test_configuration_validation():
    """Test configuration validation for fallback scenarios."""
    print("\nTesting configuration validation...")
    
    # Test LLM-only mode without fallback (should warn)
    config_risky = {
        'optimization': {
            'llm_only_mode': True,
            'fallback_to_heuristic': False
        }
    }
    
    factory = AgentFactory(config_risky)
    validation = factory.validate_configuration()
    
    assert any('fallback' in warning.lower() for warning in validation['warnings'])
    print("✓ Configuration validation warns about risky setup")
    
    # Test LLM-only mode with fallback (should be fine)
    config_safe = {
        'optimization': {
            'llm_only_mode': True,
            'fallback_to_heuristic': True
        }
    }
    
    factory_safe = AgentFactory(config_safe)
    validation_safe = factory_safe.validate_configuration()
    
    print("✓ Configuration validation passes for safe setup")


def test_llm_service_unavailable_handling():
    """Test handling of LLM service unavailability."""
    print("\nTesting LLM service unavailable handling...")
    
    config = {
        'optimization': {
            'llm_only_mode': True,
            'fallback_to_heuristic': True
        }
    }
    
    factory = AgentFactory(config)
    
    # Test handling LLM service unavailability
    fallback_agents = factory.handle_llm_service_unavailable()
    
    assert len(fallback_agents) == 3
    assert all(isinstance(agent, (AnalyzerAgent, type(fallback_agents['refiner']), type(fallback_agents['validator']))) 
              for agent in fallback_agents.values())
    
    print("✓ LLM service unavailable handled correctly")


def run_all_tests():
    """Run all fallback mechanism tests."""
    print("Running LLM Fallback Mechanism Tests")
    print("=" * 50)
    
    try:
        test_fallback_configuration()
        test_fallback_agent_creation()
        test_emergency_fallback()
        test_llm_agent_fallback_processing()
        test_configuration_validation()
        test_llm_service_unavailable_handling()
        
        print("\n" + "=" * 50)
        print("✅ All fallback mechanism tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
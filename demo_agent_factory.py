#!/usr/bin/env python3
"""
Demonstration script for AgentFactory functionality.

This script demonstrates how the AgentFactory creates different agent sets
based on configuration mode and how agents maintain consistent interfaces.
"""

import json
from agents.factory import AgentFactory


def demonstrate_agent_factory():
    """Demonstrate AgentFactory functionality with different configurations."""
    
    print("=== AgentFactory Demonstration ===\n")
    
    # Test prompt for demonstration
    test_prompt = "Write a comprehensive guide on machine learning for beginners."
    
    # 1. Demonstrate Hybrid Mode (default)
    print("1. HYBRID MODE (Default)")
    print("-" * 40)
    
    hybrid_config = {
        'optimization': {
            'llm_only_mode': False,
            'fallback_to_heuristic': True
        },
        'agents': {
            'analyzer': {
                'min_length': 10,
                'max_length': 2000,
                'clarity_weight': 0.3
            },
            'refiner': {
                'max_iterations': 3,
                'preserve_intent': True
            },
            'validator': {
                'min_quality_score': 0.6,
                'strict_mode': False
            }
        }
    }
    
    hybrid_factory = AgentFactory(hybrid_config)
    
    print(f"Mode: {hybrid_factory.get_mode_description()}")
    print(f"Available agents: {hybrid_factory.get_available_agents()}")
    print(f"Bypassed agents: {hybrid_factory.get_bypassed_agents()}")
    
    # Create and test agents
    hybrid_agents = hybrid_factory.create_agents()
    print(f"Created {len(hybrid_agents)} agents")
    
    # Test one agent from each type
    heuristic_analyzer = hybrid_agents['analyzer']
    llm_analyzer = hybrid_agents['llm_analyzer']
    
    print(f"\nTesting heuristic analyzer: {heuristic_analyzer.get_name()}")
    heuristic_result = heuristic_analyzer.process(test_prompt)
    print(f"  Success: {heuristic_result.success}")
    print(f"  Confidence: {heuristic_result.confidence_score:.2f}")
    print(f"  Suggestions: {len(heuristic_result.suggestions)}")
    
    print(f"\nTesting LLM analyzer: {llm_analyzer.get_name()}")
    llm_result = llm_analyzer.process(test_prompt)
    print(f"  Success: {llm_result.success}")
    print(f"  Confidence: {llm_result.confidence_score:.2f}")
    print(f"  Suggestions: {len(llm_result.suggestions)}")
    
    print("\n" + "="*60 + "\n")
    
    # 2. Demonstrate LLM-Only Mode
    print("2. LLM-ONLY MODE")
    print("-" * 40)
    
    llm_only_config = {
        'optimization': {
            'llm_only_mode': True,
            'fallback_to_heuristic': True
        },
        'agents': {
            'analyzer': {
                'llm_model': 'claude-3-sonnet',
                'llm_temperature': 0.3,
                'analysis_depth': 'comprehensive'
            },
            'refiner': {
                'llm_model': 'claude-3-sonnet',
                'llm_temperature': 0.4,
                'refinement_style': 'comprehensive'
            },
            'validator': {
                'llm_model': 'claude-3-sonnet',
                'llm_temperature': 0.1,
                'validation_strictness': 'moderate'
            }
        }
    }
    
    llm_factory = AgentFactory(llm_only_config)
    
    print(f"Mode: {llm_factory.get_mode_description()}")
    print(f"Available agents: {llm_factory.get_available_agents()}")
    print(f"Bypassed agents: {llm_factory.get_bypassed_agents()}")
    
    # Create and test agents
    llm_only_agents = llm_factory.create_agents()
    print(f"Created {len(llm_only_agents)} agents")
    
    # Test all LLM agents
    for agent_name, agent in llm_only_agents.items():
        print(f"\nTesting {agent_name}: {agent.get_name()}")
        result = agent.process(test_prompt)
        print(f"  Success: {result.success}")
        print(f"  Confidence: {result.confidence_score:.2f}")
        print(f"  Suggestions: {len(result.suggestions)}")
        if result.suggestions:
            print(f"  First suggestion: {result.suggestions[0][:80]}...")
    
    print("\n" + "="*60 + "\n")
    
    # 3. Demonstrate Configuration Validation
    print("3. CONFIGURATION VALIDATION")
    print("-" * 40)
    
    # Test validation for both modes
    for factory, mode_name in [(hybrid_factory, "Hybrid"), (llm_factory, "LLM-Only")]:
        validation_result = factory.validate_configuration()
        print(f"\n{mode_name} Mode Validation:")
        print(f"  Valid: {validation_result['valid']}")
        print(f"  Mode: {validation_result['mode']}")
        print(f"  Warnings: {len(validation_result['warnings'])}")
        print(f"  Errors: {len(validation_result['errors'])}")
        
        if validation_result['warnings']:
            for warning in validation_result['warnings']:
                print(f"    Warning: {warning}")
    
    print("\n" + "="*60 + "\n")
    
    # 4. Demonstrate Mode Switching
    print("4. DYNAMIC MODE SWITCHING")
    print("-" * 40)
    
    # Start with one mode and switch to another
    dynamic_factory = AgentFactory({'optimization': {'llm_only_mode': False}})
    print(f"Initial mode: {dynamic_factory.get_mode_description()}")
    print(f"Initial agents: {len(dynamic_factory.create_agents())}")
    
    # Switch to LLM-only mode
    dynamic_factory.update_config({
        'optimization': {
            'llm_only_mode': True,
            'fallback_to_heuristic': True
        }
    })
    print(f"\nAfter switch: {dynamic_factory.get_mode_description()}")
    print(f"New agents: {len(dynamic_factory.create_agents())}")
    
    # Switch back to hybrid mode
    dynamic_factory.update_config({
        'optimization': {
            'llm_only_mode': False
        }
    })
    print(f"\nAfter switch back: {dynamic_factory.get_mode_description()}")
    print(f"Final agents: {len(dynamic_factory.create_agents())}")
    
    print("\n" + "="*60 + "\n")
    
    # 5. Demonstrate Interface Consistency
    print("5. INTERFACE CONSISTENCY")
    print("-" * 40)
    
    # Show that all agents have the same interface regardless of mode
    all_agents = {}
    all_agents.update(hybrid_agents)
    all_agents.update(llm_only_agents)
    
    print("All agents implement the same interface:")
    for agent_name, agent in list(all_agents.items())[:3]:  # Show first 3 for brevity
        print(f"\n{agent_name} ({agent.get_name()}):")
        print(f"  Has process method: {hasattr(agent, 'process')}")
        print(f"  Has validate_input method: {hasattr(agent, 'validate_input')}")
        print(f"  Has get_name method: {hasattr(agent, 'get_name')}")
        print(f"  Has get_config method: {hasattr(agent, 'get_config')}")
        
        # Test that process returns AgentResult
        result = agent.process("Test prompt")
        print(f"  Returns AgentResult: {type(result).__name__ == 'AgentResult'}")
        print(f"  Result has required fields: {all(hasattr(result, field) for field in ['success', 'analysis', 'suggestions', 'confidence_score'])}")
    
    print("\n=== Demonstration Complete ===")


if __name__ == "__main__":
    demonstrate_agent_factory()
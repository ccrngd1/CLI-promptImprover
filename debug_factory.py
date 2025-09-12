#!/usr/bin/env python3
"""Debug script to check AgentFactory behavior."""

import json
from agents.factory import AgentFactory
from config_loader import get_config_loader

def main():
    # Load config
    config_loader = get_config_loader('config.json')
    config = config_loader.get_config()
    
    print("=== Configuration Debug ===")
    print(f"Full config keys: {list(config.keys())}")
    print(f"Optimization config: {json.dumps(config.get('optimization', {}), indent=2)}")
    
    # Create factory
    print("\n=== Factory Debug ===")
    factory = AgentFactory(config)
    print(f"Factory config keys: {list(factory.config.keys())}")
    print(f"Factory optimization_config: {factory.optimization_config}")
    print(f"Factory llm_only_mode: {factory.llm_only_mode}")
    print(f"Factory fallback_to_heuristic: {factory.fallback_to_heuristic}")
    print(f"Factory is_llm_only_mode(): {factory.is_llm_only_mode()}")
    print(f"Factory mode description: {factory.get_mode_description()}")
    
    # Test create_agents method decision
    print(f"\n=== Agent Creation Decision ===")
    print(f"Will call _create_llm_only_agents: {factory.llm_only_mode}")
    print(f"Will call _create_hybrid_agents: {not factory.llm_only_mode}")
    
    # Actually create agents to see what happens
    print(f"\n=== Creating Agents ===")
    try:
        agents = factory.create_agents()
        print(f"Successfully created {len(agents)} agents: {list(agents.keys())}")
        for name, agent in agents.items():
            print(f"  {name}: {type(agent).__name__}")
    except Exception as e:
        print(f"Failed to create agents: {e}")

if __name__ == "__main__":
    main()
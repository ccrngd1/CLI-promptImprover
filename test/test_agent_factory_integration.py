"""
Integration tests for AgentFactory class.

This module contains integration tests to verify that the AgentFactory
creates working agent instances that can process prompts correctly.
"""

import pytest
from agents.factory import AgentFactory
from agents.base import Agent, AgentResult
from agents.analyzer import AnalyzerAgent
from agents.llm_enhanced_analyzer import LLMAnalyzerAgent


class TestAgentFactoryIntegration:
    """Integration test cases for AgentFactory class."""
    
    def test_create_and_use_heuristic_agents(self):
        """Test creating and using heuristic agents through factory."""
        config = {
            'optimization': {
                'llm_only_mode': False
            },
            'agents': {
                'analyzer': {'min_length': 5, 'max_length': 1000},
                'refiner': {'max_iterations': 2},
                'validator': {'min_quality_score': 0.5}
            }
        }
        
        factory = AgentFactory(config)
        agents = factory.create_agents()
        
        # Test that we can use the analyzer agent
        analyzer = agents['analyzer']
        assert isinstance(analyzer, AnalyzerAgent)
        
        # Test processing a simple prompt
        test_prompt = "Write a short story about a robot."
        result = analyzer.process(test_prompt)
        
        assert isinstance(result, AgentResult)
        assert result.agent_name == "AnalyzerAgent"
        assert result.success is True
        assert isinstance(result.analysis, dict)
        assert isinstance(result.suggestions, list)
        assert 0.0 <= result.confidence_score <= 1.0
    
    def test_create_and_use_llm_only_agents(self):
        """Test creating and using LLM-only agents through factory."""
        config = {
            'optimization': {
                'llm_only_mode': True,
                'fallback_to_heuristic': True
            },
            'agents': {
                'analyzer': {
                    'llm_model': 'claude-3-sonnet',
                    'llm_temperature': 0.3,
                    'analysis_depth': 'comprehensive'
                }
            }
        }
        
        factory = AgentFactory(config)
        agents = factory.create_agents()
        
        # Test that we get LLM agents
        analyzer = agents['analyzer']
        assert isinstance(analyzer, LLMAnalyzerAgent)
        
        # Test processing a simple prompt
        test_prompt = "Explain quantum computing to a beginner."
        result = analyzer.process(test_prompt)
        
        assert isinstance(result, AgentResult)
        assert result.agent_name == "LLMAnalyzerAgent"
        assert result.success is True
        assert isinstance(result.analysis, dict)
        assert isinstance(result.suggestions, list)
        assert 0.0 <= result.confidence_score <= 1.0
    
    def test_agent_interface_consistency(self):
        """Test that agents created by factory have consistent interfaces."""
        # Test both modes create agents with same interface
        configs = [
            {'optimization': {'llm_only_mode': False}},
            {'optimization': {'llm_only_mode': True, 'fallback_to_heuristic': True}}
        ]
        
        for config in configs:
            factory = AgentFactory(config)
            agents = factory.create_agents()
            
            # All agents should inherit from Agent base class
            for agent_name, agent in agents.items():
                assert isinstance(agent, Agent)
                assert hasattr(agent, 'process')
                assert hasattr(agent, 'validate_input')
                assert hasattr(agent, 'get_name')
                assert hasattr(agent, 'get_config')
                
                # Test that process method works
                test_prompt = "Test prompt for interface consistency."
                result = agent.process(test_prompt)
                assert isinstance(result, AgentResult)
    
    def test_factory_mode_switching(self):
        """Test switching between modes and recreating agents."""
        # Start with hybrid mode
        factory = AgentFactory({'optimization': {'llm_only_mode': False}})
        hybrid_agents = factory.create_agents()
        
        # Should have 6 agents in hybrid mode
        assert len(hybrid_agents) == 6
        assert 'analyzer' in hybrid_agents
        assert 'llm_analyzer' in hybrid_agents
        
        # Switch to LLM-only mode
        factory.update_config({
            'optimization': {
                'llm_only_mode': True,
                'fallback_to_heuristic': True
            }
        })
        
        llm_only_agents = factory.create_agents()
        
        # Should have 3 agents in LLM-only mode
        assert len(llm_only_agents) == 3
        assert 'analyzer' in llm_only_agents
        assert 'llm_analyzer' not in llm_only_agents
        
        # Agents should still work
        test_prompt = "Test prompt after mode switch."
        result = llm_only_agents['analyzer'].process(test_prompt)
        assert result.success is True
    
    def test_agent_config_propagation(self):
        """Test that configuration is properly propagated to agents."""
        config = {
            'optimization': {
                'llm_only_mode': False
            },
            'agents': {
                'analyzer': {
                    'min_length': 20,
                    'max_length': 500,
                    'clarity_weight': 0.5
                }
            }
        }
        
        factory = AgentFactory(config)
        agents = factory.create_agents()
        
        analyzer = agents['analyzer']
        analyzer_config = analyzer.get_config()
        
        # Check that configuration was passed through
        assert analyzer_config['min_length'] == 20
        assert analyzer_config['max_length'] == 500
        assert analyzer_config['clarity_weight'] == 0.5
        assert analyzer_config['llm_only_mode'] is False
        assert analyzer_config['fallback_enabled'] is True
    
    def test_validation_with_real_agents(self):
        """Test configuration validation with real agent creation."""
        # Valid configuration
        valid_config = {
            'optimization': {'llm_only_mode': True},
            'agents': {
                'analyzer': {'llm_model': 'test-model'},
                'refiner': {'llm_model': 'test-model'},
                'validator': {'llm_model': 'test-model'}
            }
        }
        
        factory = AgentFactory(valid_config)
        validation_result = factory.validate_configuration()
        
        assert validation_result['valid'] is True
        assert validation_result['mode'] == 'llm_only'
        
        # Should be able to create agents successfully
        agents = factory.create_agents()
        assert len(agents) == 3
    
    def test_error_handling_in_agent_creation(self):
        """Test error handling when agent creation fails."""
        # This test verifies that the factory handles errors gracefully
        config = {
            'optimization': {
                'llm_only_mode': False
            }
        }
        
        factory = AgentFactory(config)
        
        # Should not raise exceptions even with minimal config
        try:
            agents = factory.create_agents()
            assert len(agents) > 0
        except Exception as e:
            pytest.fail(f"Agent creation should not fail with basic config: {e}")
    
    def test_agent_names_and_types(self):
        """Test that agents have correct names and types."""
        # Test LLM-only mode
        llm_config = {'optimization': {'llm_only_mode': True, 'fallback_to_heuristic': True}}
        llm_factory = AgentFactory(llm_config)
        llm_agents = llm_factory.create_agents()
        
        assert llm_agents['analyzer'].get_name() == "LLMAnalyzerAgent"
        assert llm_agents['refiner'].get_name() == "LLMRefinerAgent"
        assert llm_agents['validator'].get_name() == "LLMValidatorAgent"
        
        # Test hybrid mode
        hybrid_config = {'optimization': {'llm_only_mode': False}}
        hybrid_factory = AgentFactory(hybrid_config)
        hybrid_agents = hybrid_factory.create_agents()
        
        assert hybrid_agents['analyzer'].get_name() == "AnalyzerAgent"
        assert hybrid_agents['refiner'].get_name() == "RefinerAgent"
        assert hybrid_agents['validator'].get_name() == "ValidatorAgent"
        assert hybrid_agents['llm_analyzer'].get_name() == "LLMAnalyzerAgent"
        assert hybrid_agents['llm_refiner'].get_name() == "LLMRefinerAgent"
        assert hybrid_agents['llm_validator'].get_name() == "LLMValidatorAgent"
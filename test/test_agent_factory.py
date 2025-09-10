"""
Tests for AgentFactory class.

This module contains unit tests for the AgentFactory class to ensure
proper agent creation based on configuration mode.
"""

import pytest
from unittest.mock import patch, MagicMock
from agents.factory import AgentFactory
from agents.analyzer import AnalyzerAgent
from agents.refiner import RefinerAgent
from agents.validator import ValidatorAgent
from agents.llm_enhanced_analyzer import LLMAnalyzerAgent
from agents.llm_enhanced_refiner import LLMRefinerAgent
from agents.llm_enhanced_validator import LLMValidatorAgent


class TestAgentFactory:
    """Test cases for AgentFactory class."""
    
    def test_factory_initialization_default_config(self):
        """Test factory initialization with default configuration."""
        factory = AgentFactory()
        
        assert not factory.is_llm_only_mode()
        assert factory.is_hybrid_mode()
        assert factory.fallback_to_heuristic is True
        assert factory.get_mode_description() == "Hybrid mode (heuristic + LLM agents)"
    
    def test_factory_initialization_llm_only_mode(self):
        """Test factory initialization with LLM-only mode configuration."""
        config = {
            'optimization': {
                'llm_only_mode': True,
                'fallback_to_heuristic': False
            }
        }
        
        factory = AgentFactory(config)
        
        assert factory.is_llm_only_mode()
        assert not factory.is_hybrid_mode()
        assert factory.fallback_to_heuristic is False
        assert "LLM-only mode" in factory.get_mode_description()
    
    def test_factory_initialization_hybrid_mode(self):
        """Test factory initialization with hybrid mode configuration."""
        config = {
            'optimization': {
                'llm_only_mode': False,
                'fallback_to_heuristic': True
            }
        }
        
        factory = AgentFactory(config)
        
        assert not factory.is_llm_only_mode()
        assert factory.is_hybrid_mode()
        assert factory.fallback_to_heuristic is True
    
    @patch('agents.factory.LLMAnalyzerAgent')
    @patch('agents.factory.LLMRefinerAgent')
    @patch('agents.factory.LLMValidatorAgent')
    def test_create_llm_only_agents(self, mock_validator, mock_refiner, mock_analyzer):
        """Test creation of LLM-only agents."""
        config = {
            'optimization': {
                'llm_only_mode': True
            },
            'agents': {
                'analyzer': {'model': 'test-model'},
                'refiner': {'model': 'test-model'},
                'validator': {'model': 'test-model'}
            }
        }
        
        factory = AgentFactory(config)
        agents = factory.create_agents()
        
        # Verify correct agents were created
        assert len(agents) == 3
        assert 'analyzer' in agents
        assert 'refiner' in agents
        assert 'validator' in agents
        
        # Verify LLM agents were instantiated
        mock_analyzer.assert_called_once()
        mock_refiner.assert_called_once()
        mock_validator.assert_called_once()
    
    @patch('agents.factory.AnalyzerAgent')
    @patch('agents.factory.RefinerAgent')
    @patch('agents.factory.ValidatorAgent')
    @patch('agents.factory.LLMAnalyzerAgent')
    @patch('agents.factory.LLMRefinerAgent')
    @patch('agents.factory.LLMValidatorAgent')
    def test_create_hybrid_agents(self, mock_llm_validator, mock_llm_refiner, mock_llm_analyzer,
                                 mock_validator, mock_refiner, mock_analyzer):
        """Test creation of hybrid agents."""
        config = {
            'optimization': {
                'llm_only_mode': False
            },
            'agents': {
                'analyzer': {'model': 'test-model'},
                'refiner': {'model': 'test-model'},
                'validator': {'model': 'test-model'}
            }
        }
        
        factory = AgentFactory(config)
        agents = factory.create_agents()
        
        # Verify correct agents were created
        assert len(agents) == 6
        assert 'analyzer' in agents
        assert 'refiner' in agents
        assert 'validator' in agents
        assert 'llm_analyzer' in agents
        assert 'llm_refiner' in agents
        assert 'llm_validator' in agents
        
        # Verify both heuristic and LLM agents were instantiated
        mock_analyzer.assert_called_once()
        mock_refiner.assert_called_once()
        mock_validator.assert_called_once()
        mock_llm_analyzer.assert_called_once()
        mock_llm_refiner.assert_called_once()
        mock_llm_validator.assert_called_once()
    
    def test_get_available_agents_llm_only(self):
        """Test getting available agents in LLM-only mode."""
        config = {
            'optimization': {
                'llm_only_mode': True
            }
        }
        
        factory = AgentFactory(config)
        available_agents = factory.get_available_agents()
        
        assert available_agents == ['analyzer', 'refiner', 'validator']
    
    def test_get_available_agents_hybrid(self):
        """Test getting available agents in hybrid mode."""
        config = {
            'optimization': {
                'llm_only_mode': False
            }
        }
        
        factory = AgentFactory(config)
        available_agents = factory.get_available_agents()
        
        expected_agents = ['analyzer', 'refiner', 'validator', 'llm_analyzer', 'llm_refiner', 'llm_validator']
        assert available_agents == expected_agents
    
    def test_get_bypassed_agents_llm_only(self):
        """Test getting bypassed agents in LLM-only mode."""
        config = {
            'optimization': {
                'llm_only_mode': True
            }
        }
        
        factory = AgentFactory(config)
        bypassed_agents = factory.get_bypassed_agents()
        
        expected_bypassed = ['heuristic_analyzer', 'heuristic_refiner', 'heuristic_validator']
        assert bypassed_agents == expected_bypassed
    
    def test_get_bypassed_agents_hybrid(self):
        """Test getting bypassed agents in hybrid mode."""
        config = {
            'optimization': {
                'llm_only_mode': False
            }
        }
        
        factory = AgentFactory(config)
        bypassed_agents = factory.get_bypassed_agents()
        
        assert bypassed_agents == []
    
    def test_update_config_mode_change(self):
        """Test updating configuration and mode change."""
        # Start with hybrid mode
        factory = AgentFactory({'optimization': {'llm_only_mode': False}})
        assert not factory.is_llm_only_mode()
        
        # Update to LLM-only mode
        new_config = {
            'optimization': {
                'llm_only_mode': True,
                'fallback_to_heuristic': False
            }
        }
        factory.update_config(new_config)
        
        assert factory.is_llm_only_mode()
        assert not factory.fallback_to_heuristic
    
    def test_validate_configuration_valid(self):
        """Test configuration validation with valid config."""
        config = {
            'optimization': {
                'llm_only_mode': True
            },
            'agents': {
                'analyzer': {'model': 'test-model'},
                'refiner': {'model': 'test-model'},
                'validator': {'model': 'test-model'}
            }
        }
        
        factory = AgentFactory(config)
        validation_result = factory.validate_configuration()
        
        assert validation_result['valid'] is True
        assert validation_result['mode'] == 'llm_only'
    
    def test_validate_configuration_missing_agent_config(self):
        """Test configuration validation with missing agent config."""
        config = {
            'optimization': {
                'llm_only_mode': True
            },
            'agents': {
                'analyzer': {'model': 'test-model'}
                # Missing refiner and validator configs
            }
        }
        
        factory = AgentFactory(config)
        validation_result = factory.validate_configuration()
        
        assert validation_result['valid'] is True
        assert len(validation_result['warnings']) >= 2  # Missing refiner and validator
    
    def test_validate_configuration_llm_only_without_fallback_warning(self):
        """Test configuration validation warns about LLM-only without fallback."""
        config = {
            'optimization': {
                'llm_only_mode': True,
                'fallback_to_heuristic': False
            }
        }
        
        factory = AgentFactory(config)
        validation_result = factory.validate_configuration()
        
        assert validation_result['valid'] is True
        assert any('without fallback' in warning for warning in validation_result['warnings'])
    
    @patch('agents.factory.LLMAnalyzerAgent')
    @patch('agents.factory.AnalyzerAgent')
    def test_llm_only_fallback_on_failure(self, mock_heuristic_analyzer, mock_llm_analyzer):
        """Test fallback to heuristic agents when LLM agent creation fails."""
        # Configure LLM agent to raise exception
        mock_llm_analyzer.side_effect = Exception("LLM service unavailable")
        
        config = {
            'optimization': {
                'llm_only_mode': True,
                'fallback_to_heuristic': True
            }
        }
        
        factory = AgentFactory(config)
        
        # Should fallback to heuristic agents
        agents = factory.create_agents()
        
        # Verify heuristic agents were created as fallback
        assert len(agents) == 3
        mock_heuristic_analyzer.assert_called()
    
    @patch('agents.factory.LLMValidatorAgent')
    @patch('agents.factory.LLMRefinerAgent')
    @patch('agents.factory.LLMAnalyzerAgent')
    def test_llm_only_no_fallback_on_failure(self, mock_llm_analyzer, mock_llm_refiner, mock_llm_validator):
        """Test exception raised when all LLM agent creation fails and no fallback."""
        # Configure all LLM agents to raise exception
        mock_llm_analyzer.side_effect = Exception("LLM service unavailable")
        mock_llm_refiner.side_effect = Exception("LLM service unavailable")
        mock_llm_validator.side_effect = Exception("LLM service unavailable")
        
        config = {
            'optimization': {
                'llm_only_mode': True,
                'fallback_to_heuristic': False
            }
        }
        
        factory = AgentFactory(config)
        
        # Should raise exception without fallback when all agents fail
        with pytest.raises(Exception, match="Failed to create any LLM agents"):
            factory.create_agents()
    
    def test_get_agent_config_includes_mode_info(self):
        """Test that agent config includes mode information."""
        config = {
            'optimization': {
                'llm_only_mode': True,
                'fallback_to_heuristic': False
            },
            'agents': {
                'analyzer': {'model': 'test-model', 'temperature': 0.3}
            }
        }
        
        factory = AgentFactory(config)
        agent_config = factory._get_agent_config('analyzer')
        
        assert agent_config['llm_only_mode'] is True
        assert agent_config['fallback_enabled'] is False
        assert agent_config['model'] == 'test-model'
        assert agent_config['temperature'] == 0.3
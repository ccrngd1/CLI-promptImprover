"""
Edge case tests for LLM-only mode functionality.

This module contains additional unit tests for edge cases and specific
scenarios in the LLM-only mode implementation.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

from agents.factory import AgentFactory
from orchestration.engine import LLMOrchestrationEngine
from bedrock.executor import BedrockExecutor
from evaluation.evaluator import Evaluator
from config_loader import ConfigurationLoader


class TestLLMOnlyModeEdgeCases:
    """Test edge cases for LLM-only mode functionality."""
    
    def test_empty_optimization_config(self):
        """Test behavior with empty optimization configuration."""
        config = {'optimization': {}}
        factory = AgentFactory(config)
        
        # Should default to hybrid mode
        assert not factory.is_llm_only_mode()
        assert factory.is_hybrid_mode()
        assert factory.fallback_to_heuristic is True
    
    def test_missing_optimization_section(self):
        """Test behavior with missing optimization section entirely."""
        config = {'agents': {'analyzer': {'model': 'test-model'}}}
        factory = AgentFactory(config)
        
        # Should default to hybrid mode
        assert not factory.is_llm_only_mode()
        assert factory.is_hybrid_mode()
        assert factory.fallback_to_heuristic is True
    
    def test_none_configuration(self):
        """Test behavior with None configuration."""
        factory = AgentFactory(None)
        
        # Should default to hybrid mode
        assert not factory.is_llm_only_mode()
        assert factory.is_hybrid_mode()
        assert factory.fallback_to_heuristic is True
    
    def test_config_update_preserves_other_settings(self):
        """Test that config updates preserve other configuration settings."""
        initial_config = {
            'optimization': {
                'llm_only_mode': False,
                'fallback_to_heuristic': True
            },
            'agents': {
                'analyzer': {'model': 'test-model', 'temperature': 0.3}
            },
            'other_section': {
                'some_setting': 'value'
            }
        }
        
        factory = AgentFactory(initial_config)
        
        # Update configuration
        new_config = {
            'optimization': {
                'llm_only_mode': True,
                'fallback_to_heuristic': False
            }
        }
        factory.update_config(new_config)
        
        # Verify mode changed
        assert factory.is_llm_only_mode()
        assert not factory.fallback_to_heuristic
        
        # Verify other settings preserved
        assert factory.config['agents']['analyzer']['model'] == 'test-model'
        assert factory.config['agents']['analyzer']['temperature'] == 0.3
        assert factory.config['other_section']['some_setting'] == 'value'
    
    def test_agent_config_with_missing_agent_types(self):
        """Test agent configuration with missing agent types."""
        config = {
            'optimization': {'llm_only_mode': True},
            'agents': {
                'analyzer': {'model': 'test-model'}
                # Missing refiner and validator
            }
        }
        
        factory = AgentFactory(config)
        
        # Should still work, using defaults for missing agents
        refiner_config = factory._get_agent_config('refiner')
        validator_config = factory._get_agent_config('validator')
        
        assert refiner_config['llm_only_mode'] is True
        assert validator_config['llm_only_mode'] is True
    
    def test_mode_description_variations(self):
        """Test mode description for various configurations."""
        # LLM-only without fallback
        config1 = {
            'optimization': {
                'llm_only_mode': True,
                'fallback_to_heuristic': False
            }
        }
        factory1 = AgentFactory(config1)
        description1 = factory1.get_mode_description()
        assert "LLM-only mode without fallback" in description1
        
        # LLM-only with fallback
        config2 = {
            'optimization': {
                'llm_only_mode': True,
                'fallback_to_heuristic': True
            }
        }
        factory2 = AgentFactory(config2)
        description2 = factory2.get_mode_description()
        assert "LLM-only mode with fallback" in description2
        
        # Hybrid mode
        config3 = {
            'optimization': {
                'llm_only_mode': False
            }
        }
        factory3 = AgentFactory(config3)
        description3 = factory3.get_mode_description()
        assert "Hybrid mode" in description3
    
    @patch('agents.factory.LLMAnalyzerAgent')
    @patch('agents.factory.LLMRefinerAgent')
    @patch('agents.factory.LLMValidatorAgent')
    def test_partial_llm_agent_creation_success(self, mock_validator, mock_refiner, mock_analyzer):
        """Test partial success in LLM agent creation."""
        config = {
            'optimization': {
                'llm_only_mode': True,
                'fallback_to_heuristic': True
            }
        }
        
        # Configure some agents to succeed, others to fail
        mock_analyzer.return_value = Mock()
        mock_refiner.side_effect = Exception("Refiner failed")
        mock_validator.return_value = Mock()
        
        factory = AgentFactory(config)
        
        # Should handle partial failures gracefully
        with patch.object(factory, '_create_fallback_agents_for_failed') as mock_fallback:
            mock_fallback.return_value = {'refiner': Mock()}
            agents = factory.create_agents()
            
            # Should have created successful agents plus fallback for failed one
            assert 'analyzer' in agents
            assert 'validator' in agents
    
    def test_validation_with_custom_metrics(self):
        """Test configuration validation with custom metrics."""
        config = {
            'optimization': {
                'llm_only_mode': True,
                'fallback_to_heuristic': True,
                'custom_metric_1': 'value1',
                'custom_metric_2': 42
            },
            'agents': {
                'analyzer': {'model': 'test-model'},
                'refiner': {'model': 'test-model'},
                'validator': {'model': 'test-model'}
            }
        }
        
        factory = AgentFactory(config)
        validation = factory.validate_configuration()
        
        # Should be valid despite custom metrics
        assert validation['valid'] is True
        assert validation['mode'] == 'llm_only'
    
    def test_emergency_fallback_with_minimal_config(self):
        """Test emergency fallback creation with minimal configuration."""
        config = {'optimization': {'llm_only_mode': True}}
        factory = AgentFactory(config)
        
        with patch('agents.factory.AnalyzerAgent') as mock_analyzer, \
             patch('agents.factory.RefinerAgent') as mock_refiner, \
             patch('agents.factory.ValidatorAgent') as mock_validator:
            
            mock_analyzer.return_value = Mock()
            mock_refiner.return_value = Mock()
            mock_validator.return_value = Mock()
            
            emergency_agents = factory.create_emergency_fallback_agents()
            
            # Should create all three emergency agents
            assert len(emergency_agents) == 3
            assert 'analyzer' in emergency_agents
            assert 'refiner' in emergency_agents
            assert 'validator' in emergency_agents
            
            # Verify emergency config was passed
            analyzer_call_args = mock_analyzer.call_args[0][0]
            assert analyzer_call_args['is_emergency_fallback'] is True


class TestOrchestrationEngineEdgeCases:
    """Test edge cases for orchestration engine with LLM-only mode."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_bedrock_executor = Mock(spec=BedrockExecutor)
        self.mock_evaluator = Mock(spec=Evaluator)
    
    @patch('orchestration.engine.AgentFactory')
    def test_orchestration_engine_with_empty_agents(self, mock_agent_factory_class):
        """Test orchestration engine behavior with empty agent dictionary."""
        config = {
            'optimization': {
                'llm_only_mode': True,
                'fallback_to_heuristic': False
            }
        }
        
        # Mock agent factory to return empty agents
        mock_factory = Mock()
        mock_factory.create_agents.return_value = {}
        mock_factory.fallback_to_heuristic = False
        mock_factory.get_mode_description.return_value = "LLM-only mode without fallback"
        mock_factory.get_bypassed_agents.return_value = []
        mock_agent_factory_class.return_value = mock_factory
        
        engine = LLMOrchestrationEngine(
            bedrock_executor=self.mock_bedrock_executor,
            evaluator=self.mock_evaluator,
            config=config
        )
        
        # Should handle empty agents gracefully
        assert len(engine.agents) == 0
        assert engine.llm_only_mode is True
    
    @patch('orchestration.engine.AgentFactory')
    def test_llm_failure_count_reset_on_success(self, mock_agent_factory_class):
        """Test that LLM failure count resets on successful operations."""
        config = {
            'optimization': {
                'llm_only_mode': True,
                'fallback_to_heuristic': True
            }
        }
        
        mock_factory = Mock()
        mock_factory.create_agents.return_value = {'analyzer': Mock()}
        mock_factory.fallback_to_heuristic = True
        mock_factory.get_mode_description.return_value = "LLM-only mode with fallback"
        mock_factory.get_bypassed_agents.return_value = []
        mock_agent_factory_class.return_value = mock_factory
        
        engine = LLMOrchestrationEngine(
            bedrock_executor=self.mock_bedrock_executor,
            evaluator=self.mock_evaluator,
            config=config
        )
        
        # Simulate some failures
        engine._llm_failure_count = 3
        assert engine._llm_failure_count == 3
        
        # Reset failure count
        engine._reset_llm_failure_count()
        assert engine._llm_failure_count == 0
    
    @patch('orchestration.engine.AgentFactory')
    def test_error_recovery_strategy_selection(self, mock_agent_factory_class):
        """Test error recovery strategy selection based on error type."""
        config = {
            'optimization': {
                'llm_only_mode': True,
                'fallback_to_heuristic': True
            }
        }
        
        mock_factory = Mock()
        mock_factory.create_agents.return_value = {'analyzer': Mock()}
        mock_factory.fallback_to_heuristic = True
        mock_factory.get_mode_description.return_value = "LLM-only mode with fallback"
        mock_factory.get_bypassed_agents.return_value = []
        mock_agent_factory_class.return_value = mock_factory
        
        engine = LLMOrchestrationEngine(
            bedrock_executor=self.mock_bedrock_executor,
            evaluator=self.mock_evaluator,
            config=config
        )
        
        # Test that error recovery methods exist
        assert hasattr(engine, '_handle_llm_service_unavailable')
        assert hasattr(engine, '_handle_llm_timeout')
        assert hasattr(engine, '_handle_llm_rate_limit')
        assert hasattr(engine, '_handle_agent_failure')
        
        # Test that methods are callable
        assert callable(engine._handle_llm_service_unavailable)
        assert callable(engine._handle_llm_timeout)
        assert callable(engine._handle_llm_rate_limit)
        assert callable(engine._handle_agent_failure)


class TestConfigurationLoaderIntegration:
    """Test integration between configuration loader and LLM-only mode."""
    
    def test_config_loader_optimization_methods(self):
        """Test configuration loader optimization-specific methods."""
        # This test would require a real config file, so we'll mock it
        with patch('config_loader.ConfigurationLoader') as mock_loader_class:
            mock_loader = Mock()
            mock_loader.get_optimization_config.return_value = {
                'llm_only_mode': True,
                'fallback_to_heuristic': False
            }
            mock_loader.is_llm_only_mode.return_value = True
            mock_loader.is_fallback_enabled.return_value = False
            mock_loader_class.return_value = mock_loader
            
            loader = mock_loader_class('test_config.json')
            
            # Test optimization config methods
            opt_config = loader.get_optimization_config()
            assert opt_config['llm_only_mode'] is True
            assert opt_config['fallback_to_heuristic'] is False
            
            assert loader.is_llm_only_mode() is True
            assert loader.is_fallback_enabled() is False
    
    def test_runtime_config_update_validation(self):
        """Test runtime configuration update validation."""
        with patch('config_loader.ConfigurationLoader') as mock_loader_class:
            mock_loader = Mock()
            
            # Test valid update
            mock_loader.update_config.return_value = {
                'success': True,
                'applied_changes': [
                    {
                        'key': 'optimization.llm_only_mode',
                        'old_value': False,
                        'new_value': True
                    }
                ],
                'failed_changes': []
            }
            mock_loader_class.return_value = mock_loader
            
            loader = mock_loader_class('test_config.json')
            result = loader.update_config({'optimization.llm_only_mode': True})
            
            assert result['success'] is True
            assert len(result['applied_changes']) == 1
            assert result['applied_changes'][0]['key'] == 'optimization.llm_only_mode'
    
    def test_config_validation_edge_cases(self):
        """Test configuration validation edge cases."""
        with patch('config_loader.ConfigurationLoader') as mock_loader_class:
            mock_loader = Mock()
            
            # Test validation with warnings
            mock_loader.validate_current_config.return_value = {
                'valid': True,
                'warnings': [
                    'LLM-only mode without fallback may fail if LLM services are unavailable'
                ],
                'errors': []
            }
            mock_loader_class.return_value = mock_loader
            
            loader = mock_loader_class('test_config.json')
            validation = loader.validate_current_config()
            
            assert validation['valid'] is True
            assert len(validation['warnings']) > 0
            assert 'without fallback' in validation['warnings'][0]


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
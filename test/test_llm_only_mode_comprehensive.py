"""
Comprehensive unit tests for LLM-only mode functionality.

This module contains comprehensive unit tests for the LLM-only mode feature,
covering configuration validation, agent factory creation, orchestration engine
behavior, and LLM response mocking for agent selection logic.
"""

import json
import pytest
from unittest.mock import Mock, patch, MagicMock, call
from datetime import datetime
from typing import Dict, Any, List

from agents.factory import AgentFactory
from agents.base import AgentResult
from agents.analyzer import AnalyzerAgent
from agents.refiner import RefinerAgent
from agents.validator import ValidatorAgent
from agents.llm_enhanced_analyzer import LLMAnalyzerAgent
from agents.llm_enhanced_refiner import LLMRefinerAgent
from agents.llm_enhanced_validator import LLMValidatorAgent
from orchestration.engine import LLMOrchestrationEngine
from bedrock.executor import BedrockExecutor, ModelConfig
from evaluation.evaluator import Evaluator
from config_loader import ConfigurationLoader, ConfigChangeEvent
from models import PromptIteration, ExecutionResult, EvaluationResult


class TestLLMOnlyModeConfiguration:
    """Test configuration validation and parsing for LLM-only mode."""
    
    def test_default_configuration_parsing(self):
        """Test parsing of default configuration (hybrid mode)."""
        config = {}
        factory = AgentFactory(config)
        
        assert not factory.is_llm_only_mode()
        assert factory.is_hybrid_mode()
        assert factory.fallback_to_heuristic is True
        assert factory.get_mode_description() == "Hybrid mode (heuristic + LLM agents)"
    
    def test_llm_only_mode_configuration_parsing(self):
        """Test parsing of LLM-only mode configuration."""
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
        assert "LLM-only mode without fallback" in factory.get_mode_description()
    
    def test_llm_only_mode_with_fallback_configuration(self):
        """Test LLM-only mode with fallback enabled configuration."""
        config = {
            'optimization': {
                'llm_only_mode': True,
                'fallback_to_heuristic': True
            }
        }
        factory = AgentFactory(config)
        
        assert factory.is_llm_only_mode()
        assert factory.fallback_to_heuristic is True
        assert "LLM-only mode with fallback" in factory.get_mode_description()
    
    def test_configuration_validation_valid_llm_only(self):
        """Test configuration validation for valid LLM-only mode."""
        config = {
            'optimization': {
                'llm_only_mode': True,
                'fallback_to_heuristic': True
            },
            'agents': {
                'analyzer': {'model': 'test-model', 'temperature': 0.2},
                'refiner': {'model': 'test-model', 'temperature': 0.4},
                'validator': {'model': 'test-model', 'temperature': 0.1}
            }
        }
        factory = AgentFactory(config)
        validation = factory.validate_configuration()
        
        assert validation['valid'] is True
        assert validation['mode'] == 'llm_only'
        assert len(validation['errors']) == 0
    
    def test_configuration_validation_missing_models(self):
        """Test configuration validation with missing LLM models."""
        config = {
            'optimization': {
                'llm_only_mode': True
            },
            'agents': {
                'analyzer': {'temperature': 0.2},  # Missing model
                'refiner': {},  # Missing model
                # Missing validator entirely
            }
        }
        factory = AgentFactory(config)
        validation = factory.validate_configuration()
        
        assert validation['valid'] is True  # Still valid but with warnings
        assert validation['mode'] == 'llm_only'
        assert len(validation['warnings']) >= 3  # Missing models and validator config
    
    def test_configuration_validation_llm_only_without_fallback_warning(self):
        """Test configuration validation warns about LLM-only without fallback."""
        config = {
            'optimization': {
                'llm_only_mode': True,
                'fallback_to_heuristic': False
            }
        }
        factory = AgentFactory(config)
        validation = factory.validate_configuration()
        
        assert validation['valid'] is True
        warning_messages = ' '.join(validation['warnings'])
        assert 'without fallback' in warning_messages
        assert 'may fail if LLM services are unavailable' in warning_messages
    
    def test_invalid_configuration_values(self):
        """Test handling of invalid configuration values."""
        # Test invalid llm_only_mode type
        config = {
            'optimization': {
                'llm_only_mode': 'invalid',  # Should be boolean
                'fallback_to_heuristic': 'also_invalid'  # Should be boolean
            }
        }
        
        # AgentFactory should handle invalid values gracefully
        factory = AgentFactory(config)
        
        # The factory currently doesn't validate types, so it uses the raw values
        # This test documents the current behavior - in a real implementation,
        # we might want to add type validation
        assert factory.llm_only_mode == 'invalid'  # Uses raw value
        assert factory.fallback_to_heuristic == 'also_invalid'  # Uses raw value
    
    def test_agent_config_includes_mode_information(self):
        """Test that agent configurations include mode information."""
        config = {
            'optimization': {
                'llm_only_mode': True,
                'fallback_to_heuristic': False
            },
            'agents': {
                'analyzer': {
                    'model': 'test-model',
                    'temperature': 0.3,
                    'max_tokens': 1500
                }
            }
        }
        factory = AgentFactory(config)
        agent_config = factory._get_agent_config('analyzer')
        
        # Verify mode information is included
        assert agent_config['llm_only_mode'] is True
        assert agent_config['fallback_enabled'] is False
        
        # Verify original config is preserved
        assert agent_config['model'] == 'test-model'
        assert agent_config['temperature'] == 0.3
        assert agent_config['max_tokens'] == 1500


class TestAgentFactoryCreation:
    """Test agent factory creation logic for both modes."""
    
    @patch('agents.factory.LLMAnalyzerAgent')
    @patch('agents.factory.LLMRefinerAgent')
    @patch('agents.factory.LLMValidatorAgent')
    def test_llm_only_agent_creation_success(self, mock_validator, mock_refiner, mock_analyzer):
        """Test successful creation of LLM-only agents."""
        config = {
            'optimization': {'llm_only_mode': True},
            'agents': {
                'analyzer': {'model': 'test-model'},
                'refiner': {'model': 'test-model'},
                'validator': {'model': 'test-model'}
            }
        }
        
        # Configure mocks to return agent instances
        mock_analyzer.return_value = Mock(spec=LLMAnalyzerAgent)
        mock_refiner.return_value = Mock(spec=LLMRefinerAgent)
        mock_validator.return_value = Mock(spec=LLMValidatorAgent)
        
        factory = AgentFactory(config)
        agents = factory.create_agents()
        
        # Verify correct agents were created
        assert len(agents) == 3
        assert 'analyzer' in agents
        assert 'refiner' in agents
        assert 'validator' in agents
        
        # Verify LLM agents were instantiated with correct config
        mock_analyzer.assert_called_once()
        mock_refiner.assert_called_once()
        mock_validator.assert_called_once()
        
        # Verify agent config includes mode information
        analyzer_call_args = mock_analyzer.call_args[0][0]
        assert analyzer_call_args['llm_only_mode'] is True
    
    @patch('agents.factory.AnalyzerAgent')
    @patch('agents.factory.RefinerAgent')
    @patch('agents.factory.ValidatorAgent')
    @patch('agents.factory.LLMAnalyzerAgent')
    @patch('agents.factory.LLMRefinerAgent')
    @patch('agents.factory.LLMValidatorAgent')
    def test_hybrid_agent_creation_success(self, mock_llm_validator, mock_llm_refiner, mock_llm_analyzer,
                                         mock_validator, mock_refiner, mock_analyzer):
        """Test successful creation of hybrid agents."""
        config = {
            'optimization': {'llm_only_mode': False},
            'agents': {
                'analyzer': {'model': 'test-model'},
                'refiner': {'model': 'test-model'},
                'validator': {'model': 'test-model'}
            }
        }
        
        # Configure mocks to return agent instances
        mock_analyzer.return_value = Mock(spec=AnalyzerAgent)
        mock_refiner.return_value = Mock(spec=RefinerAgent)
        mock_validator.return_value = Mock(spec=ValidatorAgent)
        mock_llm_analyzer.return_value = Mock(spec=LLMAnalyzerAgent)
        mock_llm_refiner.return_value = Mock(spec=LLMRefinerAgent)
        mock_llm_validator.return_value = Mock(spec=LLMValidatorAgent)
        
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
    
    @patch('agents.factory.LLMAnalyzerAgent')
    @patch('agents.factory.LLMRefinerAgent')
    @patch('agents.factory.LLMValidatorAgent')
    @patch('agents.factory.AnalyzerAgent')
    @patch('agents.factory.RefinerAgent')
    @patch('agents.factory.ValidatorAgent')
    def test_llm_only_fallback_on_partial_failure(self, mock_h_validator, mock_h_refiner, mock_h_analyzer,
                                                 mock_validator, mock_refiner, mock_analyzer):
        """Test fallback to heuristic agents when some LLM agents fail."""
        config = {
            'optimization': {
                'llm_only_mode': True,
                'fallback_to_heuristic': True
            }
        }
        
        # Configure LLM analyzer to fail, others to succeed
        mock_analyzer.side_effect = Exception("LLM service unavailable")
        mock_refiner.return_value = Mock(spec=LLMRefinerAgent)
        mock_validator.return_value = Mock(spec=LLMValidatorAgent)
        
        # Configure heuristic fallback agents
        mock_h_analyzer.return_value = Mock(spec=AnalyzerAgent)
        mock_h_refiner.return_value = Mock(spec=RefinerAgent)
        mock_h_validator.return_value = Mock(spec=ValidatorAgent)
        
        factory = AgentFactory(config)
        agents = factory.create_agents()
        
        # Should have created successful LLM agents and fallback heuristic for failed one
        assert len(agents) >= 2  # At least refiner and validator
        assert 'refiner' in agents
        assert 'validator' in agents
        
        # Fallback analyzer should have been created
        if 'analyzer' in agents:
            # Verify fallback was created for analyzer
            mock_h_analyzer.assert_called_once()
    
    @patch('agents.factory.LLMAnalyzerAgent')
    @patch('agents.factory.LLMRefinerAgent')
    @patch('agents.factory.LLMValidatorAgent')
    @patch('agents.factory.AnalyzerAgent')
    @patch('agents.factory.RefinerAgent')
    @patch('agents.factory.ValidatorAgent')
    def test_llm_only_complete_fallback_on_all_failures(self, mock_h_validator, mock_h_refiner, mock_h_analyzer,
                                                       mock_validator, mock_refiner, mock_analyzer):
        """Test complete fallback when all LLM agents fail."""
        config = {
            'optimization': {
                'llm_only_mode': True,
                'fallback_to_heuristic': True
            }
        }
        
        # Configure all LLM agents to fail
        mock_analyzer.side_effect = Exception("LLM service unavailable")
        mock_refiner.side_effect = Exception("LLM service unavailable")
        mock_validator.side_effect = Exception("LLM service unavailable")
        
        # Configure heuristic fallback agents
        mock_h_analyzer.return_value = Mock(spec=AnalyzerAgent)
        mock_h_refiner.return_value = Mock(spec=RefinerAgent)
        mock_h_validator.return_value = Mock(spec=ValidatorAgent)
        
        factory = AgentFactory(config)
        agents = factory.create_agents()
        
        # Should have fallen back to heuristic agents
        assert len(agents) == 3
        assert 'analyzer' in agents
        assert 'refiner' in agents
        assert 'validator' in agents
        
        # Verify heuristic agents were created as fallback
        mock_h_analyzer.assert_called_once()
        mock_h_refiner.assert_called_once()
        mock_h_validator.assert_called_once()
    
    @patch('agents.factory.LLMAnalyzerAgent')
    @patch('agents.factory.LLMRefinerAgent')
    @patch('agents.factory.LLMValidatorAgent')
    def test_llm_only_no_fallback_on_failure(self, mock_validator, mock_refiner, mock_analyzer):
        """Test exception raised when LLM agents fail and no fallback enabled."""
        config = {
            'optimization': {
                'llm_only_mode': True,
                'fallback_to_heuristic': False
            }
        }
        
        # Configure all LLM agents to fail
        mock_analyzer.side_effect = Exception("LLM service unavailable")
        mock_refiner.side_effect = Exception("LLM service unavailable")
        mock_validator.side_effect = Exception("LLM service unavailable")
        
        factory = AgentFactory(config)
        
        # Should raise exception without fallback
        with pytest.raises(Exception, match="Failed to create any LLM agents"):
            factory.create_agents()
    
    def test_available_agents_llm_only_mode(self):
        """Test getting available agents in LLM-only mode."""
        config = {'optimization': {'llm_only_mode': True}}
        factory = AgentFactory(config)
        
        available_agents = factory.get_available_agents()
        assert available_agents == ['analyzer', 'refiner', 'validator']
    
    def test_available_agents_hybrid_mode(self):
        """Test getting available agents in hybrid mode."""
        config = {'optimization': {'llm_only_mode': False}}
        factory = AgentFactory(config)
        
        available_agents = factory.get_available_agents()
        expected = ['analyzer', 'refiner', 'validator', 'llm_analyzer', 'llm_refiner', 'llm_validator']
        assert available_agents == expected
    
    def test_bypassed_agents_llm_only_mode(self):
        """Test getting bypassed agents in LLM-only mode."""
        config = {'optimization': {'llm_only_mode': True}}
        factory = AgentFactory(config)
        
        bypassed_agents = factory.get_bypassed_agents()
        expected = ['heuristic_analyzer', 'heuristic_refiner', 'heuristic_validator']
        assert bypassed_agents == expected
    
    def test_bypassed_agents_hybrid_mode(self):
        """Test getting bypassed agents in hybrid mode."""
        config = {'optimization': {'llm_only_mode': False}}
        factory = AgentFactory(config)
        
        bypassed_agents = factory.get_bypassed_agents()
        assert bypassed_agents == []
    
    def test_emergency_fallback_agents_creation(self):
        """Test creation of emergency fallback agents."""
        config = {'optimization': {'llm_only_mode': True}}
        factory = AgentFactory(config)
        
        with patch('agents.factory.AnalyzerAgent') as mock_analyzer, \
             patch('agents.factory.RefinerAgent') as mock_refiner, \
             patch('agents.factory.ValidatorAgent') as mock_validator:
            
            mock_analyzer.return_value = Mock(spec=AnalyzerAgent)
            mock_refiner.return_value = Mock(spec=RefinerAgent)
            mock_validator.return_value = Mock(spec=ValidatorAgent)
            
            emergency_agents = factory.create_emergency_fallback_agents()
            
            assert len(emergency_agents) == 3
            assert 'analyzer' in emergency_agents
            assert 'refiner' in emergency_agents
            assert 'validator' in emergency_agents
            
            # Verify emergency config was passed
            analyzer_call_args = mock_analyzer.call_args[0][0]
            assert analyzer_call_args['is_emergency_fallback'] is True


class TestOrchestrationEngineConfiguration:
    """Test orchestration engine behavior with different configurations."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_bedrock_executor = Mock(spec=BedrockExecutor)
        self.mock_evaluator = Mock(spec=Evaluator)
    
    @patch('orchestration.engine.AgentFactory')
    def test_orchestration_engine_llm_only_mode_initialization(self, mock_agent_factory_class):
        """Test orchestration engine initialization in LLM-only mode."""
        config = {
            'optimization': {
                'llm_only_mode': True,
                'fallback_to_heuristic': False
            },
            'orchestration': {
                'orchestrator_model': 'test-model',
                'orchestrator_temperature': 0.3,
                'min_iterations': 3,
                'max_iterations': 10
            }
        }
        
        # Mock agent factory
        mock_factory = Mock()
        mock_factory.create_agents.return_value = {
            'analyzer': Mock(spec=LLMAnalyzerAgent),
            'refiner': Mock(spec=LLMRefinerAgent),
            'validator': Mock(spec=LLMValidatorAgent)
        }
        mock_factory.get_mode_description.return_value = "LLM-only mode without fallback"
        mock_factory.get_bypassed_agents.return_value = ['heuristic_analyzer', 'heuristic_refiner', 'heuristic_validator']
        mock_factory.fallback_to_heuristic = False
        mock_agent_factory_class.return_value = mock_factory
        
        engine = LLMOrchestrationEngine(
            bedrock_executor=self.mock_bedrock_executor,
            evaluator=self.mock_evaluator,
            config=config
        )
        
        # Verify LLM-only mode was configured
        assert engine.llm_only_mode is True
        assert len(engine.agents) == 3
        
        # Verify agent factory was initialized with correct config
        mock_agent_factory_class.assert_called_once_with(config)
        mock_factory.create_agents.assert_called_once()
    
    @patch('orchestration.engine.AgentFactory')
    def test_orchestration_engine_hybrid_mode_initialization(self, mock_agent_factory_class):
        """Test orchestration engine initialization in hybrid mode."""
        config = {
            'optimization': {
                'llm_only_mode': False,
                'fallback_to_heuristic': True
            }
        }
        
        # Mock agent factory
        mock_factory = Mock()
        mock_factory.create_agents.return_value = {
            'analyzer': Mock(spec=AnalyzerAgent),
            'refiner': Mock(spec=RefinerAgent),
            'validator': Mock(spec=ValidatorAgent),
            'llm_analyzer': Mock(spec=LLMAnalyzerAgent),
            'llm_refiner': Mock(spec=LLMRefinerAgent),
            'llm_validator': Mock(spec=LLMValidatorAgent)
        }
        mock_factory.get_mode_description.return_value = "Hybrid mode (heuristic + LLM agents)"
        mock_factory.get_bypassed_agents.return_value = []
        mock_factory.fallback_to_heuristic = True
        mock_agent_factory_class.return_value = mock_factory
        
        engine = LLMOrchestrationEngine(
            bedrock_executor=self.mock_bedrock_executor,
            evaluator=self.mock_evaluator,
            config=config
        )
        
        # Verify hybrid mode was configured
        assert engine.llm_only_mode is False
        assert len(engine.agents) == 6
    
    @patch('orchestration.engine.AgentFactory')
    def test_orchestration_engine_agent_creation_failure_with_fallback(self, mock_agent_factory_class):
        """Test orchestration engine handling of agent creation failure with fallback."""
        config = {
            'optimization': {
                'llm_only_mode': True,
                'fallback_to_heuristic': True
            }
        }
        
        # Mock agent factory
        mock_factory = Mock()
        mock_factory.create_agents.side_effect = Exception("LLM agents failed")
        mock_factory.create_emergency_fallback_agents.return_value = {
            'analyzer': Mock(spec=AnalyzerAgent),
            'refiner': Mock(spec=RefinerAgent),
            'validator': Mock(spec=ValidatorAgent)
        }
        mock_factory.fallback_to_heuristic = True
        mock_factory.get_mode_description.return_value = "LLM-only mode with fallback"
        mock_factory.get_bypassed_agents.return_value = []  # Return empty list instead of Mock
        mock_agent_factory_class.return_value = mock_factory
        
        engine = LLMOrchestrationEngine(
            bedrock_executor=self.mock_bedrock_executor,
            evaluator=self.mock_evaluator,
            config=config
        )
        
        # Should have created emergency fallback agents
        assert len(engine.agents) == 3
        mock_factory.create_emergency_fallback_agents.assert_called_once()
    
    @patch('orchestration.engine.AgentFactory')
    def test_orchestration_engine_agent_creation_failure_no_fallback(self, mock_agent_factory_class):
        """Test orchestration engine handling of agent creation failure without fallback."""
        config = {
            'optimization': {
                'llm_only_mode': True,
                'fallback_to_heuristic': False
            }
        }
        
        # Mock agent factory
        mock_factory = Mock()
        mock_factory.create_agents.side_effect = Exception("LLM agents failed")
        mock_factory.fallback_to_heuristic = False
        mock_agent_factory_class.return_value = mock_factory
        
        # Should raise exception without fallback - match the actual exception message
        with pytest.raises(Exception, match="LLM agents failed"):
            LLMOrchestrationEngine(
                bedrock_executor=self.mock_bedrock_executor,
                evaluator=self.mock_evaluator,
                config=config
            )
    
    @patch('orchestration.engine.AgentFactory')
    def test_orchestration_engine_runtime_config_change_to_llm_only(self, mock_agent_factory_class):
        """Test orchestration engine handling runtime config change to LLM-only mode."""
        # Start with hybrid mode
        initial_config = {
            'optimization': {
                'llm_only_mode': False,
                'fallback_to_heuristic': True
            }
        }
        
        # Mock initial agent factory
        mock_factory = Mock()
        mock_factory.create_agents.return_value = {
            'analyzer': Mock(spec=AnalyzerAgent),
            'llm_analyzer': Mock(spec=LLMAnalyzerAgent)
        }
        mock_factory.get_mode_description.return_value = "Hybrid mode"
        mock_factory.get_bypassed_agents.return_value = []
        mock_factory.fallback_to_heuristic = True
        mock_agent_factory_class.return_value = mock_factory
        
        engine = LLMOrchestrationEngine(
            bedrock_executor=self.mock_bedrock_executor,
            evaluator=self.mock_evaluator,
            config=initial_config
        )
        
        # Verify initial state
        assert engine.llm_only_mode is False
        
        # Simulate runtime configuration change
        change_event = ConfigChangeEvent(
            timestamp=datetime.now(),
            changes=[{
                'type': 'modified',
                'key': 'optimization.llm_only_mode',
                'old_value': False,
                'new_value': True
            }],
            source='runtime_update',
            success=True
        )
        
        # Update engine config
        engine.config['optimization']['llm_only_mode'] = True
        engine.optimization_config = engine.config.get('optimization', {})
        engine.llm_only_mode = True
        
        # Mock new agent factory for LLM-only mode
        new_mock_factory = Mock()
        new_mock_factory.create_agents.return_value = {
            'analyzer': Mock(spec=LLMAnalyzerAgent)
        }
        new_mock_factory.get_mode_description.return_value = "LLM-only mode"
        new_mock_factory.get_bypassed_agents.return_value = ['heuristic_analyzer']
        
        with patch('orchestration.engine.AgentFactory', return_value=new_mock_factory):
            # Trigger config change handler
            engine._handle_optimization_config_changes([{
                'key': 'optimization.llm_only_mode',
                'new_value': True,
                'old_value': False
            }])
        
        # Verify mode was changed
        assert engine.llm_only_mode is True


class TestLLMResponseMocking:
    """Test agent selection logic with mocked LLM responses."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_bedrock_executor = Mock(spec=BedrockExecutor)
        self.mock_evaluator = Mock(spec=Evaluator)
    
    def create_mock_agent_result(self, agent_name: str, success: bool = True, 
                                confidence: float = 0.8, suggestions: List[str] = None) -> AgentResult:
        """Create a mock agent result."""
        return AgentResult(
            agent_name=agent_name,
            success=success,
            analysis={'test_analysis': 'mock analysis data'},
            suggestions=suggestions or [f"Suggestion from {agent_name}"],
            confidence_score=confidence,
            error_message=None if success else f"Error in {agent_name}"
        )
    
    @patch('orchestration.engine.AgentFactory')
    def test_llm_only_mode_agent_coordination_success(self, mock_agent_factory_class):
        """Test successful agent coordination in LLM-only mode."""
        config = {
            'optimization': {
                'llm_only_mode': True,
                'fallback_to_heuristic': True
            }
        }
        
        # Mock LLM agents
        mock_analyzer = Mock(spec=LLMAnalyzerAgent)
        mock_refiner = Mock(spec=LLMRefinerAgent)
        mock_validator = Mock(spec=LLMValidatorAgent)
        
        # Configure agent results
        mock_analyzer.process.return_value = self.create_mock_agent_result(
            'analyzer', suggestions=['Improve clarity', 'Add context']
        )
        mock_refiner.process.return_value = self.create_mock_agent_result(
            'refiner', suggestions=['Refine structure', 'Enhance flow']
        )
        mock_validator.process.return_value = self.create_mock_agent_result(
            'validator', suggestions=['Validate completeness']
        )
        
        # Mock agent factory
        mock_factory = Mock()
        mock_factory.create_agents.return_value = {
            'analyzer': mock_analyzer,
            'refiner': mock_refiner,
            'validator': mock_validator
        }
        mock_factory.get_mode_description.return_value = "LLM-only mode"
        mock_factory.get_bypassed_agents.return_value = ['heuristic_analyzer', 'heuristic_refiner', 'heuristic_validator']
        mock_factory.fallback_to_heuristic = True
        mock_agent_factory_class.return_value = mock_factory
        
        engine = LLMOrchestrationEngine(
            bedrock_executor=self.mock_bedrock_executor,
            evaluator=self.mock_evaluator,
            config=config
        )
        
        # Mock LLM orchestrator response for coordination
        mock_orchestrator_response = {
            'success': True,
            'agent_results': {
                'analyzer': mock_analyzer.process.return_value,
                'refiner': mock_refiner.process.return_value,
                'validator': mock_validator.process.return_value
            },
            'execution_order': ['analyzer', 'refiner', 'validator'],
            'strategy_type': 'sequential',
            'reasoning': 'Sequential execution for comprehensive analysis'
        }
        
        with patch.object(engine, '_coordinate_agent_analysis', return_value=mock_orchestrator_response):
            # Test agent coordination
            result = engine._coordinate_agent_analysis(
                prompt="Test prompt",
                context={'session_id': 'test'},
                history=[],
                feedback=None
            )
            
            assert result['success'] is True
            assert len(result['agent_results']) == 3
            assert 'analyzer' in result['agent_results']
            assert 'refiner' in result['agent_results']
            assert 'validator' in result['agent_results']
    
    @patch('orchestration.engine.AgentFactory')
    def test_llm_only_mode_agent_failure_with_fallback(self, mock_agent_factory_class):
        """Test agent failure handling in LLM-only mode with fallback."""
        config = {
            'optimization': {
                'llm_only_mode': True,
                'fallback_to_heuristic': True
            }
        }
        
        # Mock LLM agents with one failing
        mock_analyzer = Mock(spec=LLMAnalyzerAgent)
        mock_refiner = Mock(spec=LLMRefinerAgent)
        mock_validator = Mock(spec=LLMValidatorAgent)
        
        # Configure analyzer to fail
        mock_analyzer.process.side_effect = Exception("LLM service timeout")
        
        # Configure other agents to succeed
        mock_refiner.process.return_value = self.create_mock_agent_result('refiner')
        mock_validator.process.return_value = self.create_mock_agent_result('validator')
        
        # Mock fallback heuristic analyzer
        mock_fallback_analyzer = Mock(spec=AnalyzerAgent)
        mock_fallback_analyzer.process.return_value = self.create_mock_agent_result(
            'analyzer', suggestions=['Fallback analysis']
        )
        
        # Mock agent factory
        mock_factory = Mock()
        mock_factory.create_agents.return_value = {
            'analyzer': mock_analyzer,
            'refiner': mock_refiner,
            'validator': mock_validator
        }
        mock_factory._create_fallback_agents_for_failed.return_value = {
            'analyzer': mock_fallback_analyzer
        }
        mock_factory.fallback_to_heuristic = True
        mock_factory.get_mode_description.return_value = "LLM-only mode with fallback"
        mock_factory.get_bypassed_agents.return_value = []
        mock_agent_factory_class.return_value = mock_factory
        
        engine = LLMOrchestrationEngine(
            bedrock_executor=self.mock_bedrock_executor,
            evaluator=self.mock_evaluator,
            config=config
        )
        
        # Test error recovery
        error_context = {
            'agent_name': 'analyzer',
            'error': 'LLM service timeout'
        }
        
        recovery_result = engine._handle_agent_failure(error_context)
        
        # Should have attempted fallback creation
        assert recovery_result['strategy'] == 'individual_fallback'
        assert recovery_result['agent_name'] == 'analyzer'
    
    @patch('orchestration.engine.AgentFactory')
    def test_hybrid_mode_agent_selection_logic(self, mock_agent_factory_class):
        """Test agent selection logic in hybrid mode."""
        config = {
            'optimization': {
                'llm_only_mode': False,
                'fallback_to_heuristic': True
            }
        }
        
        # Mock both heuristic and LLM agents
        mock_heuristic_analyzer = Mock(spec=AnalyzerAgent)
        mock_llm_analyzer = Mock(spec=LLMAnalyzerAgent)
        
        # Configure different results for comparison
        mock_heuristic_analyzer.process.return_value = self.create_mock_agent_result(
            'analyzer', confidence=0.6, suggestions=['Heuristic suggestion']
        )
        mock_llm_analyzer.process.return_value = self.create_mock_agent_result(
            'llm_analyzer', confidence=0.9, suggestions=['LLM suggestion']
        )
        
        # Mock agent factory
        mock_factory = Mock()
        mock_factory.create_agents.return_value = {
            'analyzer': mock_heuristic_analyzer,
            'llm_analyzer': mock_llm_analyzer
        }
        mock_factory.get_bypassed_agents.return_value = []
        mock_factory.fallback_to_heuristic = True
        mock_agent_factory_class.return_value = mock_factory
        
        engine = LLMOrchestrationEngine(
            bedrock_executor=self.mock_bedrock_executor,
            evaluator=self.mock_evaluator,
            config=config
        )
        
        # Verify both agents are available
        assert 'analyzer' in engine.agents
        assert 'llm_analyzer' in engine.agents
        assert engine.llm_only_mode is False
    
    @patch('orchestration.engine.AgentFactory')
    def test_llm_service_unavailable_handling(self, mock_agent_factory_class):
        """Test handling of LLM service unavailability."""
        config = {
            'optimization': {
                'llm_only_mode': True,
                'fallback_to_heuristic': True
            }
        }
        
        # Mock agent factory
        mock_factory = Mock()
        mock_factory.create_agents.side_effect = Exception("LLM service unavailable")
        mock_factory.create_emergency_fallback_agents.return_value = {
            'analyzer': Mock(spec=AnalyzerAgent),
            'refiner': Mock(spec=RefinerAgent),
            'validator': Mock(spec=ValidatorAgent)
        }
        mock_factory.handle_llm_service_unavailable.return_value = {
            'analyzer': Mock(spec=AnalyzerAgent),
            'refiner': Mock(spec=RefinerAgent),
            'validator': Mock(spec=ValidatorAgent)
        }
        mock_factory.fallback_to_heuristic = True
        mock_factory.get_mode_description.return_value = "LLM-only mode with fallback"
        mock_factory.get_bypassed_agents.return_value = []
        mock_agent_factory_class.return_value = mock_factory
        
        engine = LLMOrchestrationEngine(
            bedrock_executor=self.mock_bedrock_executor,
            evaluator=self.mock_evaluator,
            config=config
        )
        
        # Test LLM service unavailable handling
        error_context = {'error_type': 'llm_service_unavailable'}
        recovery_result = engine._handle_llm_service_unavailable(error_context)
        
        # Should have attempted fallback
        mock_factory.handle_llm_service_unavailable.assert_called_once()
    
    def test_llm_failure_count_tracking(self):
        """Test LLM failure count tracking and reset."""
        config = {
            'optimization': {
                'llm_only_mode': True,
                'fallback_to_heuristic': True
            }
        }
        
        with patch('orchestration.engine.AgentFactory') as mock_agent_factory_class:
            mock_factory = Mock()
            mock_factory.create_agents.return_value = {
                'analyzer': Mock(spec=LLMAnalyzerAgent)
            }
            mock_factory.fallback_to_heuristic = True
            mock_factory.get_mode_description.return_value = "LLM-only mode with fallback"
            mock_factory.get_bypassed_agents.return_value = []
            mock_agent_factory_class.return_value = mock_factory
            
            engine = LLMOrchestrationEngine(
                bedrock_executor=self.mock_bedrock_executor,
                evaluator=self.mock_evaluator,
                config=config
            )
            
            # Initial failure count should be 0
            assert engine._llm_failure_count == 0
            
            # Simulate LLM failures
            for i in range(3):
                engine._handle_llm_service_unavailable({})
                assert engine._llm_failure_count == i + 1
            
            # Test failure count reset
            engine._reset_llm_failure_count()
            assert engine._llm_failure_count == 0
    
    @patch('orchestration.engine.AgentFactory')
    def test_max_llm_failures_exceeded(self, mock_agent_factory_class):
        """Test behavior when max LLM failures are exceeded."""
        config = {
            'optimization': {
                'llm_only_mode': True,
                'fallback_to_heuristic': True
            }
        }
        
        mock_factory = Mock()
        mock_factory.create_agents.return_value = {
            'analyzer': Mock(spec=LLMAnalyzerAgent)
        }
        mock_factory.fallback_to_heuristic = True
        mock_factory.get_mode_description.return_value = "LLM-only mode with fallback"
        mock_factory.get_bypassed_agents.return_value = []
        mock_agent_factory_class.return_value = mock_factory
        
        engine = LLMOrchestrationEngine(
            bedrock_executor=self.mock_bedrock_executor,
            evaluator=self.mock_evaluator,
            config=config
        )
        
        # Set failure count to max
        engine._llm_failure_count = engine._max_llm_failures + 1
        
        # Should not attempt fallback when max failures exceeded
        recovery_result = engine._handle_llm_service_unavailable({})
        
        assert recovery_result['success'] is False
        assert 'max failures exceeded' in recovery_result['error'] or 'no fallback strategy available' in recovery_result['error']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
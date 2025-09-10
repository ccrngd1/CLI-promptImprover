"""
Tests for orchestration engine runtime configuration changes.

This module tests the orchestration engine's ability to handle
runtime configuration changes, particularly for the optimization
section and LLM-only mode switching.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from orchestration.engine import LLMOrchestrationEngine
from bedrock.executor import BedrockExecutor
from evaluation.evaluator import Evaluator
from config_loader import ConfigurationLoader, ConfigChangeEvent
from datetime import datetime


class TestOrchestrationRuntimeConfig:
    """Test orchestration engine runtime configuration handling."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create temporary config file
        self.temp_dir = tempfile.mkdtemp()
        self.config_file = Path(self.temp_dir) / 'test_config.json'
        
        # Sample configuration
        self.sample_config = {
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
        with open(self.config_file, 'w') as f:
            json.dump(self.sample_config, f, indent=2)
        
        # Create mock components
        self.mock_bedrock_executor = Mock(spec=BedrockExecutor)
        self.mock_evaluator = Mock(spec=Evaluator)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        # Clean up temporary files
        if self.config_file.exists():
            self.config_file.unlink()
        Path(self.temp_dir).rmdir()
        
        # Reset global config loader
        import config_loader
        config_loader._global_config_loader = None
    
    @patch('config_loader.get_config_loader')
    def test_orchestration_engine_initialization_with_config_loader(self, mock_get_config_loader):
        """Test orchestration engine initialization with configuration loader."""
        # Set up mock config loader
        mock_config_loader = Mock(spec=ConfigurationLoader)
        mock_config_loader.get_config.return_value = self.sample_config
        mock_get_config_loader.return_value = mock_config_loader
        
        # Initialize orchestration engine
        engine = LLMOrchestrationEngine(
            bedrock_executor=self.mock_bedrock_executor,
            evaluator=self.mock_evaluator,
            config=self.sample_config
        )
        
        # Verify config loader was used
        mock_get_config_loader.assert_called_once()
        mock_config_loader.add_change_listener.assert_called_once()
        
        # Verify initial configuration
        assert engine.llm_only_mode is False
        assert engine.optimization_config['fallback_to_heuristic'] is True
    
    def test_llm_only_mode_runtime_change(self):
        """Test runtime change of LLM-only mode."""
        # Initialize with real config loader
        config_loader = ConfigurationLoader(str(self.config_file))
        
        with patch('config_loader.get_config_loader', return_value=config_loader):
            engine = LLMOrchestrationEngine(
                bedrock_executor=self.mock_bedrock_executor,
                evaluator=self.mock_evaluator,
                config=self.sample_config
            )
            
            # Verify initial state
            assert engine.llm_only_mode is False
            initial_agents = list(engine.agents.keys())
            
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
            
            # Update config loader state
            config_loader.update_config({'optimization.llm_only_mode': True})
            
            # Trigger change handler
            engine._handle_config_change(change_event)
            
            # Verify mode was changed
            assert engine.llm_only_mode is True
            
            # Verify agents were recreated
            new_agents = list(engine.agents.keys())
            # In LLM-only mode, should have LLM agents only
            assert 'analyzer' in new_agents or 'llm_analyzer' in new_agents
    
    def test_fallback_config_runtime_change(self):
        """Test runtime change of fallback configuration."""
        config_loader = ConfigurationLoader(str(self.config_file))
        
        with patch('config_loader.get_config_loader', return_value=config_loader):
            engine = LLMOrchestrationEngine(
                bedrock_executor=self.mock_bedrock_executor,
                evaluator=self.mock_evaluator,
                config=self.sample_config
            )
            
            # Verify initial state
            assert engine.agent_factory.fallback_to_heuristic is True
            
            # Simulate runtime configuration change
            change_event = ConfigChangeEvent(
                timestamp=datetime.now(),
                changes=[{
                    'type': 'modified',
                    'key': 'optimization.fallback_to_heuristic',
                    'old_value': True,
                    'new_value': False
                }],
                source='runtime_update',
                success=True
            )
            
            # Update config loader state
            config_loader.update_config({'optimization.fallback_to_heuristic': False})
            
            # Trigger change handler
            engine._handle_config_change(change_event)
            
            # Verify fallback setting was changed
            assert engine.agent_factory.fallback_to_heuristic is False
    
    def test_orchestration_config_runtime_change(self):
        """Test runtime change of orchestration configuration."""
        config_loader = ConfigurationLoader(str(self.config_file))
        
        with patch('config_loader.get_config_loader', return_value=config_loader):
            engine = LLMOrchestrationEngine(
                bedrock_executor=self.mock_bedrock_executor,
                evaluator=self.mock_evaluator,
                config=self.sample_config
            )
            
            # Verify initial state
            assert engine.convergence_config['max_iterations'] == 10
            assert engine.orchestrator_model_config.temperature == 0.3
            
            # Simulate runtime configuration changes
            change_event = ConfigChangeEvent(
                timestamp=datetime.now(),
                changes=[
                    {
                        'type': 'modified',
                        'key': 'orchestration.max_iterations',
                        'old_value': 10,
                        'new_value': 15
                    },
                    {
                        'type': 'modified',
                        'key': 'orchestration.orchestrator_temperature',
                        'old_value': 0.3,
                        'new_value': 0.5
                    }
                ],
                source='runtime_update',
                success=True
            )
            
            # Update config loader state
            config_loader.update_config({
                'orchestration.max_iterations': 15,
                'orchestration.orchestrator_temperature': 0.5
            })
            
            # Trigger change handler
            engine._handle_config_change(change_event)
            
            # Verify orchestration settings were changed
            assert engine.convergence_config['max_iterations'] == 15
            assert engine.orchestrator_model_config.temperature == 0.5
    
    def test_manual_config_reload(self):
        """Test manual configuration reload."""
        config_loader = ConfigurationLoader(str(self.config_file))
        
        with patch('config_loader.get_config_loader', return_value=config_loader):
            engine = LLMOrchestrationEngine(
                bedrock_executor=self.mock_bedrock_executor,
                evaluator=self.mock_evaluator,
                config=self.sample_config
            )
            
            # Modify config file
            modified_config = self.sample_config.copy()
            modified_config['optimization']['llm_only_mode'] = True
            modified_config['orchestration']['max_iterations'] = 20
            
            with open(self.config_file, 'w') as f:
                json.dump(modified_config, f, indent=2)
            
            # Manually reload configuration
            result = engine.reload_configuration()
            
            # Verify reload was successful
            assert result['success'] is True
            assert result['reloaded'] is True
    
    def test_manual_config_update(self):
        """Test manual configuration update through engine."""
        config_loader = ConfigurationLoader(str(self.config_file))
        
        with patch('config_loader.get_config_loader', return_value=config_loader):
            engine = LLMOrchestrationEngine(
                bedrock_executor=self.mock_bedrock_executor,
                evaluator=self.mock_evaluator,
                config=self.sample_config
            )
            
            # Update configuration through engine
            result = engine.update_configuration({
                'optimization.llm_only_mode': True,
                'orchestration.max_iterations': 12
            })
            
            # Verify update was successful
            assert result['success'] is True
            assert len(result['applied_changes']) == 2
            
            # Verify changes were applied to engine
            assert engine.llm_only_mode is True
            assert engine.convergence_config['max_iterations'] == 12
    
    def test_failed_config_change_handling(self):
        """Test handling of failed configuration changes."""
        config_loader = ConfigurationLoader(str(self.config_file))
        
        with patch('config_loader.get_config_loader', return_value=config_loader):
            engine = LLMOrchestrationEngine(
                bedrock_executor=self.mock_bedrock_executor,
                evaluator=self.mock_evaluator,
                config=self.sample_config
            )
            
            # Simulate failed configuration change
            change_event = ConfigChangeEvent(
                timestamp=datetime.now(),
                changes=[],
                source='runtime_update',
                success=False,
                errors=['Test configuration error']
            )
            
            # Trigger change handler (should not raise exception)
            engine._handle_config_change(change_event)
            
            # Verify original configuration is unchanged
            assert engine.llm_only_mode is False
    
    def test_agent_recreation_failure_handling(self):
        """Test handling of agent recreation failures during config changes."""
        config_loader = ConfigurationLoader(str(self.config_file))
        
        with patch('config_loader.get_config_loader', return_value=config_loader):
            engine = LLMOrchestrationEngine(
                bedrock_executor=self.mock_bedrock_executor,
                evaluator=self.mock_evaluator,
                config=self.sample_config
            )
            
            # Store original agents
            original_agents = engine.agents.copy()
            
            # Mock agent factory to fail on recreation
            with patch.object(engine, 'agent_factory') as mock_factory:
                mock_factory.create_agents.side_effect = Exception("Agent creation failed")
                
                # Simulate configuration change that would trigger agent recreation
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
                
                # Update config loader state
                config_loader.update_config({'optimization.llm_only_mode': True})
                
                # Trigger change handler (should handle failure gracefully)
                with pytest.raises(Exception, match="Agent creation failed"):
                    engine._handle_config_change(change_event)
    
    def test_multiple_config_changes(self):
        """Test handling multiple configuration changes in sequence."""
        config_loader = ConfigurationLoader(str(self.config_file))
        
        with patch('config_loader.get_config_loader', return_value=config_loader):
            engine = LLMOrchestrationEngine(
                bedrock_executor=self.mock_bedrock_executor,
                evaluator=self.mock_evaluator,
                config=self.sample_config
            )
            
            # Apply multiple configuration changes
            changes = [
                {'optimization.llm_only_mode': True},
                {'optimization.fallback_to_heuristic': False},
                {'orchestration.max_iterations': 15},
                {'orchestration.orchestrator_temperature': 0.4}
            ]
            
            for change in changes:
                result = engine.update_configuration(change)
                assert result['success'] is True
            
            # Verify all changes were applied
            assert engine.llm_only_mode is True
            assert engine.agent_factory.fallback_to_heuristic is False
            assert engine.convergence_config['max_iterations'] == 15
            assert engine.orchestrator_model_config.temperature == 0.4


if __name__ == '__main__':
    pytest.main([__file__])
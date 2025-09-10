"""
Tests for the configuration loader and runtime configuration changes.

This module tests the centralized configuration loading system,
validation, and runtime configuration change capabilities.
"""

import json
import os
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from config_loader import (
    ConfigurationLoader, ConfigChangeEvent, get_config_loader,
    load_config, get_optimization_config, is_llm_only_mode, update_config_runtime
)


class TestConfigurationLoader:
    """Test the ConfigurationLoader class."""
    
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
                'min_iterations': 3,
                'max_iterations': 10,
                'score_improvement_threshold': 0.02
            },
            'optimization': {
                'llm_only_mode': False,
                'fallback_to_heuristic': True
            },
            'agents': {
                'analyzer': {
                    'enabled': True,
                    'model': 'anthropic.claude-3-sonnet-20240229-v1:0',
                    'temperature': 0.2
                }
            }
        }
        
        # Write sample config to file
        with open(self.config_file, 'w') as f:
            json.dump(self.sample_config, f, indent=2)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        # Clean up temporary files
        if self.config_file.exists():
            self.config_file.unlink()
        Path(self.temp_dir).rmdir()
    
    def test_initialization(self):
        """Test configuration loader initialization."""
        loader = ConfigurationLoader(str(self.config_file))
        
        assert loader is not None
        config = loader.get_config()
        assert config['bedrock']['region'] == 'us-east-1'
        assert config['optimization']['llm_only_mode'] is False
        assert config['optimization']['fallback_to_heuristic'] is True
    
    def test_get_config_value(self):
        """Test getting specific configuration values."""
        loader = ConfigurationLoader(str(self.config_file))
        
        # Test dot notation access
        assert loader.get_config_value('bedrock.region') == 'us-east-1'
        assert loader.get_config_value('optimization.llm_only_mode') is False
        assert loader.get_config_value('optimization.fallback_to_heuristic') is True
        
        # Test default values
        assert loader.get_config_value('nonexistent.key', 'default') == 'default'
        assert loader.get_config_value('optimization.nonexistent', False) is False
    
    def test_optimization_config_methods(self):
        """Test optimization-specific configuration methods."""
        loader = ConfigurationLoader(str(self.config_file))
        
        # Test optimization config getter
        opt_config = loader.get_optimization_config()
        assert opt_config['llm_only_mode'] is False
        assert opt_config['fallback_to_heuristic'] is True
        
        # Test convenience methods
        assert loader.is_llm_only_mode() is False
        assert loader.is_fallback_enabled() is True
    
    def test_runtime_config_update(self):
        """Test runtime configuration updates."""
        loader = ConfigurationLoader(str(self.config_file))
        
        # Test enabling LLM-only mode
        result = loader.update_config({
            'optimization.llm_only_mode': True
        })
        
        assert result['success'] is True
        assert len(result['applied_changes']) == 1
        assert result['applied_changes'][0]['key'] == 'optimization.llm_only_mode'
        assert result['applied_changes'][0]['new_value'] is True
        assert result['applied_changes'][0]['old_value'] is False
        
        # Verify the change was applied
        assert loader.is_llm_only_mode() is True
        assert loader.get_config_value('optimization.llm_only_mode') is True
    
    def test_multiple_runtime_updates(self):
        """Test multiple runtime configuration updates."""
        loader = ConfigurationLoader(str(self.config_file))
        
        # Test multiple changes at once
        changes = {
            'optimization.llm_only_mode': True,
            'optimization.fallback_to_heuristic': False,
            'orchestration.max_iterations': 15
        }
        
        result = loader.update_config(changes)
        
        assert result['success'] is True
        assert len(result['applied_changes']) == 3
        
        # Verify all changes were applied
        assert loader.is_llm_only_mode() is True
        assert loader.is_fallback_enabled() is False
        assert loader.get_config_value('orchestration.max_iterations') == 15
    
    def test_invalid_config_update(self):
        """Test handling of invalid configuration updates."""
        loader = ConfigurationLoader(str(self.config_file))
        
        # Test invalid optimization value
        result = loader.update_config({
            'optimization.llm_only_mode': 'invalid_value'
        })
        
        assert result['success'] is False
        assert len(result['failed_changes']) > 0
        
        # Verify original value is unchanged
        assert loader.is_llm_only_mode() is False
    
    def test_config_change_listeners(self):
        """Test configuration change listeners."""
        loader = ConfigurationLoader(str(self.config_file))
        
        # Set up listener
        change_events = []
        
        def test_listener(event: ConfigChangeEvent):
            change_events.append(event)
        
        loader.add_change_listener(test_listener)
        
        # Make a configuration change
        loader.update_config({'optimization.llm_only_mode': True})
        
        # Verify listener was called
        assert len(change_events) == 1
        event = change_events[0]
        assert event.success is True
        assert event.source == 'runtime_update'
        assert len(event.changes) == 1
    
    def test_config_file_reload(self):
        """Test reloading configuration from file."""
        loader = ConfigurationLoader(str(self.config_file))
        
        # Modify the config file
        modified_config = self.sample_config.copy()
        modified_config['optimization']['llm_only_mode'] = True
        modified_config['orchestration']['max_iterations'] = 20
        
        with open(self.config_file, 'w') as f:
            json.dump(modified_config, f, indent=2)
        
        # Reload configuration
        result = loader.reload_from_file()
        
        assert result['success'] is True
        assert result['reloaded'] is True
        assert len(result['changes_detected']) > 0
        
        # Verify changes were loaded
        assert loader.is_llm_only_mode() is True
        assert loader.get_config_value('orchestration.max_iterations') == 20
    
    def test_config_validation(self):
        """Test configuration validation."""
        loader = ConfigurationLoader(str(self.config_file))
        
        # Test valid configuration
        validation = loader.validate_current_config()
        assert validation['valid'] is True
        
        # Test invalid configuration update
        loader.update_config({'orchestration.max_iterations': -1}, validate=False)
        
        validation = loader.validate_current_config()
        assert validation['valid'] is False
        assert len(validation['errors']) > 0
    
    def test_convenience_methods(self):
        """Test convenience methods for optimization settings."""
        loader = ConfigurationLoader(str(self.config_file))
        
        # Test enabling LLM-only mode
        result = loader.enable_llm_only_mode(True)
        assert result['success'] is True
        assert loader.is_llm_only_mode() is True
        
        # Test disabling LLM-only mode
        result = loader.enable_llm_only_mode(False)
        assert result['success'] is True
        assert loader.is_llm_only_mode() is False
        
        # Test enabling fallback
        result = loader.enable_fallback(True)
        assert result['success'] is True
        assert loader.is_fallback_enabled() is True
        
        # Test disabling fallback
        result = loader.enable_fallback(False)
        assert result['success'] is True
        assert loader.is_fallback_enabled() is False
    
    def test_thread_safety(self):
        """Test thread safety of configuration operations."""
        loader = ConfigurationLoader(str(self.config_file))
        
        results = []
        errors = []
        
        def update_config_thread(thread_id):
            try:
                for i in range(10):
                    result = loader.update_config({
                        f'test_key_{thread_id}_{i}': f'value_{thread_id}_{i}'
                    })
                    results.append((thread_id, i, result['success']))
                    time.sleep(0.01)  # Small delay to encourage race conditions
            except Exception as e:
                errors.append((thread_id, str(e)))
        
        # Start multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=update_config_thread, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify no errors occurred
        assert len(errors) == 0, f"Thread safety errors: {errors}"
        
        # Verify all updates were successful
        successful_updates = [r for r in results if r[2]]
        assert len(successful_updates) == 50  # 5 threads * 10 updates each
    
    def test_export_config(self):
        """Test configuration export functionality."""
        loader = ConfigurationLoader(str(self.config_file))
        
        # Test JSON export
        export_file = Path(self.temp_dir) / 'exported_config.json'
        result = loader.export_config(str(export_file), 'json')
        
        assert result['success'] is True
        assert export_file.exists()
        
        # Verify exported content
        with open(export_file, 'r') as f:
            exported_config = json.load(f)
        
        assert exported_config['bedrock']['region'] == 'us-east-1'
        assert exported_config['optimization']['llm_only_mode'] is False


class TestGlobalConfigLoader:
    """Test global configuration loader functions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Reset global loader
        import config_loader
        config_loader._global_config_loader = None
    
    def test_get_global_loader(self):
        """Test getting the global configuration loader."""
        loader1 = get_config_loader()
        loader2 = get_config_loader()
        
        # Should return the same instance
        assert loader1 is loader2
    
    def test_global_convenience_functions(self):
        """Test global convenience functions."""
        # These should work without explicit initialization
        config = load_config()
        assert isinstance(config, dict)
        
        opt_config = get_optimization_config()
        assert isinstance(opt_config, dict)
        assert 'llm_only_mode' in opt_config
        
        llm_only = is_llm_only_mode()
        assert isinstance(llm_only, bool)
    
    def test_global_runtime_update(self):
        """Test global runtime configuration update."""
        # Get initial state
        initial_mode = is_llm_only_mode()
        
        # Update configuration
        result = update_config_runtime({
            'optimization.llm_only_mode': not initial_mode
        })
        
        assert result['success'] is True
        
        # Verify change was applied
        new_mode = is_llm_only_mode()
        assert new_mode != initial_mode


class TestConfigChangeEvent:
    """Test ConfigChangeEvent dataclass."""
    
    def test_config_change_event_creation(self):
        """Test creating ConfigChangeEvent instances."""
        event = ConfigChangeEvent(
            timestamp=datetime.now(),
            changes=[{'key': 'test.key', 'value': 'test_value'}],
            source='test',
            success=True,
            errors=None
        )
        
        assert event.success is True
        assert event.source == 'test'
        assert len(event.changes) == 1
        assert event.errors is None
    
    def test_config_change_event_with_errors(self):
        """Test ConfigChangeEvent with errors."""
        event = ConfigChangeEvent(
            timestamp=datetime.now(),
            changes=[],
            source='test',
            success=False,
            errors=['Test error message']
        )
        
        assert event.success is False
        assert len(event.errors) == 1
        assert event.errors[0] == 'Test error message'


if __name__ == '__main__':
    pytest.main([__file__])
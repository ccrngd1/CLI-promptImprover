"""
Centralized configuration loader for the Bedrock Prompt Optimizer.

This module provides a centralized way to load, validate, and manage
configuration across all components of the application, with support
for runtime configuration changes and the new optimization section.
"""

import json
import os
import threading
import time
from pathlib import Path
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass
from datetime import datetime

# Lazy import to avoid circular dependencies
from logging_config import get_logger


@dataclass
class ConfigChangeEvent:
    """Represents a configuration change event."""
    timestamp: datetime
    changes: List[Dict[str, Any]]
    source: str  # 'file_reload', 'runtime_update', 'initialization'
    success: bool
    errors: List[str] = None


class ConfigurationLoader:
    """
    Centralized configuration loader with runtime change support.
    
    This class provides a single point of configuration management for the
    entire application, with support for:
    - Loading configuration from files
    - Runtime configuration changes
    - Configuration validation
    - Change notifications to components
    - Thread-safe configuration access
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the configuration loader.
        
        Args:
            config_path: Optional path to configuration file
        """
        # Lazy import to avoid circular dependencies
        from cli.config import ConfigManager
        self._config_manager = ConfigManager(config_path)
        self._current_config = {}
        self._config_lock = threading.RLock()
        self._change_listeners = []
        self._last_reload_time = None
        self._file_watch_enabled = False
        self._logging_configured = False
        
        # Create logger first (it will be reconfigured later)
        self.logger = get_logger('config_loader')
        
        # Load initial configuration
        self._load_initial_config()
        
        # Initialize logging with configuration
        self._setup_logging_from_config()
        self._logging_configured = True
    
    def _load_initial_config(self):
        """Load the initial configuration."""
        try:
            with self._config_lock:
                self._current_config = self._config_manager.load_config()
                self._last_reload_time = datetime.now()
                
                # Validate initial configuration
                validation = self._config_manager.validate_config()
                if not validation['valid']:
                    self.logger.error(
                        f"Initial configuration validation failed: {validation['errors']}"
                    )
                    raise ValueError(f"Invalid configuration: {validation['errors']}")
                
                if validation['warnings']:
                    for warning in validation['warnings']:
                        self.logger.warning(f"Configuration warning: {warning}")
                
                self.logger.info(
                    "Configuration loaded successfully",
                    extra={
                        'config_sections': list(self._current_config.keys()),
                        'optimization_mode': self._current_config.get('optimization', {}).get('llm_only_mode', False)
                    }
                )
                
                # Notify listeners of initial load
                self._notify_change_listeners(ConfigChangeEvent(
                    timestamp=datetime.now(),
                    changes=[{'type': 'initialization', 'config': self._current_config}],
                    source='initialization',
                    success=True
                ))
                
        except Exception as e:
            self.logger.error(f"Failed to load initial configuration: {str(e)}")
            raise
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get the current configuration.
        
        Returns:
            Dictionary containing the current configuration
        """
        with self._config_lock:
            return self._current_config.copy()
    
    def get_config_value(self, key: str, default: Any = None) -> Any:
        """
        Get a specific configuration value using dot notation.
        
        Args:
            key: Configuration key (e.g., 'optimization.llm_only_mode')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        with self._config_lock:
            return self._config_manager.get_config_value(key, default)
    
    def update_config(self, changes: Dict[str, Any], validate: bool = True) -> Dict[str, Any]:
        """
        Update configuration at runtime.
        
        Args:
            changes: Dictionary of configuration changes
            validate: Whether to validate changes before applying
            
        Returns:
            Results of the update operation
        """
        with self._config_lock:
            self.logger.info(
                f"Applying runtime configuration changes: {list(changes.keys())}"
            )
            
            # Apply changes through config manager
            results = self._config_manager.apply_runtime_changes(changes, validate)
            
            if results['success']:
                # Reload configuration to get updated values
                self._current_config = self._config_manager.load_config()
                
                # Create change event
                change_event = ConfigChangeEvent(
                    timestamp=datetime.now(),
                    changes=results['applied_changes'],
                    source='runtime_update',
                    success=True
                )
                
                # Notify listeners
                self._notify_change_listeners(change_event)
                
                self.logger.info(
                    f"Successfully applied {len(results['applied_changes'])} configuration changes"
                )
                
                # Log specific optimization changes
                optimization_changes = [
                    change for change in results['applied_changes']
                    if change['key'].startswith('optimization.')
                ]
                
                if optimization_changes:
                    for change in optimization_changes:
                        self.logger.info(
                            f"Optimization setting changed: {change['key']} = {change['new_value']} (was {change['old_value']})"
                        )
            else:
                # Create failed change event
                change_event = ConfigChangeEvent(
                    timestamp=datetime.now(),
                    changes=[],
                    source='runtime_update',
                    success=False,
                    errors=results['failed_changes']
                )
                
                self._notify_change_listeners(change_event)
                
                self.logger.error(
                    f"Failed to apply configuration changes: {results['failed_changes']}"
                )
            
            return results
    
    def reload_from_file(self) -> Dict[str, Any]:
        """
        Reload configuration from file.
        
        Returns:
            Results of the reload operation
        """
        with self._config_lock:
            self.logger.info("Reloading configuration from file")
            
            reload_results = self._config_manager.reload_config()
            
            if reload_results['success'] and reload_results['reloaded']:
                # Update current configuration
                self._current_config = self._config_manager.load_config()
                self._last_reload_time = datetime.now()
                
                # Create change event
                change_event = ConfigChangeEvent(
                    timestamp=datetime.now(),
                    changes=reload_results['changes_detected'],
                    source='file_reload',
                    success=True
                )
                
                # Notify listeners
                self._notify_change_listeners(change_event)
                
                self.logger.info(
                    f"Configuration reloaded with {len(reload_results['changes_detected'])} changes"
                )
            elif reload_results['success'] and not reload_results['reloaded']:
                self.logger.info("Configuration file unchanged, no reload needed")
            else:
                self.logger.error(f"Configuration reload failed: {reload_results['errors']}")
            
            return reload_results
    
    def validate_current_config(self) -> Dict[str, Any]:
        """
        Validate the current configuration.
        
        Returns:
            Validation results
        """
        with self._config_lock:
            return self._config_manager.validate_config()
    
    def add_change_listener(self, listener: Callable[[ConfigChangeEvent], None]):
        """
        Add a listener for configuration changes.
        
        Args:
            listener: Function to call when configuration changes
        """
        self._change_listeners.append(listener)
        self.logger.debug(f"Added configuration change listener: {listener.__name__}")
    
    def remove_change_listener(self, listener: Callable[[ConfigChangeEvent], None]):
        """
        Remove a configuration change listener.
        
        Args:
            listener: Function to remove from listeners
        """
        if listener in self._change_listeners:
            self._change_listeners.remove(listener)
            self.logger.debug(f"Removed configuration change listener: {listener.__name__}")
    
    def _notify_change_listeners(self, event: ConfigChangeEvent):
        """
        Notify all change listeners of a configuration change.
        
        Args:
            event: Configuration change event
        """
        for listener in self._change_listeners:
            try:
                listener(event)
            except Exception as e:
                self.logger.error(
                    f"Error notifying configuration change listener {listener.__name__}: {str(e)}"
                )
    
    def get_optimization_config(self) -> Dict[str, Any]:
        """
        Get the optimization-specific configuration.
        
        Returns:
            Dictionary containing optimization configuration
        """
        with self._config_lock:
            return self._current_config.get('optimization', {
                'llm_only_mode': False,
                'fallback_to_heuristic': True
            })
    
    def is_llm_only_mode(self) -> bool:
        """
        Check if LLM-only mode is enabled.
        
        Returns:
            True if LLM-only mode is enabled
        """
        return self.get_optimization_config().get('llm_only_mode', False)
    
    def is_fallback_enabled(self) -> bool:
        """
        Check if fallback to heuristic agents is enabled.
        
        Returns:
            True if fallback is enabled
        """
        return self.get_optimization_config().get('fallback_to_heuristic', True)
    
    def enable_llm_only_mode(self, enable: bool = True) -> Dict[str, Any]:
        """
        Enable or disable LLM-only mode.
        
        Args:
            enable: Whether to enable LLM-only mode
            
        Returns:
            Results of the configuration change
        """
        return self.update_config({'optimization.llm_only_mode': enable})
    
    def enable_fallback(self, enable: bool = True) -> Dict[str, Any]:
        """
        Enable or disable fallback to heuristic agents.
        
        Args:
            enable: Whether to enable fallback
            
        Returns:
            Results of the configuration change
        """
        return self.update_config({'optimization.fallback_to_heuristic': enable})
    
    def get_last_reload_time(self) -> Optional[datetime]:
        """
        Get the timestamp of the last configuration reload.
        
        Returns:
            Datetime of last reload or None if never reloaded
        """
        return self._last_reload_time
    
    def _setup_logging_from_config(self):
        """Set up logging using configuration values."""
        try:
            # Temporarily set root logger to ERROR to suppress initialization messages
            import logging
            root_logger = logging.getLogger()
            root_logger.setLevel(logging.ERROR)
            
            # Lazy import to avoid circular dependencies
            from logging_config import setup_logging
            
            logging_config = self._current_config.get('logging', {})
            
            # Extract logging parameters
            log_level = logging_config.get('level', 'INFO')
            log_dir = logging_config.get('log_dir')
            enable_structured_logging = logging_config.get('enable_structured_logging', True)
            enable_performance_logging = logging_config.get('enable_performance_logging', True)
            llm_logging_config = logging_config.get('llm_logging', {})
            
            # Set up logging - this will override the temporary ERROR level
            setup_logging(
                log_level=log_level,
                log_dir=log_dir,
                enable_structured_logging=enable_structured_logging,
                enable_performance_logging=enable_performance_logging,
                llm_logging_config=llm_logging_config
            )
            
        except Exception as e:
            # Fallback to basic logging if configuration fails
            import logging
            logging.basicConfig(level=logging.INFO)
            print(f"Failed to configure logging from config: {str(e)}")
    
    def get_logging_config(self) -> Dict[str, Any]:
        """
        Get the logging-specific configuration.
        
        Returns:
            Dictionary containing logging configuration
        """
        with self._config_lock:
            return self._current_config.get('logging', {})
    
    def get_llm_logging_config(self) -> Dict[str, Any]:
        """
        Get the LLM logging-specific configuration.
        
        Returns:
            Dictionary containing LLM logging configuration
        """
        with self._config_lock:
            return self._current_config.get('logging', {}).get('llm_logging', {})
    
    def export_config(self, export_path: str, format_type: str = 'json') -> Dict[str, Any]:
        """
        Export current configuration to a file.
        
        Args:
            export_path: Path to export the configuration
            format_type: Format to export ('json' or 'yaml')
            
        Returns:
            Results of the export operation
        """
        try:
            with self._config_lock:
                self._config_manager.export_config(export_path, format_type)
                
                return {
                    'success': True,
                    'export_path': export_path,
                    'format': format_type,
                    'timestamp': datetime.now().isoformat()
                }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'export_path': export_path,
                'format': format_type
            }


# Global configuration loader instance
_global_config_loader: Optional[ConfigurationLoader] = None
_loader_lock = threading.Lock()


def get_config_loader(config_path: Optional[str] = None) -> ConfigurationLoader:
    """
    Get the global configuration loader instance.
    
    Args:
        config_path: Optional path to configuration file (only used on first call)
        
    Returns:
        Global ConfigurationLoader instance
    """
    global _global_config_loader
    
    with _loader_lock:
        if _global_config_loader is None:
            _global_config_loader = ConfigurationLoader(config_path)
        
        return _global_config_loader


def load_config() -> Dict[str, Any]:
    """
    Load the current configuration using the global loader.
    
    Returns:
        Current configuration dictionary
    """
    return get_config_loader().get_config()


def get_optimization_config() -> Dict[str, Any]:
    """
    Get optimization configuration using the global loader.
    
    Returns:
        Optimization configuration dictionary
    """
    return get_config_loader().get_optimization_config()


def is_llm_only_mode() -> bool:
    """
    Check if LLM-only mode is enabled using the global loader.
    
    Returns:
        True if LLM-only mode is enabled
    """
    return get_config_loader().is_llm_only_mode()


def update_config_runtime(changes: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update configuration at runtime using the global loader.
    
    Args:
        changes: Dictionary of configuration changes
        
    Returns:
        Results of the update operation
    """
    return get_config_loader().update_config(changes)


def get_llm_logging_config() -> Dict[str, Any]:
    """
    Get LLM logging configuration using the global loader.
    
    Returns:
        LLM logging configuration dictionary
    """
    return get_config_loader().get_llm_logging_config()


def get_logging_config() -> Dict[str, Any]:
    """
    Get logging configuration using the global loader.
    
    Returns:
        Logging configuration dictionary
    """
    return get_config_loader().get_logging_config()
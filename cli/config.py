"""
Configuration management for the Bedrock Prompt Optimizer CLI.

This module handles loading, saving, and managing configuration files for
AWS credentials, model preferences, and orchestration settings.
"""

import json
import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union, List


class ConfigManager:
    """Manages configuration files and settings for the CLI application."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the configuration manager.
        
        Args:
            config_path: Optional path to configuration file
        """
        self.config_path = Path(config_path) if config_path else self._get_default_config_path()
        self.config_data = {}
        self._load_config()
    
    def _get_default_config_path(self) -> Path:
        """Get the default configuration file path."""
        # Try user's home directory first
        home_config = Path.home() / '.bedrock-optimizer' / 'config.yaml'
        if home_config.exists():
            return home_config
        
        # Try current directory
        local_config = Path('./config.yaml')
        if local_config.exists():
            return local_config
        
        # Default to home directory
        return home_config
    
    def _load_config(self):
        """Load configuration from file."""
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    if self.config_path.suffix.lower() == '.json':
                        self.config_data = json.load(f)
                    else:
                        self.config_data = yaml.safe_load(f) or {}
            except Exception as e:
                print(f"Warning: Failed to load configuration from {self.config_path}: {e}")
                self.config_data = {}
        else:
            self.config_data = {}
    
    def load_config(self) -> Dict[str, Any]:
        """
        Load and return the complete configuration.
        
        Returns:
            Dictionary containing all configuration settings
        """
        # Merge with environment variables and defaults
        config = self._get_default_config()
        config.update(self.config_data)
        
        # Override with environment variables
        self._apply_environment_overrides(config)
        
        return config
    
    def save_config(self):
        """Save the current configuration to file."""
        try:
            # Ensure directory exists
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.config_path, 'w') as f:
                if self.config_path.suffix.lower() == '.json':
                    json.dump(self.config_data, f, indent=2)
                else:
                    yaml.dump(self.config_data, f, default_flow_style=False, indent=2)
        except Exception as e:
            raise Exception(f"Failed to save configuration: {e}")
    
    def get_config_value(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value using dot notation.
        
        Args:
            key: Configuration key (e.g., 'bedrock.region')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        config = self.load_config()
        keys = key.split('.')
        value = config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set_config_value(self, key: str, value: Any):
        """
        Set a configuration value using dot notation.
        
        Args:
            key: Configuration key (e.g., 'bedrock.region')
            value: Value to set
        """
        keys = key.split('.')
        config = self.config_data
        
        # Navigate to the parent of the target key
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        # Set the value
        config[keys[-1]] = value
        
        # Save the configuration
        self.save_config()
    
    def create_default_config(self):
        """Create a default configuration file."""
        default_config = self._get_default_config()
        
        # Ensure directory exists
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write default configuration
        with open(self.config_path, 'w') as f:
            yaml.dump(default_config, f, default_flow_style=False, indent=2)
        
        self.config_data = default_config
        print(f"Default configuration created at: {self.config_path}")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get the default configuration structure."""
        return {
            'bedrock': {
                'region': 'us-east-1',
                'profile': None,  # Use default AWS profile
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
            'evaluation': {
                'default_criteria': ['relevance', 'clarity', 'completeness'],
                'scoring_model': 'anthropic.claude-3-sonnet-20240229-v1:0',
                'evaluation_temperature': 0.1,
                'custom_metrics': {}
            },
            'storage': {
                'path': './prompt_history',
                'format': 'json',
                'backup_enabled': True,
                'max_history_size': 1000
            },
            'cli': {
                'default_interactive': True,
                'progress_indicators': True,
                'colored_output': True,
                'verbose_orchestration': False
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
            },
            'best_practices': {
                'repository_path': './best_practices/data',
                'auto_update': True,
                'custom_practices_enabled': True
            }
        }
    
    def _apply_environment_overrides(self, config: Dict[str, Any]):
        """Apply environment variable overrides to configuration."""
        env_mappings = {
            'AWS_REGION': 'bedrock.region',
            'AWS_PROFILE': 'bedrock.profile',
            'BEDROCK_DEFAULT_MODEL': 'bedrock.default_model',
            'OPTIMIZER_MAX_ITERATIONS': 'orchestration.max_iterations',
            'OPTIMIZER_STORAGE_PATH': 'storage.path'
        }
        
        for env_var, config_key in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value:
                self._set_nested_value(config, config_key, env_value)
    
    def _set_nested_value(self, config: Dict[str, Any], key: str, value: str):
        """Set a nested configuration value."""
        keys = key.split('.')
        current = config
        
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        
        # Convert value to appropriate type
        final_key = keys[-1]
        if final_key in ['max_iterations', 'timeout', 'max_retries']:
            try:
                value = int(value)
            except ValueError:
                pass
        elif final_key in ['temperature', 'score_improvement_threshold', 'convergence_confidence_threshold']:
            try:
                value = float(value)
            except ValueError:
                pass
        elif value.lower() in ['true', 'false']:
            value = value.lower() == 'true'
        
        current[final_key] = value
    
    def validate_config(self) -> Dict[str, Any]:
        """
        Validate the current configuration and return validation results.
        
        Returns:
            Dictionary with validation results
        """
        config = self.load_config()
        validation_results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'info': []
        }
        
        # Validate Bedrock configuration
        self._validate_bedrock_config(config.get('bedrock', {}), validation_results)
        
        # Validate orchestration configuration
        self._validate_orchestration_config(config.get('orchestration', {}), validation_results)
        
        # Validate storage configuration
        self._validate_storage_config(config.get('storage', {}), validation_results)
        
        # Validate agent configurations
        self._validate_agents_config(config.get('agents', {}), validation_results)
        
        # Validate evaluation configuration
        self._validate_evaluation_config(config.get('evaluation', {}), validation_results)
        
        # Validate best practices configuration
        self._validate_best_practices_config(config.get('best_practices', {}), validation_results)
        
        # Check for environment variable overrides
        self._check_environment_overrides(validation_results)
        
        return validation_results
    
    def _validate_bedrock_config(self, bedrock_config: Dict[str, Any], results: Dict[str, Any]) -> None:
        """Validate Bedrock-specific configuration."""
        if not bedrock_config.get('region'):
            results['errors'].append("Bedrock region not specified")
            results['valid'] = False
        else:
            # Validate region format
            region = bedrock_config['region']
            if not region.replace('-', '').replace('_', '').isalnum():
                results['warnings'].append(f"Region format may be invalid: {region}")
        
        if not bedrock_config.get('default_model'):
            results['warnings'].append("No default Bedrock model specified")
        else:
            model = bedrock_config['default_model']
            if not model.startswith(('anthropic.', 'amazon.', 'ai21.', 'cohere.', 'meta.')):
                results['warnings'].append(f"Model ID format may be invalid: {model}")
        
        # Validate timeout and retries
        timeout = bedrock_config.get('timeout', 30)
        if not isinstance(timeout, (int, float)) or timeout <= 0:
            results['errors'].append(f"Invalid timeout value: {timeout}")
            results['valid'] = False
        elif timeout > 300:
            results['warnings'].append(f"Timeout value is very high: {timeout}s")
        
        max_retries = bedrock_config.get('max_retries', 3)
        if not isinstance(max_retries, int) or max_retries < 0:
            results['errors'].append(f"Invalid max_retries value: {max_retries}")
            results['valid'] = False
        elif max_retries > 10:
            results['warnings'].append(f"max_retries is very high: {max_retries}")
    
    def _validate_orchestration_config(self, orch_config: Dict[str, Any], results: Dict[str, Any]) -> None:
        """Validate orchestration-specific configuration."""
        max_iter = orch_config.get('max_iterations', 10)
        min_iter = orch_config.get('min_iterations', 3)
        
        if not isinstance(max_iter, int) or max_iter <= 0:
            results['errors'].append(f"Invalid max_iterations: {max_iter}")
            results['valid'] = False
        elif max_iter > 50:
            results['warnings'].append("max_iterations > 50 may be excessive")
        
        if not isinstance(min_iter, int) or min_iter <= 0:
            results['errors'].append(f"Invalid min_iterations: {min_iter}")
            results['valid'] = False
        elif max_iter < min_iter:
            results['errors'].append("max_iterations must be >= min_iterations")
            results['valid'] = False
        
        # Validate threshold values
        threshold = orch_config.get('score_improvement_threshold', 0.02)
        if not isinstance(threshold, (int, float)) or threshold < 0 or threshold > 1:
            results['errors'].append(f"Invalid score_improvement_threshold: {threshold}")
            results['valid'] = False
        
        confidence = orch_config.get('convergence_confidence_threshold', 0.8)
        if not isinstance(confidence, (int, float)) or confidence < 0 or confidence > 1:
            results['errors'].append(f"Invalid convergence_confidence_threshold: {confidence}")
            results['valid'] = False
        
        # Validate orchestrator model settings
        orch_temp = orch_config.get('orchestrator_temperature', 0.3)
        if not isinstance(orch_temp, (int, float)) or orch_temp < 0 or orch_temp > 1:
            results['errors'].append(f"Invalid orchestrator_temperature: {orch_temp}")
            results['valid'] = False
    
    def _validate_storage_config(self, storage_config: Dict[str, Any], results: Dict[str, Any]) -> None:
        """Validate storage-specific configuration."""
        storage_path = storage_config.get('path')
        if storage_path:
            try:
                path_obj = Path(storage_path)
                path_obj.mkdir(parents=True, exist_ok=True)
                # Test write permissions
                test_file = path_obj / '.test_write'
                test_file.write_text('test')
                test_file.unlink()
                results['info'].append(f"Storage path verified: {storage_path}")
            except Exception as e:
                results['errors'].append(f"Cannot create or write to storage path: {e}")
                results['valid'] = False
        
        storage_format = storage_config.get('format', 'json')
        if storage_format not in ['json', 'yaml']:
            results['errors'].append(f"Invalid storage format: {storage_format}")
            results['valid'] = False
        
        max_history = storage_config.get('max_history_size', 1000)
        if not isinstance(max_history, int) or max_history <= 0:
            results['errors'].append(f"Invalid max_history_size: {max_history}")
            results['valid'] = False
        elif max_history > 10000:
            results['warnings'].append(f"max_history_size is very large: {max_history}")
    
    def _validate_agents_config(self, agents_config: Dict[str, Any], results: Dict[str, Any]) -> None:
        """Validate agent-specific configurations."""
        required_agents = ['analyzer', 'refiner', 'validator']
        
        for agent_name in required_agents:
            if agent_name not in agents_config:
                results['warnings'].append(f"Missing configuration for required agent: {agent_name}")
                continue
            
            agent_config = agents_config[agent_name]
            
            if agent_config.get('enabled', True):
                if not agent_config.get('model'):
                    results['warnings'].append(f"No model specified for {agent_name} agent")
                
                temp = agent_config.get('temperature', 0.3)
                if not isinstance(temp, (int, float)) or temp < 0 or temp > 1:
                    results['errors'].append(f"Invalid temperature for {agent_name} agent: {temp}")
                    results['valid'] = False
                
                max_tokens = agent_config.get('max_tokens', 1000)
                if not isinstance(max_tokens, int) or max_tokens <= 0:
                    results['errors'].append(f"Invalid max_tokens for {agent_name} agent: {max_tokens}")
                    results['valid'] = False
                elif max_tokens > 8000:
                    results['warnings'].append(f"max_tokens is very high for {agent_name} agent: {max_tokens}")
    
    def _validate_evaluation_config(self, eval_config: Dict[str, Any], results: Dict[str, Any]) -> None:
        """Validate evaluation-specific configuration."""
        criteria = eval_config.get('default_criteria', [])
        if not isinstance(criteria, list) or not criteria:
            results['warnings'].append("No default evaluation criteria specified")
        
        valid_criteria = ['relevance', 'clarity', 'completeness', 'consistency', 'creativity']
        for criterion in criteria:
            if criterion not in valid_criteria:
                results['warnings'].append(f"Unknown evaluation criterion: {criterion}")
        
        eval_temp = eval_config.get('evaluation_temperature', 0.1)
        if not isinstance(eval_temp, (int, float)) or eval_temp < 0 or eval_temp > 1:
            results['errors'].append(f"Invalid evaluation_temperature: {eval_temp}")
            results['valid'] = False
    
    def _validate_best_practices_config(self, bp_config: Dict[str, Any], results: Dict[str, Any]) -> None:
        """Validate best practices configuration."""
        repo_path = bp_config.get('repository_path', './best_practices/data')
        try:
            Path(repo_path).mkdir(parents=True, exist_ok=True)
            results['info'].append(f"Best practices repository path verified: {repo_path}")
        except Exception as e:
            results['errors'].append(f"Cannot create best practices repository path: {e}")
            results['valid'] = False
    
    def _check_environment_overrides(self, results: Dict[str, Any]) -> None:
        """Check for active environment variable overrides."""
        env_vars = [
            'AWS_REGION', 'AWS_PROFILE', 'BEDROCK_DEFAULT_MODEL',
            'OPTIMIZER_MAX_ITERATIONS', 'OPTIMIZER_STORAGE_PATH'
        ]
        
        active_overrides = []
        for var in env_vars:
            if os.getenv(var):
                active_overrides.append(f"{var}={os.getenv(var)}")
        
        if active_overrides:
            results['info'].append(f"Active environment overrides: {', '.join(active_overrides)}")
    
    def validate_runtime_changes(self, changes: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate configuration changes before applying them at runtime.
        
        Args:
            changes: Dictionary of configuration changes to validate
            
        Returns:
            Validation results for the proposed changes
        """
        validation_results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'safe_to_apply': True
        }
        
        # Check if changes require restart
        restart_required_keys = [
            'bedrock.region', 'bedrock.profile', 'storage.path', 'storage.format'
        ]
        
        requires_restart = any(key in changes for key in restart_required_keys)
        if requires_restart:
            validation_results['warnings'].append("Some changes require application restart to take effect")
            validation_results['safe_to_apply'] = False
        
        # Validate individual changes
        for key, value in changes.items():
            if key.startswith('orchestration.'):
                self._validate_orchestration_change(key, value, validation_results)
            elif key.startswith('agents.'):
                self._validate_agent_change(key, value, validation_results)
            elif key.startswith('evaluation.'):
                self._validate_evaluation_change(key, value, validation_results)
        
        return validation_results
    
    def _validate_orchestration_change(self, key: str, value: Any, results: Dict[str, Any]) -> None:
        """Validate orchestration configuration changes."""
        if key.endswith('_iterations') and (not isinstance(value, int) or value <= 0):
            results['errors'].append(f"Invalid value for {key}: {value}")
            results['valid'] = False
        elif key.endswith('_threshold') and (not isinstance(value, (int, float)) or value < 0 or value > 1):
            results['errors'].append(f"Invalid threshold value for {key}: {value}")
            results['valid'] = False
        elif key.endswith('_temperature') and (not isinstance(value, (int, float)) or value < 0 or value > 1):
            results['errors'].append(f"Invalid temperature value for {key}: {value}")
            results['valid'] = False
    
    def _validate_agent_change(self, key: str, value: Any, results: Dict[str, Any]) -> None:
        """Validate agent configuration changes."""
        if key.endswith('.temperature') and (not isinstance(value, (int, float)) or value < 0 or value > 1):
            results['errors'].append(f"Invalid temperature for {key}: {value}")
            results['valid'] = False
        elif key.endswith('.max_tokens') and (not isinstance(value, int) or value <= 0):
            results['errors'].append(f"Invalid max_tokens for {key}: {value}")
            results['valid'] = False
        elif key.endswith('.enabled') and not isinstance(value, bool):
            results['errors'].append(f"Invalid enabled value for {key}: {value}")
            results['valid'] = False
    
    def _validate_evaluation_change(self, key: str, value: Any, results: Dict[str, Any]) -> None:
        """Validate evaluation configuration changes."""
        if key.endswith('.evaluation_temperature') and (not isinstance(value, (int, float)) or value < 0 or value > 1):
            results['errors'].append(f"Invalid evaluation temperature: {value}")
            results['valid'] = False
    
    def get_aws_config(self) -> Dict[str, Any]:
        """
        Get AWS-specific configuration for Bedrock.
        
        Returns:
            Dictionary with AWS configuration
        """
        config = self.load_config()
        bedrock_config = config.get('bedrock', {})
        
        aws_config = {
            'region_name': bedrock_config.get('region', 'us-east-1')
        }
        
        # Add profile if specified
        profile = bedrock_config.get('profile')
        if profile:
            aws_config['profile_name'] = profile
        
        return aws_config
    
    def get_model_config(self, model_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get model configuration for Bedrock execution.
        
        Args:
            model_id: Optional specific model ID
            
        Returns:
            Dictionary with model configuration
        """
        config = self.load_config()
        bedrock_config = config.get('bedrock', {})
        
        return {
            'model_id': model_id or bedrock_config.get('default_model', 'anthropic.claude-3-sonnet-20240229-v1:0'),
            'temperature': 0.7,
            'max_tokens': 2000,
            'timeout': bedrock_config.get('timeout', 30)
        }
    
    def export_config(self, export_path: str, format_type: str = 'yaml'):
        """
        Export current configuration to a file.
        
        Args:
            export_path: Path to export the configuration
            format_type: Format to export ('yaml' or 'json')
        """
        config = self.load_config()
        export_file = Path(export_path)
        
        try:
            with open(export_file, 'w') as f:
                if format_type.lower() == 'json':
                    json.dump(config, f, indent=2, default=str)
                else:
                    yaml.dump(config, f, default_flow_style=False, indent=2)
            
            print(f"Configuration exported to: {export_file}")
        except Exception as e:
            raise Exception(f"Failed to export configuration: {e}")
    
    def import_config(self, import_path: str):
        """
        Import configuration from a file.
        
        Args:
            import_path: Path to import the configuration from
        """
        import_file = Path(import_path)
        
        if not import_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {import_file}")
        
        try:
            with open(import_file, 'r') as f:
                if import_file.suffix.lower() == '.json':
                    imported_config = json.load(f)
                else:
                    imported_config = yaml.safe_load(f)
            
            # Merge with existing configuration
            self.config_data.update(imported_config)
            self.save_config()
            
            print(f"Configuration imported from: {import_file}")
        except Exception as e:
            raise Exception(f"Failed to import configuration: {e}")
    
    def apply_runtime_changes(self, changes: Dict[str, Any], validate: bool = True) -> Dict[str, Any]:
        """
        Apply configuration changes at runtime.
        
        Args:
            changes: Dictionary of configuration changes to apply
            validate: Whether to validate changes before applying
            
        Returns:
            Results of the operation including any validation errors
        """
        results = {
            'success': True,
            'applied_changes': [],
            'failed_changes': [],
            'warnings': []
        }
        
        if validate:
            validation = self.validate_runtime_changes(changes)
            if not validation['valid']:
                results['success'] = False
                results['failed_changes'] = validation['errors']
                return results
            
            if not validation['safe_to_apply']:
                results['warnings'].extend(validation['warnings'])
        
        # Apply changes
        for key, value in changes.items():
            try:
                old_value = self.get_config_value(key)
                self.set_config_value(key, value)
                results['applied_changes'].append({
                    'key': key,
                    'old_value': old_value,
                    'new_value': value
                })
            except Exception as e:
                results['failed_changes'].append({
                    'key': key,
                    'value': value,
                    'error': str(e)
                })
                results['success'] = False
        
        return results
    
    def get_runtime_changeable_keys(self) -> List[str]:
        """
        Get list of configuration keys that can be safely changed at runtime.
        
        Returns:
            List of configuration keys that support runtime changes
        """
        return [
            'orchestration.orchestrator_temperature',
            'orchestration.orchestrator_max_tokens',
            'orchestration.min_iterations',
            'orchestration.max_iterations',
            'orchestration.score_improvement_threshold',
            'orchestration.convergence_confidence_threshold',
            'evaluation.evaluation_temperature',
            'evaluation.default_criteria',
            'agents.analyzer.temperature',
            'agents.analyzer.max_tokens',
            'agents.analyzer.enabled',
            'agents.refiner.temperature',
            'agents.refiner.max_tokens',
            'agents.refiner.enabled',
            'agents.validator.temperature',
            'agents.validator.max_tokens',
            'agents.validator.enabled',
            'cli.verbose_orchestration',
            'cli.progress_indicators',
            'cli.colored_output',
            'best_practices.auto_update',
            'best_practices.custom_practices_enabled'
        ]
    
    def backup_config(self, backup_name: Optional[str] = None) -> str:
        """
        Create a backup of the current configuration.
        
        Args:
            backup_name: Optional name for the backup file
            
        Returns:
            Path to the backup file
        """
        if backup_name is None:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"config_backup_{timestamp}"
        
        backup_path = self.config_path.parent / f"{backup_name}.yaml"
        
        try:
            with open(backup_path, 'w') as f:
                yaml.dump(self.config_data, f, default_flow_style=False, indent=2)
            
            return str(backup_path)
        except Exception as e:
            raise Exception(f"Failed to create backup: {e}")
    
    def restore_config(self, backup_path: str) -> None:
        """
        Restore configuration from a backup file.
        
        Args:
            backup_path: Path to the backup file
        """
        backup_file = Path(backup_path)
        
        if not backup_file.exists():
            raise FileNotFoundError(f"Backup file not found: {backup_file}")
        
        try:
            with open(backup_file, 'r') as f:
                backup_config = yaml.safe_load(f)
            
            # Validate backup before restoring
            temp_config = ConfigManager()
            temp_config.config_data = backup_config
            validation = temp_config.validate_config()
            
            if not validation['valid']:
                raise ValueError(f"Backup configuration is invalid: {validation['errors']}")
            
            # Create backup of current config before restoring
            current_backup = self.backup_config("pre_restore_backup")
            
            # Restore configuration
            self.config_data = backup_config
            self.save_config()
            
            print(f"Configuration restored from: {backup_file}")
            print(f"Previous configuration backed up to: {current_backup}")
            
        except Exception as e:
            raise Exception(f"Failed to restore configuration: {e}")
    
    def reset_to_defaults(self) -> None:
        """Reset configuration to default values."""
        # Create backup before resetting
        backup_path = self.backup_config("pre_reset_backup")
        
        # Reset to defaults
        self.config_data = self._get_default_config()
        self.save_config()
        
        print("Configuration reset to defaults")
        print(f"Previous configuration backed up to: {backup_path}")
    
    def get_config_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the current configuration.
        
        Returns:
            Dictionary with configuration summary
        """
        config = self.load_config()
        
        return {
            'bedrock': {
                'region': config.get('bedrock', {}).get('region'),
                'default_model': config.get('bedrock', {}).get('default_model'),
                'profile': config.get('bedrock', {}).get('profile')
            },
            'orchestration': {
                'max_iterations': config.get('orchestration', {}).get('max_iterations'),
                'orchestrator_model': config.get('orchestration', {}).get('orchestrator_model')
            },
            'agents': {
                name: {
                    'enabled': agent_config.get('enabled', True),
                    'model': agent_config.get('model')
                }
                for name, agent_config in config.get('agents', {}).items()
            },
            'storage': {
                'path': config.get('storage', {}).get('path'),
                'format': config.get('storage', {}).get('format')
            },
            'config_file': str(self.config_path),
            'environment_overrides': self._get_active_env_overrides()
        }
    
    def _get_active_env_overrides(self) -> Dict[str, str]:
        """Get currently active environment variable overrides."""
        env_mappings = {
            'AWS_REGION': 'bedrock.region',
            'AWS_PROFILE': 'bedrock.profile',
            'BEDROCK_DEFAULT_MODEL': 'bedrock.default_model',
            'OPTIMIZER_MAX_ITERATIONS': 'orchestration.max_iterations',
            'OPTIMIZER_STORAGE_PATH': 'storage.path'
        }
        
        active_overrides = {}
        for env_var, config_key in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value:
                active_overrides[env_var] = env_value
        
        return active_overrides


def create_sample_config():
    """Create a sample configuration file for reference."""
    config_manager = ConfigManager()
    sample_path = Path('./config.sample.yaml')
    
    sample_config = config_manager._get_default_config()
    
    # Add comments to the sample configuration
    sample_content = """# Bedrock Prompt Optimizer Configuration
# This is a sample configuration file showing all available options

# AWS Bedrock Configuration
bedrock:
  region: us-east-1                    # AWS region for Bedrock
  profile: null                        # AWS profile (null for default)
  default_model: anthropic.claude-3-sonnet-20240229-v1:0  # Default model ID
  timeout: 30                          # Request timeout in seconds
  max_retries: 3                       # Maximum retry attempts

# LLM Orchestration Settings
orchestration:
  orchestrator_model: anthropic.claude-3-sonnet-20240229-v1:0
  orchestrator_temperature: 0.3        # Temperature for orchestration decisions
  orchestrator_max_tokens: 2000        # Max tokens for orchestration
  min_iterations: 3                    # Minimum optimization iterations
  max_iterations: 10                   # Maximum optimization iterations
  score_improvement_threshold: 0.02    # Threshold for convergence detection
  stability_window: 3                  # Window for stability analysis
  convergence_confidence_threshold: 0.8 # Confidence threshold for convergence

# Evaluation Configuration
evaluation:
  default_criteria: [relevance, clarity, completeness]  # Default evaluation criteria
  scoring_model: anthropic.claude-3-sonnet-20240229-v1:0  # Model for evaluation
  evaluation_temperature: 0.1          # Temperature for evaluation
  custom_metrics: {}                   # Custom evaluation metrics

# Storage Configuration
storage:
  path: ./prompt_history               # Path for storing session history
  format: json                         # Storage format (json/yaml)
  backup_enabled: true                 # Enable automatic backups
  max_history_size: 1000              # Maximum number of stored sessions

# CLI Configuration
cli:
  default_interactive: true            # Default to interactive mode
  progress_indicators: true            # Show progress indicators
  colored_output: true                 # Enable colored output
  verbose_orchestration: false         # Show detailed orchestration info

# Agent Configuration
agents:
  analyzer:
    enabled: true
    model: anthropic.claude-3-sonnet-20240229-v1:0
    temperature: 0.2
    max_tokens: 1500
  refiner:
    enabled: true
    model: anthropic.claude-3-sonnet-20240229-v1:0
    temperature: 0.4
    max_tokens: 2000
  validator:
    enabled: true
    model: anthropic.claude-3-sonnet-20240229-v1:0
    temperature: 0.1
    max_tokens: 1000

# Best Practices Configuration
best_practices:
  repository_path: ./best_practices/data  # Path to best practices repository
  auto_update: true                    # Auto-update best practices
  custom_practices_enabled: true      # Enable custom practices
"""
    
    with open(sample_path, 'w') as f:
        f.write(sample_content)
    
    print(f"Sample configuration created at: {sample_path}")


if __name__ == '__main__':
    # Create sample configuration when run directly
    create_sample_config()
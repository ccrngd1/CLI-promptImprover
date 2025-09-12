"""
Test suite for LLM logging configuration options and security features.

This module tests the implementation of configurable log levels, output options,
response truncation, sensitive data filtering, and separate log file configuration
for LLM interactions.
"""

import json
import os
import tempfile
import unittest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from typing import Dict, Any

# Import the modules to test
from agents.llm_agent_logger import LLMAgentLogger
from logging_config import setup_logging, LLMInteractionFormatter, _parse_file_size
from config_loader import ConfigurationLoader


class TestLLMLoggingConfiguration(unittest.TestCase):
    """Test LLM logging configuration options and security features."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_config = {
            'logging': {
                'level': 'INFO',
                'log_dir': self.temp_dir,
                'enable_structured_logging': True,
                'enable_performance_logging': True,
                'llm_logging': {
                    'level': 'DEBUG',
                    'log_raw_responses': True,
                    'log_prompts': False,
                    'max_response_log_length': 1000,
                    'max_prompt_log_length': 500,
                    'max_reasoning_log_length': 750,
                    'separate_log_files': True,
                    'enable_security_filtering': True,
                    'sensitive_data_patterns': [
                        'password', 'api_key', 'secret', 'token', 'credential'
                    ],
                    'truncation_indicator': '... [TRUNCATED]',
                    'log_file_prefix': 'test_llm_',
                    'log_file_max_size': '5MB',
                    'log_file_backup_count': 3
                }
            }
        }
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_file_size_parsing(self):
        """Test file size parsing function."""
        # Test various size formats
        self.assertEqual(_parse_file_size('5MB'), 5 * 1024 * 1024)
        self.assertEqual(_parse_file_size('1GB'), 1024 * 1024 * 1024)
        self.assertEqual(_parse_file_size('500KB'), 500 * 1024)
        self.assertEqual(_parse_file_size('1024B'), 1024)
        
        # Test invalid formats (should default to 20MB)
        self.assertEqual(_parse_file_size('invalid'), 20 * 1024 * 1024)
        self.assertEqual(_parse_file_size(''), 20 * 1024 * 1024)
    
    def test_llm_interaction_formatter_configuration(self):
        """Test LLMInteractionFormatter with configuration."""
        llm_config = self.test_config['logging']['llm_logging']
        formatter = LLMInteractionFormatter(llm_config)
        
        # Test configuration loading
        self.assertEqual(formatter.max_response_length, 1000)
        self.assertEqual(formatter.max_prompt_length, 500)
        self.assertEqual(formatter.max_reasoning_length, 750)
        self.assertTrue(formatter.enable_security_filtering)
        self.assertFalse(formatter.log_prompts)
        self.assertTrue(formatter.log_raw_responses)
        self.assertEqual(formatter.truncation_indicator, '... [TRUNCATED]')
    
    def test_security_filtering_in_formatter(self):
        """Test security filtering in LLMInteractionFormatter."""
        llm_config = self.test_config['logging']['llm_logging']
        formatter = LLMInteractionFormatter(llm_config)
        
        # Test sensitive content filtering
        sensitive_text = "Here is my password: secret123 and api_key=abc123"
        filtered_text = formatter._filter_sensitive_content(sensitive_text)
        
        self.assertIn('[FILTERED]', filtered_text)
        self.assertNotIn('secret123', filtered_text)
        self.assertNotIn('abc123', filtered_text)
    
    def test_security_filtering_in_dict(self):
        """Test security filtering in dictionary structures."""
        llm_config = self.test_config['logging']['llm_logging']
        formatter = LLMInteractionFormatter(llm_config)
        
        test_data = {
            'username': 'testuser',
            'password': 'secret123',
            'api_key': 'abc123',
            'normal_field': 'normal_value',
            'nested': {
                'token': 'xyz789',
                'safe_field': 'safe_value'
            }
        }
        
        filtered_data = formatter._filter_sensitive_data_dict(test_data)
        
        self.assertEqual(filtered_data['username'], 'testuser')
        self.assertEqual(filtered_data['password'], '[FILTERED]')
        self.assertEqual(filtered_data['api_key'], '[FILTERED]')
        self.assertEqual(filtered_data['normal_field'], 'normal_value')
        self.assertEqual(filtered_data['nested']['token'], '[FILTERED]')
        self.assertEqual(filtered_data['nested']['safe_field'], 'safe_value')
    
    def test_llm_agent_logger_configuration(self):
        """Test LLMAgentLogger with configuration."""
        llm_config = self.test_config['logging']['llm_logging']
        logger = LLMAgentLogger('test_agent', config=llm_config)
        
        # Test configuration loading
        self.assertEqual(logger.max_response_length, 1000)
        self.assertEqual(logger.max_prompt_length, 500)
        self.assertEqual(logger.max_reasoning_length, 750)
        self.assertTrue(logger.enable_security_filtering)
        self.assertFalse(logger.log_prompts)
        self.assertTrue(logger.log_raw_responses)
        self.assertEqual(logger.truncation_indicator, '... [TRUNCATED]')
    
    def test_response_truncation(self):
        """Test response truncation in LLMAgentLogger."""
        llm_config = self.test_config['logging']['llm_logging']
        logger = LLMAgentLogger('test_agent', config=llm_config)
        
        # Create a long response
        long_response = 'A' * 1500  # Longer than max_response_log_length (1000)
        
        with patch.object(logger.logger, 'info') as mock_log:
            logger.log_llm_response({
                'success': True,
                'response': long_response,
                'model': 'test-model'
            })
            
            # Check that the log was called
            mock_log.assert_called_once()
            
            # Get the logged data
            call_args = mock_log.call_args
            extra_data = call_args[1]['extra']
            
            # Verify truncation occurred
            self.assertTrue(len(extra_data['response']) < len(long_response))
            self.assertIn('... [TRUNCATED]', extra_data['response'])
    
    def test_prompt_logging_configuration(self):
        """Test prompt logging based on configuration."""
        # Test with prompts disabled
        llm_config = self.test_config['logging']['llm_logging'].copy()
        llm_config['log_prompts'] = False
        logger = LLMAgentLogger('test_agent', config=llm_config)
        
        with patch.object(logger.logger, 'info') as mock_log:
            logger.log_llm_call('test prompt', session_id='test_session')
            
            call_args = mock_log.call_args
            extra_data = call_args[1]['extra']
            
            # Prompt should not be logged
            self.assertNotIn('prompt', extra_data)
            self.assertFalse(extra_data.get('prompt_logged', True))
        
        # Test with prompts enabled
        llm_config['log_prompts'] = True
        logger = LLMAgentLogger('test_agent', config=llm_config)
        
        with patch.object(logger.logger, 'info') as mock_log:
            logger.log_llm_call('test prompt', session_id='test_session')
            
            call_args = mock_log.call_args
            extra_data = call_args[1]['extra']
            
            # Prompt should be logged
            self.assertIn('prompt', extra_data)
            self.assertEqual(extra_data['prompt'], 'test prompt')
    
    def test_sensitive_data_filtering_in_logger(self):
        """Test sensitive data filtering in LLMAgentLogger."""
        llm_config = self.test_config['logging']['llm_logging']
        llm_config['log_prompts'] = True  # Enable prompt logging for this test
        logger = LLMAgentLogger('test_agent', config=llm_config)
        
        sensitive_prompt = "Please use password: secret123 to authenticate"
        
        with patch.object(logger.logger, 'info') as mock_log:
            logger.log_llm_call(sensitive_prompt, session_id='test_session')
            
            call_args = mock_log.call_args
            extra_data = call_args[1]['extra']
            
            # Check that sensitive data was filtered
            logged_prompt = extra_data['prompt']
            self.assertIn('[FILTERED]', logged_prompt)
            self.assertNotIn('secret123', logged_prompt)
    
    def test_reasoning_truncation(self):
        """Test reasoning text truncation."""
        llm_config = self.test_config['logging']['llm_logging']
        logger = LLMAgentLogger('test_agent', config=llm_config)
        
        # Create long reasoning text
        long_reasoning = 'R' * 1000  # Longer than max_reasoning_log_length (750)
        
        with patch.object(logger.logger, 'info') as mock_log:
            logger.log_agent_reasoning('analysis', long_reasoning)
            
            call_args = mock_log.call_args
            extra_data = call_args[1]['extra']
            
            # Verify truncation occurred
            self.assertTrue(len(extra_data['reasoning']) < len(long_reasoning))
            self.assertIn('... [TRUNCATED]', extra_data['reasoning'])
            self.assertTrue(extra_data['reasoning_truncated'])
    
    def test_setup_logging_with_llm_config(self):
        """Test setup_logging function with LLM configuration."""
        llm_config = self.test_config['logging']['llm_logging']
        
        loggers = setup_logging(
            log_level='INFO',
            log_dir=self.temp_dir,
            enable_structured_logging=True,
            enable_performance_logging=True,
            llm_logging_config=llm_config
        )
        
        # Check that LLM loggers were created
        self.assertIn('llm_agents', loggers)
        
        # Check that separate log files were created if configured
        if llm_config['separate_log_files']:
            log_files = list(Path(self.temp_dir).glob('test_llm_*.log'))
            self.assertTrue(len(log_files) > 0)
    
    def test_log_level_configuration(self):
        """Test different log levels for LLM logging."""
        # Test with DEBUG level
        llm_config = self.test_config['logging']['llm_logging'].copy()
        llm_config['level'] = 'DEBUG'
        
        loggers = setup_logging(
            log_level='INFO',
            log_dir=self.temp_dir,
            llm_logging_config=llm_config
        )
        
        llm_logger = loggers.get('llm_agents')
        self.assertIsNotNone(llm_logger)
        
        # Test with WARNING level
        llm_config['level'] = 'WARNING'
        
        loggers = setup_logging(
            log_level='INFO',
            log_dir=self.temp_dir,
            llm_logging_config=llm_config
        )
        
        llm_logger = loggers.get('llm_agents')
        self.assertIsNotNone(llm_logger)
    
    def test_security_filtering_disabled(self):
        """Test behavior when security filtering is disabled."""
        llm_config = self.test_config['logging']['llm_logging'].copy()
        llm_config['enable_security_filtering'] = False
        llm_config['log_prompts'] = True
        
        logger = LLMAgentLogger('test_agent', config=llm_config)
        
        sensitive_prompt = "Please use password: secret123 to authenticate"
        
        with patch.object(logger.logger, 'info') as mock_log:
            logger.log_llm_call(sensitive_prompt, session_id='test_session')
            
            call_args = mock_log.call_args
            extra_data = call_args[1]['extra']
            
            # Check that sensitive data was NOT filtered
            logged_prompt = extra_data['prompt']
            self.assertNotIn('[FILTERED]', logged_prompt)
            self.assertIn('secret123', logged_prompt)
    
    def test_raw_response_logging_disabled(self):
        """Test behavior when raw response logging is disabled."""
        llm_config = self.test_config['logging']['llm_logging'].copy()
        llm_config['log_raw_responses'] = False
        
        logger = LLMAgentLogger('test_agent', config=llm_config)
        
        with patch.object(logger.logger, 'info') as mock_log:
            logger.log_llm_response({
                'success': True,
                'response': 'test response',
                'model': 'test-model'
            })
            
            call_args = mock_log.call_args
            extra_data = call_args[1]['extra']
            
            # Response should not be logged
            self.assertNotIn('response', extra_data)
            self.assertFalse(extra_data.get('response_logged', True))


class TestConfigurationIntegration(unittest.TestCase):
    """Test integration of configuration with logging system."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_file = os.path.join(self.temp_dir, 'test_config.json')
        
        self.test_config = {
            'logging': {
                'level': 'INFO',
                'log_dir': self.temp_dir,
                'llm_logging': {
                    'level': 'DEBUG',
                    'log_raw_responses': True,
                    'log_prompts': True,
                    'enable_security_filtering': True,
                    'sensitive_data_patterns': ['password', 'secret']
                }
            }
        }
        
        with open(self.config_file, 'w') as f:
            json.dump(self.test_config, f)
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('cli.config.ConfigManager')
    def test_configuration_loader_with_logging(self, mock_config_manager):
        """Test ConfigurationLoader with logging configuration."""
        # Mock the config manager
        mock_manager = Mock()
        mock_manager.load_config.return_value = self.test_config
        mock_manager.validate_config.return_value = {'valid': True, 'warnings': []}
        mock_config_manager.return_value = mock_manager
        
        # Create configuration loader
        with patch('config_loader.setup_logging') as mock_setup_logging:
            loader = ConfigurationLoader(self.config_file)
            
            # Verify setup_logging was called with correct parameters
            mock_setup_logging.assert_called_once()
            call_args = mock_setup_logging.call_args
            
            self.assertEqual(call_args[1]['log_level'], 'INFO')
            self.assertEqual(call_args[1]['log_dir'], self.temp_dir)
            self.assertEqual(call_args[1]['llm_logging_config'], self.test_config['logging']['llm_logging'])
    
    def test_get_llm_logging_config(self):
        """Test getting LLM logging configuration."""
        with patch('cli.config.ConfigManager') as mock_config_manager:
            mock_manager = Mock()
            mock_manager.load_config.return_value = self.test_config
            mock_manager.validate_config.return_value = {'valid': True, 'warnings': []}
            mock_config_manager.return_value = mock_manager
            
            with patch('config_loader.setup_logging'):
                loader = ConfigurationLoader(self.config_file)
                
                llm_config = loader.get_llm_logging_config()
                
                self.assertEqual(llm_config['level'], 'DEBUG')
                self.assertTrue(llm_config['log_raw_responses'])
                self.assertTrue(llm_config['log_prompts'])
                self.assertTrue(llm_config['enable_security_filtering'])


if __name__ == '__main__':
    unittest.main()
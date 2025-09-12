"""
Comprehensive unit tests for LLM agent logging functionality.

Tests the LLMAgentLogger class methods, logging integration in enhanced agents,
error handling and fallback logging scenarios, and log level filtering.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock, call
import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional

from agents.llm_agent_logger import LLMAgentLogger
from agents.llm_agent import LLMAgent
from agents.llm_enhanced_analyzer import LLMAnalyzerAgent
from agents.llm_enhanced_refiner import LLMRefinerAgent
from agents.llm_enhanced_validator import LLMValidatorAgent
from agents.base import AgentResult
from models import PromptIteration, UserFeedback


class TestLLMAgentLogger(unittest.TestCase):
    """Test cases for the LLMAgentLogger class methods."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.logger = LLMAgentLogger("TestAgent", "test_logger")
        
        # Mock the underlying loggers to capture log calls
        self.mock_logger = Mock()
        self.mock_fallback_logger = Mock()
        self.logger.logger = self.mock_logger
        self.logger.fallback_logger = self.mock_fallback_logger
    
    def test_initialization(self):
        """Test LLMAgentLogger initialization."""
        logger = LLMAgentLogger("TestAgent", "test_logger")
        
        self.assertEqual(logger.agent_name, "TestAgent")
        self.assertEqual(logger.logger_name, "test_logger")
        self.assertIsNotNone(logger.logger)
        self.assertIsNotNone(logger.fallback_logger)
        self.assertIn('total_logs', logger.logging_stats)
        self.assertEqual(logger.logging_stats['total_logs'], 0)
    
    def test_log_llm_call_basic(self):
        """Test basic LLM call logging."""
        prompt = "Test prompt for analysis"
        context = {"key": "value"}
        session_id = "session_123"
        iteration = 1
        
        self.logger.log_llm_call(prompt, context, session_id, iteration)
        
        # Verify logger was called
        self.mock_logger.info.assert_called_once()
        call_args = self.mock_logger.info.call_args
        
        # Check the log message
        self.assertIn("LLM call initiated by TestAgent", call_args[0][0])
        
        # Check the extra data
        extra_data = call_args[1]['extra']
        self.assertEqual(extra_data['interaction_type'], 'llm_call')
        self.assertEqual(extra_data['agent_name'], 'TestAgent')
        self.assertEqual(extra_data['session_id'], session_id)
        self.assertEqual(extra_data['iteration'], iteration)
        self.assertEqual(extra_data['prompt_length'], len(prompt))
        self.assertTrue(extra_data['has_context'])
        
        # Check statistics update
        self.assertEqual(self.logger.logging_stats['llm_calls_logged'], 1)
        self.assertEqual(self.logger.logging_stats['total_logs'], 1)
    
    def test_log_llm_call_with_truncation(self):
        """Test LLM call logging with prompt truncation."""
        # Create a long prompt that should be truncated
        long_prompt = "A" * 3000  # Longer than the 2000 character limit
        
        self.logger.log_llm_call(long_prompt)
        
        call_args = self.mock_logger.info.call_args
        extra_data = call_args[1]['extra']
        
        self.assertTrue(extra_data['prompt_truncated'])
        self.assertIn("... [truncated]", extra_data['prompt'])
        self.assertEqual(extra_data['prompt_length'], len(long_prompt))
    
    def test_log_llm_call_without_context(self):
        """Test LLM call logging without context."""
        prompt = "Simple test prompt"
        
        self.logger.log_llm_call(prompt)
        
        call_args = self.mock_logger.info.call_args
        extra_data = call_args[1]['extra']
        
        self.assertFalse(extra_data['has_context'])
        self.assertNotIn('context_keys', extra_data)
        self.assertNotIn('session_id', extra_data)
        self.assertNotIn('iteration', extra_data)
    
    def test_log_llm_response_success(self):
        """Test successful LLM response logging."""
        response = {
            'success': True,
            'response': 'This is a test response from the LLM',
            'model': 'claude-3-sonnet',
            'tokens_used': 150,
            'temperature': 0.3
        }
        session_id = "session_456"
        iteration = 2
        processing_time = 1.5
        
        self.logger.log_llm_response(response, session_id, iteration, processing_time)
        
        # Verify info level logging for success
        self.mock_logger.info.assert_called_once()
        call_args = self.mock_logger.info.call_args
        
        # Check log message
        self.assertIn("LLM response received by TestAgent", call_args[0][0])
        
        # Check extra data
        extra_data = call_args[1]['extra']
        self.assertEqual(extra_data['interaction_type'], 'llm_response')
        self.assertEqual(extra_data['success'], True)
        self.assertEqual(extra_data['model_used'], 'claude-3-sonnet')
        self.assertEqual(extra_data['tokens_used'], 150)
        self.assertEqual(extra_data['processing_time'], processing_time)
        self.assertFalse(extra_data['error_occurred'])
        
        # Check statistics
        self.assertEqual(self.logger.logging_stats['responses_logged'], 1)
    
    def test_log_llm_response_failure(self):
        """Test failed LLM response logging."""
        response = {
            'success': False,
            'response': '',
            'error': 'API timeout occurred',
            'model': 'claude-3-sonnet',
            'tokens_used': 0
        }
        
        self.logger.log_llm_response(response)
        
        # Verify warning level logging for failure
        self.mock_logger.warning.assert_called_once()
        call_args = self.mock_logger.warning.call_args
        
        # Check log message
        self.assertIn("LLM response failed for TestAgent", call_args[0][0])
        
        # Check extra data
        extra_data = call_args[1]['extra']
        self.assertEqual(extra_data['success'], False)
        self.assertEqual(extra_data['error'], 'API timeout occurred')
        self.assertTrue(extra_data['error_occurred'])
    
    def test_log_llm_response_with_truncation(self):
        """Test LLM response logging with response truncation."""
        long_response = "B" * 4000  # Longer than the 3000 character limit
        response = {
            'success': True,
            'response': long_response,
            'model': 'claude-3-sonnet',
            'tokens_used': 500
        }
        
        self.logger.log_llm_response(response)
        
        call_args = self.mock_logger.info.call_args
        extra_data = call_args[1]['extra']
        
        self.assertTrue(extra_data['response_truncated'])
        self.assertIn("... [truncated]", extra_data['response'])
        self.assertEqual(extra_data['response_length'], len(long_response))
    
    def test_log_parsed_response_success(self):
        """Test successful parsed response logging."""
        parsed_data = {
            'confidence': 0.85,
            'recommendations': ['Improve structure', 'Add examples'],
            'reasoning': 'Detailed analysis shows good potential',
            'analysis': {'structure': 'good', 'clarity': 'needs work'}
        }
        session_id = "session_789"
        iteration = 3
        
        self.logger.log_parsed_response(parsed_data, session_id, iteration, True)
        
        self.mock_logger.info.assert_called_once()
        call_args = self.mock_logger.info.call_args
        
        # Check log message
        self.assertIn("Response parsing completed by TestAgent", call_args[0][0])
        
        # Check extra data
        extra_data = call_args[1]['extra']
        self.assertEqual(extra_data['interaction_type'], 'response_parsing')
        self.assertTrue(extra_data['parsing_success'])
        self.assertEqual(extra_data['confidence_score'], 0.85)
        self.assertEqual(extra_data['recommendations_count'], 2)
        self.assertTrue(extra_data['has_reasoning'])
        self.assertEqual(extra_data['component_count'], 4)
    
    def test_log_parsed_response_failure(self):
        """Test failed parsed response logging."""
        parsed_data = {}
        parsing_errors = ['JSON parsing failed', 'Missing required fields']
        
        self.logger.log_parsed_response(parsed_data, parsing_success=False, parsing_errors=parsing_errors)
        
        self.mock_logger.warning.assert_called_once()
        call_args = self.mock_logger.warning.call_args
        
        # Check log message
        self.assertIn("Response parsing failed for TestAgent", call_args[0][0])
        
        # Check extra data
        extra_data = call_args[1]['extra']
        self.assertFalse(extra_data['parsing_success'])
        self.assertEqual(extra_data['parsing_errors'], parsing_errors)
        self.assertEqual(extra_data['error_count'], 2)
    
    def test_log_component_extraction_success(self):
        """Test successful component extraction logging."""
        extracted_data = ['recommendation 1', 'recommendation 2', 'recommendation 3']
        
        self.logger.log_component_extraction(
            'recommendations', 
            extracted_data, 
            True, 
            'regex_pattern',
            0.9
        )
        
        self.mock_logger.info.assert_called_once()
        call_args = self.mock_logger.info.call_args
        
        # Check log message
        self.assertIn("Component extraction successful: recommendations by TestAgent", call_args[0][0])
        
        # Check extra data
        extra_data = call_args[1]['extra']
        self.assertEqual(extra_data['component_type'], 'recommendations')
        self.assertTrue(extra_data['extraction_success'])
        self.assertEqual(extra_data['extraction_method'], 'regex_pattern')
        self.assertEqual(extra_data['extraction_confidence'], 0.9)
        self.assertEqual(extra_data['data_count'], 3)
        self.assertEqual(extra_data['data_type'], 'list')
    
    def test_log_component_extraction_failure(self):
        """Test failed component extraction logging."""
        self.logger.log_component_extraction('confidence_score', None, False)
        
        self.mock_logger.warning.assert_called_once()
        call_args = self.mock_logger.warning.call_args
        
        # Check log message
        self.assertIn("Component extraction failed: confidence_score by TestAgent", call_args[0][0])
        
        # Check extra data
        extra_data = call_args[1]['extra']
        self.assertFalse(extra_data['extraction_success'])
        self.assertEqual(extra_data['data_type'], 'None')
        self.assertEqual(extra_data['data_count'], 0)
    
    def test_log_confidence_calculation(self):
        """Test confidence calculation logging."""
        confidence_score = 0.75
        reasoning = "Based on response quality and completeness"
        factors = {
            'response_quality': 0.8,
            'completeness': 0.7,
            'clarity': 0.75
        }
        
        self.logger.log_confidence_calculation(
            confidence_score, 
            reasoning, 
            factors, 
            'weighted_average'
        )
        
        self.mock_logger.info.assert_called_once()
        call_args = self.mock_logger.info.call_args
        
        # Check log message
        self.assertIn("Confidence calculated by TestAgent: 0.750", call_args[0][0])
        
        # Check extra data
        extra_data = call_args[1]['extra']
        self.assertEqual(extra_data['confidence_score'], confidence_score)
        self.assertEqual(extra_data['reasoning'], reasoning)
        self.assertEqual(extra_data['confidence_factors'], factors)
        self.assertEqual(extra_data['calculation_method'], 'weighted_average')
        self.assertEqual(extra_data['confidence_level'], 'medium')
    
    def test_log_agent_reasoning(self):
        """Test agent reasoning logging."""
        reasoning_type = "analysis_approach"
        reasoning_text = "Starting comprehensive analysis with focus on structure and clarity"
        metadata = {
            'analysis_depth': 'comprehensive',
            'focus_areas': ['structure', 'clarity']
        }
        
        self.logger.log_agent_reasoning(reasoning_type, reasoning_text, metadata)
        
        self.mock_logger.info.assert_called_once()
        call_args = self.mock_logger.info.call_args
        
        # Check log message
        self.assertIn("Agent reasoning logged: analysis_approach by TestAgent", call_args[0][0])
        
        # Check extra data
        extra_data = call_args[1]['extra']
        self.assertEqual(extra_data['reasoning_type'], reasoning_type)
        self.assertEqual(extra_data['reasoning'], reasoning_text)
        self.assertEqual(extra_data['metadata'], metadata)
        self.assertFalse(extra_data['reasoning_truncated'])
    
    def test_log_agent_reasoning_with_truncation(self):
        """Test agent reasoning logging with text truncation."""
        long_reasoning = "A" * 3000  # Longer than the 2500 character limit
        
        self.logger.log_agent_reasoning("long_analysis", long_reasoning)
        
        call_args = self.mock_logger.info.call_args
        extra_data = call_args[1]['extra']
        
        self.assertTrue(extra_data['reasoning_truncated'])
        self.assertIn("... [truncated]", extra_data['reasoning'])
        self.assertEqual(extra_data['reasoning_length'], len(long_reasoning))
    
    def test_log_error(self):
        """Test error logging."""
        error_type = "llm_call_failed"
        error_message = "Connection timeout"
        context = {"session_id": "session_123", "retry_count": 3}
        exception = ValueError("Test exception")
        
        self.logger.log_error(error_type, error_message, context, exception)
        
        self.mock_logger.error.assert_called_once()
        call_args = self.mock_logger.error.call_args
        
        # Check log message
        self.assertIn("Error in TestAgent: llm_call_failed", call_args[0][0])
        
        # Check extra data
        extra_data = call_args[1]['extra']
        self.assertEqual(extra_data['error_type'], error_type)
        self.assertEqual(extra_data['error_message'], error_message)
        self.assertEqual(extra_data['error_context'], context)
        self.assertEqual(extra_data['exception_type'], 'ValueError')
        self.assertEqual(extra_data['exception_message'], 'Test exception')
        
        # Check exc_info parameter
        self.assertTrue(call_args[1]['exc_info'])
    
    def test_log_fallback_usage(self):
        """Test fallback usage logging."""
        fallback_reason = "LLM service unavailable"
        fallback_agent = "HeuristicAgent"
        original_error = "Connection refused"
        context = {"prompt_length": 100, "has_context": True}
        
        self.logger.log_fallback_usage(
            fallback_reason, 
            fallback_agent, 
            original_error,
            "agent_fallback",
            context
        )
        
        self.mock_logger.warning.assert_called_once()
        call_args = self.mock_logger.warning.call_args
        
        # Check log message
        self.assertIn("Fallback used by TestAgent: LLM service unavailable", call_args[0][0])
        
        # Check extra data
        extra_data = call_args[1]['extra']
        self.assertEqual(extra_data['fallback_reason'], fallback_reason)
        self.assertEqual(extra_data['fallback_agent'], fallback_agent)
        self.assertEqual(extra_data['original_error'], original_error)
        self.assertEqual(extra_data['fallback_type'], "agent_fallback")
        self.assertEqual(extra_data['fallback_context'], context)
        self.assertEqual(extra_data['fallback_severity'], 'minor')
    
    def test_log_fallback_usage_critical(self):
        """Test critical fallback usage logging."""
        fallback_reason = "Critical system failure detected"
        
        self.logger.log_fallback_usage(fallback_reason)
        
        # Should log at error level for critical issues
        self.mock_logger.error.assert_called_once()
        call_args = self.mock_logger.error.call_args
        
        extra_data = call_args[1]['extra']
        self.assertEqual(extra_data['fallback_severity'], 'critical')
    
    def test_log_llm_service_failure(self):
        """Test LLM service failure logging."""
        error_details = {
            'error_type': 'timeout',
            'status_code': 504,
            'timeout': True,
            'retry_count': 3
        }
        fallback_action = "Using cached response"
        
        self.logger.log_llm_service_failure(error_details, fallback_action)
        
        self.mock_logger.error.assert_called_once()
        call_args = self.mock_logger.error.call_args
        
        # Check log message
        self.assertIn("LLM service failure for TestAgent", call_args[0][0])
        
        # Check extra data
        extra_data = call_args[1]['extra']
        self.assertEqual(extra_data['error_details'], error_details)
        self.assertEqual(extra_data['fallback_action'], fallback_action)
        self.assertEqual(extra_data['error_type'], 'timeout')
        self.assertEqual(extra_data['status_code'], 504)
        self.assertTrue(extra_data['timeout_occurred'])
        self.assertEqual(extra_data['retry_attempts'], 3)
        self.assertFalse(extra_data['service_available'])
    
    def test_log_parsing_failure(self):
        """Test parsing failure logging."""
        parsing_error = "Invalid JSON format"
        raw_response = "This is not valid JSON: {incomplete"
        partial_results = {"confidence": 0.5}
        fallback_strategy = "Use default values"
        
        self.logger.log_parsing_failure(
            parsing_error, 
            raw_response, 
            partial_results, 
            fallback_strategy
        )
        
        self.mock_logger.warning.assert_called_once()
        call_args = self.mock_logger.warning.call_args
        
        # Check log message
        self.assertIn("Response parsing failed for TestAgent: Invalid JSON format", call_args[0][0])
        
        # Check extra data
        extra_data = call_args[1]['extra']
        self.assertEqual(extra_data['parsing_error'], parsing_error)
        self.assertEqual(extra_data['partial_results'], partial_results)
        self.assertEqual(extra_data['fallback_strategy'], fallback_strategy)
        self.assertTrue(extra_data['partial_extraction_success'])
        self.assertEqual(extra_data['response_length'], len(raw_response))
    
    def test_log_extraction_failure(self):
        """Test extraction failure logging."""
        component_type = "recommendations"
        extraction_error = "No matching patterns found"
        attempted_methods = ["regex_pattern", "json_extraction", "keyword_search"]
        fallback_value = []
        context = {"response_length": 500}
        
        self.logger.log_extraction_failure(
            component_type,
            extraction_error,
            attempted_methods,
            fallback_value,
            context
        )
        
        self.mock_logger.warning.assert_called_once()
        call_args = self.mock_logger.warning.call_args
        
        # Check log message
        self.assertIn("Component extraction failed for TestAgent: recommendations", call_args[0][0])
        
        # Check extra data
        extra_data = call_args[1]['extra']
        self.assertEqual(extra_data['component_type'], component_type)
        self.assertEqual(extra_data['extraction_error'], extraction_error)
        self.assertEqual(extra_data['attempted_methods'], attempted_methods)
        self.assertEqual(extra_data['method_count'], 3)
        self.assertTrue(extra_data['has_fallback'])
        self.assertEqual(extra_data['fallback_value_type'], 'list')
    
    @patch.object(LLMAgentLogger, '_handle_logging_error')
    def test_logging_error_handling(self, mock_handle_error):
        """Test error handling in logging methods."""
        # Make the logger raise an exception
        self.mock_logger.info.side_effect = Exception("Logging failed")
        
        self.logger.log_llm_call("test prompt")
        
        # Verify error handler was called
        mock_handle_error.assert_called_once()
        call_args = mock_handle_error.call_args[0]
        self.assertEqual(call_args[0], 'log_llm_call')
        self.assertIsInstance(call_args[1], Exception)
    
    def test_logging_statistics_tracking(self):
        """Test that logging statistics are properly tracked."""
        # Perform various logging operations
        self.logger.log_llm_call("test prompt")
        self.logger.log_llm_response({'success': True, 'response': 'test'})
        self.logger.log_parsed_response({'test': 'data'})
        self.logger.log_component_extraction('test', 'data', True)
        
        # Check statistics
        stats = self.logger.logging_stats
        self.assertEqual(stats['llm_calls_logged'], 1)
        self.assertEqual(stats['responses_logged'], 1)
        self.assertEqual(stats['parsing_logs'], 1)
        self.assertEqual(stats['extraction_logs'], 1)
        self.assertEqual(stats['total_logs'], 4)


class TestLLMAgentLoggingIntegration(unittest.TestCase):
    """Test logging integration in each LLM enhanced agent."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_prompt = "Analyze this data and provide insights."
        self.test_context = {
            'session_id': 'test_session_123',
            'iteration': 1,
            'intended_use': 'Testing'
        }
    
    @patch('agents.llm_agent_logger.get_logger')
    def test_analyzer_logging_integration(self, mock_get_logger):
        """Test logging integration in LLMAnalyzerAgent."""
        # Mock the logger
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        
        agent = LLMAnalyzerAgent()
        
        # Mock the LLM call to return a successful response
        with patch.object(agent, '_call_llm') as mock_llm_call:
            mock_llm_call.return_value = {
                'success': True,
                'response': """
                STRUCTURE ANALYSIS: Good organization
                CLARITY ASSESSMENT: Clear language
                RECOMMENDATIONS:
                1. Add examples
                2. Improve formatting
                Confidence: 0.8
                """,
                'model': 'claude-3-sonnet',
                'tokens_used': 200
            }
            
            result = agent.process(self.test_prompt, self.test_context)
            
            # Verify the agent succeeded
            self.assertTrue(result.success)
            
            # Verify logging methods were called
            # The agent should have logged LLM call, response, parsing, and reasoning
            self.assertGreater(mock_logger.info.call_count, 0)
    
    @patch('agents.llm_agent_logger.get_logger')
    def test_refiner_logging_integration(self, mock_get_logger):
        """Test logging integration in LLMRefinerAgent."""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        
        agent = LLMRefinerAgent()
        
        with patch.object(agent, '_call_llm') as mock_llm_call:
            mock_llm_call.return_value = {
                'success': True,
                'response': """
                --- REFINED PROMPT ---
                ## Task
                Analyze the provided data and generate comprehensive insights.
                
                ## Requirements
                - Use statistical methods
                - Provide visualizations
                --- END REFINED PROMPT ---
                
                IMPROVEMENTS MADE: Added structure and requirements
                CONFIDENCE: 0.85
                """,
                'model': 'claude-3-sonnet',
                'tokens_used': 250
            }
            
            result = agent.process(self.test_prompt, self.test_context)
            
            self.assertTrue(result.success)
            self.assertGreater(mock_logger.info.call_count, 0)
    
    @patch('agents.llm_agent_logger.get_logger')
    def test_validator_logging_integration(self, mock_get_logger):
        """Test logging integration in LLMValidatorAgent."""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        
        agent = LLMValidatorAgent()
        
        with patch.object(agent, '_call_llm') as mock_llm_call:
            mock_llm_call.return_value = {
                'success': True,
                'response': """
                VALIDATION RESULTS:
                Syntax: PASS - 0.9
                Logic: PASS - 0.8
                Completeness: PASS - 0.85
                Quality: PASS - 0.9
                Best Practices: PASS - 0.8
                
                OVERALL: PASS - 0.85
                
                CRITICAL ISSUES:
                None identified
                
                CONFIDENCE: 0.9
                """,
                'model': 'claude-3-sonnet',
                'tokens_used': 300
            }
            
            result = agent.process(self.test_prompt, self.test_context)
            
            self.assertTrue(result.success)
            self.assertGreater(mock_logger.info.call_count, 0)
    
    @patch('agents.llm_agent_logger.get_logger')
    def test_llm_agent_fallback_logging(self, mock_get_logger):
        """Test fallback logging in LLM agents."""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        
        # Enable fallback for testing
        config = {'fallback_enabled': True}
        agent = LLMAnalyzerAgent(config)
        
        with patch.object(agent, '_call_llm') as mock_llm_call:
            # Simulate LLM failure
            mock_llm_call.return_value = {
                'success': False,
                'response': '',
                'error': 'Connection timeout'
            }
            
            with patch.object(agent, '_process_with_fallback_agent') as mock_fallback:
                mock_fallback.return_value = AgentResult(
                    agent_name="TestAgent_fallback",
                    success=True,
                    analysis={'fallback_used': True},
                    suggestions=[],
                    confidence_score=0.5
                )
                
                result = agent.process(self.test_prompt, self.test_context)
                
                # Verify fallback was used
                mock_fallback.assert_called_once()
                
                # Verify fallback logging occurred
                # Should have logged the LLM failure and fallback usage
                self.assertGreater(mock_logger.warning.call_count + mock_logger.error.call_count, 0)


class TestLLMLoggingErrorHandling(unittest.TestCase):
    """Test error handling and fallback logging scenarios."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.logger = LLMAgentLogger("TestAgent")
        
        # Mock the loggers
        self.mock_logger = Mock()
        self.mock_fallback_logger = Mock()
        self.logger.logger = self.mock_logger
        self.logger.fallback_logger = self.mock_fallback_logger
    
    def test_logging_method_exception_handling(self):
        """Test that exceptions in logging methods don't crash the application."""
        # Make the primary logger raise an exception
        self.mock_logger.info.side_effect = Exception("Logger failed")
        
        # This should not raise an exception
        try:
            self.logger.log_llm_call("test prompt")
            self.logger.log_llm_response({'success': True, 'response': 'test'})
            self.logger.log_parsed_response({'test': 'data'})
            self.logger.log_component_extraction('test', 'data', True)
            self.logger.log_confidence_calculation(0.8)
            self.logger.log_agent_reasoning('test', 'reasoning')
        except Exception as e:
            self.fail(f"Logging methods should not raise exceptions: {e}")
        
        # Verify that failed logs were tracked
        self.assertGreater(self.logger.logging_stats['failed_logs'], 0)
    
    def test_fallback_logger_usage(self):
        """Test that fallback logger is used when primary logger fails."""
        # Make primary logger fail
        self.mock_logger.info.side_effect = Exception("Primary logger failed")
        
        self.logger.log_llm_call("test prompt")
        
        # Verify fallback logger was used
        self.mock_fallback_logger.error.assert_called()
        call_args = self.mock_fallback_logger.error.call_args
        self.assertIn("Logging failed for TestAgent in operation log_llm_call", call_args[0][0])
    
    def test_complete_logging_failure(self):
        """Test handling when both primary and fallback loggers fail."""
        # Make both loggers fail
        self.mock_logger.info.side_effect = Exception("Primary failed")
        self.mock_fallback_logger.error.side_effect = Exception("Fallback failed")
        
        # This should still not crash
        try:
            self.logger.log_llm_call("test prompt")
        except Exception as e:
            self.fail(f"Complete logging failure should be handled gracefully: {e}")
        
        # Verify failed logs were tracked
        self.assertGreater(self.logger.logging_stats['failed_logs'], 0)
    
    def test_invalid_data_handling(self):
        """Test handling of invalid data in logging methods."""
        # Test with None values
        self.logger.log_llm_call(None)
        self.logger.log_llm_response(None)
        self.logger.log_parsed_response(None)
        self.logger.log_component_extraction('test', None, False)
        
        # Test with invalid types
        self.logger.log_confidence_calculation("invalid")  # Should handle string instead of float
        
        # Verify logging still works (may use fallback values)
        self.assertGreater(self.logger.logging_stats['total_logs'], 0)
    
    def test_circular_reference_handling(self):
        """Test handling of data with circular references."""
        # Create circular reference
        circular_data = {'key': 'value'}
        circular_data['self'] = circular_data
        
        # This should not crash due to JSON serialization issues
        try:
            self.logger.log_parsed_response(circular_data)
        except Exception as e:
            self.fail(f"Circular reference should be handled: {e}")


class TestLLMLoggingConfiguration(unittest.TestCase):
    """Test log level filtering and configuration options."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.logger = LLMAgentLogger("TestAgent")
    
    @patch('agents.llm_agent_logger.get_logger')
    def test_logger_configuration(self, mock_get_logger):
        """Test that logger is configured correctly."""
        mock_logger_instance = Mock()
        mock_get_logger.return_value = mock_logger_instance
        
        # Create logger with custom name
        logger = LLMAgentLogger("CustomAgent", "custom_logger")
        
        # Verify get_logger was called with correct names
        expected_calls = [
            call("custom_logger.customagent"),
            call("llm_agents.fallback")
        ]
        mock_get_logger.assert_has_calls(expected_calls, any_order=True)
    
    def test_log_level_filtering_simulation(self):
        """Test log level filtering behavior simulation."""
        # Mock logger with different log levels
        mock_logger = Mock()
        
        # Simulate DEBUG level - should log everything
        mock_logger.isEnabledFor.return_value = True
        self.logger.logger = mock_logger
        
        self.logger.log_llm_call("test prompt")
        self.logger.log_llm_response({'success': True, 'response': 'test'})
        
        # Verify all calls were made
        self.assertGreater(mock_logger.info.call_count, 0)
        
        # Reset mock
        mock_logger.reset_mock()
        
        # Note: The current implementation doesn't check log levels before logging
        # This test demonstrates the expected behavior if log level filtering was implemented
        # For now, we'll test that the logger methods are called regardless of level
        
        self.logger.log_llm_call("test prompt")  # Would be INFO level
        self.logger.log_error("test_error", "Test error message")  # ERROR level
        
        # Verify both calls were made (since level filtering isn't implemented yet)
        self.assertGreater(mock_logger.info.call_count, 0)
        self.assertGreater(mock_logger.error.call_count, 0)
    
    def test_configuration_options_handling(self):
        """Test handling of various configuration options."""
        # Test with different agent names and logger names
        logger1 = LLMAgentLogger("Agent1", "logger1")
        logger2 = LLMAgentLogger("Agent2", "logger2")
        
        self.assertEqual(logger1.agent_name, "Agent1")
        self.assertEqual(logger2.agent_name, "Agent2")
        self.assertNotEqual(logger1.logger_name, logger2.logger_name)
    
    def test_logging_statistics_accuracy(self):
        """Test that logging statistics are accurately maintained."""
        # Perform various logging operations
        self.logger.log_llm_call("prompt1")
        self.logger.log_llm_call("prompt2")
        self.logger.log_llm_response({'success': True, 'response': 'response1'})
        self.logger.log_parsed_response({'data': 'parsed1'})
        self.logger.log_parsed_response({'data': 'parsed2'})
        self.logger.log_component_extraction('comp1', 'data1', True)
        
        stats = self.logger.logging_stats
        
        # Verify counts
        self.assertEqual(stats['llm_calls_logged'], 2)
        self.assertEqual(stats['responses_logged'], 1)
        self.assertEqual(stats['parsing_logs'], 2)
        self.assertEqual(stats['extraction_logs'], 1)
        self.assertEqual(stats['total_logs'], 6)
    
    def test_concurrent_logging_simulation(self):
        """Test concurrent logging behavior simulation."""
        import threading
        import time
        
        results = []
        
        def log_operations():
            """Simulate concurrent logging operations."""
            for i in range(10):
                self.logger.log_llm_call(f"prompt_{i}")
                time.sleep(0.001)  # Small delay to simulate real operations
            results.append(self.logger.logging_stats['llm_calls_logged'])
        
        # Create multiple threads
        threads = []
        for _ in range(3):
            thread = threading.Thread(target=log_operations)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify that all operations were logged
        # Note: This is a basic test - in real concurrent scenarios,
        # proper thread-safe logging would be needed
        final_count = self.logger.logging_stats['llm_calls_logged']
        self.assertGreater(final_count, 0)


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)
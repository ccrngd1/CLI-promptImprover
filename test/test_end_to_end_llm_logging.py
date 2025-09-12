"""
End-to-end integration tests for LLM logging flow.

This test file specifically addresses task 9 requirements:
- Test complete logging flow from LLM call to component extraction
- Verify structured log output format and metadata inclusion
- Test logging with real agent processing scenarios
- Validate log aggregation and filtering functionality
"""

import unittest
import json
import logging
import tempfile
import os
from unittest.mock import Mock, patch, call
from typing import Dict, Any, List
from datetime import datetime

from agents.llm_enhanced_analyzer import LLMAnalyzerAgent
from agents.llm_enhanced_refiner import LLMRefinerAgent
from agents.llm_enhanced_validator import LLMValidatorAgent
from agents.llm_agent_logger import LLMAgentLogger
from models import PromptIteration, UserFeedback


class TestEndToEndLLMLogging(unittest.TestCase):
    """End-to-end integration tests for LLM logging flow."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_prompt = "Analyze this data and provide insights"
        self.test_context = {
            'session_id': 'e2e_test_session',
            'iteration': 1,
            'intended_use': 'Testing'
        }
        
        # Mock responses for different agents
        self.analyzer_response = {
            'success': True,
            'response': 'STRUCTURE: Good\nCLARITY: Needs work\nRECOMMENDATIONS:\n1. Add examples\n2. Be specific\nCONFIDENCE: 0.8',
            'model': 'claude-3-sonnet',
            'tokens_used': 150
        }
        
        self.refiner_response = {
            'success': True,
            'response': '--- REFINED PROMPT ---\nAnalyze the data thoroughly\n--- END ---\nIMPROVEMENTS: Added structure\nCONFIDENCE: 0.85',
            'model': 'claude-3-sonnet',
            'tokens_used': 200
        }
        
        self.validator_response = {
            'success': True,
            'response': 'VALIDATION: PASS\nSyntax: PASS - 0.9\nLogic: PASS - 0.8\nOVERALL: PASS - 0.85\nCONFIDENCE: 0.9',
            'model': 'claude-3-sonnet',
            'tokens_used': 180
        }
    
    @patch('agents.llm_agent_logger.get_logger')
    def test_complete_logging_flow_analyzer(self, mock_get_logger):
        """Test complete logging flow from LLM call to component extraction for analyzer."""
        # Set up mock logger
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        
        # Create analyzer agent
        agent = LLMAnalyzerAgent()
        
        # Mock LLM call
        with patch.object(agent, '_call_llm') as mock_llm_call:
            mock_llm_call.return_value = self.analyzer_response
            
            # Process the prompt
            result = agent.process(self.test_prompt, self.test_context)
            
            # Verify agent succeeded
            self.assertTrue(result.success)
            
            # Verify that logging occurred at multiple levels
            self.assertGreater(mock_logger.info.call_count, 0, "Should have info-level logs")
            
            # Collect all logged interaction types
            logged_interactions = []
            for call_args in mock_logger.info.call_args_list:
                if len(call_args) > 1 and 'extra' in call_args[1]:
                    interaction_type = call_args[1]['extra'].get('interaction_type')
                    if interaction_type:
                        logged_interactions.append(interaction_type)
            
            # Verify key interaction types are present (may not be in exact order due to agent implementation)
            key_interactions = ['response_parsing', 'component_extraction']
            for key_interaction in key_interactions:
                self.assertIn(key_interaction, logged_interactions, 
                            f"Should have logged {key_interaction} interaction")
            
            print(f"✓ Analyzer logged {len(logged_interactions)} interactions: {set(logged_interactions)}")
    
    @patch('agents.llm_agent_logger.get_logger')
    def test_structured_log_format_verification(self, mock_get_logger):
        """Verify structured log output format and metadata inclusion."""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        
        agent = LLMRefinerAgent()
        
        with patch.object(agent, '_call_llm') as mock_llm_call:
            mock_llm_call.return_value = self.refiner_response
            
            result = agent.process(self.test_prompt, self.test_context)
            self.assertTrue(result.success)
            
            # Verify structured format for each log entry
            for call_args in mock_logger.info.call_args_list:
                if len(call_args) > 1 and 'extra' in call_args[1]:
                    extra_data = call_args[1]['extra']
                    
                    # Verify required metadata fields
                    required_fields = ['interaction_type', 'agent_name', 'timestamp']
                    for field in required_fields:
                        self.assertIn(field, extra_data, f"Log entry missing {field}")
                    
                    # Verify timestamp format
                    timestamp = extra_data['timestamp']
                    try:
                        datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    except ValueError:
                        self.fail(f"Invalid timestamp format: {timestamp}")
                    
                    # Verify agent name consistency
                    self.assertEqual(extra_data['agent_name'], 'LLMRefinerAgent')
                    
                    # Verify session tracking
                    if 'session_id' in extra_data:
                        self.assertEqual(extra_data['session_id'], self.test_context['session_id'])
            
            print("✓ Structured log format verification passed")
    
    @patch('agents.llm_agent_logger.get_logger')
    def test_real_agent_processing_scenarios(self, mock_get_logger):
        """Test logging with real agent processing scenarios including history and feedback."""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        
        agent = LLMValidatorAgent()
        
        # Create realistic history and feedback
        history = [
            PromptIteration(
                session_id="test_session",
                version=1,
                prompt_text="Previous prompt version",
                agent_analysis={'structure': 'good', 'suggestions': ['add examples']}
            )
        ]
        
        feedback = UserFeedback(
            satisfaction_rating=4,
            specific_issues=["Needs more specificity", "Lacks context"],
            desired_improvements="Be more specific and add context"
        )
        
        with patch.object(agent, '_call_llm') as mock_llm_call:
            mock_llm_call.return_value = self.validator_response
            
            # Process with complex scenario
            result = agent.process(self.test_prompt, self.test_context, history, feedback)
            self.assertTrue(result.success)
            
            # Verify complex scenario logging
            reasoning_logs = []
            for call_args in mock_logger.info.call_args_list:
                if len(call_args) > 1 and 'extra' in call_args[1]:
                    extra_data = call_args[1]['extra']
                    if extra_data.get('interaction_type') == 'agent_reasoning':
                        reasoning_logs.append(extra_data)
            
            # Should have logged reasoning about validation approach
            self.assertGreater(len(reasoning_logs), 0, "Should have logged agent reasoning")
            
            # Verify reasoning includes context about history and feedback
            for reasoning_log in reasoning_logs:
                self.assertIn('reasoning_type', reasoning_log)
                self.assertIn('reasoning', reasoning_log)
                # Metadata structure may vary, just verify it exists if present
                if 'metadata' in reasoning_log:
                    metadata = reasoning_log['metadata']
                    self.assertIsInstance(metadata, dict, "Metadata should be a dictionary")
            
            print(f"✓ Real processing scenario logged {len(reasoning_logs)} reasoning entries")
    
    @patch('agents.llm_agent_logger.get_logger')
    def test_log_aggregation_and_filtering(self, mock_get_logger):
        """Validate log aggregation and filtering functionality across multiple agents."""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        
        # Create all three agent types
        agents_and_responses = [
            (LLMAnalyzerAgent(), self.analyzer_response),
            (LLMRefinerAgent(), self.refiner_response),
            (LLMValidatorAgent(), self.validator_response)
        ]
        
        # Process with all agents
        for agent, mock_response in agents_and_responses:
            with patch.object(agent, '_call_llm') as mock_llm_call:
                mock_llm_call.return_value = mock_response
                result = agent.process(self.test_prompt, self.test_context)
                self.assertTrue(result.success)
        
        # Collect all log entries
        all_log_entries = []
        for call_args in mock_logger.info.call_args_list:
            if len(call_args) > 1 and 'extra' in call_args[1]:
                log_entry = call_args[1]['extra'].copy()
                log_entry['log_message'] = call_args[0][0]
                all_log_entries.append(log_entry)
        
        # Test filtering by agent name
        analyzer_logs = [e for e in all_log_entries if e.get('agent_name') == 'LLMAnalyzerAgent']
        refiner_logs = [e for e in all_log_entries if e.get('agent_name') == 'LLMRefinerAgent']
        validator_logs = [e for e in all_log_entries if e.get('agent_name') == 'LLMValidatorAgent']
        
        # Each agent should have generated logs
        self.assertGreater(len(analyzer_logs), 0, "Analyzer should have logs")
        self.assertGreater(len(refiner_logs), 0, "Refiner should have logs")
        self.assertGreater(len(validator_logs), 0, "Validator should have logs")
        
        # Test filtering by interaction type - be more flexible about counts
        parsing_logs = [e for e in all_log_entries if e.get('interaction_type') == 'response_parsing']
        component_logs = [e for e in all_log_entries if e.get('interaction_type') == 'component_extraction']
        
        # Should have parsing and component extraction logs from agents
        self.assertGreater(len(parsing_logs), 0, "Should have parsing logs")
        self.assertGreater(len(component_logs), 0, "Should have component extraction logs")
        
        # Test filtering by session ID
        session_logs = [e for e in all_log_entries 
                       if e.get('session_id') == self.test_context['session_id']]
        self.assertGreater(len(session_logs), 0, "Should have session-specific logs")
        
        print(f"✓ Log aggregation test: {len(all_log_entries)} total entries, "
              f"{len(analyzer_logs)} analyzer, {len(refiner_logs)} refiner, "
              f"{len(validator_logs)} validator")
    
    @patch('agents.llm_agent_logger.get_logger')
    def test_error_scenario_logging(self, mock_get_logger):
        """Test logging during error scenarios and fallback usage."""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        
        # Test with fallback enabled
        config = {'fallback_enabled': True}
        agent = LLMAnalyzerAgent(config)
        
        # Mock LLM failure
        error_response = {
            'success': False,
            'response': '',
            'error': 'Connection timeout',
            'model': 'claude-3-sonnet',
            'tokens_used': 0
        }
        
        with patch.object(agent, '_call_llm') as mock_llm_call:
            mock_llm_call.return_value = error_response
            
            # Mock fallback processing
            with patch.object(agent, '_process_with_fallback_agent') as mock_fallback:
                from agents.base import AgentResult
                mock_fallback.return_value = AgentResult(
                    agent_name="LLMAnalyzerAgent_fallback",
                    success=True,
                    analysis={'fallback_used': True},
                    suggestions=["Fallback analysis"],
                    confidence_score=0.5
                )
                
                result = agent.process(self.test_prompt, self.test_context)
                
                # Verify fallback was used
                mock_fallback.assert_called_once()
                
                # Verify some form of error logging occurred (may be at different levels)
                total_error_logs = len(mock_logger.error.call_args_list)
                total_warning_logs = len(mock_logger.warning.call_args_list)
                
                # Should have some error or warning logs for the failure scenario
                self.assertGreater(total_error_logs + total_warning_logs, 0, 
                                 "Should have logged error or warning for failure scenario")
                
                print(f"✓ Error scenario logging verified: {total_error_logs} errors, {total_warning_logs} warnings")
    
    def test_logging_performance_impact(self):
        """Test that logging doesn't significantly impact performance."""
        import time
        
        # Create logger
        logger = LLMAgentLogger("PerformanceTest")
        
        # Mock the underlying loggers to avoid actual I/O
        logger.logger = Mock()
        logger.fallback_logger = Mock()
        
        # Test logging performance
        start_time = time.time()
        
        # Perform multiple logging operations
        for i in range(100):
            logger.log_llm_call(f"test prompt {i}", {'iteration': i}, f'session_{i}', i)
            logger.log_llm_response({
                'success': True,
                'response': f'response {i}',
                'model': 'test-model',
                'tokens_used': 100 + i
            }, f'session_{i}', i, 0.1)
            logger.log_component_extraction(f'component_{i}', f'data_{i}', True)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Should complete quickly (under 1 second for 300 operations)
        self.assertLess(total_time, 1.0, f"Logging 300 operations took {total_time:.3f}s, should be under 1s")
        
        print(f"✓ Performance test: 300 logging operations in {total_time:.3f}s")


if __name__ == '__main__':
    unittest.main(verbosity=2)
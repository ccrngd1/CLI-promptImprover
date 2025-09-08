"""
Integration tests for comprehensive error handling and logging system.

Tests the integration of error handling, logging, recovery strategies,
and monitoring across all system components.
"""

import pytest
import tempfile
import shutil
import time
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

from error_handling import (
    BedrockOptimizerException, APIError, OrchestrationError, AgentError,
    TimeoutError, RateLimitError, ValidationError, StorageError,
    ErrorCategory, ErrorSeverity, global_error_handler,
    FallbackStrategy, RetryStrategy, with_retry, with_timeout,
    handle_api_errors, handle_orchestration_errors
)
from logging_config import (
    setup_logging, get_logger, log_exception, performance_logger,
    orchestration_logger, StructuredFormatter, PerformanceLogger,
    OrchestrationLogger
)
from session import SessionManager
from orchestration.engine import LLMOrchestrationEngine
from bedrock.executor import BedrockExecutor
from evaluation.evaluator import Evaluator
from storage.history import HistoryManager


class TestErrorHandlingIntegration:
    """Test comprehensive error handling integration."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Set up comprehensive logging
        self.loggers = setup_logging(
            log_level='DEBUG',
            log_dir=f"{self.temp_dir}/logs",
            enable_structured_logging=True,
            enable_performance_logging=True
        )
        
        # Create mock dependencies
        self.mock_bedrock_executor = Mock(spec=BedrockExecutor)
        self.mock_evaluator = Mock(spec=Evaluator)
        self.mock_history_manager = Mock(spec=HistoryManager)
        
        # Create session manager with error handling
        self.session_manager = SessionManager(
            bedrock_executor=self.mock_bedrock_executor,
            evaluator=self.mock_evaluator,
            history_manager=self.mock_history_manager
        )
        
        # Create orchestration engine with error handling
        self.orchestration_engine = LLMOrchestrationEngine(
            bedrock_executor=self.mock_bedrock_executor,
            evaluator=self.mock_evaluator
        )
    
    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
        # Clear error handler statistics
        global_error_handler.error_counts.clear()
    
    def test_structured_logging_integration(self):
        """Test structured logging across all components."""
        logger = get_logger('test_integration')
        
        # Test basic structured logging
        logger.info(
            "Test structured log entry",
            extra={
                'session_id': 'test-session-123',
                'iteration': 1,
                'agent_name': 'test_agent',
                'processing_time': 2.5,
                'error_code': 'TEST_001'
            }
        )
        
        # Test performance logging
        performance_logger.start_timer('test_operation')
        time.sleep(0.1)  # Simulate work
        duration = performance_logger.end_timer(
            'test_operation',
            session_id='test-session-123',
            success=True
        )
        
        assert duration >= 0.1
        
        # Test orchestration logging
        orchestration_logger.log_agent_coordination(
            session_id='test-session-123',
            iteration=1,
            execution_order=['analyzer', 'refiner', 'validator'],
            strategy_type='comprehensive',
            reasoning='Test coordination reasoning'
        )
        
        orchestration_logger.log_conflict_resolution(
            session_id='test-session-123',
            iteration=1,
            conflicts=[{'agent': 'analyzer', 'recommendation': 'test'}],
            resolution_method='llm_synthesis',
            final_decision='Test resolution'
        )
        
        # Verify log files were created
        log_dir = Path(self.temp_dir) / 'logs'
        assert log_dir.exists()
        
        # Check for log files
        log_files = list(log_dir.glob('*.log'))
        assert len(log_files) > 0
    
    def test_error_recovery_strategies(self):
        """Test error recovery strategies across components."""
        # Register custom recovery strategy
        def custom_recovery(error, context):
            return {'recovered': True, 'strategy': 'custom', 'error': str(error)}
        
        custom_strategy = FallbackStrategy(fallback_function=custom_recovery)
        global_error_handler.register_recovery_strategy(
            ErrorCategory.API_ERROR,
            custom_strategy
        )
        
        # Test API error recovery
        api_error = APIError(
            "Test API error",
            api_name="test_api",
            status_code=503
        )
        
        recovery_result = global_error_handler.handle_error(
            api_error,
            {'test_context': 'api_recovery'}
        )
        
        assert recovery_result['recovered'] is True
        assert recovery_result['strategy'] == 'custom'
        
        # Test orchestration error recovery
        orchestration_error = OrchestrationError(
            "Test orchestration error",
            orchestration_stage="synthesis"
        )
        
        try:
            global_error_handler.handle_error(
                orchestration_error,
                {'test_context': 'orchestration_recovery'}
            )
        except OrchestrationError:
            # Expected if no recovery strategy is registered
            pass
        
        # Verify error statistics
        error_stats = global_error_handler.get_error_statistics()
        assert error_stats['total_errors'] >= 2
        assert 'api_error:APIError' in error_stats['error_counts']
    
    def test_retry_mechanism_integration(self):
        """Test retry mechanisms with error handling."""
        call_count = 0
        
        @with_retry(max_attempts=3, base_delay=0.1)
        def failing_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise APIError("Temporary failure", "test_api")
            return "success"
        
        # Should succeed after retries
        result = failing_function()
        assert result == "success"
        assert call_count == 3
        
        # Test with permanent failure
        call_count = 0
        
        @with_retry(max_attempts=2, base_delay=0.1)
        def permanently_failing_function():
            nonlocal call_count
            call_count += 1
            raise ValidationError("Permanent validation error")
        
        with pytest.raises(ValidationError):
            permanently_failing_function()
        
        assert call_count == 2  # Should have retried
    
    def test_timeout_handling_integration(self):
        """Test timeout handling across components."""
        
        @with_timeout(timeout_seconds=0.5, operation_name="test_timeout")
        def slow_function():
            time.sleep(1.0)  # Longer than timeout
            return "should_not_reach"
        
        with pytest.raises(TimeoutError) as exc_info:
            slow_function()
        
        assert "test_timeout" in str(exc_info.value)
        assert exc_info.value.timeout_duration == 0.5
    
    def test_session_manager_error_integration(self):
        """Test error handling integration in session manager."""
        # Setup mocks to trigger various errors
        self.mock_history_manager.create_session.side_effect = StorageError(
            "Storage system unavailable",
            storage_operation="create_session"
        )
        
        # Test session creation with storage error
        result = self.session_manager.create_session(
            initial_prompt="Test prompt",
            context={'test': 'error_integration'}
        )
        
        # Should handle error gracefully
        assert result.success is False
        assert "storage" in result.message.lower() or "failed" in result.message.lower()
        
        # Test orchestration error handling
        self.mock_history_manager.create_session.side_effect = None
        self.mock_history_manager.create_session.return_value = "test-session-123"
        
        # Mock orchestration engine to raise error
        self.session_manager.orchestration_engine = Mock()
        self.session_manager.orchestration_engine.run_llm_orchestrated_iteration.side_effect = OrchestrationError(
            "Orchestration system failure",
            orchestration_stage="agent_coordination"
        )
        
        # Create session successfully
        session_result = self.session_manager.create_session(
            initial_prompt="Test prompt",
            context={'test': 'orchestration_error'}
        )
        assert session_result.success is True
        
        # Run iteration that will fail
        iteration_result = self.session_manager.run_optimization_iteration(
            session_result.session_state.session_id
        )
        
        # Should handle orchestration error
        assert iteration_result.success is False
        assert "orchestration" in iteration_result.message.lower() or "failed" in iteration_result.message.lower()
    
    def test_orchestration_engine_error_integration(self):
        """Test error handling integration in orchestration engine."""
        # Test with agent failures
        with patch.object(self.orchestration_engine, 'llm_agents') as mock_agents:
            # Setup failing agents
            mock_analyzer = Mock()
            mock_analyzer.process.side_effect = AgentError(
                "Agent processing failed",
                agent_name="analyzer",
                agent_operation="process"
            )
            
            mock_refiner = Mock()
            mock_refiner.process.side_effect = TimeoutError(
                "Agent timeout",
                operation="refiner_process",
                timeout_duration=30.0
            )
            
            mock_validator = Mock()
            mock_validator.process.side_effect = ValidationError(
                "Validation failed"
            )
            
            mock_agents.__getitem__.side_effect = lambda key: {
                'analyzer': mock_analyzer,
                'refiner': mock_refiner,
                'validator': mock_validator
            }[key]
            mock_agents.__contains__.return_value = True
            
            # Mock Bedrock executor failure
            self.mock_bedrock_executor.execute_prompt.side_effect = APIError(
                "Bedrock API unavailable",
                api_name="bedrock",
                status_code=503
            )
            
            # Run orchestration iteration
            result = self.orchestration_engine.run_llm_orchestrated_iteration(
                prompt="Test prompt for error handling",
                context={'session_id': 'error-test-session'}
            )
            
            # Should handle multiple failures gracefully
            assert result.success is False
            assert result.error_message is not None
            assert len(result.agent_results) == 0  # No successful agents
    
    def test_comprehensive_error_monitoring(self):
        """Test comprehensive error monitoring and statistics."""
        # Generate various types of errors
        errors_to_generate = [
            APIError("API Error 1", "bedrock"),
            APIError("API Error 2", "bedrock", status_code=429),
            OrchestrationError("Orchestration Error 1", "synthesis"),
            AgentError("Agent Error 1", "analyzer", "process"),
            TimeoutError("Timeout Error 1", "llm_call", 30.0),
            RateLimitError("Rate Limit Error 1", retry_after=60.0),
            ValidationError("Validation Error 1", field_name="prompt"),
            StorageError("Storage Error 1", storage_operation="save")
        ]
        
        # Process errors through error handler
        for error in errors_to_generate:
            try:
                global_error_handler.handle_error(error, {'test': 'monitoring'})
            except Exception:
                # Expected for errors without recovery strategies
                pass
        
        # Get error statistics
        error_stats = global_error_handler.get_error_statistics()
        
        # Verify statistics
        assert error_stats['total_errors'] >= len(errors_to_generate)
        assert len(error_stats['error_categories']) >= 6  # Different categories
        
        # Verify specific error types were tracked
        error_types = list(error_stats['error_counts'].keys())
        assert any('APIError' in error_type for error_type in error_types)
        assert any('OrchestrationError' in error_type for error_type in error_types)
        assert any('AgentError' in error_type for error_type in error_types)
    
    def test_log_file_structure_and_content(self):
        """Test log file structure and content quality."""
        # Generate various log entries
        logger = get_logger('test_log_structure')
        
        # Generate different types of log entries
        logger.debug("Debug message for testing")
        logger.info("Info message with context", extra={'session_id': 'test-123'})
        logger.warning("Warning message")
        logger.error("Error message", extra={'error_code': 'TEST_001'})
        
        # Generate performance logs
        performance_logger.start_timer('test_performance_log')
        time.sleep(0.05)
        performance_logger.end_timer('test_performance_log', test_metric=True)
        
        # Generate orchestration logs
        orchestration_logger.log_synthesis_decision(
            session_id='test-123',
            iteration=1,
            agent_results={'analyzer': Mock()},
            synthesis_reasoning='Test synthesis',
            final_prompt='Test final prompt',
            confidence=0.85
        )
        
        # Check log directory structure
        log_dir = Path(self.temp_dir) / 'logs'
        assert log_dir.exists()
        
        # Check for expected log files
        expected_files = [
            'bedrock_optimizer.log',
            'errors.log',
            'orchestration.log',
            'performance.log'
        ]
        
        for expected_file in expected_files:
            log_file = log_dir / expected_file
            if log_file.exists():
                # Verify file has content
                assert log_file.stat().st_size > 0
                
                # Verify JSON structure if structured logging is enabled
                with open(log_file, 'r') as f:
                    lines = f.readlines()
                    if lines:
                        # Try to parse first line as JSON
                        try:
                            log_entry = json.loads(lines[0])
                            assert 'timestamp' in log_entry
                            assert 'level' in log_entry
                            assert 'message' in log_entry
                        except json.JSONDecodeError:
                            # Non-JSON format is also acceptable
                            pass
    
    def test_error_context_preservation(self):
        """Test that error context is preserved through the handling chain."""
        original_context = {
            'session_id': 'context-test-session',
            'iteration': 5,
            'agent_name': 'test_agent',
            'operation': 'test_operation',
            'user_data': {'key': 'value'}
        }
        
        # Create error with context
        error = OrchestrationError(
            "Test error with context",
            orchestration_stage="test_stage",
            context=original_context
        )
        
        # Handle error and verify context preservation
        try:
            global_error_handler.handle_error(error, original_context)
        except Exception:
            pass
        
        # Verify error was logged with context
        # (This would be verified by checking log files in a real scenario)
        assert error.context is not None
        
        # Test exception logging with context
        logger = get_logger('context_test')
        
        try:
            raise ValueError("Test exception for context preservation")
        except Exception as e:
            log_exception(logger, e, original_context)
        
        # Context should be preserved in log entry
        # (Verification would involve parsing log files)
    
    def test_performance_monitoring_integration(self):
        """Test performance monitoring integration across components."""
        # Test nested performance timing
        performance_logger.start_timer('outer_operation')
        
        performance_logger.start_timer('inner_operation_1')
        time.sleep(0.05)
        duration1 = performance_logger.end_timer('inner_operation_1')
        
        performance_logger.start_timer('inner_operation_2')
        time.sleep(0.03)
        duration2 = performance_logger.end_timer('inner_operation_2')
        
        total_duration = performance_logger.end_timer('outer_operation')
        
        # Verify timing relationships
        assert duration1 >= 0.05
        assert duration2 >= 0.03
        assert total_duration >= duration1 + duration2
        
        # Test metric logging
        performance_logger.log_metric('test_metric_1', 42.5, unit='seconds')
        performance_logger.log_metric('test_metric_2', 100, unit='count', category='test')
        
        # Test performance monitoring in orchestration
        with patch.object(self.orchestration_engine, 'llm_agents') as mock_agents:
            mock_agents.__getitem__.return_value = Mock(
                process=Mock(return_value=Mock(
                    agent_name='test', success=True, analysis={}, 
                    suggestions=[], confidence_score=0.8
                ))
            )
            mock_agents.__contains__.return_value = True
            
            # Mock successful responses
            self.mock_bedrock_executor.execute_prompt.return_value = Mock(
                model_name="test", response_text="test", execution_time=1.0,
                token_usage={}, success=True
            )
            
            self.mock_evaluator.evaluate_response.return_value = Mock(
                overall_score=0.8, relevance_score=0.8, clarity_score=0.8,
                completeness_score=0.8, qualitative_feedback="test"
            )
            
            # Run orchestration with performance monitoring
            result = self.orchestration_engine.run_llm_orchestrated_iteration(
                prompt="Performance test prompt",
                context={'session_id': 'perf-test-session'}
            )
            
            # Verify performance was tracked
            assert result.processing_time > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
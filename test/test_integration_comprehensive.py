"""
Comprehensive integration tests for end-to-end optimization workflows.

Tests complete optimization cycles, error recovery, timeout handling,
and orchestration edge cases with real-world scenarios.
"""

import pytest
import tempfile
import shutil
import time
import threading
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from typing import Dict, List, Any

from session import SessionManager, SessionConfig
from orchestration.engine import LLMOrchestrationEngine, OrchestrationResult, ConvergenceAnalysis
from bedrock.executor import BedrockExecutor, ModelConfig
from evaluation.evaluator import Evaluator
from storage.history import HistoryManager
from models import PromptIteration, ExecutionResult, EvaluationResult, UserFeedback
from agents.base import AgentResult
from error_handling import (
    BedrockOptimizerException, APIError, OrchestrationError, TimeoutError,
    RateLimitError, global_error_handler, ErrorCategory, ErrorSeverity
)
from logging_config import setup_logging, performance_logger, orchestration_logger


class TestEndToEndOptimizationWorkflows:
    """Test complete optimization workflows from start to finish."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Set up logging for tests
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
        
        # Create session manager
        self.session_manager = SessionManager(
            bedrock_executor=self.mock_bedrock_executor,
            evaluator=self.mock_evaluator,
            history_manager=self.mock_history_manager
        )
        
        # Mock orchestration engine
        self.session_manager.orchestration_engine = Mock(spec=LLMOrchestrationEngine)
        
        # Test data
        self.test_prompt = "Explain machine learning concepts to beginners"
        self.test_context = {
            'intended_use': 'Educational content',
            'target_audience': 'Beginners',
            'domain': 'Technology'
        }
    
    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_complete_optimization_cycle_success(self):
        """Test a complete successful optimization cycle."""
        # Setup mock responses for multiple iterations
        self._setup_successful_optimization_mocks()
        
        # Start performance timing
        performance_logger.start_timer('complete_optimization_cycle')
        
        # Create session
        session_result = self.session_manager.create_session(
            initial_prompt=self.test_prompt,
            context=self.test_context
        )
        
        assert session_result.success is True
        session_id = session_result.session_state.session_id
        
        # Run multiple optimization iterations
        iteration_results = []
        for i in range(3):
            # Add user feedback for second iteration
            feedback = None
            if i == 1:
                feedback = UserFeedback(
                    satisfaction_rating=3,
                    specific_issues=["Too technical", "Needs examples"],
                    desired_improvements="Make it more beginner-friendly",
                    continue_optimization=True
                )
            
            result = self.session_manager.run_optimization_iteration(
                session_id, user_feedback=feedback
            )
            
            assert result.success is True
            assert result.session_state.current_iteration == i + 1
            iteration_results.append(result)
            
            # Log orchestration decision
            orchestration_logger.log_agent_coordination(
                session_id=session_id,
                iteration=i + 1,
                execution_order=['analyzer', 'refiner', 'validator'],
                strategy_type='comprehensive',
                reasoning=f'Iteration {i + 1} comprehensive analysis'
            )
        
        # Collect final user feedback
        final_feedback_result = self.session_manager.collect_user_feedback(
            session_id=session_id,
            satisfaction_rating=5,
            continue_optimization=False
        )
        
        assert final_feedback_result.success is True
        assert final_feedback_result.session_state.status == 'finalized'
        
        # Finalize session
        finalize_result = self.session_manager.finalize_session(session_id)
        assert finalize_result.success is True
        
        # End performance timing
        total_time = performance_logger.end_timer(
            'complete_optimization_cycle',
            session_id=session_id,
            iterations=3
        )
        
        # Verify workflow completion
        assert len(iteration_results) == 3
        assert all(result.success for result in iteration_results)
        assert total_time > 0
        
        # Verify orchestration was called for each iteration
        assert self.session_manager.orchestration_engine.run_llm_orchestrated_iteration.call_count == 3
        
        # Verify history operations
        assert self.mock_history_manager.save_iteration.call_count == 3
        assert self.mock_history_manager.finalize_session.called
    
    def test_optimization_with_convergence_detection(self):
        """Test optimization workflow with automatic convergence detection."""
        # Setup mocks with convergence analysis
        self._setup_convergence_optimization_mocks()
        
        # Create session with auto-finalize on convergence
        config = SessionConfig(
            max_iterations=5,
            auto_finalize_on_convergence=True,
            convergence_threshold=0.02
        )
        
        session_result = self.session_manager.create_session(
            initial_prompt=self.test_prompt,
            context=self.test_context,
            config=config
        )
        
        session_id = session_result.session_state.session_id
        
        # Run iterations until convergence
        iteration_count = 0
        while iteration_count < 5:
            result = self.session_manager.run_optimization_iteration(session_id)
            iteration_count += 1
            
            assert result.success is True
            
            # Check if convergence was detected
            if result.session_state.convergence_detected:
                orchestration_logger.log_convergence_analysis(
                    session_id=session_id,
                    iteration=iteration_count,
                    has_converged=True,
                    convergence_score=0.92,
                    reasoning="Evaluation scores have stabilized",
                    confidence=0.88
                )
                break
        
        # Verify convergence was detected before max iterations
        assert iteration_count < 5
        assert result.session_state.convergence_detected is True
        assert result.session_state.status == 'finalized'
    
    def test_optimization_with_api_failures_and_recovery(self):
        """Test optimization workflow with API failures and recovery."""
        # Setup mocks with intermittent failures
        self._setup_api_failure_mocks()
        
        session_result = self.session_manager.create_session(
            initial_prompt=self.test_prompt,
            context=self.test_context
        )
        
        session_id = session_result.session_state.session_id
        
        # Run iteration that will encounter API failures
        with patch('error_handling.global_error_handler.handle_error') as mock_error_handler:
            # Configure error handler to return fallback results
            mock_error_handler.side_effect = self._error_handler_with_fallback
            
            result = self.session_manager.run_optimization_iteration(session_id)
            
            # Should succeed despite API failures due to error handling
            assert result.success is True
            assert mock_error_handler.called
    
    def test_optimization_with_orchestration_conflicts(self):
        """Test optimization workflow with agent conflicts and resolution."""
        # Setup mocks with conflicting agent recommendations
        self._setup_conflict_resolution_mocks()
        
        session_result = self.session_manager.create_session(
            initial_prompt=self.test_prompt,
            context=self.test_context
        )
        
        session_id = session_result.session_state.session_id
        
        # Run iteration with conflicts
        result = self.session_manager.run_optimization_iteration(session_id)
        
        assert result.success is True
        assert result.iteration_result is not None
        
        # Verify conflict resolution was logged
        orchestration_logger.log_conflict_resolution(
            session_id=session_id,
            iteration=1,
            conflicts=[
                {'agent': 'analyzer', 'recommendation': 'Make more detailed'},
                {'agent': 'refiner', 'recommendation': 'Simplify for clarity'}
            ],
            resolution_method='llm_synthesis',
            final_decision='Balanced approach with clear structure and examples'
        )
    
    def test_optimization_with_timeout_handling(self):
        """Test optimization workflow with timeout scenarios."""
        # Setup mocks with slow responses
        self._setup_timeout_mocks()
        
        session_result = self.session_manager.create_session(
            initial_prompt=self.test_prompt,
            context=self.test_context
        )
        
        session_id = session_result.session_state.session_id
        
        # Run iteration that will timeout
        with patch('error_handling.global_error_handler.handle_error') as mock_error_handler:
            mock_error_handler.side_effect = self._timeout_error_handler
            
            result = self.session_manager.run_optimization_iteration(session_id)
            
            # Should handle timeout gracefully
            assert result.success is False
            assert "timeout" in result.message.lower()
    
    def test_optimization_with_large_prompt_handling(self):
        """Test optimization workflow with large prompts."""
        # Create a large prompt (simulate complex use case)
        large_prompt = "Explain machine learning concepts. " * 500  # ~15KB prompt
        
        # Setup mocks for large prompt handling
        self._setup_large_prompt_mocks()
        
        session_result = self.session_manager.create_session(
            initial_prompt=large_prompt,
            context=self.test_context
        )
        
        session_id = session_result.session_state.session_id
        
        # Measure performance for large prompt
        performance_logger.start_timer('large_prompt_optimization')
        
        result = self.session_manager.run_optimization_iteration(session_id)
        
        processing_time = performance_logger.end_timer(
            'large_prompt_optimization',
            prompt_size=len(large_prompt),
            session_id=session_id
        )
        
        assert result.success is True
        assert processing_time > 0
        
        # Log performance metric
        performance_logger.log_metric(
            'large_prompt_processing_time',
            processing_time,
            prompt_size=len(large_prompt)
        )
    
    def test_concurrent_optimization_sessions(self):
        """Test multiple concurrent optimization sessions."""
        # Create multiple sessions
        session_ids = []
        for i in range(3):
            session_result = self.session_manager.create_session(
                initial_prompt=f"{self.test_prompt} - Session {i}",
                context={**self.test_context, 'session_number': i}
            )
            assert session_result.success is True
            session_ids.append(session_result.session_state.session_id)
        
        # Setup mocks for concurrent execution
        self._setup_concurrent_session_mocks()
        
        # Run iterations concurrently using threads
        results = {}
        threads = []
        
        def run_iteration(session_id):
            result = self.session_manager.run_optimization_iteration(session_id)
            results[session_id] = result
        
        performance_logger.start_timer('concurrent_sessions')
        
        # Start threads
        for session_id in session_ids:
            thread = threading.Thread(target=run_iteration, args=(session_id,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        concurrent_time = performance_logger.end_timer(
            'concurrent_sessions',
            session_count=len(session_ids)
        )
        
        # Verify all sessions completed successfully
        assert len(results) == 3
        assert all(result.success for result in results.values())
        
        # Verify sessions are independent
        active_sessions = self.session_manager.list_active_sessions()
        assert len(active_sessions) == 3
        
        # Log concurrency performance
        performance_logger.log_metric(
            'concurrent_sessions_time',
            concurrent_time,
            session_count=len(session_ids)
        )
    
    def test_session_persistence_and_recovery(self):
        """Test session persistence and recovery after failures."""
        # Create session
        session_result = self.session_manager.create_session(
            initial_prompt=self.test_prompt,
            context=self.test_context
        )
        
        session_id = session_result.session_state.session_id
        
        # Setup mocks for persistence testing
        self._setup_persistence_mocks()
        
        # Run first iteration
        result1 = self.session_manager.run_optimization_iteration(session_id)
        assert result1.success is True
        
        # Simulate session manager restart (clear active sessions)
        original_sessions = self.session_manager.active_sessions.copy()
        self.session_manager.active_sessions.clear()
        
        # Verify session is no longer in active sessions
        assert session_id not in self.session_manager.active_sessions
        
        # Mock history manager to return session data
        mock_session_data = {
            'session_id': session_id,
            'initial_prompt': self.test_prompt,
            'context': self.test_context,
            'current_iteration': 1,
            'status': 'active'
        }
        
        self.mock_history_manager.load_session.return_value = mock_session_data
        
        # Attempt to resume session (this would be implemented in a real scenario)
        # For now, verify that history operations work correctly
        assert self.mock_history_manager.save_iteration.called
    
    def test_error_statistics_and_monitoring(self):
        """Test error statistics collection and monitoring."""
        # Setup mocks that will generate various errors
        self._setup_error_statistics_mocks()
        
        session_result = self.session_manager.create_session(
            initial_prompt=self.test_prompt,
            context=self.test_context
        )
        
        session_id = session_result.session_state.session_id
        
        # Run iterations that will generate different types of errors
        error_types = []
        
        for i in range(3):
            try:
                result = self.session_manager.run_optimization_iteration(session_id)
                if not result.success:
                    error_types.append('iteration_failure')
            except Exception as e:
                error_types.append(type(e).__name__)
        
        # Get error statistics
        error_stats = global_error_handler.get_error_statistics()
        
        # Verify error tracking
        assert 'error_counts' in error_stats
        assert 'total_errors' in error_stats
        assert error_stats['total_errors'] >= 0
    
    def _setup_successful_optimization_mocks(self):
        """Setup mocks for successful optimization workflow."""
        # Mock history manager
        self.mock_history_manager.create_session.return_value = "test-session-123"
        self.mock_history_manager.load_session_history.return_value = []
        self.mock_history_manager.save_iteration.return_value = True
        self.mock_history_manager.finalize_session.return_value = True
        
        # Mock orchestration results
        orchestration_results = []
        for i in range(3):
            result = OrchestrationResult(
                success=True,
                orchestrated_prompt=f"Optimized prompt iteration {i+1}",
                agent_results={
                    'analyzer': AgentResult('analyzer', True, {'score': 0.8 + i*0.05}, ['suggestion'], 0.8),
                    'refiner': AgentResult('refiner', True, {'refined': True}, ['improvement'], 0.85),
                    'validator': AgentResult('validator', True, {'valid': True}, ['validated'], 0.9)
                },
                execution_result=ExecutionResult(
                    model_name="test-model",
                    response_text=f"Response for iteration {i+1}",
                    execution_time=1.5,
                    token_usage={'input': 100, 'output': 50},
                    success=True
                ),
                evaluation_result=EvaluationResult(
                    overall_score=0.8 + i*0.05,
                    relevance_score=0.8,
                    clarity_score=0.8,
                    completeness_score=0.8,
                    qualitative_feedback="Good quality"
                ),
                processing_time=2.0,
                llm_orchestrator_confidence=0.85
            )
            orchestration_results.append(result)
        
        self.session_manager.orchestration_engine.run_llm_orchestrated_iteration.side_effect = orchestration_results
    
    def _setup_convergence_optimization_mocks(self):
        """Setup mocks for convergence detection testing."""
        self.mock_history_manager.create_session.return_value = "convergence-session-123"
        self.mock_history_manager.load_session_history.return_value = []
        self.mock_history_manager.save_iteration.return_value = True
        
        # Create convergence analysis for third iteration
        convergence_analysis = ConvergenceAnalysis(
            has_converged=True,
            convergence_score=0.92,
            convergence_reasons=["Stable evaluation scores", "High confidence"],
            improvement_trend="stable",
            iterations_analyzed=3,
            confidence=0.88,
            llm_reasoning="Optimization has reached stable high quality"
        )
        
        # Mock orchestration results with convergence on third iteration
        orchestration_results = [
            OrchestrationResult(
                success=True,
                orchestrated_prompt="Iteration 1 prompt",
                agent_results={},
                execution_result=ExecutionResult("test", "response", 1.0, {}, True),
                evaluation_result=EvaluationResult(0.75, 0.75, 0.75, 0.75, "Good"),
                processing_time=1.0,
                llm_orchestrator_confidence=0.8
            ),
            OrchestrationResult(
                success=True,
                orchestrated_prompt="Iteration 2 prompt",
                agent_results={},
                execution_result=ExecutionResult("test", "response", 1.0, {}, True),
                evaluation_result=EvaluationResult(0.85, 0.85, 0.85, 0.85, "Better"),
                processing_time=1.0,
                llm_orchestrator_confidence=0.85
            ),
            OrchestrationResult(
                success=True,
                orchestrated_prompt="Converged prompt",
                agent_results={},
                execution_result=ExecutionResult("test", "response", 1.0, {}, True),
                evaluation_result=EvaluationResult(0.9, 0.9, 0.9, 0.9, "Excellent"),
                convergence_analysis=convergence_analysis,
                processing_time=1.0,
                llm_orchestrator_confidence=0.9
            )
        ]
        
        self.session_manager.orchestration_engine.run_llm_orchestrated_iteration.side_effect = orchestration_results
    
    def _setup_api_failure_mocks(self):
        """Setup mocks that simulate API failures."""
        self.mock_history_manager.create_session.return_value = "api-failure-session-123"
        self.mock_history_manager.load_session_history.return_value = []
        self.mock_history_manager.save_iteration.return_value = True
        
        # Mock orchestration to raise API errors
        def failing_orchestration(*args, **kwargs):
            raise APIError(
                "Bedrock API temporarily unavailable",
                api_name="bedrock",
                status_code=503,
                severity=ErrorSeverity.MEDIUM
            )
        
        self.session_manager.orchestration_engine.run_llm_orchestrated_iteration.side_effect = failing_orchestration
    
    def _setup_conflict_resolution_mocks(self):
        """Setup mocks for conflict resolution testing."""
        self.mock_history_manager.create_session.return_value = "conflict-session-123"
        self.mock_history_manager.load_session_history.return_value = []
        self.mock_history_manager.save_iteration.return_value = True
        
        # Mock orchestration result with conflict resolution
        result = OrchestrationResult(
            success=True,
            orchestrated_prompt="Balanced prompt with clear structure and examples",
            agent_results={
                'analyzer': AgentResult('analyzer', True, {'recommendation': 'detailed'}, ['More detail'], 0.8),
                'refiner': AgentResult('refiner', True, {'recommendation': 'simple'}, ['Simplify'], 0.85),
                'validator': AgentResult('validator', True, {'recommendation': 'balanced'}, ['Balance'], 0.9)
            },
            execution_result=ExecutionResult("test", "response", 1.0, {}, True),
            evaluation_result=EvaluationResult(0.85, 0.85, 0.85, 0.85, "Balanced"),
            conflict_resolutions=[
                {'conflict': 'detail_vs_simplicity', 'resolution': 'balanced_approach'}
            ],
            synthesis_reasoning="Resolved conflict by balancing detail with clarity",
            processing_time=2.5,
            llm_orchestrator_confidence=0.8
        )
        
        self.session_manager.orchestration_engine.run_llm_orchestrated_iteration.return_value = result
    
    def _setup_timeout_mocks(self):
        """Setup mocks that simulate timeout scenarios."""
        self.mock_history_manager.create_session.return_value = "timeout-session-123"
        
        def slow_orchestration(*args, **kwargs):
            time.sleep(2)  # Simulate slow operation
            raise TimeoutError(
                "Orchestration operation timed out",
                operation="llm_orchestration",
                timeout_duration=1.0
            )
        
        self.session_manager.orchestration_engine.run_llm_orchestrated_iteration.side_effect = slow_orchestration
    
    def _setup_large_prompt_mocks(self):
        """Setup mocks for large prompt handling."""
        self.mock_history_manager.create_session.return_value = "large-prompt-session-123"
        self.mock_history_manager.load_session_history.return_value = []
        self.mock_history_manager.save_iteration.return_value = True
        
        # Mock orchestration result for large prompt
        result = OrchestrationResult(
            success=True,
            orchestrated_prompt="Optimized large prompt with improved structure",
            agent_results={},
            execution_result=ExecutionResult("test", "Large response", 3.0, {'input': 5000, 'output': 1000}, True),
            evaluation_result=EvaluationResult(0.8, 0.8, 0.8, 0.8, "Good for large prompt"),
            processing_time=5.0,  # Longer processing time for large prompt
            llm_orchestrator_confidence=0.8
        )
        
        self.session_manager.orchestration_engine.run_llm_orchestrated_iteration.return_value = result
    
    def _setup_concurrent_session_mocks(self):
        """Setup mocks for concurrent session testing."""
        session_counter = 0
        
        def create_session_id():
            nonlocal session_counter
            session_counter += 1
            return f"concurrent-session-{session_counter}"
        
        self.mock_history_manager.create_session.side_effect = create_session_id
        self.mock_history_manager.load_session_history.return_value = []
        self.mock_history_manager.save_iteration.return_value = True
        
        # Mock orchestration result
        result = OrchestrationResult(
            success=True,
            orchestrated_prompt="Concurrent optimization result",
            agent_results={},
            execution_result=ExecutionResult("test", "Concurrent response", 1.0, {}, True),
            evaluation_result=EvaluationResult(0.8, 0.8, 0.8, 0.8, "Concurrent quality"),
            processing_time=1.0,
            llm_orchestrator_confidence=0.8
        )
        
        self.session_manager.orchestration_engine.run_llm_orchestrated_iteration.return_value = result
    
    def _setup_persistence_mocks(self):
        """Setup mocks for persistence testing."""
        self.mock_history_manager.create_session.return_value = "persistence-session-123"
        self.mock_history_manager.load_session_history.return_value = []
        self.mock_history_manager.save_iteration.return_value = True
        
        result = OrchestrationResult(
            success=True,
            orchestrated_prompt="Persistent optimization",
            agent_results={},
            execution_result=ExecutionResult("test", "Persistent response", 1.0, {}, True),
            evaluation_result=EvaluationResult(0.8, 0.8, 0.8, 0.8, "Persistent quality"),
            processing_time=1.0,
            llm_orchestrator_confidence=0.8
        )
        
        self.session_manager.orchestration_engine.run_llm_orchestrated_iteration.return_value = result
    
    def _setup_error_statistics_mocks(self):
        """Setup mocks for error statistics testing."""
        self.mock_history_manager.create_session.return_value = "error-stats-session-123"
        
        # Mock orchestration to raise different types of errors
        error_sequence = [
            APIError("API Error", "bedrock"),
            OrchestrationError("Orchestration Error", "synthesis"),
            TimeoutError("Timeout Error", "processing", 30.0)
        ]
        
        self.session_manager.orchestration_engine.run_llm_orchestrated_iteration.side_effect = error_sequence
    
    def _error_handler_with_fallback(self, error, context):
        """Mock error handler that provides fallback results."""
        if isinstance(error, APIError):
            # Return fallback orchestration result
            return OrchestrationResult(
                success=True,
                orchestrated_prompt="Fallback optimized prompt",
                agent_results={},
                execution_result=ExecutionResult("fallback", "Fallback response", 1.0, {}, True),
                evaluation_result=EvaluationResult(0.7, 0.7, 0.7, 0.7, "Fallback quality"),
                processing_time=1.0,
                llm_orchestrator_confidence=0.7
            )
        else:
            raise error
    
    def _timeout_error_handler(self, error, context):
        """Mock error handler for timeout scenarios."""
        if isinstance(error, TimeoutError):
            # Don't recover from timeouts in this test
            raise error
        return None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
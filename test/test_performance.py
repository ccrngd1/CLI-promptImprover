"""
Performance tests for concurrent sessions, large prompt handling, and system scalability.

Tests system performance under various load conditions, memory usage,
response times, and concurrent operation handling.
"""

import pytest
import time
import threading
import multiprocessing
import psutil
import tempfile
import shutil
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import Mock, patch
from typing import List, Dict, Any
import json

from session import SessionManager, SessionConfig
from orchestration.engine import LLMOrchestrationEngine, OrchestrationResult
from bedrock.executor import BedrockExecutor, ModelConfig
from evaluation.evaluator import Evaluator
from storage.history import HistoryManager
from models import PromptIteration, ExecutionResult, EvaluationResult, UserFeedback
from agents.base import AgentResult
from logging_config import setup_logging, performance_logger
from error_handling import BedrockOptimizerException, TimeoutError


class PerformanceMetrics:
    """Class to collect and analyze performance metrics."""
    
    def __init__(self):
        self.metrics = {
            'response_times': [],
            'memory_usage': [],
            'cpu_usage': [],
            'concurrent_operations': [],
            'error_rates': [],
            'throughput': []
        }
        self.start_time = None
        self.end_time = None
    
    def start_measurement(self):
        """Start performance measurement."""
        self.start_time = time.time()
        self.metrics['memory_usage'].append(psutil.virtual_memory().percent)
        self.metrics['cpu_usage'].append(psutil.cpu_percent())
    
    def end_measurement(self):
        """End performance measurement."""
        self.end_time = time.time()
        self.metrics['memory_usage'].append(psutil.virtual_memory().percent)
        self.metrics['cpu_usage'].append(psutil.cpu_percent())
    
    def record_response_time(self, response_time: float):
        """Record a response time measurement."""
        self.metrics['response_times'].append(response_time)
    
    def record_error(self, error_type: str):
        """Record an error occurrence."""
        self.metrics['error_rates'].append(error_type)
    
    def record_throughput(self, operations_per_second: float):
        """Record throughput measurement."""
        self.metrics['throughput'].append(operations_per_second)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get performance metrics summary."""
        response_times = self.metrics['response_times']
        
        summary = {
            'total_duration': self.end_time - self.start_time if self.end_time and self.start_time else 0,
            'response_time_stats': {
                'mean': statistics.mean(response_times) if response_times else 0,
                'median': statistics.median(response_times) if response_times else 0,
                'min': min(response_times) if response_times else 0,
                'max': max(response_times) if response_times else 0,
                'std_dev': statistics.stdev(response_times) if len(response_times) > 1 else 0
            },
            'memory_usage': {
                'initial': self.metrics['memory_usage'][0] if self.metrics['memory_usage'] else 0,
                'final': self.metrics['memory_usage'][-1] if self.metrics['memory_usage'] else 0,
                'peak': max(self.metrics['memory_usage']) if self.metrics['memory_usage'] else 0
            },
            'cpu_usage': {
                'average': statistics.mean(self.metrics['cpu_usage']) if self.metrics['cpu_usage'] else 0,
                'peak': max(self.metrics['cpu_usage']) if self.metrics['cpu_usage'] else 0
            },
            'error_count': len(self.metrics['error_rates']),
            'throughput_stats': {
                'average': statistics.mean(self.metrics['throughput']) if self.metrics['throughput'] else 0,
                'peak': max(self.metrics['throughput']) if self.metrics['throughput'] else 0
            }
        }
        
        return summary


class TestPerformance:
    """Performance test suite."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Set up performance logging
        self.loggers = setup_logging(
            log_level='INFO',
            log_dir=f"{self.temp_dir}/logs",
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
        
        # Performance metrics
        self.metrics = PerformanceMetrics()
    
    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @pytest.mark.performance
    def test_concurrent_session_performance(self):
        """Test performance with multiple concurrent sessions."""
        num_sessions = 10
        iterations_per_session = 3
        
        # Setup mocks for concurrent execution
        self._setup_performance_mocks()
        
        self.metrics.start_measurement()
        
        # Create sessions concurrently
        session_ids = []
        with ThreadPoolExecutor(max_workers=num_sessions) as executor:
            futures = []
            
            for i in range(num_sessions):
                future = executor.submit(
                    self._create_test_session,
                    f"Concurrent test prompt {i}",
                    {'session_id': i}
                )
                futures.append(future)
            
            # Collect session IDs
            for future in as_completed(futures):
                session_id = future.result()
                if session_id:
                    session_ids.append(session_id)
        
        assert len(session_ids) == num_sessions
        
        # Run iterations concurrently
        all_results = []
        with ThreadPoolExecutor(max_workers=num_sessions * 2) as executor:
            futures = []
            
            for session_id in session_ids:
                for iteration in range(iterations_per_session):
                    future = executor.submit(
                        self._run_timed_iteration,
                        session_id
                    )
                    futures.append(future)
            
            # Collect results
            for future in as_completed(futures):
                result, response_time = future.result()
                all_results.append(result)
                self.metrics.record_response_time(response_time)
        
        self.metrics.end_measurement()
        
        # Analyze results
        successful_operations = sum(1 for result in all_results if result and result.success)
        total_operations = len(all_results)
        success_rate = successful_operations / total_operations if total_operations > 0 else 0
        
        # Performance assertions
        assert success_rate >= 0.95  # At least 95% success rate
        
        summary = self.metrics.get_summary()
        
        # Response time should be reasonable
        assert summary['response_time_stats']['mean'] < 5.0  # Average under 5 seconds
        assert summary['response_time_stats']['max'] < 10.0  # Max under 10 seconds
        
        # Memory usage should not grow excessively
        memory_growth = summary['memory_usage']['final'] - summary['memory_usage']['initial']
        assert memory_growth < 20.0  # Less than 20% memory growth
        
        # Log performance metrics
        performance_logger.log_metric(
            'concurrent_sessions_success_rate',
            success_rate,
            num_sessions=num_sessions,
            iterations_per_session=iterations_per_session
        )
        
        performance_logger.log_metric(
            'concurrent_sessions_avg_response_time',
            summary['response_time_stats']['mean'],
            num_sessions=num_sessions
        )
    
    @pytest.mark.performance
    def test_large_prompt_performance(self):
        """Test performance with large prompts."""
        # Create prompts of varying sizes
        prompt_sizes = [1000, 5000, 10000, 20000, 50000]  # Characters
        
        self._setup_performance_mocks()
        
        results = {}
        
        for size in prompt_sizes:
            # Create large prompt
            base_prompt = "Explain machine learning concepts in detail. "
            large_prompt = base_prompt * (size // len(base_prompt))
            
            self.metrics.start_measurement()
            
            # Create session
            session_result = self.session_manager.create_session(
                initial_prompt=large_prompt,
                context={'prompt_size': size}
            )
            
            assert session_result.success is True
            session_id = session_result.session_state.session_id
            
            # Run iteration and measure performance
            start_time = time.time()
            iteration_result = self.session_manager.run_optimization_iteration(session_id)
            end_time = time.time()
            
            response_time = end_time - start_time
            
            self.metrics.end_measurement()
            
            # Record results
            results[size] = {
                'success': iteration_result.success,
                'response_time': response_time,
                'memory_usage': psutil.virtual_memory().percent
            }
            
            # Log performance for this size
            performance_logger.log_metric(
                'large_prompt_response_time',
                response_time,
                prompt_size=size
            )
        
        # Analyze scaling behavior
        response_times = [results[size]['response_time'] for size in prompt_sizes]
        
        # Response time should scale reasonably with prompt size
        # (not exponentially)
        max_response_time = max(response_times)
        min_response_time = min(response_times)
        scaling_factor = max_response_time / min_response_time
        
        assert scaling_factor < 10.0  # Should not scale more than 10x
        assert all(results[size]['success'] for size in prompt_sizes)  # All should succeed
        
        # Log scaling analysis
        performance_logger.log_metric(
            'large_prompt_scaling_factor',
            scaling_factor,
            max_size=max(prompt_sizes),
            min_size=min(prompt_sizes)
        )
    
    @pytest.mark.performance
    def test_memory_usage_under_load(self):
        """Test memory usage under sustained load."""
        num_iterations = 50
        
        self._setup_performance_mocks()
        
        # Create session
        session_result = self.session_manager.create_session(
            initial_prompt="Memory usage test prompt",
            context={'test_type': 'memory_load'}
        )
        
        session_id = session_result.session_state.session_id
        
        # Record initial memory usage
        initial_memory = psutil.virtual_memory().percent
        memory_readings = [initial_memory]
        
        # Run many iterations
        for i in range(num_iterations):
            result = self.session_manager.run_optimization_iteration(session_id)
            assert result.success is True
            
            # Record memory usage every 10 iterations
            if i % 10 == 0:
                current_memory = psutil.virtual_memory().percent
                memory_readings.append(current_memory)
                
                performance_logger.log_metric(
                    'memory_usage_during_load',
                    current_memory,
                    iteration=i
                )
        
        # Final memory reading
        final_memory = psutil.virtual_memory().percent
        memory_readings.append(final_memory)
        
        # Analyze memory usage
        memory_growth = final_memory - initial_memory
        max_memory = max(memory_readings)
        
        # Memory should not grow excessively
        assert memory_growth < 15.0  # Less than 15% growth
        assert max_memory < 90.0  # Should not exceed 90% memory usage
        
        # Check for memory leaks (memory should stabilize)
        if len(memory_readings) >= 3:
            recent_readings = memory_readings[-3:]
            memory_variance = statistics.variance(recent_readings)
            assert memory_variance < 5.0  # Low variance indicates stability
        
        performance_logger.log_metric(
            'memory_growth_under_load',
            memory_growth,
            iterations=num_iterations
        )
    
    @pytest.mark.performance
    def test_response_time_consistency(self):
        """Test response time consistency across multiple operations."""
        num_operations = 30
        
        self._setup_performance_mocks()
        
        # Create session
        session_result = self.session_manager.create_session(
            initial_prompt="Response time consistency test",
            context={'test_type': 'response_time'}
        )
        
        session_id = session_result.session_state.session_id
        
        response_times = []
        
        # Run multiple iterations
        for i in range(num_operations):
            start_time = time.time()
            result = self.session_manager.run_optimization_iteration(session_id)
            end_time = time.time()
            
            response_time = end_time - start_time
            response_times.append(response_time)
            
            assert result.success is True
            
            performance_logger.log_metric(
                'individual_response_time',
                response_time,
                operation=i
            )
        
        # Analyze response time consistency
        mean_time = statistics.mean(response_times)
        std_dev = statistics.stdev(response_times)
        coefficient_of_variation = std_dev / mean_time if mean_time > 0 else 0
        
        # Response times should be consistent (low coefficient of variation)
        assert coefficient_of_variation < 0.5  # CV should be less than 50%
        assert std_dev < 2.0  # Standard deviation should be reasonable
        
        # No response time should be excessively long
        max_time = max(response_times)
        assert max_time < mean_time * 3  # No response more than 3x the mean
        
        performance_logger.log_metric(
            'response_time_consistency',
            coefficient_of_variation,
            operations=num_operations
        )
    
    @pytest.mark.performance
    def test_throughput_measurement(self):
        """Test system throughput under various conditions."""
        test_scenarios = [
            {'name': 'light_load', 'concurrent_sessions': 2, 'iterations': 5},
            {'name': 'medium_load', 'concurrent_sessions': 5, 'iterations': 10},
            {'name': 'heavy_load', 'concurrent_sessions': 10, 'iterations': 15}
        ]
        
        self._setup_performance_mocks()
        
        throughput_results = {}
        
        for scenario in test_scenarios:
            scenario_name = scenario['name']
            num_sessions = scenario['concurrent_sessions']
            num_iterations = scenario['iterations']
            
            start_time = time.time()
            
            # Create sessions
            session_ids = []
            for i in range(num_sessions):
                session_result = self.session_manager.create_session(
                    initial_prompt=f"Throughput test {scenario_name} session {i}",
                    context={'scenario': scenario_name, 'session': i}
                )
                session_ids.append(session_result.session_state.session_id)
            
            # Run iterations concurrently
            total_operations = 0
            with ThreadPoolExecutor(max_workers=num_sessions) as executor:
                futures = []
                
                for session_id in session_ids:
                    for iteration in range(num_iterations):
                        future = executor.submit(
                            self.session_manager.run_optimization_iteration,
                            session_id
                        )
                        futures.append(future)
                
                # Wait for completion
                for future in as_completed(futures):
                    result = future.result()
                    if result and result.success:
                        total_operations += 1
            
            end_time = time.time()
            duration = end_time - start_time
            
            # Calculate throughput
            throughput = total_operations / duration if duration > 0 else 0
            throughput_results[scenario_name] = {
                'throughput': throughput,
                'total_operations': total_operations,
                'duration': duration,
                'success_rate': total_operations / (num_sessions * num_iterations)
            }
            
            performance_logger.log_metric(
                f'throughput_{scenario_name}',
                throughput,
                sessions=num_sessions,
                iterations=num_iterations
            )
        
        # Verify throughput scales appropriately
        light_throughput = throughput_results['light_load']['throughput']
        heavy_throughput = throughput_results['heavy_load']['throughput']
        
        # Heavy load should have higher absolute throughput
        # (even if per-session throughput is lower)
        assert heavy_throughput > light_throughput * 0.5  # At least 50% of scaled expectation
        
        # All scenarios should have good success rates
        for scenario_name, results in throughput_results.items():
            assert results['success_rate'] >= 0.9  # At least 90% success rate
    
    @pytest.mark.performance
    def test_error_handling_performance_impact(self):
        """Test performance impact of error handling and recovery."""
        num_operations = 20
        error_rate = 0.3  # 30% of operations will fail
        
        # Setup mocks with intermittent failures
        self._setup_error_performance_mocks(error_rate)
        
        session_result = self.session_manager.create_session(
            initial_prompt="Error handling performance test",
            context={'test_type': 'error_performance'}
        )
        
        session_id = session_result.session_state.session_id
        
        successful_operations = 0
        failed_operations = 0
        response_times = []
        
        for i in range(num_operations):
            start_time = time.time()
            
            try:
                result = self.session_manager.run_optimization_iteration(session_id)
                if result.success:
                    successful_operations += 1
                else:
                    failed_operations += 1
            except Exception:
                failed_operations += 1
            
            end_time = time.time()
            response_time = end_time - start_time
            response_times.append(response_time)
        
        # Analyze error handling performance
        actual_error_rate = failed_operations / num_operations
        mean_response_time = statistics.mean(response_times)
        
        # Error rate should be close to expected
        assert abs(actual_error_rate - error_rate) < 0.2  # Within 20% of expected
        
        # Response times should still be reasonable despite errors
        assert mean_response_time < 3.0  # Average under 3 seconds
        
        # Some operations should still succeed
        assert successful_operations > 0
        
        performance_logger.log_metric(
            'error_handling_response_time',
            mean_response_time,
            error_rate=actual_error_rate
        )
    
    def _create_test_session(self, prompt: str, context: Dict[str, Any]) -> str:
        """Helper to create a test session."""
        try:
            result = self.session_manager.create_session(
                initial_prompt=prompt,
                context=context
            )
            return result.session_state.session_id if result.success else None
        except Exception:
            return None
    
    def _run_timed_iteration(self, session_id: str) -> tuple:
        """Helper to run a timed iteration."""
        start_time = time.time()
        try:
            result = self.session_manager.run_optimization_iteration(session_id)
            end_time = time.time()
            return result, end_time - start_time
        except Exception as e:
            end_time = time.time()
            self.metrics.record_error(type(e).__name__)
            return None, end_time - start_time
    
    def _setup_performance_mocks(self):
        """Setup mocks optimized for performance testing."""
        # Mock history manager with fast responses
        session_counter = 0
        
        def create_session_id():
            nonlocal session_counter
            session_counter += 1
            return f"perf-session-{session_counter}"
        
        self.mock_history_manager.create_session.side_effect = create_session_id
        self.mock_history_manager.load_session_history.return_value = []
        self.mock_history_manager.save_iteration.return_value = True
        
        # Mock orchestration with fast, consistent responses
        def fast_orchestration(*args, **kwargs):
            # Simulate some processing time
            time.sleep(0.1)  # 100ms processing time
            
            return OrchestrationResult(
                success=True,
                orchestrated_prompt="Performance test optimized prompt",
                agent_results={
                    'analyzer': AgentResult('analyzer', True, {'score': 0.8}, ['fast'], 0.8),
                    'refiner': AgentResult('refiner', True, {'refined': True}, ['quick'], 0.85),
                    'validator': AgentResult('validator', True, {'valid': True}, ['validated'], 0.9)
                },
                execution_result=ExecutionResult(
                    model_name="perf-test-model",
                    response_text="Fast performance test response",
                    execution_time=0.5,
                    token_usage={'input': 100, 'output': 50},
                    success=True
                ),
                evaluation_result=EvaluationResult(
                    overall_score=0.8,
                    relevance_score=0.8,
                    clarity_score=0.8,
                    completeness_score=0.8,
                    qualitative_feedback="Good performance"
                ),
                processing_time=0.1,
                llm_orchestrator_confidence=0.8
            )
        
        self.session_manager.orchestration_engine.run_llm_orchestrated_iteration.side_effect = fast_orchestration
    
    def _setup_error_performance_mocks(self, error_rate: float):
        """Setup mocks with controlled error rate for performance testing."""
        self.mock_history_manager.create_session.return_value = "error-perf-session-123"
        self.mock_history_manager.load_session_history.return_value = []
        self.mock_history_manager.save_iteration.return_value = True
        
        call_count = 0
        
        def error_prone_orchestration(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            
            # Introduce errors based on error rate
            if (call_count % int(1 / error_rate)) == 0:
                time.sleep(0.2)  # Simulate error handling overhead
                raise BedrockOptimizerException("Simulated error for performance testing")
            
            # Normal successful response
            time.sleep(0.1)
            return OrchestrationResult(
                success=True,
                orchestrated_prompt="Error performance test prompt",
                agent_results={},
                execution_result=ExecutionResult("test", "response", 0.5, {}, True),
                evaluation_result=EvaluationResult(0.8, 0.8, 0.8, 0.8, "Good"),
                processing_time=0.1,
                llm_orchestrator_confidence=0.8
            )
        
        self.session_manager.orchestration_engine.run_llm_orchestrated_iteration.side_effect = error_prone_orchestration


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "performance"])
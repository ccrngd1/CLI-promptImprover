"""
Unit tests for the SessionManager class.

Tests session lifecycle management, orchestration integration, user feedback
collection, and session finalization with reasoning explanations.
"""

import pytest
import tempfile
import shutil
import json
from datetime import datetime
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path

from session import SessionManager, SessionConfig, SessionState, SessionResult
from models import PromptIteration, ExecutionResult, EvaluationResult, UserFeedback
from storage.history import HistoryManager
from bedrock.executor import BedrockExecutor, ModelConfig
from evaluation.evaluator import Evaluator
from orchestration.engine import LLMOrchestrationEngine, OrchestrationResult, ConvergenceAnalysis
from agents.base import AgentResult


class TestSessionManager:
    """Test cases for SessionManager functionality."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        # Create temporary directory for test storage
        self.temp_dir = tempfile.mkdtemp()
        
        # Create mock dependencies
        self.mock_bedrock_executor = Mock(spec=BedrockExecutor)
        self.mock_evaluator = Mock(spec=Evaluator)
        self.mock_history_manager = Mock(spec=HistoryManager)
        
        # Create SessionManager instance
        self.session_manager = SessionManager(
            bedrock_executor=self.mock_bedrock_executor,
            evaluator=self.mock_evaluator,
            history_manager=self.mock_history_manager
        )
        
        # Mock orchestration engine
        self.session_manager.orchestration_engine = Mock(spec=LLMOrchestrationEngine)
    
    def teardown_method(self):
        """Clean up after each test method."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_create_session_success(self):
        """Test successful session creation."""
        # Setup
        initial_prompt = "Test prompt for optimization"
        context = {"intended_use": "testing", "domain": "general"}
        
        # Mock history manager response
        self.mock_history_manager.create_session.return_value = "test-session-id"
        
        # Execute
        result = self.session_manager.create_session(
            initial_prompt=initial_prompt,
            context=context
        )
        
        # Verify
        assert result.success is True
        assert result.session_state is not None
        assert result.session_state.initial_prompt == initial_prompt
        assert result.session_state.current_prompt == initial_prompt
        assert result.session_state.context == context
        assert result.session_state.status == 'active'
        assert result.session_state.current_iteration == 0
        assert "created successfully" in result.message
        
        # Verify history manager was called
        self.mock_history_manager.create_session.assert_called_once()
        
        # Verify session is in active sessions
        assert result.session_state.session_id in self.session_manager.active_sessions
    
    def test_create_session_with_custom_config(self):
        """Test session creation with custom configuration."""
        # Setup
        initial_prompt = "Test prompt"
        custom_config = SessionConfig(
            max_iterations=5,
            min_iterations=2,
            convergence_threshold=0.05,
            auto_finalize_on_convergence=True
        )
        
        self.mock_history_manager.create_session.return_value = "test-session-id"
        
        # Execute
        result = self.session_manager.create_session(
            initial_prompt=initial_prompt,
            config=custom_config
        )
        
        # Verify
        assert result.success is True
        assert result.session_state.config.max_iterations == 5
        assert result.session_state.config.min_iterations == 2
        assert result.session_state.config.convergence_threshold == 0.05
        assert result.session_state.config.auto_finalize_on_convergence is True
    
    def test_create_session_failure(self):
        """Test session creation failure handling."""
        # Setup
        self.mock_history_manager.create_session.side_effect = Exception("Storage error")
        
        # Execute
        result = self.session_manager.create_session("Test prompt")
        
        # Verify
        assert result.success is False
        assert result.session_state is None
        assert "Failed to create session" in result.message
    
    def test_run_optimization_iteration_success(self):
        """Test successful optimization iteration."""
        # Setup
        session_id = self._create_test_session()
        
        # Mock orchestration result
        mock_orchestration_result = self._create_mock_orchestration_result(success=True)
        self.session_manager.orchestration_engine.run_llm_orchestrated_iteration.return_value = mock_orchestration_result
        
        # Mock history operations
        self.mock_history_manager.load_session_history.return_value = []
        self.mock_history_manager.save_iteration.return_value = True
        
        # Execute
        result = self.session_manager.run_optimization_iteration(session_id)
        
        # Verify
        assert result.success is True
        assert result.session_state is not None
        assert result.session_state.current_iteration == 1
        assert result.session_state.current_prompt == "Optimized prompt text"
        assert result.iteration_result == mock_orchestration_result
        assert "completed successfully" in result.message
        
        # Verify orchestration engine was called
        self.session_manager.orchestration_engine.run_llm_orchestrated_iteration.assert_called_once()
        
        # Verify history operations
        self.mock_history_manager.load_session_history.assert_called_once_with(session_id)
        self.mock_history_manager.save_iteration.assert_called_once()
    
    def test_run_optimization_iteration_with_feedback(self):
        """Test optimization iteration with user feedback."""
        # Setup
        session_id = self._create_test_session()
        user_feedback = UserFeedback(
            satisfaction_rating=3,
            specific_issues=["Too verbose", "Unclear instructions"],
            desired_improvements="Make it more concise",
            continue_optimization=True
        )
        
        mock_orchestration_result = self._create_mock_orchestration_result(success=True)
        self.session_manager.orchestration_engine.run_llm_orchestrated_iteration.return_value = mock_orchestration_result
        
        self.mock_history_manager.load_session_history.return_value = []
        self.mock_history_manager.save_iteration.return_value = True
        
        # Execute
        result = self.session_manager.run_optimization_iteration(session_id, user_feedback)
        
        # Verify
        assert result.success is True
        
        # Verify feedback was passed to orchestration
        call_args = self.session_manager.orchestration_engine.run_llm_orchestrated_iteration.call_args
        assert call_args[1]['feedback'] == user_feedback
    
    def test_run_optimization_iteration_session_not_found(self):
        """Test iteration with non-existent session."""
        # Execute
        result = self.session_manager.run_optimization_iteration("non-existent-session")
        
        # Verify
        assert result.success is False
        assert "not found" in result.message
    
    def test_run_optimization_iteration_max_iterations_reached(self):
        """Test iteration when maximum iterations are reached."""
        # Setup
        session_id = self._create_test_session()
        session_state = self.session_manager.active_sessions[session_id]
        session_state.current_iteration = session_state.config.max_iterations
        
        # Execute
        result = self.session_manager.run_optimization_iteration(session_id)
        
        # Verify
        assert result.success is True
        assert session_state.status == 'finalized'
        assert "Maximum iterations reached" in result.message
        assert "Export final prompt" in result.suggested_actions
    
    def test_run_optimization_iteration_orchestration_failure(self):
        """Test iteration with orchestration failure."""
        # Setup
        session_id = self._create_test_session()
        
        mock_orchestration_result = self._create_mock_orchestration_result(
            success=False, 
            error_message="Orchestration failed"
        )
        self.session_manager.orchestration_engine.run_llm_orchestrated_iteration.return_value = mock_orchestration_result
        
        self.mock_history_manager.load_session_history.return_value = []
        
        # Execute
        result = self.session_manager.run_optimization_iteration(session_id)
        
        # Verify
        assert result.success is False
        assert result.session_state.status == 'error'
        assert "Iteration failed" in result.message
        assert result.session_state.error_message == "Orchestration failed"
    
    def test_run_optimization_iteration_save_failure(self):
        """Test iteration with history save failure."""
        # Setup
        session_id = self._create_test_session()
        
        mock_orchestration_result = self._create_mock_orchestration_result(success=True)
        self.session_manager.orchestration_engine.run_llm_orchestrated_iteration.return_value = mock_orchestration_result
        
        self.mock_history_manager.load_session_history.return_value = []
        self.mock_history_manager.save_iteration.return_value = False
        
        # Execute
        result = self.session_manager.run_optimization_iteration(session_id)
        
        # Verify
        assert result.success is False
        assert "Failed to save iteration" in result.message
    
    def test_collect_user_feedback_success(self):
        """Test successful user feedback collection."""
        # Setup
        session_id = self._create_test_session()
        
        # Mock latest iteration
        mock_iteration = Mock(spec=PromptIteration)
        self.mock_history_manager.get_latest_iteration.return_value = mock_iteration
        self.mock_history_manager.save_iteration.return_value = True
        
        # Execute
        result = self.session_manager.collect_user_feedback(
            session_id=session_id,
            satisfaction_rating=4,
            specific_issues=["Minor formatting issue"],
            desired_improvements="Better formatting",
            continue_optimization=True
        )
        
        # Verify
        assert result.success is True
        assert "feedback collected successfully" in result.message
        assert result.requires_user_input is False
        
        # Verify feedback was saved to iteration
        self.mock_history_manager.save_iteration.assert_called_once_with(mock_iteration)
        assert mock_iteration.user_feedback is not None
        assert mock_iteration.user_feedback.satisfaction_rating == 4
    
    def test_collect_user_feedback_high_satisfaction(self):
        """Test feedback collection with high satisfaction rating."""
        # Setup
        session_id = self._create_test_session()
        self.mock_history_manager.get_latest_iteration.return_value = Mock()
        self.mock_history_manager.save_iteration.return_value = True
        
        # Execute
        result = self.session_manager.collect_user_feedback(
            session_id=session_id,
            satisfaction_rating=5,
            continue_optimization=True
        )
        
        # Verify
        assert result.success is True
        session_state = result.session_state
        assert session_state.convergence_detected is True
        assert "High user satisfaction" in session_state.convergence_reason
        assert any("finalizing" in action.lower() for action in result.suggested_actions)
    
    def test_collect_user_feedback_stop_optimization(self):
        """Test feedback collection when user wants to stop optimization."""
        # Setup
        session_id = self._create_test_session()
        self.mock_history_manager.get_latest_iteration.return_value = Mock()
        self.mock_history_manager.save_iteration.return_value = True
        
        # Execute
        result = self.session_manager.collect_user_feedback(
            session_id=session_id,
            satisfaction_rating=4,
            continue_optimization=False
        )
        
        # Verify
        assert result.success is True
        session_state = result.session_state
        assert session_state.status == 'finalized'
        assert "User requested finalization" in session_state.convergence_reason
        assert "Finalize session" in result.suggested_actions
    
    def test_collect_user_feedback_invalid_rating(self):
        """Test feedback collection with invalid satisfaction rating."""
        # Setup
        session_id = self._create_test_session()
        
        # Execute
        result = self.session_manager.collect_user_feedback(
            session_id=session_id,
            satisfaction_rating=6  # Invalid rating (should be 1-5)
        )
        
        # Verify
        assert result.success is False
        assert "Invalid user feedback" in result.message
    
    def test_finalize_session_success(self):
        """Test successful session finalization."""
        # Setup
        session_id = self._create_test_session()
        self.mock_history_manager.finalize_session.return_value = True
        
        # Execute
        result = self.session_manager.finalize_session(session_id)
        
        # Verify
        assert result.success is True
        assert "finalized successfully" in result.message
        
        # Verify session state updated
        assert session_id not in self.session_manager.active_sessions
        
        # Verify history manager called
        self.mock_history_manager.finalize_session.assert_called_once()
    
    def test_finalize_session_history_failure(self):
        """Test session finalization with history manager failure."""
        # Setup
        session_id = self._create_test_session()
        self.mock_history_manager.finalize_session.return_value = False
        
        # Execute
        result = self.session_manager.finalize_session(session_id)
        
        # Verify
        assert result.success is False
        assert "Failed to finalize session in history" in result.message
    
    def test_pause_and_resume_session(self):
        """Test pausing and resuming a session."""
        # Setup
        session_id = self._create_test_session()
        
        # Test pause
        pause_result = self.session_manager.pause_session(session_id)
        assert pause_result.success is True
        assert pause_result.session_state.status == 'paused'
        assert "paused successfully" in pause_result.message
        
        # Test resume
        resume_result = self.session_manager.resume_session(session_id)
        assert resume_result.success is True
        assert resume_result.session_state.status == 'active'
        assert "resumed successfully" in resume_result.message
    
    def test_pause_session_invalid_status(self):
        """Test pausing a session with invalid status."""
        # Setup
        session_id = self._create_test_session()
        session_state = self.session_manager.active_sessions[session_id]
        session_state.status = 'finalized'
        
        # Execute
        result = self.session_manager.pause_session(session_id)
        
        # Verify
        assert result.success is False
        assert "Cannot pause session with status" in result.message
    
    def test_get_session_state(self):
        """Test getting session state."""
        # Setup
        session_id = self._create_test_session()
        
        # Execute
        session_state = self.session_manager.get_session_state(session_id)
        
        # Verify
        assert session_state is not None
        assert session_state.session_id == session_id
        assert session_state.status == 'active'
    
    def test_list_active_sessions(self):
        """Test listing active sessions."""
        # Setup
        session_id1 = self._create_test_session()
        session_id2 = self._create_test_session()
        
        # Execute
        active_sessions = self.session_manager.list_active_sessions()
        
        # Verify
        assert len(active_sessions) == 2
        session_ids = [session.session_id for session in active_sessions]
        assert session_id1 in session_ids
        assert session_id2 in session_ids
    
    def test_export_session_with_reasoning(self):
        """Test exporting session with orchestration reasoning."""
        # Setup
        session_id = self._create_test_session()
        export_path = f"{self.temp_dir}/export.json"
        
        # Mock history manager export
        self.mock_history_manager.export_session.return_value = True
        
        # Add orchestration history
        self.session_manager.orchestration_engine.orchestration_history = [
            self._create_mock_orchestration_result(success=True)
        ]
        
        # Mock file operations
        with patch('builtins.open', create=True) as mock_open:
            mock_file = MagicMock()
            mock_open.return_value.__enter__.return_value = mock_file
            mock_file.read.return_value = '{"session_info": {}, "iterations": []}'
            
            # Execute
            result = self.session_manager.export_session_with_reasoning(
                session_id=session_id,
                export_path=export_path,
                include_orchestration_details=True
            )
        
        # Verify
        assert result.success is True
        assert "exported successfully" in result.message
        
        # Verify history manager was called
        self.mock_history_manager.export_session.assert_called_once_with(session_id, export_path)
    
    def test_convergence_detection_with_orchestration(self):
        """Test convergence detection using orchestration analysis."""
        # Setup
        session_id = self._create_test_session()
        
        # Create convergence analysis
        convergence_analysis = ConvergenceAnalysis(
            has_converged=True,
            convergence_score=0.9,
            convergence_reasons=["Stable evaluation scores", "High confidence"],
            improvement_trend="stable",
            iterations_analyzed=3,
            confidence=0.85,
            llm_reasoning="The prompt has reached optimal quality based on evaluation metrics."
        )
        
        mock_orchestration_result = self._create_mock_orchestration_result(
            success=True,
            convergence_analysis=convergence_analysis
        )
        
        self.session_manager.orchestration_engine.run_llm_orchestrated_iteration.return_value = mock_orchestration_result
        self.mock_history_manager.load_session_history.return_value = []
        self.mock_history_manager.save_iteration.return_value = True
        
        # Execute
        result = self.session_manager.run_optimization_iteration(session_id)
        
        # Verify
        assert result.success is True
        session_state = result.session_state
        assert session_state.convergence_detected is True
        assert "LLM orchestration detected convergence" in session_state.convergence_reason
        assert any("finalizing" in action.lower() for action in result.suggested_actions)
    
    def test_orchestration_summary_update(self):
        """Test orchestration summary updates in session state."""
        # Setup
        session_id = self._create_test_session()
        
        mock_orchestration_result = self._create_mock_orchestration_result(success=True)
        mock_orchestration_result.llm_orchestrator_confidence = 0.8
        mock_orchestration_result.processing_time = 2.5
        mock_orchestration_result.conflict_resolutions = [{"test": "conflict"}]
        
        self.session_manager.orchestration_engine.run_llm_orchestrated_iteration.return_value = mock_orchestration_result
        self.mock_history_manager.load_session_history.return_value = []
        self.mock_history_manager.save_iteration.return_value = True
        
        # Execute
        result = self.session_manager.run_optimization_iteration(session_id)
        
        # Verify
        assert result.success is True
        session_state = result.session_state
        
        # Check orchestration summary
        assert 'iterations' in session_state.orchestration_summary
        assert len(session_state.orchestration_summary['iterations']) == 1
        
        iteration_summary = session_state.orchestration_summary['iterations'][0]
        assert iteration_summary['iteration'] == 1
        assert iteration_summary['success'] is True
        assert iteration_summary['confidence'] == 0.8
        assert iteration_summary['processing_time'] == 2.5
        assert iteration_summary['conflicts_resolved'] == 1
        
        # Check overall statistics
        assert session_state.orchestration_summary['total_iterations'] == 1
        assert session_state.orchestration_summary['average_confidence'] == 0.8
        assert session_state.orchestration_summary['total_processing_time'] == 2.5
    
    def test_session_config_serialization(self):
        """Test SessionConfig serialization and deserialization."""
        # Setup
        model_config = ModelConfig(
            model_id="test-model",
            temperature=0.7,
            max_tokens=1000
        )
        
        config = SessionConfig(
            max_iterations=5,
            min_iterations=2,
            convergence_threshold=0.05,
            auto_finalize_on_convergence=True,
            model_config=model_config,
            orchestration_config={"test": "value"}
        )
        
        # Execute
        config_dict = config.to_dict()
        
        # Verify
        assert config_dict['max_iterations'] == 5
        assert config_dict['min_iterations'] == 2
        assert config_dict['convergence_threshold'] == 0.05
        assert config_dict['auto_finalize_on_convergence'] is True
        assert config_dict['model_config'] is not None
        assert config_dict['orchestration_config'] == {"test": "value"}
    
    def test_session_state_serialization(self):
        """Test SessionState serialization."""
        # Setup
        session_id = self._create_test_session()
        session_state = self.session_manager.active_sessions[session_id]
        
        # Execute
        state_dict = session_state.to_dict()
        
        # Verify
        assert state_dict['session_id'] == session_id
        assert state_dict['status'] == 'active'
        assert state_dict['current_iteration'] == 0
        assert 'created_at' in state_dict
        assert 'last_updated' in state_dict
        assert isinstance(state_dict['created_at'], str)  # Should be ISO format
        assert isinstance(state_dict['last_updated'], str)
    
    def _create_test_session(self) -> str:
        """Helper method to create a test session."""
        self.mock_history_manager.create_session.return_value = f"test-session-{len(self.session_manager.active_sessions)}"
        
        result = self.session_manager.create_session(
            initial_prompt="Test prompt for optimization",
            context={"intended_use": "testing"}
        )
        
        assert result.success is True
        return result.session_state.session_id
    
    def _create_mock_orchestration_result(self,
                                        success: bool = True,
                                        error_message: str = None,
                                        convergence_analysis: ConvergenceAnalysis = None) -> OrchestrationResult:
        """Helper method to create mock orchestration results."""
        
        # Create mock agent results
        agent_results = {
            'analyzer': AgentResult(
                agent_name='analyzer',
                success=True,
                analysis={'structure_score': 0.8, 'clarity_issues': ['Minor formatting']},
                suggestions=['Improve clarity', 'Add examples'],
                confidence_score=0.8
            ),
            'refiner': AgentResult(
                agent_name='refiner',
                success=True,
                analysis={'refined_prompt': 'Optimized prompt text', 'improvements': ['Better structure']},
                suggestions=['Refined version available'],
                confidence_score=0.85
            ),
            'validator': AgentResult(
                agent_name='validator',
                success=True,
                analysis={'validation_passed': True, 'issues_found': []},
                suggestions=['Validation passed'],
                confidence_score=0.9
            )
        }
        
        # Create mock execution and evaluation results
        execution_result = ExecutionResult(
            model_name="test-model",
            response_text="Test response",
            execution_time=1.5,
            token_usage={"input": 100, "output": 50},
            success=True
        ) if success else None
        
        evaluation_result = EvaluationResult(
            overall_score=0.8,
            relevance_score=0.85,
            clarity_score=0.75,
            completeness_score=0.8,
            qualitative_feedback="Good quality response"
        ) if success else None
        
        return OrchestrationResult(
            success=success,
            orchestrated_prompt="Optimized prompt text" if success else "Original prompt",
            agent_results=agent_results,
            execution_result=execution_result,
            evaluation_result=evaluation_result,
            conflict_resolutions=[],
            synthesis_reasoning="Synthesized agent recommendations successfully",
            orchestration_decisions=["Used analyzer feedback", "Applied refiner suggestions"],
            convergence_analysis=convergence_analysis,
            processing_time=2.0,
            llm_orchestrator_confidence=0.8,
            error_message=error_message
        )


if __name__ == "__main__":
    pytest.main([__file__])
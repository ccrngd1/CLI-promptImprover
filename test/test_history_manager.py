"""
Unit tests for the HistoryManager class.

Tests cover all storage operations including session management, iteration persistence,
and error handling scenarios using temporary directories.
"""

import json
import os
import tempfile
import shutil
import unittest
from datetime import datetime
from pathlib import Path

from storage.history import HistoryManager
from models import PromptIteration, ExecutionResult, EvaluationResult, UserFeedback


class TestHistoryManager(unittest.TestCase):
    """Test cases for HistoryManager functionality."""
    
    def setUp(self):
        """Set up test environment with temporary directory."""
        self.temp_dir = tempfile.mkdtemp()
        self.history_manager = HistoryManager(storage_dir=self.temp_dir)
    
    def tearDown(self):
        """Clean up temporary directory after tests."""
        shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Test HistoryManager initialization and directory creation."""
        # Check that directories are created
        self.assertTrue(Path(self.temp_dir).exists())
        self.assertTrue((Path(self.temp_dir) / "sessions").exists())
        self.assertTrue((Path(self.temp_dir) / "iterations").exists())
    
    def test_create_session(self):
        """Test session creation with unique identifiers."""
        initial_prompt = "Test prompt for optimization"
        context = "Testing context"
        
        session_id = self.history_manager.create_session(initial_prompt, context)
        
        # Verify session ID is generated
        self.assertIsInstance(session_id, str)
        self.assertTrue(len(session_id) > 0)
        
        # Verify session file is created
        session_file = Path(self.temp_dir) / "sessions" / f"{session_id}.json"
        self.assertTrue(session_file.exists())
        
        # Verify session data
        session_info = self.history_manager.get_session_info(session_id)
        self.assertIsNotNone(session_info)
        self.assertEqual(session_info['initial_prompt'], initial_prompt)
        self.assertEqual(session_info['context'], context)
        self.assertEqual(session_info['status'], 'active')
        self.assertEqual(session_info['iteration_count'], 0)
    
    def test_create_session_without_context(self):
        """Test session creation without context."""
        initial_prompt = "Test prompt without context"
        
        session_id = self.history_manager.create_session(initial_prompt)
        session_info = self.history_manager.get_session_info(session_id)
        
        self.assertEqual(session_info['context'], "")
    
    def test_save_and_load_iteration(self):
        """Test saving and loading prompt iterations."""
        # Create a session first
        session_id = self.history_manager.create_session("Initial prompt")
        
        # Create test iteration
        execution_result = ExecutionResult(
            model_name="claude-3-sonnet",
            response_text="Test response",
            execution_time=1.5,
            token_usage={"input": 10, "output": 20},
            success=True
        )
        
        evaluation_result = EvaluationResult(
            overall_score=0.8,
            relevance_score=0.9,
            clarity_score=0.7,
            completeness_score=0.8
        )
        
        user_feedback = UserFeedback(
            satisfaction_rating=4,
            specific_issues=["Could be clearer"],
            desired_improvements="More specific examples"
        )
        
        iteration = PromptIteration(
            session_id=session_id,
            version=1,
            prompt_text="Optimized prompt version 1",
            execution_result=execution_result,
            evaluation_scores=evaluation_result,
            user_feedback=user_feedback
        )
        
        # Save iteration
        success = self.history_manager.save_iteration(iteration)
        self.assertTrue(success)
        
        # Load iteration
        loaded_iteration = self.history_manager.load_iteration(session_id, 1)
        self.assertIsNotNone(loaded_iteration)
        self.assertEqual(loaded_iteration.session_id, session_id)
        self.assertEqual(loaded_iteration.version, 1)
        self.assertEqual(loaded_iteration.prompt_text, "Optimized prompt version 1")
        
        # Verify nested objects
        self.assertIsNotNone(loaded_iteration.execution_result)
        self.assertEqual(loaded_iteration.execution_result.model_name, "claude-3-sonnet")
        self.assertIsNotNone(loaded_iteration.evaluation_scores)
        self.assertEqual(loaded_iteration.evaluation_scores.overall_score, 0.8)
        self.assertIsNotNone(loaded_iteration.user_feedback)
        self.assertEqual(loaded_iteration.user_feedback.satisfaction_rating, 4)
    
    def test_save_invalid_iteration(self):
        """Test saving invalid iteration returns False."""
        # Create iteration with invalid data
        iteration = PromptIteration(
            session_id="",  # Invalid empty session_id
            version=0,      # Invalid version (must be >= 1)
            prompt_text=""  # Invalid empty prompt
        )
        
        success = self.history_manager.save_iteration(iteration)
        self.assertFalse(success)
    
    def test_load_nonexistent_iteration(self):
        """Test loading non-existent iteration returns None."""
        result = self.history_manager.load_iteration("nonexistent", 1)
        self.assertIsNone(result)
    
    def test_load_session_history(self):
        """Test loading complete session history."""
        session_id = self.history_manager.create_session("Initial prompt")
        
        # Create multiple iterations
        iterations = []
        for i in range(1, 4):
            iteration = PromptIteration(
                session_id=session_id,
                version=i,
                prompt_text=f"Prompt version {i}",
                execution_result=ExecutionResult(
                    model_name="test-model",
                    response_text=f"Response {i}",
                    execution_time=1.0,
                    token_usage={"input": 10, "output": 20},
                    success=True
                )
            )
            iterations.append(iteration)
            self.history_manager.save_iteration(iteration)
        
        # Load session history
        loaded_history = self.history_manager.load_session_history(session_id)
        
        self.assertEqual(len(loaded_history), 3)
        # Verify they're sorted by version
        for i, iteration in enumerate(loaded_history):
            self.assertEqual(iteration.version, i + 1)
            self.assertEqual(iteration.prompt_text, f"Prompt version {i + 1}")
    
    def test_load_empty_session_history(self):
        """Test loading history for session with no iterations."""
        session_id = self.history_manager.create_session("Initial prompt")
        history = self.history_manager.load_session_history(session_id)
        self.assertEqual(len(history), 0)
    
    def test_get_session_info(self):
        """Test retrieving session information."""
        initial_prompt = "Test prompt"
        context = "Test context"
        session_id = self.history_manager.create_session(initial_prompt, context)
        
        session_info = self.history_manager.get_session_info(session_id)
        
        self.assertIsNotNone(session_info)
        self.assertEqual(session_info['id'], session_id)
        self.assertEqual(session_info['initial_prompt'], initial_prompt)
        self.assertEqual(session_info['context'], context)
        self.assertEqual(session_info['status'], 'active')
    
    def test_get_nonexistent_session_info(self):
        """Test retrieving info for non-existent session."""
        result = self.history_manager.get_session_info("nonexistent")
        self.assertIsNone(result)
    
    def test_list_sessions(self):
        """Test listing all sessions."""
        # Create multiple sessions
        session_ids = []
        for i in range(3):
            session_id = self.history_manager.create_session(f"Prompt {i}")
            session_ids.append(session_id)
        
        sessions = self.history_manager.list_sessions()
        
        self.assertEqual(len(sessions), 3)
        # Verify all session IDs are present
        listed_ids = [session['id'] for session in sessions]
        for session_id in session_ids:
            self.assertIn(session_id, listed_ids)
    
    def test_finalize_session(self):
        """Test finalizing a session."""
        session_id = self.history_manager.create_session("Initial prompt")
        final_prompt = "Final optimized prompt"
        
        success = self.history_manager.finalize_session(session_id, final_prompt)
        self.assertTrue(success)
        
        session_info = self.history_manager.get_session_info(session_id)
        self.assertEqual(session_info['status'], 'finalized')
        self.assertEqual(session_info['final_prompt'], final_prompt)
        self.assertIsNotNone(session_info['finalized_at'])
    
    def test_finalize_nonexistent_session(self):
        """Test finalizing non-existent session returns False."""
        success = self.history_manager.finalize_session("nonexistent", "final prompt")
        self.assertFalse(success)
    
    def test_delete_session(self):
        """Test deleting a session and its iterations."""
        session_id = self.history_manager.create_session("Test prompt")
        
        # Add some iterations
        for i in range(1, 3):
            iteration = PromptIteration(
                session_id=session_id,
                version=i,
                prompt_text=f"Prompt {i}",
                execution_result=ExecutionResult(
                    model_name="test-model",
                    response_text=f"Response {i}",
                    execution_time=1.0,
                    token_usage={"input": 10, "output": 20},
                    success=True
                )
            )
            self.history_manager.save_iteration(iteration)
        
        # Verify session exists
        self.assertIsNotNone(self.history_manager.get_session_info(session_id))
        self.assertEqual(len(self.history_manager.load_session_history(session_id)), 2)
        
        # Delete session
        success = self.history_manager.delete_session(session_id)
        self.assertTrue(success)
        
        # Verify session is deleted
        self.assertIsNone(self.history_manager.get_session_info(session_id))
        self.assertEqual(len(self.history_manager.load_session_history(session_id)), 0)
    
    def test_export_session(self):
        """Test exporting a complete session."""
        session_id = self.history_manager.create_session("Test prompt", "Test context")
        
        # Add iteration
        iteration = PromptIteration(
            session_id=session_id,
            version=1,
            prompt_text="Test prompt",
            execution_result=ExecutionResult(
                model_name="test-model",
                response_text="Test response",
                execution_time=1.0,
                token_usage={"input": 10, "output": 20},
                success=True
            )
        )
        self.history_manager.save_iteration(iteration)
        
        # Export session
        export_path = os.path.join(self.temp_dir, "exported_session.json")
        success = self.history_manager.export_session(session_id, export_path)
        self.assertTrue(success)
        
        # Verify export file
        self.assertTrue(os.path.exists(export_path))
        
        with open(export_path, 'r', encoding='utf-8') as f:
            export_data = json.load(f)
        
        self.assertIn('session_info', export_data)
        self.assertIn('iterations', export_data)
        self.assertEqual(export_data['session_info']['id'], session_id)
        self.assertEqual(len(export_data['iterations']), 1)
    
    def test_export_nonexistent_session(self):
        """Test exporting non-existent session returns False."""
        export_path = os.path.join(self.temp_dir, "nonexistent.json")
        success = self.history_manager.export_session("nonexistent", export_path)
        self.assertFalse(success)
    
    def test_get_latest_iteration(self):
        """Test getting the latest iteration for a session."""
        session_id = self.history_manager.create_session("Test prompt")
        
        # Initially no iterations
        latest = self.history_manager.get_latest_iteration(session_id)
        self.assertIsNone(latest)
        
        # Add iterations
        for i in range(1, 4):
            iteration = PromptIteration(
                session_id=session_id,
                version=i,
                prompt_text=f"Prompt version {i}",
                execution_result=ExecutionResult(
                    model_name="test-model",
                    response_text=f"Response {i}",
                    execution_time=1.0,
                    token_usage={"input": 10, "output": 20},
                    success=True
                )
            )
            self.history_manager.save_iteration(iteration)
        
        # Get latest iteration
        latest = self.history_manager.get_latest_iteration(session_id)
        self.assertIsNotNone(latest)
        self.assertEqual(latest.version, 3)
        self.assertEqual(latest.prompt_text, "Prompt version 3")
    
    def test_iteration_count_update(self):
        """Test that session iteration count is updated correctly."""
        session_id = self.history_manager.create_session("Test prompt")
        
        # Add iterations
        for i in range(1, 4):
            iteration = PromptIteration(
                session_id=session_id,
                version=i,
                prompt_text=f"Prompt version {i}",
                execution_result=ExecutionResult(
                    model_name="test-model",
                    response_text=f"Response {i}",
                    execution_time=1.0,
                    token_usage={"input": 10, "output": 20},
                    success=True
                )
            )
            self.history_manager.save_iteration(iteration)
        
        # Check iteration count in session info
        session_info = self.history_manager.get_session_info(session_id)
        self.assertEqual(session_info['iteration_count'], 3)


if __name__ == '__main__':
    unittest.main()
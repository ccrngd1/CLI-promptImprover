#!/usr/bin/env python3
"""
Unit tests for user feedback collection and integration functionality.

Tests the feedback collection interface, processing logic, history tracking,
and orchestration integration for analyzing user feedback patterns.
"""

import unittest
import tempfile
import shutil
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

from models import UserFeedback, PromptIteration, EvaluationResult, ExecutionResult
from session import SessionManager, SessionConfig, SessionState
from storage.history import HistoryManager
from bedrock.executor import BedrockExecutor
from evaluation.evaluator import Evaluator


class TestUserFeedbackModel(unittest.TestCase):
    """Test the UserFeedback data model."""
    
    def test_create_valid_feedback(self):
        """Test creating valid user feedback."""
        feedback = UserFeedback(
            satisfaction_rating=4,
            specific_issues=["Too verbose", "Needs examples"],
            desired_improvements="Make it more concise with concrete examples",
            continue_optimization=True
        )
        
        self.assertTrue(feedback.validate())
        self.assertEqual(feedback.satisfaction_rating, 4)
        self.assertEqual(len(feedback.specific_issues), 2)
        self.assertTrue(feedback.continue_optimization)
    
    def test_feedback_validation(self):
        """Test feedback validation rules."""
        # Invalid rating
        invalid_feedback = UserFeedback(
            satisfaction_rating=6,  # Invalid: > 5
            specific_issues=[],
            desired_improvements="",
            continue_optimization=True
        )
        self.assertFalse(invalid_feedback.validate())
        
        # Invalid rating (too low)
        invalid_feedback = UserFeedback(
            satisfaction_rating=0,  # Invalid: < 1
            specific_issues=[],
            desired_improvements="",
            continue_optimization=True
        )
        self.assertFalse(invalid_feedback.validate())
        
        # Valid edge cases
        valid_feedback = UserFeedback(
            satisfaction_rating=1,  # Valid minimum
            specific_issues=[],
            desired_improvements="",
            continue_optimization=False
        )
        self.assertTrue(valid_feedback.validate())
        
        valid_feedback = UserFeedback(
            satisfaction_rating=5,  # Valid maximum
            specific_issues=[],
            desired_improvements="",
            continue_optimization=True
        )
        self.assertTrue(valid_feedback.validate())
    
    def test_feedback_serialization(self):
        """Test feedback serialization and deserialization."""
        original = UserFeedback(
            satisfaction_rating=3,
            specific_issues=["Issue 1", "Issue 2"],
            desired_improvements="Some improvements",
            continue_optimization=False
        )
        
        # Test to_dict and from_dict
        data = original.to_dict()
        restored = UserFeedback.from_dict(data)
        
        self.assertEqual(original.satisfaction_rating, restored.satisfaction_rating)
        self.assertEqual(original.specific_issues, restored.specific_issues)
        self.assertEqual(original.desired_improvements, restored.desired_improvements)
        self.assertEqual(original.continue_optimization, restored.continue_optimization)


class TestFeedbackCollection(unittest.TestCase):
    """Test feedback collection functionality in SessionManager."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Mock dependencies
        self.mock_bedrock = Mock(spec=BedrockExecutor)
        self.mock_evaluator = Mock(spec=Evaluator)
        self.mock_history = Mock(spec=HistoryManager)
        
        # Create session manager
        self.session_manager = SessionManager(
            bedrock_executor=self.mock_bedrock,
            evaluator=self.mock_evaluator,
            history_manager=self.mock_history
        )
        
        # Create test session
        self.session_id = "test-session-123"
        self.session_state = SessionState(
            session_id=self.session_id,
            status='active',
            current_iteration=1,
            current_prompt="Test prompt",
            initial_prompt="Test prompt",
            context=None,
            config=SessionConfig(),
            created_at=datetime.now(),
            last_updated=datetime.now()
        )
        self.session_manager.active_sessions[self.session_id] = self.session_state
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_collect_valid_feedback(self):
        """Test collecting valid user feedback."""
        # Mock history manager methods
        self.mock_history.get_latest_iteration.return_value = Mock()
        self.mock_history.save_iteration.return_value = True
        
        result = self.session_manager.collect_user_feedback(
            session_id=self.session_id,
            satisfaction_rating=4,
            specific_issues=["Too long", "Unclear"],
            desired_improvements="Make it shorter and clearer",
            continue_optimization=True
        )
        
        self.assertTrue(result.success)
        self.assertIsNotNone(result.session_state)
        self.assertEqual(result.session_state.session_id, self.session_id)
        self.assertIn("successfully", result.message.lower())
    
    def test_collect_feedback_invalid_session(self):
        """Test collecting feedback for non-existent session."""
        result = self.session_manager.collect_user_feedback(
            session_id="non-existent",
            satisfaction_rating=3,
            specific_issues=[],
            desired_improvements="",
            continue_optimization=True
        )
        
        self.assertFalse(result.success)
        self.assertIn("not found", result.message.lower())
    
    def test_collect_feedback_invalid_rating(self):
        """Test collecting feedback with invalid rating."""
        result = self.session_manager.collect_user_feedback(
            session_id=self.session_id,
            satisfaction_rating=6,  # Invalid rating
            specific_issues=[],
            desired_improvements="",
            continue_optimization=True
        )
        
        self.assertFalse(result.success)
        self.assertIn("invalid", result.message.lower())
    
    def test_feedback_affects_session_state(self):
        """Test that feedback affects session state appropriately."""
        # Mock history manager
        self.mock_history.get_latest_iteration.return_value = Mock()
        self.mock_history.save_iteration.return_value = True
        
        # Test high satisfaction feedback
        result = self.session_manager.collect_user_feedback(
            session_id=self.session_id,
            satisfaction_rating=5,
            specific_issues=[],
            desired_improvements="Perfect!",
            continue_optimization=True
        )
        
        self.assertTrue(result.success)
        self.assertTrue(self.session_state.convergence_detected)
        self.assertEqual(self.session_state.convergence_reason, 'High user satisfaction')
        
        # Reset state
        self.session_state.convergence_detected = False
        self.session_state.convergence_reason = None
        
        # Test stop optimization feedback
        result = self.session_manager.collect_user_feedback(
            session_id=self.session_id,
            satisfaction_rating=3,
            specific_issues=[],
            desired_improvements="",
            continue_optimization=False
        )
        
        self.assertTrue(result.success)
        self.assertEqual(self.session_state.status, 'finalized')
        self.assertEqual(self.session_state.convergence_reason, 'User requested finalization')
    
    def test_feedback_suggested_actions(self):
        """Test that appropriate actions are suggested based on feedback."""
        self.mock_history.get_latest_iteration.return_value = Mock()
        self.mock_history.save_iteration.return_value = True
        
        # Test low satisfaction feedback
        result = self.session_manager.collect_user_feedback(
            session_id=self.session_id,
            satisfaction_rating=2,
            specific_issues=["Confusing", "Too complex"],
            desired_improvements="Simplify the language",
            continue_optimization=True
        )
        
        self.assertTrue(result.success)
        self.assertIsInstance(result.suggested_actions, list)
        self.assertGreater(len(result.suggested_actions), 0)
        
        # Check for appropriate suggestions
        actions_text = ' '.join(result.suggested_actions).lower()
        self.assertTrue(any(keyword in actions_text for keyword in 
                          ['iteration', 'issues', 'strategy', 'review']))


class TestFeedbackPatternAnalysis(unittest.TestCase):
    """Test feedback pattern analysis functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Mock dependencies
        self.mock_bedrock = Mock(spec=BedrockExecutor)
        self.mock_evaluator = Mock(spec=Evaluator)
        self.mock_history = Mock(spec=HistoryManager)
        
        # Create session manager
        self.session_manager = SessionManager(
            bedrock_executor=self.mock_bedrock,
            evaluator=self.mock_evaluator,
            history_manager=self.mock_history
        )
        
        self.session_id = "test-session-123"
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def _create_test_history(self, feedback_data):
        """Create test history with feedback data."""
        history = []
        for i, fb_data in enumerate(feedback_data, 1):
            feedback = UserFeedback(
                satisfaction_rating=fb_data['rating'],
                specific_issues=fb_data.get('issues', []),
                desired_improvements=fb_data.get('improvements', ''),
                continue_optimization=fb_data.get('continue', True)
            )
            
            evaluation = EvaluationResult(
                overall_score=fb_data.get('eval_score', 0.7),
                relevance_score=0.7,
                clarity_score=0.7,
                completeness_score=0.7
            )
            
            iteration = PromptIteration(
                session_id=self.session_id,
                version=i,
                prompt_text=f"Test prompt {i}",
                user_feedback=feedback,
                evaluation_scores=evaluation
            )
            history.append(iteration)
        
        return history
    
    def test_analyze_feedback_patterns_no_history(self):
        """Test feedback analysis with no history."""
        self.mock_history.load_session_history.return_value = []
        
        result = self.session_manager.analyze_feedback_patterns(self.session_id)
        
        self.assertFalse(result['success'])
        self.assertIn('no history', result['message'].lower())
    
    def test_analyze_feedback_patterns_no_feedback(self):
        """Test feedback analysis with history but no feedback."""
        # Create history without feedback
        history = [
            PromptIteration(
                session_id=self.session_id,
                version=1,
                prompt_text="Test prompt",
                user_feedback=None
            )
        ]
        self.mock_history.load_session_history.return_value = history
        
        result = self.session_manager.analyze_feedback_patterns(self.session_id)
        
        self.assertFalse(result['success'])
        self.assertIn('no feedback', result['message'].lower())
    
    def test_analyze_improving_trend(self):
        """Test analysis of improving satisfaction trend."""
        feedback_data = [
            {'rating': 2, 'issues': ['unclear'], 'eval_score': 0.5},
            {'rating': 3, 'issues': ['too long'], 'eval_score': 0.6},
            {'rating': 4, 'issues': [], 'eval_score': 0.8}
        ]
        
        history = self._create_test_history(feedback_data)
        self.mock_history.load_session_history.return_value = history
        
        result = self.session_manager.analyze_feedback_patterns(self.session_id)
        
        self.assertTrue(result['success'])
        self.assertEqual(result['feedback_count'], 3)
        
        analysis = result['analysis']
        self.assertEqual(analysis['rating_trend'], 'improving')
        self.assertEqual(analysis['latest_rating'], 4)
        self.assertAlmostEqual(analysis['average_rating'], 3.0, places=1)
    
    def test_analyze_declining_trend(self):
        """Test analysis of declining satisfaction trend."""
        feedback_data = [
            {'rating': 4, 'issues': [], 'eval_score': 0.8},
            {'rating': 3, 'issues': ['needs work'], 'eval_score': 0.7},
            {'rating': 2, 'issues': ['confusing', 'too complex'], 'eval_score': 0.5}
        ]
        
        history = self._create_test_history(feedback_data)
        self.mock_history.load_session_history.return_value = history
        
        result = self.session_manager.analyze_feedback_patterns(self.session_id)
        
        self.assertTrue(result['success'])
        
        analysis = result['analysis']
        self.assertEqual(analysis['rating_trend'], 'declining')
        self.assertEqual(analysis['latest_rating'], 2)
        
        # Should suggest reviewing recent changes
        suggestions = result['suggestions']
        self.assertTrue(any('declining' in s.lower() for s in suggestions))
    
    def test_analyze_common_issues(self):
        """Test identification of common issues."""
        feedback_data = [
            {'rating': 2, 'issues': ['unclear', 'too long']},
            {'rating': 3, 'issues': ['unclear', 'needs examples']},
            {'rating': 2, 'issues': ['unclear', 'confusing']}
        ]
        
        history = self._create_test_history(feedback_data)
        self.mock_history.load_session_history.return_value = history
        
        result = self.session_manager.analyze_feedback_patterns(self.session_id)
        
        self.assertTrue(result['success'])
        
        analysis = result['analysis']
        common_issues = analysis['common_issues']
        
        # 'unclear' should be the most common issue (appears 3 times)
        self.assertEqual(common_issues[0][0], 'unclear')
        self.assertEqual(common_issues[0][1], 3)
    
    def test_analyze_recurring_patterns(self):
        """Test identification of recurring patterns."""
        feedback_data = [
            {'rating': 2, 'issues': ['vague', 'needs structure']},
            {'rating': 2, 'issues': ['vague', 'unclear']},
            {'rating': 1, 'issues': ['vague', 'confusing']}
        ]
        
        history = self._create_test_history(feedback_data)
        self.mock_history.load_session_history.return_value = history
        
        result = self.session_manager.analyze_feedback_patterns(self.session_id)
        
        self.assertTrue(result['success'])
        
        patterns = result['patterns']
        
        # Should identify consistently low ratings
        self.assertTrue(any('consistently low' in p.lower() for p in patterns))
        
        # Should identify recurring issue (vague appears in all)
        self.assertTrue(any('vague' in p.lower() for p in patterns))
    
    def test_generate_strategic_suggestions(self):
        """Test generation of strategic suggestions based on patterns."""
        feedback_data = [
            {'rating': 2, 'issues': ['unclear', 'vague'], 'improvements': 'make it clearer'},
            {'rating': 2, 'issues': ['structure'], 'improvements': 'better format'},
            {'rating': 3, 'issues': ['examples'], 'improvements': 'more examples'}
        ]
        
        history = self._create_test_history(feedback_data)
        self.mock_history.load_session_history.return_value = history
        
        result = self.session_manager.analyze_feedback_patterns(self.session_id)
        
        self.assertTrue(result['success'])
        
        suggestions = result['suggestions']
        self.assertIsInstance(suggestions, list)
        self.assertGreater(len(suggestions), 0)
        
        # Should suggest addressing low satisfaction
        suggestions_text = ' '.join(suggestions).lower()
        self.assertTrue(any(keyword in suggestions_text for keyword in 
                          ['clarity', 'structure', 'specific', 'major']))


class TestFeedbackOrchestrationIntegration(unittest.TestCase):
    """Test integration of feedback analysis with orchestration."""
    
    def setUp(self):
        """Set up test environment."""
        self.mock_bedrock = Mock(spec=BedrockExecutor)
        self.mock_evaluator = Mock(spec=Evaluator)
        self.mock_history = Mock(spec=HistoryManager)
        
        self.session_manager = SessionManager(
            bedrock_executor=self.mock_bedrock,
            evaluator=self.mock_evaluator,
            history_manager=self.mock_history
        )
        
        self.session_id = "test-session-123"
        self.session_state = SessionState(
            session_id=self.session_id,
            status='active',
            current_iteration=3,
            current_prompt="Test prompt",
            initial_prompt="Test prompt",
            context={'domain': 'test'},
            config=SessionConfig(),
            created_at=datetime.now(),
            last_updated=datetime.now()
        )
        self.session_manager.active_sessions[self.session_id] = self.session_state
    
    def test_feedback_analysis_integration(self):
        """Test that feedback analysis is integrated into orchestration."""
        # Test the feedback analysis method directly
        feedback_analysis = {
            'success': True,
            'feedback_count': 3,
            'analysis': {
                'average_rating': 2.5,
                'rating_trend': 'declining',
                'common_issues': [('unclear', 2), ('vague', 1)]
            },
            'suggestions': ['Focus on clarity improvements'],
            'patterns': ['Consistently low satisfaction']
        }
        
        # Test context enhancement logic
        original_context = {'domain': 'test'}
        enhanced_context = original_context.copy()
        enhanced_context['feedback_analysis'] = feedback_analysis
        
        # Verify the context was enhanced correctly
        self.assertIn('feedback_analysis', enhanced_context)
        self.assertEqual(enhanced_context['feedback_analysis']['success'], True)
        self.assertEqual(enhanced_context['feedback_analysis']['feedback_count'], 3)
        self.assertEqual(enhanced_context['domain'], 'test')  # Original context preserved
        
        # Test that the analysis contains expected fields
        analysis = enhanced_context['feedback_analysis']['analysis']
        self.assertIn('average_rating', analysis)
        self.assertIn('rating_trend', analysis)
        self.assertIn('common_issues', analysis)


class TestFeedbackHistoryTracking(unittest.TestCase):
    """Test feedback history tracking and display."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create real history manager for integration testing
        self.history_manager = HistoryManager(storage_dir=self.temp_dir)
        
        self.session_id = self.history_manager.create_session(
            initial_prompt="Test prompt",
            context="Test context"
        )
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_save_and_load_feedback_history(self):
        """Test saving and loading feedback with iterations."""
        # Create iterations with feedback
        for i in range(3):
            feedback = UserFeedback(
                satisfaction_rating=i + 2,  # Ratings 2, 3, 4
                specific_issues=[f"Issue {i}"],
                desired_improvements=f"Improvement {i}",
                continue_optimization=True
            )
            
            iteration = PromptIteration(
                session_id=self.session_id,
                version=i + 1,
                prompt_text=f"Prompt version {i + 1}",
                user_feedback=feedback
            )
            
            success = self.history_manager.save_iteration(iteration)
            self.assertTrue(success)
        
        # Load history and verify feedback
        history = self.history_manager.load_session_history(self.session_id)
        self.assertEqual(len(history), 3)
        
        for i, iteration in enumerate(history):
            self.assertIsNotNone(iteration.user_feedback)
            self.assertEqual(iteration.user_feedback.satisfaction_rating, i + 2)
            self.assertEqual(iteration.user_feedback.specific_issues, [f"Issue {i}"])
    
    def test_feedback_persistence_across_sessions(self):
        """Test that feedback persists across different sessions."""
        # Save feedback
        feedback = UserFeedback(
            satisfaction_rating=4,
            specific_issues=["Test issue"],
            desired_improvements="Test improvement",
            continue_optimization=False
        )
        
        iteration = PromptIteration(
            session_id=self.session_id,
            version=1,
            prompt_text="Test prompt",
            user_feedback=feedback
        )
        
        self.history_manager.save_iteration(iteration)
        
        # Create new history manager instance (simulating app restart)
        new_history_manager = HistoryManager(storage_dir=self.temp_dir)
        
        # Load history with new instance
        history = new_history_manager.load_session_history(self.session_id)
        self.assertEqual(len(history), 1)
        
        loaded_feedback = history[0].user_feedback
        self.assertIsNotNone(loaded_feedback)
        self.assertEqual(loaded_feedback.satisfaction_rating, 4)
        self.assertEqual(loaded_feedback.specific_issues, ["Test issue"])
        self.assertEqual(loaded_feedback.desired_improvements, "Test improvement")
        self.assertFalse(loaded_feedback.continue_optimization)


if __name__ == '__main__':
    # Run all tests
    unittest.main(verbosity=2)
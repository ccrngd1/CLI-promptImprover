"""
Unit tests for the automated evaluation system.

Tests the Evaluator class with known input/output pairs to verify
scoring algorithms, version comparison, and report generation functionality.
"""

import unittest
from datetime import datetime, timedelta
from typing import Dict, Any
from evaluation.evaluator import Evaluator, EvaluationCriteria
from models import EvaluationResult, PromptIteration, ExecutionResult, UserFeedback


class TestEvaluator(unittest.TestCase):
    """Test cases for the Evaluator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.evaluator = Evaluator()
        self.custom_evaluator = Evaluator(
            EvaluationCriteria(
                relevance_weight=0.4,
                clarity_weight=0.3,
                completeness_weight=0.2,
                custom_weights={'technical_accuracy': 0.1}
            )
        )
    
    def test_evaluate_response_high_quality(self):
        """Test evaluation of a high-quality response."""
        prompt = "Explain how machine learning algorithms work and provide examples."
        response = """Machine learning algorithms are computational methods that enable systems to learn patterns from data without explicit programming. 

For example, supervised learning algorithms like linear regression learn to predict outcomes by training on labeled datasets. The algorithm analyzes input-output pairs to identify relationships and make predictions on new data.

Another example is neural networks, which use interconnected nodes to process information similarly to biological neurons. These networks excel at tasks like image recognition and natural language processing by learning complex patterns through multiple layers of computation."""
        
        result = self.evaluator.evaluate_response(prompt, response)
        
        # Verify high scores for quality response
        self.assertGreater(result.overall_score, 0.7)
        self.assertGreater(result.relevance_score, 0.7)
        self.assertGreater(result.clarity_score, 0.6)
        self.assertGreater(result.completeness_score, 0.7)
        
        # Verify feedback is generated
        self.assertIsInstance(result.qualitative_feedback, str)
        self.assertGreater(len(result.qualitative_feedback), 0)
        self.assertIsInstance(result.improvement_suggestions, list)
    
    def test_evaluate_response_low_quality(self):
        """Test evaluation of a low-quality response."""
        prompt = "Explain how machine learning algorithms work and provide examples."
        response = "ML is good. It works with data. Maybe neural networks."
        
        result = self.evaluator.evaluate_response(prompt, response)
        
        # Verify low scores for poor response
        self.assertLess(result.overall_score, 0.5)
        self.assertLess(result.completeness_score, 0.4)
        
        # Verify improvement suggestions are provided
        self.assertGreater(len(result.improvement_suggestions), 0)
        self.assertIn("more detailed", " ".join(result.improvement_suggestions).lower())
    
    def test_evaluate_response_empty_input(self):
        """Test evaluation with empty or invalid input."""
        result = self.evaluator.evaluate_response("", "")
        
        self.assertEqual(result.overall_score, 0.0)
        self.assertEqual(result.relevance_score, 0.0)
        self.assertEqual(result.clarity_score, 0.0)
        self.assertEqual(result.completeness_score, 0.0)
        self.assertIn("Invalid input", result.qualitative_feedback)
    
    def test_evaluate_response_with_custom_metrics(self):
        """Test evaluation with custom metrics."""
        def technical_accuracy_evaluator(prompt: str, response: str) -> float:
            # Simple mock evaluator that checks for technical terms
            technical_terms = ['algorithm', 'data', 'model', 'training', 'prediction']
            found_terms = sum(1 for term in technical_terms if term in response.lower())
            return min(found_terms / len(technical_terms), 1.0)
        
        prompt = "Explain machine learning algorithms."
        response = "Machine learning algorithms use data to train models for prediction tasks."
        
        context = {
            'custom_evaluators': {
                'technical_accuracy': technical_accuracy_evaluator
            }
        }
        
        result = self.evaluator.evaluate_response(prompt, response, context)
        
        self.assertIn('technical_accuracy', result.custom_metrics)
        self.assertGreater(result.custom_metrics['technical_accuracy'], 0.5)
    
    def test_relevance_score_calculation(self):
        """Test relevance score calculation with known cases."""
        # High relevance case
        prompt = "What are the benefits of renewable energy?"
        response = "Renewable energy offers several benefits including reduced carbon emissions, energy independence, and long-term cost savings."
        
        result = self.evaluator.evaluate_response(prompt, response)
        self.assertGreater(result.relevance_score, 0.7)
        
        # Low relevance case
        prompt = "What are the benefits of renewable energy?"
        response = "Cooking pasta requires boiling water and adding salt for flavor."
        
        result = self.evaluator.evaluate_response(prompt, response)
        self.assertLess(result.relevance_score, 0.3)
    
    def test_clarity_score_calculation(self):
        """Test clarity score calculation with different response styles."""
        # Clear, well-structured response
        clear_response = "First, renewable energy reduces environmental impact. Second, it provides energy security. Finally, it offers economic benefits through job creation."
        
        result = self.evaluator.evaluate_response("Explain renewable energy benefits.", clear_response)
        clarity_score_clear = result.clarity_score
        
        # Unclear, poorly structured response
        unclear_response = "Well, maybe renewable energy is kind of good I think because perhaps it might help with stuff and things that could be better somehow."
        
        result = self.evaluator.evaluate_response("Explain renewable energy benefits.", unclear_response)
        clarity_score_unclear = result.clarity_score
        
        self.assertGreater(clarity_score_clear, clarity_score_unclear)
        self.assertGreater(clarity_score_clear, 0.6)
        self.assertLess(clarity_score_unclear, 0.6)  # Adjusted expectation
    
    def test_completeness_score_calculation(self):
        """Test completeness score calculation."""
        prompt = "List three benefits of exercise and explain each one."
        
        # Complete response
        complete_response = """Here are three key benefits of exercise:

1. Cardiovascular health: Regular exercise strengthens the heart muscle and improves circulation, reducing the risk of heart disease.

2. Mental well-being: Physical activity releases endorphins that improve mood and reduce stress and anxiety levels.

3. Weight management: Exercise burns calories and builds muscle mass, helping maintain a healthy body weight."""
        
        result = self.evaluator.evaluate_response(prompt, complete_response)
        completeness_score_high = result.completeness_score
        
        # Incomplete response
        incomplete_response = "Exercise is good for health."
        
        result = self.evaluator.evaluate_response(prompt, incomplete_response)
        completeness_score_low = result.completeness_score
        
        self.assertGreater(completeness_score_high, completeness_score_low)
        self.assertGreater(completeness_score_high, 0.7)
        self.assertLess(completeness_score_low, 0.3)
    
    def test_compare_versions_improvement(self):
        """Test version comparison showing improvement."""
        # Create previous iteration with lower scores
        previous_eval = EvaluationResult(
            overall_score=0.6,
            relevance_score=0.5,
            clarity_score=0.6,
            completeness_score=0.7,
            qualitative_feedback="Previous version feedback"
        )
        
        previous_iteration = PromptIteration(
            id="prev-1",
            session_id="session-1",
            version=1,
            prompt_text="Previous prompt version",
            timestamp=datetime.now() - timedelta(hours=1),
            evaluation_scores=previous_eval
        )
        
        # Create current iteration with higher scores
        current_eval = EvaluationResult(
            overall_score=0.8,
            relevance_score=0.8,
            clarity_score=0.7,
            completeness_score=0.9,
            qualitative_feedback="Current version feedback"
        )
        
        current_iteration = PromptIteration(
            id="curr-1",
            session_id="session-1",
            version=2,
            prompt_text="Improved prompt version",
            timestamp=datetime.now(),
            evaluation_scores=current_eval
        )
        
        comparison = self.evaluator.compare_versions(current_iteration, previous_iteration)
        
        self.assertTrue(comparison['comparison_possible'])
        self.assertGreater(comparison['score_changes']['overall'], 0)
        self.assertGreater(comparison['improvement_percentage'], 0)
        self.assertIn('improvement', comparison['improvement_status'])
        self.assertEqual(comparison['current_version'], 2)
        self.assertEqual(comparison['previous_version'], 1)
    
    def test_compare_versions_decline(self):
        """Test version comparison showing decline."""
        # Create previous iteration with higher scores
        previous_eval = EvaluationResult(
            overall_score=0.8,
            relevance_score=0.8,
            clarity_score=0.7,
            completeness_score=0.9
        )
        
        previous_iteration = PromptIteration(
            id="prev-1",
            session_id="session-1",
            version=1,
            prompt_text="Better prompt version",
            timestamp=datetime.now() - timedelta(hours=1),
            evaluation_scores=previous_eval
        )
        
        # Create current iteration with lower scores
        current_eval = EvaluationResult(
            overall_score=0.5,
            relevance_score=0.4,
            clarity_score=0.5,
            completeness_score=0.6
        )
        
        current_iteration = PromptIteration(
            id="curr-1",
            session_id="session-1",
            version=2,
            prompt_text="Worse prompt version",
            timestamp=datetime.now(),
            evaluation_scores=current_eval
        )
        
        comparison = self.evaluator.compare_versions(current_iteration, previous_iteration)
        
        self.assertTrue(comparison['comparison_possible'])
        self.assertLess(comparison['score_changes']['overall'], 0)
        self.assertLess(comparison['improvement_percentage'], 0)
        self.assertIn('decline', comparison['improvement_status'])
    
    def test_compare_versions_missing_scores(self):
        """Test version comparison with missing evaluation scores."""
        iteration_without_scores = PromptIteration(
            id="no-scores",
            session_id="session-1",
            version=1,
            prompt_text="Prompt without evaluation",
            timestamp=datetime.now()
        )
        
        iteration_with_scores = PromptIteration(
            id="with-scores",
            session_id="session-1",
            version=2,
            prompt_text="Prompt with evaluation",
            timestamp=datetime.now(),
            evaluation_scores=EvaluationResult(
                overall_score=0.7,
                relevance_score=0.6,
                clarity_score=0.7,
                completeness_score=0.8
            )
        )
        
        comparison = self.evaluator.compare_versions(iteration_with_scores, iteration_without_scores)
        
        self.assertFalse(comparison['comparison_possible'])
        self.assertIn('Missing evaluation scores', comparison['reason'])
    
    def test_generate_evaluation_report(self):
        """Test comprehensive evaluation report generation."""
        # Create multiple iterations with different scores
        iterations = []
        base_time = datetime.now()
        
        for i in range(5):
            eval_result = EvaluationResult(
                overall_score=0.5 + (i * 0.1),  # Improving trend
                relevance_score=0.4 + (i * 0.1),
                clarity_score=0.6 + (i * 0.05),
                completeness_score=0.5 + (i * 0.08),
                custom_metrics={'technical_accuracy': 0.3 + (i * 0.15)}
            )
            
            iteration = PromptIteration(
                id=f"iter-{i}",
                session_id="session-1",
                version=i + 1,
                prompt_text=f"Prompt version {i + 1}",
                timestamp=base_time + timedelta(hours=i),
                evaluation_scores=eval_result
            )
            iterations.append(iteration)
        
        report = self.evaluator.generate_evaluation_report(iterations)
        
        self.assertEqual(report['total_iterations'], 5)
        self.assertEqual(report['evaluated_iterations'], 5)
        self.assertIn('trend_analysis', report)
        self.assertIn('best_iteration', report)
        self.assertIn('worst_iteration', report)
        self.assertIn('average_scores', report)
        self.assertIn('insights', report)
        
        # Verify trend analysis
        self.assertEqual(report['trend_analysis']['trend'], 'improving')
        
        # Verify best/worst identification
        self.assertEqual(report['best_iteration']['version'], 5)  # Last iteration has highest score
        self.assertEqual(report['worst_iteration']['version'], 1)  # First iteration has lowest score
    
    def test_generate_evaluation_report_empty_input(self):
        """Test evaluation report with empty input."""
        report = self.evaluator.generate_evaluation_report([])
        self.assertIn('error', report)
        self.assertIn('No iterations provided', report['error'])
    
    def test_generate_evaluation_report_no_evaluations(self):
        """Test evaluation report with iterations lacking evaluation scores."""
        iterations = [
            PromptIteration(
                id="iter-1",
                session_id="session-1",
                version=1,
                prompt_text="Prompt without evaluation",
                timestamp=datetime.now()
            )
        ]
        
        report = self.evaluator.generate_evaluation_report(iterations)
        self.assertIn('error', report)
        self.assertIn('No iterations with evaluation scores', report['error'])
    
    def test_custom_evaluation_criteria(self):
        """Test evaluator with custom criteria weights."""
        prompt = "Explain quantum computing."
        response = "Quantum computing uses quantum mechanical phenomena like superposition and entanglement to process information in ways that classical computers cannot."
        
        # Test with custom weights
        result = self.custom_evaluator.evaluate_response(prompt, response)
        
        # Verify the evaluation completes successfully
        self.assertIsInstance(result, EvaluationResult)
        self.assertGreaterEqual(result.overall_score, 0.0)
        self.assertLessEqual(result.overall_score, 1.0)
    
    def test_evaluation_criteria_validation(self):
        """Test evaluation criteria validation."""
        # Test invalid criteria (negative weights)
        with self.assertRaises(ValueError):
            EvaluationCriteria(
                relevance_weight=-0.1,
                clarity_weight=0.3,
                completeness_weight=0.3
            )
        
        # Test zero weights
        with self.assertRaises(ValueError):
            EvaluationCriteria(
                relevance_weight=0.0,
                clarity_weight=0.0,
                completeness_weight=0.0
            )
    
    def test_key_term_extraction(self):
        """Test key term extraction functionality."""
        text = "Machine learning algorithms analyze data to make predictions and decisions."
        key_terms = self.evaluator._extract_key_terms(text)
        
        # Should extract meaningful terms, excluding stop words
        expected_terms = ['machine', 'learning', 'algorithms', 'analyze', 'data', 'make', 'predictions', 'decisions']
        
        for term in expected_terms:
            self.assertIn(term, key_terms)
        
        # Should not include stop words
        stop_words = ['the', 'to', 'and']
        for stop_word in stop_words:
            self.assertNotIn(stop_word, key_terms)
    
    def test_question_addressing_detection(self):
        """Test question addressing detection."""
        prompt_with_questions = "What is machine learning? How does it work?"
        
        # Response that addresses questions
        good_response = "Machine learning is a subset of AI. It works by training algorithms on data."
        addressing_score_good = self.evaluator._check_question_addressing(prompt_with_questions, good_response)
        
        # Response that doesn't address questions
        poor_response = "Technology is advancing rapidly in many fields."
        addressing_score_poor = self.evaluator._check_question_addressing(prompt_with_questions, poor_response)
        
        self.assertGreater(addressing_score_good, addressing_score_poor)
    
    def test_improvement_suggestions_generation(self):
        """Test improvement suggestions generation."""
        prompt = "Explain the benefits of renewable energy."
        
        # Test with a poor response that should generate many suggestions
        poor_response = "Good."
        
        result = self.evaluator.evaluate_response(prompt, poor_response)
        suggestions = result.improvement_suggestions
        
        self.assertGreater(len(suggestions), 0)
        
        # Should suggest expanding the response
        suggestion_text = " ".join(suggestions).lower()
        self.assertTrue(
            any(keyword in suggestion_text for keyword in ['expand', 'detail', 'more', 'address'])
        )
    
    def test_trend_analysis_calculation(self):
        """Test trend analysis calculation."""
        # Create iterations with improving scores
        improving_iterations = []
        for i in range(5):
            eval_result = EvaluationResult(
                overall_score=0.4 + (i * 0.1),
                relevance_score=0.5,
                clarity_score=0.5,
                completeness_score=0.5
            )
            iteration = PromptIteration(
                id=f"iter-{i}",
                session_id="session-1",
                version=i + 1,
                prompt_text=f"Prompt {i + 1}",
                timestamp=datetime.now() + timedelta(hours=i),
                evaluation_scores=eval_result
            )
            improving_iterations.append(iteration)
        
        trend_analysis = self.evaluator._calculate_trend_analysis(improving_iterations)
        
        self.assertEqual(trend_analysis['trend'], 'improving')
        self.assertGreater(trend_analysis['slope'], 0)
        self.assertIsInstance(trend_analysis['volatility'], float)
        self.assertIn('score_range', trend_analysis)
        self.assertIn('average_score', trend_analysis)


class TestEvaluationCriteria(unittest.TestCase):
    """Test cases for EvaluationCriteria class."""
    
    def test_default_criteria(self):
        """Test default evaluation criteria."""
        criteria = EvaluationCriteria()
        
        self.assertEqual(criteria.relevance_weight, 0.3)
        self.assertEqual(criteria.clarity_weight, 0.3)
        self.assertEqual(criteria.completeness_weight, 0.3)
        self.assertEqual(criteria.custom_weights, {})
    
    def test_custom_criteria(self):
        """Test custom evaluation criteria."""
        custom_weights = {'technical_accuracy': 0.2, 'creativity': 0.1}
        criteria = EvaluationCriteria(
            relevance_weight=0.4,
            clarity_weight=0.2,
            completeness_weight=0.1,
            custom_weights=custom_weights
        )
        
        self.assertEqual(criteria.relevance_weight, 0.4)
        self.assertEqual(criteria.clarity_weight, 0.2)
        self.assertEqual(criteria.completeness_weight, 0.1)
        self.assertEqual(criteria.custom_weights, custom_weights)


if __name__ == '__main__':
    unittest.main()
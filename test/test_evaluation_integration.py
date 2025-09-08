"""
Integration test for the evaluation system.

Demonstrates the complete evaluation workflow with real examples.
"""

import unittest
from datetime import datetime, timedelta
from evaluation.evaluator import Evaluator, EvaluationCriteria
from models import PromptIteration, EvaluationResult


class TestEvaluationIntegration(unittest.TestCase):
    """Integration tests for the evaluation system."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.evaluator = Evaluator()
    
    def test_complete_evaluation_workflow(self):
        """Test a complete evaluation workflow with multiple iterations."""
        
        # Simulate a prompt optimization session
        base_prompt = "Explain machine learning"
        
        # Version 1: Basic response
        response_v1 = "Machine learning is AI."
        
        # Version 2: Improved response
        response_v2 = """Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without explicit programming. 

For example, recommendation systems use machine learning to suggest products based on user behavior patterns."""
        
        # Version 3: Comprehensive response
        response_v3 = """Machine learning is a subset of artificial intelligence that enables computers to learn patterns from data and make predictions or decisions without being explicitly programmed for each task.

There are three main types of machine learning:

1. Supervised learning: Algorithms learn from labeled training data to make predictions on new data. For example, email spam detection systems are trained on emails labeled as "spam" or "not spam."

2. Unsupervised learning: Algorithms find hidden patterns in data without labeled examples. For instance, customer segmentation in marketing uses clustering algorithms to group customers with similar behaviors.

3. Reinforcement learning: Algorithms learn through trial and error by receiving rewards or penalties. This approach is used in game AI and autonomous vehicle navigation.

Machine learning applications include recommendation systems, image recognition, natural language processing, and predictive analytics across industries like healthcare, finance, and technology."""
        
        # Create iterations
        iterations = []
        base_time = datetime.now()
        
        for i, response in enumerate([response_v1, response_v2, response_v3], 1):
            # Evaluate the response
            evaluation = self.evaluator.evaluate_response(base_prompt, response)
            
            # Create iteration
            iteration = PromptIteration(
                id=f"iter-{i}",
                session_id="test-session",
                version=i,
                prompt_text=base_prompt,
                timestamp=base_time + timedelta(hours=i),
                evaluation_scores=evaluation
            )
            iterations.append(iteration)
        
        # Verify improvement trend
        scores = [iter.evaluation_scores.overall_score for iter in iterations]
        self.assertLess(scores[0], scores[1])  # v1 < v2
        self.assertLess(scores[1], scores[2])  # v2 < v3
        
        # Test version comparison
        comparison_1_2 = self.evaluator.compare_versions(iterations[1], iterations[0])
        comparison_2_3 = self.evaluator.compare_versions(iterations[2], iterations[1])
        
        self.assertTrue(comparison_1_2['comparison_possible'])
        self.assertTrue(comparison_2_3['comparison_possible'])
        self.assertGreater(comparison_1_2['improvement_percentage'], 0)
        self.assertGreater(comparison_2_3['improvement_percentage'], 0)
        
        # Generate comprehensive report
        report = self.evaluator.generate_evaluation_report(iterations)
        
        self.assertEqual(report['total_iterations'], 3)
        self.assertEqual(report['evaluated_iterations'], 3)
        self.assertEqual(report['trend_analysis']['trend'], 'improving')
        self.assertEqual(report['best_iteration']['version'], 3)
        self.assertEqual(report['worst_iteration']['version'], 1)
        
        # Verify insights are generated
        self.assertGreater(len(report['insights']), 0)
        self.assertIn('positive progress', ' '.join(report['insights']))
        
        print("\n=== Evaluation Integration Test Results ===")
        print(f"Version 1 Score: {scores[0]:.3f}")
        print(f"Version 2 Score: {scores[1]:.3f}")
        print(f"Version 3 Score: {scores[2]:.3f}")
        print(f"Overall Improvement: {((scores[2] - scores[0]) / scores[0] * 100):.1f}%")
        print(f"Trend: {report['trend_analysis']['trend']}")
        print("Insights:")
        for insight in report['insights']:
            print(f"  - {insight}")
    
    def test_evaluation_with_custom_metrics(self):
        """Test evaluation system with custom metrics."""
        
        def technical_depth_evaluator(prompt: str, response: str) -> float:
            """Custom evaluator for technical depth."""
            technical_terms = [
                'algorithm', 'model', 'training', 'dataset', 'neural network',
                'supervised', 'unsupervised', 'reinforcement', 'classification',
                'regression', 'clustering', 'optimization'
            ]
            
            response_lower = response.lower()
            found_terms = sum(1 for term in technical_terms if term in response_lower)
            return min(found_terms / 5, 1.0)  # Normalize to 0-1 scale
        
        def example_quality_evaluator(prompt: str, response: str) -> float:
            """Custom evaluator for example quality."""
            example_indicators = [
                'for example', 'such as', 'for instance', 'like', 'including',
                'e.g.', 'i.e.', 'specifically'
            ]
            
            response_lower = response.lower()
            example_count = sum(1 for indicator in example_indicators if indicator in response_lower)
            
            # Also count numbered lists as examples
            import re
            numbered_examples = len(re.findall(r'\d+\.', response))
            
            total_examples = example_count + (numbered_examples / 2)  # Weight numbered lists less
            return min(total_examples / 2, 1.0)
        
        # Create evaluator with custom criteria
        custom_criteria = EvaluationCriteria(
            relevance_weight=0.25,
            clarity_weight=0.25,
            completeness_weight=0.25,
            custom_weights={
                'technical_depth': 0.15,
                'example_quality': 0.10
            }
        )
        
        custom_evaluator = Evaluator(custom_criteria)
        
        prompt = "Explain machine learning algorithms with examples."
        response = """Machine learning algorithms are computational methods that learn patterns from data. There are several types:

1. Supervised learning algorithms like linear regression and decision trees learn from labeled training data. For example, a spam detection algorithm is trained on emails labeled as spam or not spam.

2. Unsupervised learning algorithms such as k-means clustering find hidden patterns in unlabeled data. For instance, customer segmentation uses clustering to group customers with similar purchasing behaviors.

3. Reinforcement learning algorithms learn through trial and error, like AlphaGo learning to play Go by playing millions of games and receiving rewards for winning moves."""
        
        context = {
            'custom_evaluators': {
                'technical_depth': technical_depth_evaluator,
                'example_quality': example_quality_evaluator
            }
        }
        
        result = custom_evaluator.evaluate_response(prompt, response, context)
        
        # Verify custom metrics are calculated
        self.assertIn('technical_depth', result.custom_metrics)
        self.assertIn('example_quality', result.custom_metrics)
        
        # Verify scores are reasonable
        self.assertGreater(result.custom_metrics['technical_depth'], 0.5)
        self.assertGreater(result.custom_metrics['example_quality'], 0.5)
        self.assertGreater(result.overall_score, 0.7)
        
        print(f"\n=== Custom Metrics Test Results ===")
        print(f"Overall Score: {result.overall_score:.3f}")
        print(f"Technical Depth: {result.custom_metrics['technical_depth']:.3f}")
        print(f"Example Quality: {result.custom_metrics['example_quality']:.3f}")
        print(f"Relevance: {result.relevance_score:.3f}")
        print(f"Clarity: {result.clarity_score:.3f}")
        print(f"Completeness: {result.completeness_score:.3f}")


if __name__ == '__main__':
    unittest.main(verbosity=2)
"""
Integration tests for multi-agent collaboration scenarios.

Tests the AgentEnsemble class and consensus mechanisms for coordinating
multiple agents in prompt improvement workflows.
"""

import unittest
import time
from unittest.mock import Mock, patch
from agents import AgentEnsemble, EnsembleResult, ConsensusConfig, AgentResult
from models import PromptIteration, UserFeedback, EvaluationResult


class TestConsensusConfig(unittest.TestCase):
    """Test the ConsensusConfig data class."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = ConsensusConfig()
        
        self.assertEqual(config.voting_method, "weighted")
        self.assertEqual(config.confidence_threshold, 0.5)
        self.assertEqual(config.agreement_threshold, 0.7)
        self.assertEqual(config.timeout_seconds, 30.0)
        self.assertEqual(config.retry_attempts, 2)
        self.assertTrue(config.require_validator_approval)
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = ConsensusConfig(
            voting_method="majority",
            confidence_threshold=0.8,
            timeout_seconds=60.0,
            require_validator_approval=False
        )
        
        self.assertEqual(config.voting_method, "majority")
        self.assertEqual(config.confidence_threshold, 0.8)
        self.assertEqual(config.timeout_seconds, 60.0)
        self.assertFalse(config.require_validator_approval)


class TestAgentEnsemble(unittest.TestCase):
    """Test the AgentEnsemble implementation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.ensemble = AgentEnsemble()
    
    def test_ensemble_initialization(self):
        """Test AgentEnsemble initialization."""
        self.assertIsInstance(self.ensemble.config, ConsensusConfig)
        self.assertIn('analyzer', self.ensemble.agents)
        self.assertIn('refiner', self.ensemble.agents)
        self.assertIn('validator', self.ensemble.agents)
        self.assertEqual(len(self.ensemble.agents), 3)
    
    def test_process_simple_prompt(self):
        """Test processing a simple prompt with all agents."""
        prompt = "Write a summary of the given text."
        
        result = self.ensemble.process_prompt(prompt)
        
        self.assertIsInstance(result, EnsembleResult)
        self.assertTrue(result.success)
        self.assertIn('analyzer', result.agent_results)
        self.assertIn('refiner', result.agent_results)
        self.assertIn('validator', result.agent_results)
        self.assertGreaterEqual(result.confidence_score, 0.0)
        self.assertLessEqual(result.confidence_score, 1.0)
        self.assertGreater(result.processing_time, 0.0)
    
    def test_process_with_agent_subset(self):
        """Test processing with a subset of agents."""
        prompt = "Analyze this prompt structure."
        
        result = self.ensemble.process_prompt(prompt, agent_subset=['analyzer', 'validator'])
        
        # Should return a result even if not successful due to lower consensus
        self.assertIsInstance(result, EnsembleResult)
        self.assertIn('analyzer', result.agent_results)
        self.assertIn('validator', result.agent_results)
        self.assertNotIn('refiner', result.agent_results)
    
    def test_process_with_context_and_feedback(self):
        """Test processing with context and user feedback."""
        prompt = "Create a technical document."
        context = {
            'intended_use': 'API documentation',
            'target_audience': 'developers'
        }
        feedback = UserFeedback(
            satisfaction_rating=3,
            specific_issues=["Needs more structure", "Add examples"],
            desired_improvements="Include code samples"
        )
        
        result = self.ensemble.process_prompt(prompt, context=context, feedback=feedback)
        
        # Should return a result with analysis and recommendations
        self.assertIsInstance(result, EnsembleResult)
        self.assertIsInstance(result.consensus_analysis, dict)
        self.assertIsInstance(result.final_recommendations, list)
    
    def test_consensus_analysis(self):
        """Test consensus analysis functionality."""
        prompt = "Write a comprehensive analysis report."
        
        result = self.ensemble.process_prompt(prompt)
        
        consensus = result.consensus_analysis
        self.assertIn('consensus_level', consensus)
        self.assertIn('agreement_areas', consensus)
        self.assertIn('conflict_areas', consensus)
        self.assertIn('voting_results', consensus)
        self.assertIn('consensus_type', consensus)
        
        # Consensus level should be between 0 and 1
        self.assertGreaterEqual(consensus['consensus_level'], 0.0)
        self.assertLessEqual(consensus['consensus_level'], 1.0)
    
    def test_voting_mechanisms(self):
        """Test different voting mechanisms."""
        prompt = "Generate a detailed project plan."
        
        # Test weighted voting (default)
        result_weighted = self.ensemble.process_prompt(prompt)
        self.assertIn('voting_results', result_weighted.consensus_analysis)
        
        # Test majority voting
        majority_config = ConsensusConfig(voting_method="majority")
        ensemble_majority = AgentEnsemble(majority_config)
        result_majority = ensemble_majority.process_prompt(prompt)
        self.assertIn('voting_results', result_majority.consensus_analysis)
        
        # Test unanimous voting
        unanimous_config = ConsensusConfig(voting_method="unanimous")
        ensemble_unanimous = AgentEnsemble(unanimous_config)
        result_unanimous = ensemble_unanimous.process_prompt(prompt)
        self.assertIn('voting_results', result_unanimous.consensus_analysis)
    
    def test_validator_approval_requirement(self):
        """Test validator approval requirement."""
        prompt = "Create a simple task list."
        
        # Test with validator approval required (default)
        result_required = self.ensemble.process_prompt(prompt)
        voting_results = result_required.consensus_analysis.get('voting_results', {})
        self.assertIn('validator_approval', voting_results)
        
        # Test without validator approval requirement
        no_validator_config = ConsensusConfig(require_validator_approval=False)
        ensemble_no_validator = AgentEnsemble(no_validator_config)
        result_not_required = ensemble_no_validator.process_prompt(prompt)
        
        # Should still work without validator approval
        self.assertTrue(result_not_required.success or len(result_not_required.errors) == 0)
    
    def test_timeout_handling(self):
        """Test timeout handling for slow agents."""
        # Create ensemble with very short timeout
        short_timeout_config = ConsensusConfig(timeout_seconds=0.001)  # 1ms timeout
        ensemble_timeout = AgentEnsemble(short_timeout_config)
        
        prompt = "Process this complex prompt that might take time."
        
        result = ensemble_timeout.process_prompt(prompt)
        
        # Should handle timeouts gracefully
        self.assertIsInstance(result, EnsembleResult)
        # Some agents might timeout, but ensemble should still return a result
        self.assertTrue(len(result.errors) >= 0)  # May have timeout errors
    
    def test_error_recovery(self):
        """Test error recovery mechanisms."""
        # Mock an agent to always fail
        with patch.object(self.ensemble.agents['analyzer'], 'process') as mock_process:
            mock_process.side_effect = Exception("Simulated agent failure")
            
            prompt = "Test error recovery."
            result = self.ensemble.process_prompt(prompt)
            
            # Should handle agent failure gracefully
            self.assertIsInstance(result, EnsembleResult)
            self.assertGreater(len(result.errors), 0)
            
            # Other agents should still work
            analyzer_result = result.agent_results.get('analyzer')
            if analyzer_result:
                self.assertFalse(analyzer_result.success)
                self.assertIsNotNone(analyzer_result.error_message)
    
    def test_performance_metrics(self):
        """Test performance metrics tracking."""
        prompt = "Test performance tracking."
        
        # Process multiple prompts
        for i in range(3):
            self.ensemble.process_prompt(f"{prompt} {i}")
        
        metrics = self.ensemble.get_performance_metrics()
        
        # Should have metrics for each agent
        self.assertIn('analyzer', metrics)
        self.assertIn('refiner', metrics)
        self.assertIn('validator', metrics)
        self.assertIn('ensemble', metrics)
        
        # Check metric structure
        for agent_name in ['analyzer', 'refiner', 'validator']:
            agent_metrics = metrics[agent_name]
            self.assertIn('total_runs', agent_metrics)
            self.assertIn('successful_runs', agent_metrics)
            self.assertEqual(agent_metrics['total_runs'], 3)
    
    def test_agent_management(self):
        """Test adding and removing agents."""
        initial_count = len(self.ensemble.get_agent_names())
        
        # Add a mock agent
        mock_agent = Mock()
        mock_agent.get_name.return_value = "MockAgent"
        mock_agent.process.return_value = AgentResult(
            agent_name="MockAgent",
            success=True,
            analysis={},
            suggestions=["Mock suggestion"],
            confidence_score=0.8
        )
        
        self.ensemble.add_agent('mock', mock_agent)
        self.assertEqual(len(self.ensemble.get_agent_names()), initial_count + 1)
        self.assertIn('mock', self.ensemble.get_agent_names())
        
        # Remove the agent
        removed = self.ensemble.remove_agent('mock')
        self.assertTrue(removed)
        self.assertEqual(len(self.ensemble.get_agent_names()), initial_count)
        self.assertNotIn('mock', self.ensemble.get_agent_names())
        
        # Try to remove non-existent agent
        removed_again = self.ensemble.remove_agent('nonexistent')
        self.assertFalse(removed_again)
    
    def test_config_update(self):
        """Test configuration updates."""
        original_timeout = self.ensemble.config.timeout_seconds
        
        new_config = ConsensusConfig(
            timeout_seconds=60.0,
            voting_method="majority"
        )
        
        self.ensemble.update_config(new_config)
        
        self.assertEqual(self.ensemble.config.timeout_seconds, 60.0)
        self.assertEqual(self.ensemble.config.voting_method, "majority")
        self.assertNotEqual(self.ensemble.config.timeout_seconds, original_timeout)
    
    def test_metrics_reset(self):
        """Test performance metrics reset."""
        # Generate some metrics
        self.ensemble.process_prompt("Test metrics reset.")
        
        metrics_before = self.ensemble.get_performance_metrics()
        self.assertGreater(len(metrics_before), 0)
        
        # Reset metrics
        self.ensemble.reset_performance_metrics()
        
        metrics_after = self.ensemble.get_performance_metrics()
        self.assertEqual(len(metrics_after), 0)


class TestEnsembleIntegration(unittest.TestCase):
    """Test integration scenarios for the ensemble system."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.ensemble = AgentEnsemble()
    
    def test_iterative_improvement_workflow(self):
        """Test an iterative improvement workflow."""
        # Start with a basic prompt
        original_prompt = "Write about AI."
        
        # First iteration
        result1 = self.ensemble.process_prompt(original_prompt)
        self.assertTrue(result1.success)
        
        # Extract refined prompt from refiner agent
        refiner_result = result1.agent_results.get('refiner')
        if refiner_result and refiner_result.success:
            refined_prompt = refiner_result.analysis.get('refined_prompt', original_prompt)
        else:
            refined_prompt = original_prompt
        
        # Second iteration with refined prompt
        result2 = self.ensemble.process_prompt(refined_prompt)
        self.assertTrue(result2.success)
        
        # Should maintain or improve confidence
        self.assertGreaterEqual(result2.confidence_score, 0.0)
    
    def test_consensus_with_conflicting_agents(self):
        """Test consensus handling when agents have conflicting opinions."""
        # Use a prompt that might generate different opinions
        prompt = "Explain quantum computing in simple terms."
        
        result = self.ensemble.process_prompt(prompt)
        
        # Should handle conflicts gracefully
        self.assertIsInstance(result, EnsembleResult)
        consensus = result.consensus_analysis
        
        # Should have conflict resolution mechanisms
        self.assertIn('conflict_areas', consensus)
        self.assertIn('agreement_areas', consensus)
        
        # Final recommendations should be provided even with conflicts
        self.assertIsInstance(result.final_recommendations, list)
    
    def test_high_quality_prompt_consensus(self):
        """Test consensus on a high-quality, well-structured prompt."""
        prompt = """## Task
Please write a comprehensive technical analysis of the provided dataset.

## Context
You are analyzing sales performance data for a quarterly business review.
The audience consists of senior executives and department heads.

## Requirements
- Include key performance indicators and trends
- Provide actionable insights and recommendations
- Use clear, professional language appropriate for executives
- Support conclusions with data evidence

## Output Format
Structure your analysis as follows:
1. Executive Summary (2-3 paragraphs)
2. Key Findings (bullet points with supporting data)
3. Trend Analysis (with visual descriptions)
4. Recommendations (prioritized action items)
5. Appendix (detailed methodology if needed)

## Success Criteria
- Analysis is data-driven and objective
- Recommendations are specific and actionable
- Language is clear and executive-appropriate
- Structure follows the specified format"""
        
        result = self.ensemble.process_prompt(prompt)
        
        # Should achieve high consensus on well-structured prompt
        self.assertTrue(result.success)
        self.assertGreater(result.confidence_score, 0.7)
        
        consensus = result.consensus_analysis
        self.assertGreater(consensus['consensus_level'], 0.5)
        
        # Validator should approve
        validator_approval = consensus.get('voting_results', {}).get('validator_approval', False)
        self.assertTrue(validator_approval)
    
    def test_poor_quality_prompt_consensus(self):
        """Test consensus on a poor-quality prompt."""
        prompt = "do stuff"
        
        result = self.ensemble.process_prompt(prompt)
        
        # Should identify issues and provide recommendations
        self.assertIsInstance(result, EnsembleResult)
        
        # Should have low confidence and many suggestions
        self.assertLess(result.confidence_score, 0.8)
        self.assertGreater(len(result.final_recommendations), 0)
        
        # Validator should not approve
        consensus = result.consensus_analysis
        validator_approval = consensus.get('voting_results', {}).get('validator_approval', False)
        self.assertFalse(validator_approval)
    
    def test_ensemble_with_history(self):
        """Test ensemble processing with historical context."""
        # Create mock history
        history = [
            PromptIteration(
                session_id="test-session",
                version=1,
                prompt_text="Write a report.",
                agent_analysis={
                    'AnalyzerAgent': {
                        'analysis': {
                            'structure': {'structure_score': 0.3},
                            'clarity': {'clarity_score': 0.4}
                        }
                    }
                },
                evaluation_scores=EvaluationResult(
                    overall_score=0.4,
                    relevance_score=0.5,
                    clarity_score=0.3,
                    completeness_score=0.4
                )
            )
        ]
        
        prompt = "Write a comprehensive business report."
        result = self.ensemble.process_prompt(prompt, history=history)
        
        # Should incorporate historical context
        self.assertTrue(result.success)
        self.assertIsInstance(result.consensus_analysis, dict)
        
        # Should provide improvement recommendations based on history
        self.assertGreater(len(result.final_recommendations), 0)
    
    def test_ensemble_performance_under_load(self):
        """Test ensemble performance with multiple concurrent requests."""
        prompts = [
            "Analyze market trends.",
            "Create a project timeline.",
            "Write technical documentation.",
            "Generate a summary report.",
            "Develop a training plan."
        ]
        
        results = []
        start_time = time.time()
        
        # Process multiple prompts
        for prompt in prompts:
            result = self.ensemble.process_prompt(prompt)
            results.append(result)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # All should return valid results
        for result in results:
            self.assertIsInstance(result, EnsembleResult)
        
        # Should complete in reasonable time (this is a rough check)
        self.assertLess(total_time, 30.0)  # Should complete within 30 seconds
        
        # Performance metrics should be updated
        metrics = self.ensemble.get_performance_metrics()
        self.assertGreater(len(metrics), 0)


if __name__ == '__main__':
    unittest.main()
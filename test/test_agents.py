"""
Unit tests for the multi-agent system components.

Tests the base Agent class and specialized agent implementations
(AnalyzerAgent, RefinerAgent, ValidatorAgent).
"""

import unittest
from unittest.mock import Mock, patch
from agents import Agent, AgentResult, AnalyzerAgent, RefinerAgent, ValidatorAgent
from models import PromptIteration, UserFeedback, EvaluationResult


class TestAgentResult(unittest.TestCase):
    """Test the AgentResult data class."""
    
    def test_agent_result_creation(self):
        """Test creating an AgentResult instance."""
        result = AgentResult(
            agent_name="TestAgent",
            success=True,
            analysis={"test": "data"},
            suggestions=["suggestion1", "suggestion2"],
            confidence_score=0.8
        )
        
        self.assertEqual(result.agent_name, "TestAgent")
        self.assertTrue(result.success)
        self.assertEqual(result.analysis, {"test": "data"})
        self.assertEqual(result.suggestions, ["suggestion1", "suggestion2"])
        self.assertEqual(result.confidence_score, 0.8)
        self.assertIsNone(result.error_message)
    
    def test_agent_result_validation(self):
        """Test AgentResult validation."""
        # Valid result
        valid_result = AgentResult(
            agent_name="TestAgent",
            success=True,
            analysis={},
            suggestions=[],
            confidence_score=0.5
        )
        self.assertTrue(valid_result.validate())
        
        # Invalid confidence score
        invalid_result = AgentResult(
            agent_name="TestAgent",
            success=True,
            analysis={},
            suggestions=[],
            confidence_score=1.5  # Invalid: > 1.0
        )
        self.assertFalse(invalid_result.validate())


class TestAnalyzerAgent(unittest.TestCase):
    """Test the AnalyzerAgent implementation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.agent = AnalyzerAgent()
    
    def test_analyzer_initialization(self):
        """Test AnalyzerAgent initialization."""
        self.assertEqual(self.agent.name, "AnalyzerAgent")
        self.assertIsInstance(self.agent.config, dict)
    
    def test_analyze_simple_prompt(self):
        """Test analyzing a simple prompt."""
        prompt = "Please write a summary of the given text."
        result = self.agent.process(prompt)
        
        self.assertTrue(result.success)
        self.assertEqual(result.agent_name, "AnalyzerAgent")
        self.assertIn('structure', result.analysis)
        self.assertIn('clarity', result.analysis)
        self.assertIn('completeness', result.analysis)
        self.assertIsInstance(result.suggestions, list)
        self.assertGreaterEqual(result.confidence_score, 0.0)
        self.assertLessEqual(result.confidence_score, 1.0)
    
    def test_analyze_well_structured_prompt(self):
        """Test analyzing a well-structured prompt."""
        prompt = """## Task
Please write a comprehensive summary of the given text.

## Context
You are summarizing a technical document for a general audience.

## Requirements
- Keep the summary under 200 words
- Use simple language
- Include key points and conclusions

## Output Format
Provide the summary as a single paragraph."""
        
        result = self.agent.process(prompt)
        
        self.assertTrue(result.success)
        structure = result.analysis['structure']
        self.assertTrue(structure['has_clear_instruction'])
        self.assertTrue(structure['has_context_section'])
        self.assertTrue(structure['has_constraints'])
        self.assertGreater(structure['structure_score'], 0.7)
    
    def test_analyze_with_feedback(self):
        """Test analyzing with user feedback."""
        prompt = "Write something good."
        feedback = UserFeedback(
            satisfaction_rating=2,
            specific_issues=["Too vague", "Unclear requirements"],
            desired_improvements="Need more specific instructions"
        )
        
        result = self.agent.process(prompt, feedback=feedback)
        
        self.assertTrue(result.success)
        self.assertIn('feedback_considerations', result.analysis)
        feedback_analysis = result.analysis['feedback_considerations']
        self.assertTrue(feedback_analysis['low_satisfaction'])
        self.assertTrue(feedback_analysis['has_specific_issues'])
    
    def test_invalid_input(self):
        """Test handling invalid input."""
        result = self.agent.process("")
        
        self.assertFalse(result.success)
        self.assertIsNotNone(result.error_message)
        self.assertEqual(result.confidence_score, 0.0)


class TestRefinerAgent(unittest.TestCase):
    """Test the RefinerAgent implementation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.agent = RefinerAgent()
    
    def test_refiner_initialization(self):
        """Test RefinerAgent initialization."""
        self.assertEqual(self.agent.name, "RefinerAgent")
        self.assertIsInstance(self.agent.config, dict)
    
    def test_refine_simple_prompt(self):
        """Test refining a simple prompt."""
        prompt = "Write something about cats."
        result = self.agent.process(prompt)
        
        self.assertTrue(result.success)
        self.assertEqual(result.agent_name, "RefinerAgent")
        self.assertIn('refined_prompt', result.analysis)
        self.assertIn('improvement_areas', result.analysis)
        self.assertIn('improvements_made', result.analysis)
        
        refined_prompt = result.analysis['refined_prompt']
        self.assertNotEqual(refined_prompt, prompt)  # Should be different
        self.assertGreater(len(refined_prompt), len(prompt))  # Should be more detailed
    
    def test_refine_with_feedback(self):
        """Test refining with user feedback."""
        prompt = "Write about dogs."
        feedback = UserFeedback(
            satisfaction_rating=2,
            specific_issues=["Need more structure", "Too vague"],
            desired_improvements="Add specific requirements and examples"
        )
        
        result = self.agent.process(prompt, feedback=feedback)
        
        self.assertTrue(result.success)
        improvement_areas = result.analysis['improvement_areas']
        self.assertTrue(improvement_areas['structure'])
        self.assertTrue(improvement_areas['specificity'])
        
        refined_prompt = result.analysis['refined_prompt']
        self.assertIn('##', refined_prompt)  # Should have structure headers
    
    def test_refine_with_context(self):
        """Test refining with context information."""
        prompt = "Explain the concept."
        context = {
            'intended_use': 'educational content',
            'target_audience': 'students'
        }
        
        result = self.agent.process(prompt, context=context)
        
        self.assertTrue(result.success)
        refined_prompt = result.analysis['refined_prompt']
        self.assertGreater(len(refined_prompt), len(prompt))
    
    def test_invalid_input(self):
        """Test handling invalid input."""
        result = self.agent.process(None)
        
        self.assertFalse(result.success)
        self.assertIsNotNone(result.error_message)


class TestValidatorAgent(unittest.TestCase):
    """Test the ValidatorAgent implementation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.agent = ValidatorAgent()
    
    def test_validator_initialization(self):
        """Test ValidatorAgent initialization."""
        self.assertEqual(self.agent.name, "ValidatorAgent")
        self.assertIsInstance(self.agent.config, dict)
    
    def test_validate_good_prompt(self):
        """Test validating a well-formed prompt."""
        prompt = """## Task
Please write a comprehensive analysis of the provided data.

## Context
You are analyzing sales data for a quarterly business review.

## Requirements
- Include key trends and insights
- Provide actionable recommendations
- Use clear, professional language

## Output Format
Provide your analysis in a structured report format with sections for:
1. Executive Summary
2. Key Findings
3. Recommendations"""
        
        result = self.agent.process(prompt)
        
        self.assertTrue(result.success)
        self.assertEqual(result.agent_name, "ValidatorAgent")
        
        analysis = result.analysis
        self.assertIn('syntax', analysis)
        self.assertIn('logical_consistency', analysis)
        self.assertIn('completeness', analysis)
        self.assertIn('quality', analysis)
        self.assertIn('passes_validation', analysis)
        
        # Should pass validation
        self.assertTrue(analysis['passes_validation'])
        self.assertGreater(analysis['validation_score'], 0.6)
    
    def test_validate_poor_prompt(self):
        """Test validating a poorly formed prompt."""
        prompt = "do something"
        
        result = self.agent.process(prompt)
        
        self.assertTrue(result.success)
        analysis = result.analysis
        
        # Should not pass validation
        self.assertFalse(analysis['passes_validation'])
        self.assertLess(analysis['validation_score'], 0.6)
        self.assertGreater(len(result.suggestions), 0)
    
    def test_validate_syntax_issues(self):
        """Test detecting syntax issues."""
        prompt = "Write a report (but don't include charts"  # Unbalanced parentheses
        
        result = self.agent.process(prompt)
        
        self.assertTrue(result.success)
        syntax_issues = result.analysis['syntax']['issues']
        self.assertGreater(len(syntax_issues), 0)
        self.assertTrue(any('bracket' in issue.lower() for issue in syntax_issues))
    
    def test_validate_logical_consistency(self):
        """Test detecting logical consistency issues."""
        prompt = "Write a report but do not write anything. Always include charts but never add visuals."
        
        result = self.agent.process(prompt)
        
        self.assertTrue(result.success)
        logical_issues = result.analysis['logical_consistency']['issues']
        self.assertGreater(len(logical_issues), 0)
    
    def test_validate_completeness(self):
        """Test completeness validation."""
        prompt = "Do something with the data."  # Missing many components
        
        result = self.agent.process(prompt)
        
        self.assertTrue(result.success)
        completeness = result.analysis['completeness']
        self.assertGreater(len(completeness['missing_components']), 0)
        self.assertLess(completeness['completeness_score'], 0.8)
    
    def test_validate_with_context(self):
        """Test validation with context."""
        prompt = "Analyze the data."
        context = {
            'intended_use': 'financial analysis',
            'target_audience': 'executives'
        }
        
        result = self.agent.process(prompt, context=context)
        
        self.assertTrue(result.success)
        completeness = result.analysis['completeness']
        self.assertIn('context_alignment_score', completeness)
    
    def test_invalid_input(self):
        """Test handling invalid input."""
        result = self.agent.process("")
        
        self.assertFalse(result.success)
        self.assertIsNotNone(result.error_message)


class TestAgentIntegration(unittest.TestCase):
    """Test integration between different agents."""
    
    def test_agent_workflow(self):
        """Test a complete workflow using all agents."""
        # Start with a basic prompt
        original_prompt = "Write about machine learning."
        
        # Analyze the prompt
        analyzer = AnalyzerAgent()
        analysis_result = analyzer.process(original_prompt)
        self.assertTrue(analysis_result.success)
        
        # Refine based on analysis
        refiner = RefinerAgent()
        refinement_result = refiner.process(original_prompt)
        self.assertTrue(refinement_result.success)
        
        refined_prompt = refinement_result.analysis['refined_prompt']
        
        # Validate the refined prompt
        validator = ValidatorAgent()
        validation_result = validator.process(refined_prompt)
        self.assertTrue(validation_result.success)
        
        # Refined prompt should be better than original
        original_validation = validator.process(original_prompt)
        self.assertGreaterEqual(
            validation_result.analysis['validation_score'],
            original_validation.analysis['validation_score']
        )
    
    def test_agent_with_history(self):
        """Test agents working with historical data."""
        # Create mock history
        history = [
            PromptIteration(
                session_id="test-session",
                version=1,
                prompt_text="Write something.",
                agent_analysis={
                    'AnalyzerAgent': {
                        'analysis': {
                            'structure': {'structure_score': 0.3},
                            'clarity': {'clarity_score': 0.4},
                            'completeness': {'completeness_score': 0.2}
                        }
                    }
                },
                evaluation_scores=EvaluationResult(
                    overall_score=0.3,
                    relevance_score=0.4,
                    clarity_score=0.3,
                    completeness_score=0.2
                )
            )
        ]
        
        # Test refiner with history
        refiner = RefinerAgent()
        result = refiner.process("Write something better.", history=history)
        
        self.assertTrue(result.success)
        improvement_areas = result.analysis['improvement_areas']
        # Should identify areas for improvement based on history
        self.assertTrue(any(improvement_areas.values()))


if __name__ == '__main__':
    unittest.main()
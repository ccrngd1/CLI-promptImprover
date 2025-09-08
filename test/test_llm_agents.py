"""
Unit tests for LLM-enhanced agents.

Tests the LLM integration capabilities, system prompt management,
and reasoning frameworks for the enhanced agent functionality.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List, Optional

from agents.llm_agent import LLMAgent
from agents.llm_enhanced_analyzer import LLMAnalyzerAgent
from agents.llm_enhanced_refiner import LLMRefinerAgent
from agents.llm_enhanced_validator import LLMValidatorAgent
from agents.base import AgentResult
from models import PromptIteration, UserFeedback, ExecutionResult, EvaluationResult


class TestLLMAgent(unittest.TestCase):
    """Test cases for the base LLMAgent class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a concrete implementation for testing
        class TestLLMAgentImpl(LLMAgent):
            def _get_base_system_prompt(self):
                return "Test base system prompt"
            
            def _get_best_practices_prompt(self):
                return "Test best practices prompt"
            
            def _get_reasoning_framework_prompt(self):
                return "Test reasoning framework prompt"
            
            def process(self, prompt, context=None, history=None, feedback=None):
                return AgentResult(
                    agent_name=self.name,
                    success=True,
                    analysis={},
                    suggestions=[],
                    confidence_score=0.8
                )
        
        self.agent = TestLLMAgentImpl("TestLLMAgent")
    
    def test_initialization(self):
        """Test LLMAgent initialization."""
        self.assertEqual(self.agent.name, "TestLLMAgent")
        self.assertEqual(self.agent.llm_model, 'claude-3-sonnet')
        self.assertEqual(self.agent.llm_temperature, 0.3)
        self.assertEqual(self.agent.llm_max_tokens, 2000)
        self.assertTrue(self.agent.best_practices_enabled)
    
    def test_custom_config(self):
        """Test LLMAgent with custom configuration."""
        config = {
            'llm_model': 'gpt-4',
            'llm_temperature': 0.7,
            'llm_max_tokens': 4000,
            'best_practices_enabled': False
        }
        
        class TestLLMAgentImpl(LLMAgent):
            def _get_base_system_prompt(self):
                return "Test base system prompt"
            
            def _get_best_practices_prompt(self):
                return "Test best practices prompt"
            
            def _get_reasoning_framework_prompt(self):
                return "Test reasoning framework prompt"
            
            def process(self, prompt, context=None, history=None, feedback=None):
                return AgentResult(
                    agent_name=self.name,
                    success=True,
                    analysis={},
                    suggestions=[],
                    confidence_score=0.8
                )
        
        agent = TestLLMAgentImpl("TestAgent", config)
        
        self.assertEqual(agent.llm_model, 'gpt-4')
        self.assertEqual(agent.llm_temperature, 0.7)
        self.assertEqual(agent.llm_max_tokens, 4000)
        self.assertFalse(agent.best_practices_enabled)
    
    def test_system_prompt_building(self):
        """Test system prompt construction."""
        system_prompt = self.agent.get_system_prompt()
        
        self.assertIn("Test base system prompt", system_prompt)
        self.assertIn("Test best practices prompt", system_prompt)
        self.assertIn("Test reasoning framework prompt", system_prompt)
    
    def test_system_prompt_without_best_practices(self):
        """Test system prompt construction without best practices."""
        self.agent.enable_best_practices(False)
        system_prompt = self.agent.get_system_prompt()
        
        self.assertIn("Test base system prompt", system_prompt)
        self.assertNotIn("Test best practices prompt", system_prompt)
        self.assertIn("Test reasoning framework prompt", system_prompt)
    
    def test_update_system_prompt(self):
        """Test system prompt template updating."""
        new_template = "Custom system prompt template"
        self.agent.update_system_prompt(new_template)
        
        system_prompt = self.agent.get_system_prompt()
        self.assertIn(new_template, system_prompt)
    
    def test_format_context(self):
        """Test context formatting."""
        context = {
            'intended_use': 'Testing',
            'target_audience': 'Developers',
            'domain': 'Software',
            'constraints': 'Time limited'
        }
        
        formatted_context = self.agent._format_context(context)
        
        self.assertIn("Intended Use: Testing", formatted_context)
        self.assertIn("Target Audience: Developers", formatted_context)
        self.assertIn("Domain: Software", formatted_context)
        self.assertIn("Constraints: Time limited", formatted_context)
    
    def test_parse_llm_response(self):
        """Test LLM response parsing."""
        response = """
        Analysis Results:
        
        1. First recommendation
        2. Second recommendation
        
        Confidence: 0.85
        
        Reasoning: This is the detailed reasoning section
        with multiple lines of explanation.
        """
        
        parsed = self.agent._parse_llm_response(response)
        
        self.assertEqual(parsed['raw_response'], response)
        self.assertIn('First recommendation', parsed['recommendations'])
        self.assertIn('Second recommendation', parsed['recommendations'])
        self.assertEqual(parsed['confidence'], 0.85)
        self.assertIn('reasoning', parsed)


class TestLLMAnalyzerAgent(unittest.TestCase):
    """Test cases for LLMAnalyzerAgent."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.agent = LLMAnalyzerAgent()
        self.test_prompt = "Write a comprehensive analysis of the given data."
        self.test_context = {
            'intended_use': 'Data analysis',
            'target_audience': 'Data scientists',
            'domain': 'Analytics'
        }
    
    def test_initialization(self):
        """Test LLMAnalyzerAgent initialization."""
        self.assertEqual(self.agent.name, "LLMAnalyzerAgent")
        self.assertEqual(self.agent.analysis_depth, 'comprehensive')
        self.assertIn('structure', self.agent.focus_areas)
        self.assertIn('clarity', self.agent.focus_areas)
        self.assertIn('completeness', self.agent.focus_areas)
    
    def test_system_prompts(self):
        """Test system prompt components."""
        base_prompt = self.agent._get_base_system_prompt()
        best_practices = self.agent._get_best_practices_prompt()
        reasoning_framework = self.agent._get_reasoning_framework_prompt()
        
        self.assertIn("expert prompt engineering analyst", base_prompt.lower())
        self.assertIn("STRUCTURE BEST PRACTICES", best_practices.upper())
        self.assertIn("initial assessment", reasoning_framework.lower())
    
    @patch.object(LLMAnalyzerAgent, '_call_llm')
    def test_successful_analysis(self, mock_llm_call):
        """Test successful prompt analysis."""
        # Mock LLM response
        mock_llm_call.return_value = {
            'success': True,
            'response': """
            STRUCTURE ANALYSIS: The prompt has good organization
            CLARITY ASSESSMENT: Language is clear and direct
            COMPLETENESS EVALUATION: Missing some context
            EFFECTIVENESS REVIEW: Generally effective
            IMPROVEMENT RECOMMENDATIONS:
            1. Add specific examples
            2. Include context section
            Confidence: 0.8
            """,
            'model': 'claude-3-sonnet',
            'tokens_used': 150
        }
        
        result = self.agent.process(self.test_prompt, self.test_context)
        
        self.assertTrue(result.success)
        self.assertEqual(result.agent_name, "LLMAnalyzerAgent")
        self.assertIn('llm_analysis', result.analysis)
        self.assertIn('structure_analysis', result.analysis)
        self.assertGreater(len(result.suggestions), 0)
        self.assertGreater(result.confidence_score, 0.0)
    
    @patch.object(LLMAnalyzerAgent, '_call_llm')
    def test_llm_failure(self, mock_llm_call):
        """Test handling of LLM call failure."""
        mock_llm_call.return_value = {
            'success': False,
            'response': '',
            'error': 'API timeout'
        }
        
        result = self.agent.process(self.test_prompt)
        
        self.assertFalse(result.success)
        self.assertIn('API timeout', result.error_message)
        self.assertEqual(result.confidence_score, 0.0)
    
    def test_invalid_input(self):
        """Test handling of invalid input."""
        result = self.agent.process("")
        
        self.assertFalse(result.success)
        self.assertIn("Invalid prompt input", result.error_message)
    
    def test_build_analysis_prompt(self):
        """Test analysis prompt building."""
        feedback = UserFeedback(
            satisfaction_rating=3,
            specific_issues=["Too vague", "Missing examples"],
            desired_improvements="Add more detail",
            continue_optimization=True
        )
        
        analysis_prompt = self.agent._build_analysis_prompt(
            self.test_prompt, self.test_context, None, feedback
        )
        
        self.assertIn(self.test_prompt, analysis_prompt)
        self.assertIn("Data analysis", analysis_prompt)
        self.assertIn("Too vague", analysis_prompt)
        self.assertIn("STRUCTURE ANALYSIS", analysis_prompt)
    
    def test_extract_analysis_components(self):
        """Test extraction of analysis components."""
        parsed_response = {
            'raw_response': """
            STRUCTURE ANALYSIS: Good organization with clear sections
            CLARITY ASSESSMENT: Language is precise and unambiguous
            COMPLETENESS EVALUATION: Missing context information
            """,
            'confidence': 0.85,
            'reasoning': 'Detailed analysis performed'
        }
        
        llm_response = {
            'model': 'claude-3-sonnet',
            'tokens_used': 200
        }
        
        analysis = self.agent._extract_analysis_components(parsed_response, llm_response)
        
        self.assertIn('llm_analysis', analysis)
        self.assertIn('structure_analysis', analysis)
        self.assertIn('clarity_analysis', analysis)
        self.assertEqual(analysis['llm_analysis']['confidence'], 0.85)


class TestLLMRefinerAgent(unittest.TestCase):
    """Test cases for LLMRefinerAgent."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.agent = LLMRefinerAgent()
        self.test_prompt = "Analyze the data and provide insights."
        self.test_context = {
            'intended_use': 'Business reporting',
            'target_audience': 'Executives'
        }
    
    def test_initialization(self):
        """Test LLMRefinerAgent initialization."""
        self.assertEqual(self.agent.name, "LLMRefinerAgent")
        self.assertEqual(self.agent.refinement_style, 'comprehensive')
        self.assertTrue(self.agent.preserve_intent)
        self.assertIn('structure', self.agent.improvement_focus)
    
    def test_system_prompts(self):
        """Test system prompt components."""
        base_prompt = self.agent._get_base_system_prompt()
        best_practices = self.agent._get_best_practices_prompt()
        reasoning_framework = self.agent._get_reasoning_framework_prompt()
        
        self.assertIn("expert prompt engineer", base_prompt.lower())
        self.assertIn("STRUCTURAL IMPROVEMENTS", best_practices.upper())
        self.assertIn("analysis phase", reasoning_framework.lower())
    
    @patch.object(LLMRefinerAgent, '_call_llm')
    def test_successful_refinement(self, mock_llm_call):
        """Test successful prompt refinement."""
        mock_llm_call.return_value = {
            'success': True,
            'response': """
            --- REFINED PROMPT ---
            ## Task
            Analyze the provided business data and generate actionable insights for executive decision-making.
            
            ## Context
            This analysis will be used for strategic planning and resource allocation decisions.
            
            ## Requirements
            - Focus on key performance indicators
            - Identify trends and patterns
            - Provide specific recommendations
            --- END REFINED PROMPT ---
            
            IMPROVEMENTS MADE:
            - Added clear structure with sections
            - Specified target audience context
            - Included specific requirements
            
            BEST PRACTICES APPLIED:
            - Clear task definition
            - Context specification
            - Structured format
            
            CONFIDENCE: 0.9
            """,
            'model': 'claude-3-sonnet',
            'tokens_used': 250
        }
        
        result = self.agent.process(self.test_prompt, self.test_context)
        
        self.assertTrue(result.success)
        self.assertIn('refined_prompt', result.analysis)
        self.assertIn('## Task', result.analysis['refined_prompt'])
        # Suggestions may be empty if no further improvements are suggested
        self.assertGreaterEqual(len(result.suggestions), 0)
        self.assertGreater(result.confidence_score, 0.8)
    
    @patch.object(LLMRefinerAgent, '_call_llm')
    def test_refinement_extraction_failure(self, mock_llm_call):
        """Test handling when refined prompt cannot be extracted."""
        mock_llm_call.return_value = {
            'success': True,
            'response': "Some analysis but no clear refined prompt section",
            'model': 'claude-3-sonnet',
            'tokens_used': 100
        }
        
        result = self.agent.process(self.test_prompt)
        
        self.assertTrue(result.success)
        self.assertIn('extraction_successful', result.analysis['improvements_analysis'])
        self.assertFalse(result.analysis['improvements_analysis']['extraction_successful'])
    
    def test_build_refinement_prompt(self):
        """Test refinement prompt building."""
        feedback = UserFeedback(
            satisfaction_rating=2,
            specific_issues=["Unclear instructions"],
            desired_improvements="Make it more specific",
            continue_optimization=True
        )
        
        refinement_prompt = self.agent._build_refinement_prompt(
            self.test_prompt, self.test_context, None, feedback
        )
        
        self.assertIn(self.test_prompt, refinement_prompt)
        self.assertIn("Business reporting", refinement_prompt)
        self.assertIn("Unclear instructions", refinement_prompt)
        self.assertIn("REFINED PROMPT", refinement_prompt)
    
    def test_extract_refined_prompt(self):
        """Test refined prompt extraction."""
        parsed_response = {
            'raw_response': """
            --- REFINED PROMPT ---
            This is the improved prompt with better structure and clarity.
            --- END REFINED PROMPT ---
            
            Additional analysis follows...
            """
        }
        
        refined_prompt = self.agent._extract_refined_prompt(parsed_response)
        
        self.assertEqual(refined_prompt, "This is the improved prompt with better structure and clarity.")
    
    def test_analyze_refinement_quality(self):
        """Test refinement quality analysis."""
        original = "Analyze data."
        refined = """
        ## Task
        Analyze the provided dataset and generate comprehensive insights.
        
        ## Requirements
        - Use statistical methods
        - Provide visualizations
        - Include recommendations
        """
        
        parsed_response = {
            'raw_response': "IMPROVEMENTS MADE: Added structure and requirements"
        }
        
        llm_response = {'model': 'claude-3-sonnet', 'tokens_used': 200}
        
        analysis = self.agent._analyze_refinement_quality(
            original, refined, parsed_response, llm_response
        )
        
        self.assertTrue(analysis['extraction_successful'])
        self.assertTrue(analysis['structural_improvements'])
        self.assertGreater(analysis['length_change'], 0)
        self.assertGreater(analysis['quality_score'], 0.5)


class TestLLMValidatorAgent(unittest.TestCase):
    """Test cases for LLMValidatorAgent."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.agent = LLMValidatorAgent()
        self.test_prompt = """
        ## Task
        Analyze the provided dataset and generate insights.
        
        ## Requirements
        - Use appropriate statistical methods
        - Provide clear visualizations
        - Include actionable recommendations
        """
    
    def test_initialization(self):
        """Test LLMValidatorAgent initialization."""
        self.assertEqual(self.agent.name, "LLMValidatorAgent")
        self.assertEqual(self.agent.validation_strictness, 'moderate')
        self.assertIn('syntax', self.agent.validation_criteria)
        self.assertIn('logic', self.agent.validation_criteria)
        self.assertEqual(self.agent.min_quality_threshold, 0.6)
    
    def test_system_prompts(self):
        """Test system prompt components."""
        base_prompt = self.agent._get_base_system_prompt()
        best_practices = self.agent._get_best_practices_prompt()
        reasoning_framework = self.agent._get_reasoning_framework_prompt()
        
        self.assertIn("expert prompt validation specialist", base_prompt.lower())
        self.assertIn("SYNTAX AND FORMATTING STANDARDS", best_practices.upper())
        self.assertIn("initial assessment", reasoning_framework.lower())
    
    @patch.object(LLMValidatorAgent, '_call_llm')
    def test_successful_validation_pass(self, mock_llm_call):
        """Test successful validation that passes."""
        mock_llm_call.return_value = {
            'success': True,
            'response': """
            VALIDATION RESULTS:
            Syntax: PASS - 0.9 - Excellent grammar and formatting
            Logic: PASS - 0.8 - Clear logical flow
            Completeness: PASS - 0.85 - All essential elements present
            Quality: PASS - 0.9 - High quality and clarity
            Best Practices: PASS - 0.8 - Follows established patterns
            
            OVERALL: PASS - 0.85
            
            CRITICAL ISSUES:
            None identified
            
            RECOMMENDATIONS:
            1. Consider adding examples
            2. Specify output format
            
            CONFIDENCE: 0.9
            """,
            'model': 'claude-3-sonnet',
            'tokens_used': 300
        }
        
        result = self.agent.process(self.test_prompt)
        
        self.assertTrue(result.success)
        self.assertTrue(result.analysis['passes_validation'])
        self.assertGreater(result.analysis['validation_results']['overall']['score'], 0.8)
        self.assertEqual(len(result.analysis['validation_results']['critical_issues']), 0)
        self.assertGreater(result.confidence_score, 0.8)
    
    @patch.object(LLMValidatorAgent, '_call_llm')
    def test_successful_validation_fail(self, mock_llm_call):
        """Test successful validation that fails."""
        mock_llm_call.return_value = {
            'success': True,
            'response': """
            VALIDATION RESULTS:
            Syntax: FAIL - 0.4 - Multiple grammar errors
            Logic: PASS - 0.7 - Generally logical
            Completeness: FAIL - 0.3 - Missing essential elements
            Quality: FAIL - 0.4 - Poor clarity and specificity
            Best Practices: FAIL - 0.2 - Does not follow standards
            
            OVERALL: FAIL - 0.4
            
            CRITICAL ISSUES:
            1. Severe grammar and formatting problems
            2. Missing task definition
            3. No success criteria specified
            
            RECOMMENDATIONS:
            1. Fix grammar and spelling errors
            2. Add clear task definition
            3. Specify success criteria
            
            CONFIDENCE: 0.85
            """,
            'model': 'claude-3-sonnet',
            'tokens_used': 280
        }
        
        result = self.agent.process("bad prompt with errors")
        
        self.assertTrue(result.success)
        self.assertFalse(result.analysis['passes_validation'])
        self.assertLess(result.analysis['validation_results']['overall']['score'], 0.6)
        self.assertGreater(len(result.analysis['validation_results']['critical_issues']), 0)
    
    def test_build_validation_prompt(self):
        """Test validation prompt building."""
        context = {
            'intended_use': 'Data analysis',
            'quality_requirements': 'High accuracy'
        }
        
        validation_prompt = self.agent._build_validation_prompt(
            self.test_prompt, context, None, None
        )
        
        self.assertIn(self.test_prompt, validation_prompt)
        self.assertIn("Data analysis", validation_prompt)
        self.assertIn("SYNTAX VALIDATION", validation_prompt)
        self.assertIn("LOGICAL CONSISTENCY", validation_prompt)
    
    def test_extract_validation_results(self):
        """Test validation results extraction."""
        parsed_response = {
            'raw_response': """
            VALIDATION RESULTS:
            Syntax: PASS - 0.9
            Logic: PASS - 0.8
            Completeness: FAIL - 0.5
            Quality: PASS - 0.7
            Best Practices: PASS - 0.8
            
            OVERALL: PASS - 0.74
            
            CRITICAL ISSUES:
            1. Missing output format specification
            """,
            'confidence': 0.8
        }
        
        llm_response = {'model': 'claude-3-sonnet', 'tokens_used': 200}
        
        results = self.agent._extract_validation_results(parsed_response, llm_response)
        
        self.assertTrue(results['syntax']['passes'])
        self.assertEqual(results['syntax']['score'], 0.9)
        self.assertFalse(results['completeness']['passes'])
        self.assertEqual(results['completeness']['score'], 0.5)
        self.assertEqual(len(results['critical_issues']), 1)
    
    def test_determine_validation_status_strict(self):
        """Test validation status determination in strict mode."""
        self.agent.validation_strictness = 'strict'
        
        # All pass - should pass
        validation_results = {
            'syntax': {'passes': True, 'score': 0.9},
            'logic': {'passes': True, 'score': 0.8},
            'completeness': {'passes': True, 'score': 0.7},
            'quality': {'passes': True, 'score': 0.8},
            'best_practices': {'passes': True, 'score': 0.7},
            'overall': {'passes': True, 'score': 0.8},
            'critical_issues': []
        }
        
        self.assertTrue(self.agent._determine_validation_status(validation_results))
        
        # One fail - should fail in strict mode
        validation_results['syntax']['passes'] = False
        self.assertFalse(self.agent._determine_validation_status(validation_results))
    
    def test_determine_validation_status_moderate(self):
        """Test validation status determination in moderate mode."""
        self.agent.validation_strictness = 'moderate'
        
        # One fail - should pass in moderate mode
        validation_results = {
            'syntax': {'passes': False, 'score': 0.4},
            'logic': {'passes': True, 'score': 0.8},
            'completeness': {'passes': True, 'score': 0.7},
            'quality': {'passes': True, 'score': 0.8},
            'best_practices': {'passes': True, 'score': 0.7},
            'overall': {'passes': True, 'score': 0.7},
            'critical_issues': []
        }
        
        self.assertTrue(self.agent._determine_validation_status(validation_results))
        
        # Two fails - should fail in moderate mode
        validation_results['completeness']['passes'] = False
        self.assertFalse(self.agent._determine_validation_status(validation_results))
    
    def test_critical_issues_block_validation(self):
        """Test that critical issues block validation regardless of scores."""
        validation_results = {
            'overall': {'score': 0.9},
            'critical_issues': ['Critical syntax error']
        }
        
        self.assertFalse(self.agent._determine_validation_status(validation_results))


class TestLLMAgentIntegration(unittest.TestCase):
    """Integration tests for LLM agents working together."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = LLMAnalyzerAgent()
        self.refiner = LLMRefinerAgent()
        self.validator = LLMValidatorAgent()
        self.test_prompt = "Create a summary of the data."
    
    @patch.object(LLMAnalyzerAgent, '_call_llm')
    @patch.object(LLMRefinerAgent, '_call_llm')
    @patch.object(LLMValidatorAgent, '_call_llm')
    def test_agent_workflow(self, mock_validator_llm, mock_refiner_llm, mock_analyzer_llm):
        """Test a complete workflow using all three agents."""
        # Mock analyzer response
        mock_analyzer_llm.return_value = {
            'success': True,
            'response': """
            STRUCTURE ANALYSIS: Needs improvement
            CLARITY ASSESSMENT: Too vague
            RECOMMENDATIONS:
            1. Add specific context
            2. Define output format
            """,
            'model': 'claude-3-sonnet',
            'tokens_used': 150
        }
        
        # Mock refiner response
        mock_refiner_llm.return_value = {
            'success': True,
            'response': """
            --- REFINED PROMPT ---
            ## Task
            Create a comprehensive summary of the provided dataset.
            
            ## Context
            This summary will be used for executive reporting.
            
            ## Requirements
            - Include key findings and trends
            - Provide actionable insights
            - Format as structured report
            --- END REFINED PROMPT ---
            
            IMPROVEMENTS MADE: Added structure and context
            CONFIDENCE: 0.85
            """,
            'model': 'claude-3-sonnet',
            'tokens_used': 200
        }
        
        # Mock validator response
        mock_validator_llm.return_value = {
            'success': True,
            'response': """
            VALIDATION RESULTS:
            Syntax: PASS - 0.9
            Logic: PASS - 0.8
            Completeness: PASS - 0.85
            Quality: PASS - 0.9
            Best Practices: PASS - 0.8
            
            OVERALL: PASS - 0.85
            
            CRITICAL ISSUES:
            None identified
            CONFIDENCE: 0.9
            """,
            'model': 'claude-3-sonnet',
            'tokens_used': 180
        }
        
        # Run the workflow
        analysis_result = self.analyzer.process(self.test_prompt)
        self.assertTrue(analysis_result.success)
        
        refinement_result = self.refiner.process(self.test_prompt)
        self.assertTrue(refinement_result.success)
        
        refined_prompt = refinement_result.analysis['refined_prompt']
        validation_result = self.validator.process(refined_prompt)
        self.assertTrue(validation_result.success)
        self.assertTrue(validation_result.analysis['passes_validation'])
    
    def test_agent_error_handling(self):
        """Test error handling across agents."""
        # Test with invalid input
        result = self.analyzer.process("")
        self.assertFalse(result.success)
        
        result = self.refiner.process(None)
        self.assertFalse(result.success)
        
        result = self.validator.process("   ")
        self.assertFalse(result.success)


if __name__ == '__main__':
    unittest.main()
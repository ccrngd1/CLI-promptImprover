"""
AnalyzerAgent for prompt structure and clarity analysis.

This agent specializes in analyzing prompt structure, clarity, and identifying
potential issues that could affect prompt performance.
"""

import re
from typing import Dict, Any, List, Optional
from agents.base import Agent, AgentResult
from models import PromptIteration, UserFeedback


class AnalyzerAgent(Agent):
    """
    Agent that analyzes prompt structure and clarity.
    
    Focuses on:
    - Prompt structure and organization
    - Clarity and readability
    - Potential ambiguities
    - Missing context or instructions
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the AnalyzerAgent."""
        super().__init__("AnalyzerAgent", config)
        
        # Analysis thresholds and weights
        self.min_length = self.config.get('min_length', 10)
        self.max_length = self.config.get('max_length', 2000)
        self.clarity_weight = self.config.get('clarity_weight', 0.3)
        self.structure_weight = self.config.get('structure_weight', 0.3)
        self.completeness_weight = self.config.get('completeness_weight', 0.4)
    
    def process(self, 
                prompt: str, 
                context: Optional[Dict[str, Any]] = None,
                history: Optional[List[PromptIteration]] = None,
                feedback: Optional[UserFeedback] = None) -> AgentResult:
        """
        Analyze prompt structure and clarity.
        
        Args:
            prompt: The prompt text to analyze
            context: Optional context about the prompt's intended use
            history: Optional list of previous prompt iterations
            feedback: Optional user feedback from previous iterations
            
        Returns:
            AgentResult containing analysis and improvement suggestions
        """
        if not self.validate_input(prompt):
            return AgentResult(
                agent_name=self.name,
                success=False,
                analysis={},
                suggestions=[],
                confidence_score=0.0,
                error_message="Invalid prompt input"
            )
        
        try:
            # Perform various analyses
            structure_analysis = self._analyze_structure(prompt)
            clarity_analysis = self._analyze_clarity(prompt)
            completeness_analysis = self._analyze_completeness(prompt, context)
            
            # Consider feedback from previous iterations
            feedback_analysis = self._analyze_feedback(feedback) if feedback else {}
            
            # Calculate overall confidence score
            confidence_score = self._calculate_confidence(
                structure_analysis, clarity_analysis, completeness_analysis
            )
            
            # Generate suggestions based on analysis
            suggestions = self._generate_suggestions(
                structure_analysis, clarity_analysis, completeness_analysis, feedback_analysis
            )
            
            # Compile full analysis
            analysis = {
                'structure': structure_analysis,
                'clarity': clarity_analysis,
                'completeness': completeness_analysis,
                'feedback_considerations': feedback_analysis,
                'overall_assessment': self._generate_overall_assessment(
                    structure_analysis, clarity_analysis, completeness_analysis
                )
            }
            
            return AgentResult(
                agent_name=self.name,
                success=True,
                analysis=analysis,
                suggestions=suggestions,
                confidence_score=confidence_score
            )
            
        except Exception as e:
            return AgentResult(
                agent_name=self.name,
                success=False,
                analysis={},
                suggestions=[],
                confidence_score=0.0,
                error_message=f"Analysis failed: {str(e)}"
            )
    
    def _analyze_structure(self, prompt: str) -> Dict[str, Any]:
        """Analyze the structural aspects of the prompt."""
        lines = prompt.split('\n')
        sentences = re.split(r'[.!?]+', prompt)
        
        # Count various structural elements
        has_clear_instruction = any(
            keyword in prompt.lower() 
            for keyword in ['please', 'write', 'generate', 'create', 'analyze', 'explain']
        )
        
        has_context_section = any(
            keyword in prompt.lower() 
            for keyword in ['context:', 'background:', 'given:', 'scenario:', '## context', '## background']
        )
        
        has_examples = any(
            keyword in prompt.lower() 
            for keyword in ['example:', 'for example', 'such as', 'like:']
        )
        
        has_constraints = any(
            keyword in prompt.lower() 
            for keyword in ['must', 'should', 'cannot', 'limit', 'constraint', 'requirement']
        )
        
        return {
            'length': len(prompt),
            'line_count': len(lines),
            'sentence_count': len([s for s in sentences if s.strip()]),
            'has_clear_instruction': has_clear_instruction,
            'has_context_section': has_context_section,
            'has_examples': has_examples,
            'has_constraints': has_constraints,
            'structure_score': self._calculate_structure_score(
                has_clear_instruction, has_context_section, has_examples, has_constraints
            )
        }
    
    def _analyze_clarity(self, prompt: str) -> Dict[str, Any]:
        """Analyze the clarity and readability of the prompt."""
        words = prompt.split()
        sentences = re.split(r'[.!?]+', prompt)
        
        # Calculate readability metrics
        avg_sentence_length = len(words) / max(len(sentences), 1)
        
        # Check for ambiguous language
        ambiguous_words = ['maybe', 'perhaps', 'might', 'could be', 'possibly', 'somewhat']
        ambiguity_count = sum(1 for word in ambiguous_words if word in prompt.lower())
        
        # Check for vague terms
        vague_terms = ['thing', 'stuff', 'something', 'anything', 'everything', 'good', 'bad']
        vague_count = sum(1 for term in vague_terms if term in prompt.lower())
        
        # Check for clear action words
        action_words = ['write', 'create', 'generate', 'analyze', 'explain', 'describe', 'list']
        action_count = sum(1 for word in action_words if word in prompt.lower())
        
        clarity_score = self._calculate_clarity_score(
            avg_sentence_length, ambiguity_count, vague_count, action_count
        )
        
        return {
            'word_count': len(words),
            'avg_sentence_length': avg_sentence_length,
            'ambiguity_count': ambiguity_count,
            'vague_count': vague_count,
            'action_count': action_count,
            'clarity_score': clarity_score
        }
    
    def _analyze_completeness(self, prompt: str, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze the completeness of the prompt."""
        # Check for essential components
        has_task_definition = any(
            keyword in prompt.lower() 
            for keyword in ['task:', 'goal:', 'objective:', 'write', 'create', 'generate']
        )
        
        has_output_format = any(
            keyword in prompt.lower() 
            for keyword in ['format:', 'output:', 'response:', 'json', 'list', 'paragraph']
        )
        
        has_success_criteria = any(
            keyword in prompt.lower() 
            for keyword in ['criteria:', 'requirements:', 'must include', 'should contain']
        )
        
        # Consider context if provided
        context_alignment = 1.0
        if context:
            intended_use = context.get('intended_use', '')
            if intended_use and intended_use.lower() not in prompt.lower():
                context_alignment = 0.5
        
        completeness_score = self._calculate_completeness_score(
            has_task_definition, has_output_format, has_success_criteria, context_alignment
        )
        
        return {
            'has_task_definition': has_task_definition,
            'has_output_format': has_output_format,
            'has_success_criteria': has_success_criteria,
            'context_alignment': context_alignment,
            'completeness_score': completeness_score
        }
    
    def _analyze_feedback(self, feedback: UserFeedback) -> Dict[str, Any]:
        """Analyze user feedback to identify areas for improvement."""
        if not feedback:
            return {}
        
        # Extract insights from feedback
        low_satisfaction = feedback.satisfaction_rating < 3
        has_specific_issues = len(feedback.specific_issues) > 0
        wants_improvements = bool(feedback.desired_improvements.strip())
        
        return {
            'low_satisfaction': low_satisfaction,
            'has_specific_issues': has_specific_issues,
            'wants_improvements': wants_improvements,
            'specific_issues': feedback.specific_issues,
            'desired_improvements': feedback.desired_improvements
        }
    
    def _calculate_structure_score(self, has_instruction: bool, has_context: bool, 
                                 has_examples: bool, has_constraints: bool) -> float:
        """Calculate a structure quality score."""
        score = 0.0
        if has_instruction:
            score += 0.4
        if has_context:
            score += 0.2
        if has_examples:
            score += 0.2
        if has_constraints:
            score += 0.2
        return min(score, 1.0)
    
    def _calculate_clarity_score(self, avg_sentence_length: float, ambiguity_count: int,
                               vague_count: int, action_count: int) -> float:
        """Calculate a clarity quality score."""
        score = 1.0
        
        # Penalize very long sentences
        if avg_sentence_length > 25:
            score -= 0.2
        elif avg_sentence_length > 15:
            score -= 0.1
        
        # Penalize ambiguous and vague language
        score -= min(ambiguity_count * 0.1, 0.3)
        score -= min(vague_count * 0.1, 0.3)
        
        # Reward clear action words
        if action_count > 0:
            score += 0.1
        
        return max(score, 0.0)
    
    def _calculate_completeness_score(self, has_task: bool, has_format: bool,
                                    has_criteria: bool, context_alignment: float) -> float:
        """Calculate a completeness quality score."""
        score = 0.0
        if has_task:
            score += 0.4
        if has_format:
            score += 0.3
        if has_criteria:
            score += 0.3
        
        # Apply context alignment factor
        score *= context_alignment
        
        return min(score, 1.0)
    
    def _calculate_confidence(self, structure: Dict[str, Any], clarity: Dict[str, Any],
                            completeness: Dict[str, Any]) -> float:
        """Calculate overall confidence in the analysis."""
        structure_score = structure.get('structure_score', 0.0)
        clarity_score = clarity.get('clarity_score', 0.0)
        completeness_score = completeness.get('completeness_score', 0.0)
        
        weighted_score = (
            structure_score * self.structure_weight +
            clarity_score * self.clarity_weight +
            completeness_score * self.completeness_weight
        )
        
        return weighted_score
    
    def _generate_suggestions(self, structure: Dict[str, Any], clarity: Dict[str, Any],
                            completeness: Dict[str, Any], feedback: Dict[str, Any]) -> List[str]:
        """Generate improvement suggestions based on analysis."""
        suggestions = []
        
        # Structure suggestions
        if not structure.get('has_clear_instruction'):
            suggestions.append("Add a clear instruction or task definition (e.g., 'Please write...', 'Generate...')")
        
        if not structure.get('has_context_section'):
            suggestions.append("Consider adding a context section to provide background information")
        
        if not structure.get('has_examples'):
            suggestions.append("Include examples to clarify the expected output format")
        
        if not structure.get('has_constraints'):
            suggestions.append("Add specific constraints or requirements to guide the response")
        
        # Clarity suggestions
        if clarity.get('ambiguity_count', 0) > 2:
            suggestions.append("Reduce ambiguous language (maybe, perhaps, might) for clearer instructions")
        
        if clarity.get('vague_count', 0) > 2:
            suggestions.append("Replace vague terms (thing, stuff, something) with specific language")
        
        if clarity.get('avg_sentence_length', 0) > 25:
            suggestions.append("Break down long sentences for better readability")
        
        # Completeness suggestions
        if not completeness.get('has_task_definition'):
            suggestions.append("Clearly define the main task or objective")
        
        if not completeness.get('has_output_format'):
            suggestions.append("Specify the desired output format (JSON, list, paragraph, etc.)")
        
        if not completeness.get('has_success_criteria'):
            suggestions.append("Define success criteria or requirements for the response")
        
        # Feedback-based suggestions
        if feedback.get('low_satisfaction'):
            suggestions.append("Address user satisfaction concerns from previous feedback")
        
        if feedback.get('specific_issues'):
            for issue in feedback.get('specific_issues', []):
                suggestions.append(f"Address specific issue: {issue}")
        
        return suggestions
    
    def _generate_overall_assessment(self, structure: Dict[str, Any], clarity: Dict[str, Any],
                                   completeness: Dict[str, Any]) -> str:
        """Generate an overall assessment of the prompt."""
        structure_score = structure.get('structure_score', 0.0)
        clarity_score = clarity.get('clarity_score', 0.0)
        completeness_score = completeness.get('completeness_score', 0.0)
        
        overall_score = (
            structure_score * self.structure_weight +
            clarity_score * self.clarity_weight +
            completeness_score * self.completeness_weight
        )
        
        if overall_score >= 0.8:
            return "Excellent prompt with strong structure, clarity, and completeness"
        elif overall_score >= 0.6:
            return "Good prompt with minor areas for improvement"
        elif overall_score >= 0.4:
            return "Adequate prompt that would benefit from significant improvements"
        else:
            return "Poor prompt requiring major restructuring and clarification"
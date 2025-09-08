"""
RefinerAgent for generating improved prompt versions.

This agent specializes in taking analysis results and user feedback to generate
improved versions of prompts with better structure, clarity, and effectiveness.
"""

import re
from typing import Dict, Any, List, Optional
from agents.base import Agent, AgentResult
from models import PromptIteration, UserFeedback


class RefinerAgent(Agent):
    """
    Agent that generates improved prompt versions.
    
    Focuses on:
    - Incorporating analysis feedback
    - Restructuring prompts for better clarity
    - Adding missing components
    - Optimizing for specific use cases
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the RefinerAgent."""
        super().__init__("RefinerAgent", config)
        
        # Refinement parameters
        self.max_iterations = self.config.get('max_iterations', 3)
        self.improvement_threshold = self.config.get('improvement_threshold', 0.1)
        self.preserve_intent = self.config.get('preserve_intent', True)
    
    def process(self, 
                prompt: str, 
                context: Optional[Dict[str, Any]] = None,
                history: Optional[List[PromptIteration]] = None,
                feedback: Optional[UserFeedback] = None) -> AgentResult:
        """
        Generate an improved version of the prompt.
        
        Args:
            prompt: The original prompt text to refine
            context: Optional context about the prompt's intended use
            history: Optional list of previous prompt iterations
            feedback: Optional user feedback from previous iterations
            
        Returns:
            AgentResult containing the refined prompt and improvement analysis
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
            # Extract improvement areas from history and feedback
            improvement_areas = self._identify_improvement_areas(history, feedback)
            
            # Generate refined prompt
            refined_prompt = self._refine_prompt(prompt, improvement_areas, context)
            
            # Analyze the improvements made
            improvements_analysis = self._analyze_improvements(prompt, refined_prompt, improvement_areas)
            
            # Calculate confidence in the refinement
            confidence_score = self._calculate_refinement_confidence(improvements_analysis)
            
            # Generate suggestions for further improvements
            suggestions = self._generate_refinement_suggestions(improvements_analysis)
            
            analysis = {
                'original_prompt': prompt,
                'refined_prompt': refined_prompt,
                'improvement_areas': improvement_areas,
                'improvements_made': improvements_analysis,
                'refinement_quality': self._assess_refinement_quality(improvements_analysis)
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
                error_message=f"Refinement failed: {str(e)}"
            )
    
    def _identify_improvement_areas(self, history: Optional[List[PromptIteration]], 
                                  feedback: Optional[UserFeedback]) -> Dict[str, Any]:
        """Identify areas that need improvement based on history and feedback."""
        improvement_areas = {
            'structure': True,  # Default to improving structure for simple prompts
            'clarity': True,    # Default to improving clarity
            'completeness': True,  # Default to improving completeness
            'specificity': False,
            'examples': False,
            'constraints': False,
            'format': False
        }
        
        # Analyze feedback
        if feedback:
            if feedback.satisfaction_rating < 3:
                improvement_areas['structure'] = True
                improvement_areas['clarity'] = True
            
            for issue in feedback.specific_issues:
                issue_lower = issue.lower()
                if any(word in issue_lower for word in ['unclear', 'confusing', 'ambiguous']):
                    improvement_areas['clarity'] = True
                if any(word in issue_lower for word in ['missing', 'incomplete', 'need more']):
                    improvement_areas['completeness'] = True
                if any(word in issue_lower for word in ['vague', 'general', 'specific']):
                    improvement_areas['specificity'] = True
                if any(word in issue_lower for word in ['example', 'sample']):
                    improvement_areas['examples'] = True
                if any(word in issue_lower for word in ['format', 'structure', 'organize']):
                    improvement_areas['format'] = True
        
        # Analyze history patterns
        if history and len(history) > 1:
            latest_iteration = history[-1]
            if latest_iteration.agent_analysis:
                analyzer_results = latest_iteration.agent_analysis.get('AnalyzerAgent', {})
                if analyzer_results:
                    analysis = analyzer_results.get('analysis', {})
                    
                    # Check structure issues
                    structure = analysis.get('structure', {})
                    if structure.get('structure_score', 1.0) < 0.6:
                        improvement_areas['structure'] = True
                    
                    # Check clarity issues
                    clarity = analysis.get('clarity', {})
                    if clarity.get('clarity_score', 1.0) < 0.6:
                        improvement_areas['clarity'] = True
                    
                    # Check completeness issues
                    completeness = analysis.get('completeness', {})
                    if completeness.get('completeness_score', 1.0) < 0.6:
                        improvement_areas['completeness'] = True
        
        return improvement_areas
    
    def _refine_prompt(self, prompt: str, improvement_areas: Dict[str, Any], 
                      context: Optional[Dict[str, Any]]) -> str:
        """Generate a refined version of the prompt."""
        refined_prompt = prompt
        
        # Apply improvements based on identified areas
        if improvement_areas.get('structure'):
            refined_prompt = self._improve_structure(refined_prompt)
        
        if improvement_areas.get('clarity'):
            refined_prompt = self._improve_clarity(refined_prompt)
        
        if improvement_areas.get('completeness'):
            refined_prompt = self._improve_completeness(refined_prompt, context)
        
        if improvement_areas.get('specificity'):
            refined_prompt = self._improve_specificity(refined_prompt)
        
        if improvement_areas.get('examples'):
            refined_prompt = self._add_examples(refined_prompt, context)
        
        if improvement_areas.get('format'):
            refined_prompt = self._improve_format(refined_prompt)
        
        return refined_prompt.strip()
    
    def _improve_structure(self, prompt: str) -> str:
        """Improve the structural organization of the prompt."""
        lines = prompt.split('\n')
        
        # Check if prompt has clear sections
        has_task_section = any('task:' in line.lower() or 'objective:' in line.lower() for line in lines)
        has_context_section = any('context:' in line.lower() or 'background:' in line.lower() for line in lines)
        
        if not has_task_section and not has_context_section:
            # Add structure to unstructured prompt
            structured_prompt = "## Task\n"
            
            # Extract main instruction
            main_instruction = self._extract_main_instruction(prompt)
            if main_instruction:
                structured_prompt += f"{main_instruction}\n\n"
            
            # Add context section if there's background info
            context_info = self._extract_context_info(prompt)
            if context_info:
                structured_prompt += "## Context\n"
                structured_prompt += f"{context_info}\n\n"
            
            # Add requirements section
            requirements = self._extract_requirements(prompt)
            if requirements:
                structured_prompt += "## Requirements\n"
                structured_prompt += f"{requirements}\n\n"
            
            return structured_prompt
        
        return prompt
    
    def _improve_clarity(self, prompt: str) -> str:
        """Improve the clarity and readability of the prompt."""
        # Replace ambiguous words
        ambiguous_replacements = {
            'maybe': 'if applicable',
            'perhaps': 'if relevant',
            'might': 'should',
            'could be': 'is',
            'possibly': 'if appropriate'
        }
        
        improved_prompt = prompt
        for ambiguous, clear in ambiguous_replacements.items():
            improved_prompt = re.sub(r'\b' + re.escape(ambiguous) + r'\b', clear, improved_prompt, flags=re.IGNORECASE)
        
        # Replace vague terms
        vague_replacements = {
            'thing': 'item',
            'stuff': 'content',
            'something': 'a specific item',
            'good': 'high-quality',
            'bad': 'low-quality'
        }
        
        for vague, specific in vague_replacements.items():
            improved_prompt = re.sub(r'\b' + re.escape(vague) + r'\b', specific, improved_prompt, flags=re.IGNORECASE)
        
        return improved_prompt
    
    def _improve_completeness(self, prompt: str, context: Optional[Dict[str, Any]]) -> str:
        """Improve the completeness of the prompt."""
        improved_prompt = prompt
        
        # Add task definition if missing
        if not any(keyword in prompt.lower() for keyword in ['write', 'create', 'generate', 'analyze', 'explain']):
            improved_prompt = "Please " + improved_prompt
        
        # Add output format specification if missing
        if not any(keyword in prompt.lower() for keyword in ['format:', 'output:', 'json', 'list', 'paragraph']):
            improved_prompt += "\n\nOutput Format: Provide your response in a clear, structured format."
        
        # Add success criteria if missing
        if not any(keyword in prompt.lower() for keyword in ['must', 'should', 'requirement', 'criteria']):
            improved_prompt += "\n\nRequirements: Ensure your response is accurate, relevant, and comprehensive."
        
        return improved_prompt
    
    def _improve_specificity(self, prompt: str) -> str:
        """Make the prompt more specific and actionable."""
        # Add specific action verbs if missing
        if not re.search(r'\b(write|create|generate|analyze|explain|describe|list|summarize)\b', prompt, re.IGNORECASE):
            prompt = "Please create " + prompt
        
        # Add specific constraints
        if 'length' not in prompt.lower() and 'word' not in prompt.lower():
            prompt += " Aim for a comprehensive response of appropriate length."
        
        return prompt
    
    def _add_examples(self, prompt: str, context: Optional[Dict[str, Any]]) -> str:
        """Add examples to clarify expectations."""
        if 'example' not in prompt.lower():
            example_section = "\n\nExample: [Provide a brief example of the expected output format or style]"
            return prompt + example_section
        return prompt
    
    def _improve_format(self, prompt: str) -> str:
        """Improve the formatting and presentation of the prompt."""
        lines = prompt.split('\n')
        
        # Add proper spacing between sections
        formatted_lines = []
        for i, line in enumerate(lines):
            formatted_lines.append(line)
            
            # Add spacing after headers
            if line.strip().endswith(':') and i < len(lines) - 1:
                if lines[i + 1].strip():  # Only add space if next line isn't empty
                    formatted_lines.append('')
        
        return '\n'.join(formatted_lines)
    
    def _extract_main_instruction(self, prompt: str) -> str:
        """Extract the main instruction from the prompt."""
        # Look for sentences with action verbs
        sentences = re.split(r'[.!?]+', prompt)
        for sentence in sentences:
            if any(verb in sentence.lower() for verb in ['write', 'create', 'generate', 'analyze', 'explain']):
                return sentence.strip()
        
        # If no clear instruction found, return first sentence
        return sentences[0].strip() if sentences else ""
    
    def _extract_context_info(self, prompt: str) -> str:
        """Extract context information from the prompt."""
        # Look for background information patterns
        context_patterns = [
            r'given\s+(.+?)(?:\.|$)',
            r'context[:\s]+(.+?)(?:\.|$)',
            r'background[:\s]+(.+?)(?:\.|$)',
            r'scenario[:\s]+(.+?)(?:\.|$)'
        ]
        
        for pattern in context_patterns:
            match = re.search(pattern, prompt, re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(1).strip()
        
        return ""
    
    def _extract_requirements(self, prompt: str) -> str:
        """Extract requirements from the prompt."""
        # Look for requirement patterns
        requirement_patterns = [
            r'must\s+(.+?)(?:\.|$)',
            r'should\s+(.+?)(?:\.|$)',
            r'requirement[s]?[:\s]+(.+?)(?:\.|$)',
            r'ensure\s+(.+?)(?:\.|$)'
        ]
        
        requirements = []
        for pattern in requirement_patterns:
            matches = re.findall(pattern, prompt, re.IGNORECASE)
            requirements.extend(matches)
        
        return '\n'.join(f"- {req.strip()}" for req in requirements if req.strip())
    
    def _analyze_improvements(self, original: str, refined: str, improvement_areas: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the improvements made to the prompt."""
        return {
            'length_change': len(refined) - len(original),
            'structure_improved': improvement_areas.get('structure', False),
            'clarity_improved': improvement_areas.get('clarity', False),
            'completeness_improved': improvement_areas.get('completeness', False),
            'specificity_improved': improvement_areas.get('specificity', False),
            'examples_added': improvement_areas.get('examples', False),
            'format_improved': improvement_areas.get('format', False),
            'sections_added': refined.count('##') - original.count('##'),
            'action_verbs_added': self._count_action_verbs(refined) - self._count_action_verbs(original)
        }
    
    def _count_action_verbs(self, text: str) -> int:
        """Count action verbs in the text."""
        action_verbs = ['write', 'create', 'generate', 'analyze', 'explain', 'describe', 'list', 'summarize']
        count = 0
        for verb in action_verbs:
            count += len(re.findall(r'\b' + re.escape(verb) + r'\b', text, re.IGNORECASE))
        return count
    
    def _calculate_refinement_confidence(self, improvements: Dict[str, Any]) -> float:
        """Calculate confidence in the refinement quality."""
        confidence = 0.5  # Base confidence
        
        # Increase confidence based on improvements made
        if improvements.get('structure_improved'):
            confidence += 0.15
        if improvements.get('clarity_improved'):
            confidence += 0.15
        if improvements.get('completeness_improved'):
            confidence += 0.1
        if improvements.get('specificity_improved'):
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _generate_refinement_suggestions(self, improvements: Dict[str, Any]) -> List[str]:
        """Generate suggestions for further refinement."""
        suggestions = []
        
        if not improvements.get('structure_improved'):
            suggestions.append("Consider adding clear section headers to organize the prompt better")
        
        if not improvements.get('examples_added'):
            suggestions.append("Add specific examples to clarify expected output")
        
        if improvements.get('length_change', 0) > 500:
            suggestions.append("Consider condensing the prompt to maintain focus")
        
        if improvements.get('action_verbs_added', 0) == 0:
            suggestions.append("Add more specific action verbs to make instructions clearer")
        
        return suggestions
    
    def _assess_refinement_quality(self, improvements: Dict[str, Any]) -> str:
        """Assess the overall quality of the refinement."""
        improvement_count = sum(1 for key, value in improvements.items() 
                              if key.endswith('_improved') and value)
        
        if improvement_count >= 4:
            return "Significant improvements made across multiple areas"
        elif improvement_count >= 2:
            return "Moderate improvements made to key areas"
        elif improvement_count >= 1:
            return "Minor improvements made"
        else:
            return "Limited improvements - consider more substantial changes"
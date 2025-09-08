"""
ValidatorAgent for syntax and logical consistency checking.

This agent specializes in validating prompt syntax, logical consistency,
and ensuring the prompt meets quality standards before execution.
"""

import re
from typing import Dict, Any, List, Optional, Set
from agents.base import Agent, AgentResult
from models import PromptIteration, UserFeedback


class ValidatorAgent(Agent):
    """
    Agent that validates prompt syntax and logical consistency.
    
    Focuses on:
    - Syntax validation and formatting
    - Logical consistency and coherence
    - Completeness verification
    - Quality assurance checks
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the ValidatorAgent."""
        super().__init__("ValidatorAgent", config)
        
        # Validation thresholds
        self.min_prompt_length = self.config.get('min_prompt_length', 10)
        self.max_prompt_length = self.config.get('max_prompt_length', 5000)
        self.min_quality_score = self.config.get('min_quality_score', 0.6)
        self.strict_mode = self.config.get('strict_mode', False)
    
    def process(self, 
                prompt: str, 
                context: Optional[Dict[str, Any]] = None,
                history: Optional[List[PromptIteration]] = None,
                feedback: Optional[UserFeedback] = None) -> AgentResult:
        """
        Validate prompt syntax and logical consistency.
        
        Args:
            prompt: The prompt text to validate
            context: Optional context about the prompt's intended use
            history: Optional list of previous prompt iterations
            feedback: Optional user feedback from previous iterations
            
        Returns:
            AgentResult containing validation results and issues found
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
            # Perform comprehensive validation
            syntax_validation = self._validate_syntax(prompt)
            logical_validation = self._validate_logical_consistency(prompt)
            completeness_validation = self._validate_completeness(prompt, context)
            quality_validation = self._validate_quality(prompt)
            
            # Check for common issues
            common_issues = self._check_common_issues(prompt)
            
            # Validate against previous iterations if available
            regression_check = self._check_regression(prompt, history) if history else {}
            
            # Calculate overall validation score
            validation_score = self._calculate_validation_score(
                syntax_validation, logical_validation, completeness_validation, quality_validation
            )
            
            # Determine if prompt passes validation
            passes_validation = validation_score >= self.min_quality_score
            
            # Generate validation suggestions
            suggestions = self._generate_validation_suggestions(
                syntax_validation, logical_validation, completeness_validation, 
                quality_validation, common_issues
            )
            
            # Compile validation analysis
            analysis = {
                'syntax': syntax_validation,
                'logical_consistency': logical_validation,
                'completeness': completeness_validation,
                'quality': quality_validation,
                'common_issues': common_issues,
                'regression_check': regression_check,
                'validation_score': validation_score,
                'passes_validation': passes_validation,
                'validation_summary': self._generate_validation_summary(
                    validation_score, passes_validation, len(suggestions)
                )
            }
            
            return AgentResult(
                agent_name=self.name,
                success=True,
                analysis=analysis,
                suggestions=suggestions,
                confidence_score=validation_score
            )
            
        except Exception as e:
            return AgentResult(
                agent_name=self.name,
                success=False,
                analysis={},
                suggestions=[],
                confidence_score=0.0,
                error_message=f"Validation failed: {str(e)}"
            )
    
    def _validate_syntax(self, prompt: str) -> Dict[str, Any]:
        """Validate the syntax and formatting of the prompt."""
        issues = []
        
        # Check length constraints
        if len(prompt) < self.min_prompt_length:
            issues.append(f"Prompt too short (minimum {self.min_prompt_length} characters)")
        elif len(prompt) > self.max_prompt_length:
            issues.append(f"Prompt too long (maximum {self.max_prompt_length} characters)")
        
        # For very short prompts, add more issues to ensure they fail validation
        if len(prompt) < 20:
            issues.append("Prompt lacks sufficient detail and context")
            issues.append("No clear task definition provided")
        
        # Check for balanced brackets and quotes
        bracket_pairs = {'(': ')', '[': ']', '{': '}'}
        quote_chars = ['"', "'", '`']
        
        for open_char, close_char in bracket_pairs.items():
            open_count = prompt.count(open_char)
            close_count = prompt.count(close_char)
            if open_count != close_count:
                issues.append(f"Unbalanced {open_char}{close_char} brackets")
        
        for quote_char in quote_chars:
            if prompt.count(quote_char) % 2 != 0:
                issues.append(f"Unbalanced {quote_char} quotes")
        
        # Check for proper sentence structure
        sentences = re.split(r'[.!?]+', prompt)
        incomplete_sentences = []
        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if sentence and not re.match(r'^[A-Z]', sentence):
                incomplete_sentences.append(i + 1)
        
        if incomplete_sentences:
            issues.append(f"Sentences not starting with capital letters: {incomplete_sentences}")
        
        # Check for excessive whitespace
        if re.search(r'\s{3,}', prompt):
            issues.append("Excessive whitespace found")
        
        # Check for trailing whitespace
        lines = prompt.split('\n')
        trailing_whitespace_lines = [i + 1 for i, line in enumerate(lines) if line.rstrip() != line]
        if trailing_whitespace_lines:
            issues.append(f"Trailing whitespace on lines: {trailing_whitespace_lines}")
        
        return {
            'issues': issues,
            'syntax_score': 1.0 - (len(issues) * 0.1),
            'character_count': len(prompt),
            'line_count': len(lines),
            'sentence_count': len([s for s in sentences if s.strip()])
        }
    
    def _validate_logical_consistency(self, prompt: str) -> Dict[str, Any]:
        """Validate the logical consistency and coherence of the prompt."""
        issues = []
        
        # Check for contradictory instructions
        contradictions = self._find_contradictions(prompt)
        if contradictions:
            issues.extend([f"Contradiction found: {c}" for c in contradictions])
        
        # Check for circular references
        circular_refs = self._find_circular_references(prompt)
        if circular_refs:
            issues.extend([f"Circular reference: {ref}" for ref in circular_refs])
        
        # Check for incomplete instructions
        incomplete_instructions = self._find_incomplete_instructions(prompt)
        if incomplete_instructions:
            issues.extend([f"Incomplete instruction: {inst}" for inst in incomplete_instructions])
        
        # Check for logical flow
        flow_issues = self._check_logical_flow(prompt)
        if flow_issues:
            issues.extend(flow_issues)
        
        return {
            'issues': issues,
            'consistency_score': 1.0 - (len(issues) * 0.15),
            'contradictions': contradictions,
            'circular_references': circular_refs,
            'incomplete_instructions': incomplete_instructions
        }
    
    def _validate_completeness(self, prompt: str, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate the completeness of the prompt."""
        missing_components = []
        
        # Check for essential components
        if not self._has_clear_task(prompt):
            missing_components.append("Clear task definition")
        
        if not self._has_context_info(prompt) and not context:
            missing_components.append("Context or background information")
        
        if not self._has_success_criteria(prompt):
            missing_components.append("Success criteria or requirements")
        
        if not self._has_output_specification(prompt):
            missing_components.append("Output format specification")
        
        # Check context alignment if provided
        context_alignment_score = 1.0
        if context:
            intended_use = context.get('intended_use', '')
            target_audience = context.get('target_audience', '')
            
            if intended_use and not self._aligns_with_use_case(prompt, intended_use):
                missing_components.append(f"Alignment with intended use: {intended_use}")
                context_alignment_score -= 0.3
            
            if target_audience and not self._appropriate_for_audience(prompt, target_audience):
                missing_components.append(f"Appropriate language for audience: {target_audience}")
                context_alignment_score -= 0.2
        
        return {
            'missing_components': missing_components,
            'completeness_score': max(0.0, 1.0 - (len(missing_components) * 0.2)),
            'context_alignment_score': max(0.0, context_alignment_score),
            'has_task': self._has_clear_task(prompt),
            'has_context': self._has_context_info(prompt),
            'has_criteria': self._has_success_criteria(prompt),
            'has_output_spec': self._has_output_specification(prompt)
        }
    
    def _validate_quality(self, prompt: str) -> Dict[str, Any]:
        """Validate the overall quality of the prompt."""
        quality_issues = []
        
        # Check readability
        readability_score = self._calculate_readability(prompt)
        if readability_score < 0.6:
            quality_issues.append("Poor readability - consider simplifying language")
        
        # Check specificity
        specificity_score = self._calculate_specificity(prompt)
        if specificity_score < 0.5:
            quality_issues.append("Lacks specificity - add more detailed instructions")
        
        # Check actionability
        actionability_score = self._calculate_actionability(prompt)
        if actionability_score < 0.6:
            quality_issues.append("Not actionable enough - add clear action items")
        
        # Check for best practices
        best_practice_issues = self._check_best_practices(prompt)
        quality_issues.extend(best_practice_issues)
        
        overall_quality = (readability_score + specificity_score + actionability_score) / 3
        
        return {
            'quality_issues': quality_issues,
            'overall_quality_score': overall_quality,
            'readability_score': readability_score,
            'specificity_score': specificity_score,
            'actionability_score': actionability_score,
            'best_practice_violations': best_practice_issues
        }
    
    def _find_contradictions(self, prompt: str) -> List[str]:
        """Find contradictory statements in the prompt."""
        contradictions = []
        
        # Look for explicit contradictions
        contradiction_patterns = [
            (r'do not.*but.*do', "Contradictory do/don't instructions"),
            (r'must.*cannot', "Contradictory must/cannot requirements"),
            (r'always.*never', "Contradictory always/never statements"),
            (r'include.*exclude.*same', "Contradictory include/exclude for same item")
        ]
        
        for pattern, description in contradiction_patterns:
            if re.search(pattern, prompt, re.IGNORECASE):
                contradictions.append(description)
        
        return contradictions
    
    def _find_circular_references(self, prompt: str) -> List[str]:
        """Find circular references in the prompt."""
        # This is a simplified check - could be enhanced with more sophisticated analysis
        circular_refs = []
        
        # Look for self-referential patterns
        if re.search(r'use this prompt to.*prompt', prompt, re.IGNORECASE):
            circular_refs.append("Self-referential prompt instruction")
        
        return circular_refs
    
    def _find_incomplete_instructions(self, prompt: str) -> List[str]:
        """Find incomplete or ambiguous instructions."""
        incomplete = []
        
        # Look for incomplete patterns
        incomplete_patterns = [
            r'such as\s*$',
            r'for example\s*$',
            r'including\s*$',
            r'like\s*$'
        ]
        
        for pattern in incomplete_patterns:
            if re.search(pattern, prompt, re.IGNORECASE | re.MULTILINE):
                incomplete.append(f"Incomplete instruction ending with '{pattern}'")
        
        return incomplete
    
    def _check_logical_flow(self, prompt: str) -> List[str]:
        """Check the logical flow of the prompt."""
        flow_issues = []
        
        # Check if instructions are in logical order
        sentences = re.split(r'[.!?]+', prompt)
        
        # Look for temporal inconsistencies
        temporal_words = ['first', 'then', 'next', 'finally', 'before', 'after']
        temporal_sentences = []
        
        for i, sentence in enumerate(sentences):
            for word in temporal_words:
                if word in sentence.lower():
                    temporal_sentences.append((i, word, sentence.strip()))
        
        # Check if temporal order makes sense (simplified check)
        if len(temporal_sentences) > 1:
            order_words = ['first', 'then', 'next', 'finally']
            found_order = [word for _, word, _ in temporal_sentences if word in order_words]
            expected_order = [word for word in order_words if word in found_order]
            
            if found_order != expected_order:
                flow_issues.append("Temporal instructions may be out of logical order")
        
        return flow_issues
    
    def _has_clear_task(self, prompt: str) -> bool:
        """Check if the prompt has a clear task definition."""
        task_indicators = ['write', 'create', 'generate', 'analyze', 'explain', 'describe', 'list', 'summarize']
        return any(indicator in prompt.lower() for indicator in task_indicators)
    
    def _has_context_info(self, prompt: str) -> bool:
        """Check if the prompt has context information."""
        context_indicators = ['context:', 'background:', 'given:', 'scenario:', 'situation:']
        return any(indicator in prompt.lower() for indicator in context_indicators)
    
    def _has_success_criteria(self, prompt: str) -> bool:
        """Check if the prompt has success criteria."""
        criteria_indicators = ['must', 'should', 'requirement', 'criteria', 'ensure', 'make sure']
        return any(indicator in prompt.lower() for indicator in criteria_indicators)
    
    def _has_output_specification(self, prompt: str) -> bool:
        """Check if the prompt specifies output format."""
        output_indicators = ['format:', 'output:', 'response:', 'json', 'list', 'paragraph', 'table']
        return any(indicator in prompt.lower() for indicator in output_indicators)
    
    def _aligns_with_use_case(self, prompt: str, intended_use: str) -> bool:
        """Check if prompt aligns with intended use case."""
        # Simple keyword matching - could be enhanced with semantic analysis
        use_keywords = intended_use.lower().split()
        prompt_lower = prompt.lower()
        
        matching_keywords = sum(1 for keyword in use_keywords if keyword in prompt_lower)
        return matching_keywords >= len(use_keywords) * 0.5
    
    def _appropriate_for_audience(self, prompt: str, target_audience: str) -> bool:
        """Check if prompt language is appropriate for target audience."""
        # Simplified check based on audience type
        audience_lower = target_audience.lower()
        
        if 'technical' in audience_lower or 'developer' in audience_lower:
            # Technical audience - technical terms are okay
            return True
        elif 'general' in audience_lower or 'non-technical' in audience_lower:
            # General audience - check for overly technical language
            technical_terms = ['api', 'json', 'xml', 'sql', 'regex', 'algorithm']
            technical_count = sum(1 for term in technical_terms if term in prompt.lower())
            return technical_count <= 2
        
        return True  # Default to appropriate if audience type unclear
    
    def _calculate_readability(self, prompt: str) -> float:
        """Calculate readability score (simplified)."""
        words = prompt.split()
        sentences = re.split(r'[.!?]+', prompt)
        
        if not sentences or not words:
            return 0.0
        
        avg_sentence_length = len(words) / len(sentences)
        
        # Penalize very long sentences
        if avg_sentence_length > 25:
            return 0.3
        elif avg_sentence_length > 20:
            return 0.5
        elif avg_sentence_length > 15:
            return 0.7
        else:
            return 0.9
    
    def _calculate_specificity(self, prompt: str) -> float:
        """Calculate specificity score."""
        # Count specific vs vague terms
        specific_terms = ['exactly', 'precisely', 'specifically', 'detailed', 'comprehensive']
        vague_terms = ['thing', 'stuff', 'something', 'good', 'bad', 'nice']
        
        specific_count = sum(1 for term in specific_terms if term in prompt.lower())
        vague_count = sum(1 for term in vague_terms if term in prompt.lower())
        
        if specific_count + vague_count == 0:
            return 0.7  # Neutral score
        
        return specific_count / (specific_count + vague_count)
    
    def _calculate_actionability(self, prompt: str) -> float:
        """Calculate actionability score."""
        action_verbs = ['write', 'create', 'generate', 'analyze', 'explain', 'describe', 'list', 'summarize']
        action_count = sum(1 for verb in action_verbs if verb in prompt.lower())
        
        # Score based on number of clear actions
        if action_count >= 3:
            return 1.0
        elif action_count >= 2:
            return 0.8
        elif action_count >= 1:
            return 0.6
        else:
            return 0.2
    
    def _check_best_practices(self, prompt: str) -> List[str]:
        """Check adherence to prompt engineering best practices."""
        violations = []
        
        # Check for negative instructions (what not to do)
        negative_count = len(re.findall(r'\bdo not\b|\bdon\'t\b|\bavoid\b|\bnever\b', prompt, re.IGNORECASE))
        if negative_count > 2:
            violations.append("Too many negative instructions - focus on what TO do")
        
        # Check for multiple tasks in one prompt
        task_verbs = ['write', 'create', 'generate', 'analyze', 'explain', 'describe', 'list', 'summarize']
        task_count = sum(1 for verb in task_verbs if verb in prompt.lower())
        if task_count > 3:
            violations.append("Multiple tasks in one prompt - consider splitting")
        
        # Check for unclear pronouns
        unclear_pronouns = ['it', 'this', 'that', 'these', 'those']
        pronoun_count = sum(1 for pronoun in unclear_pronouns if f' {pronoun} ' in prompt.lower())
        if pronoun_count > 3:
            violations.append("Unclear pronoun usage - be more specific")
        
        return violations
    
    def _check_common_issues(self, prompt: str) -> List[str]:
        """Check for common prompt issues."""
        issues = []
        
        # Check for typos (simplified)
        common_typos = {
            'teh': 'the',
            'adn': 'and',
            'recieve': 'receive',
            'seperate': 'separate'
        }
        
        for typo, correct in common_typos.items():
            if typo in prompt.lower():
                issues.append(f"Possible typo: '{typo}' should be '{correct}'")
        
        # Check for formatting issues
        if prompt.count('\n\n') > prompt.count('\n') * 0.3:
            issues.append("Excessive blank lines")
        
        return issues
    
    def _check_regression(self, prompt: str, history: List[PromptIteration]) -> Dict[str, Any]:
        """Check if current prompt is worse than previous versions."""
        if not history:
            return {}
        
        # Compare with the most recent iteration
        latest = history[-1]
        if not latest.evaluation_scores:
            return {}
        
        # This is a simplified regression check
        # In practice, you'd want more sophisticated comparison
        regression_issues = []
        
        # Check if prompt became significantly shorter (potential information loss)
        if latest.prompt_text and len(prompt) < len(latest.prompt_text) * 0.7:
            regression_issues.append("Prompt significantly shorter than previous version")
        
        return {
            'regression_issues': regression_issues,
            'has_regression': len(regression_issues) > 0
        }
    
    def _calculate_validation_score(self, syntax: Dict[str, Any], logical: Dict[str, Any],
                                  completeness: Dict[str, Any], quality: Dict[str, Any]) -> float:
        """Calculate overall validation score."""
        syntax_score = syntax.get('syntax_score', 0.0)
        logical_score = logical.get('consistency_score', 0.0)
        completeness_score = completeness.get('completeness_score', 0.0)
        quality_score = quality.get('overall_quality_score', 0.0)
        
        # Weighted average
        weights = {'syntax': 0.2, 'logical': 0.3, 'completeness': 0.3, 'quality': 0.2}
        
        total_score = (
            syntax_score * weights['syntax'] +
            logical_score * weights['logical'] +
            completeness_score * weights['completeness'] +
            quality_score * weights['quality']
        )
        
        return max(0.0, min(1.0, total_score))
    
    def _generate_validation_suggestions(self, syntax: Dict[str, Any], logical: Dict[str, Any],
                                       completeness: Dict[str, Any], quality: Dict[str, Any],
                                       common_issues: List[str]) -> List[str]:
        """Generate suggestions based on validation results."""
        suggestions = []
        
        # Syntax suggestions
        for issue in syntax.get('issues', []):
            suggestions.append(f"Syntax: {issue}")
        
        # Logical consistency suggestions
        for issue in logical.get('issues', []):
            suggestions.append(f"Logic: {issue}")
        
        # Completeness suggestions
        for component in completeness.get('missing_components', []):
            suggestions.append(f"Add missing component: {component}")
        
        # Quality suggestions
        for issue in quality.get('quality_issues', []):
            suggestions.append(f"Quality: {issue}")
        
        # Common issues
        for issue in common_issues:
            suggestions.append(f"Issue: {issue}")
        
        return suggestions
    
    def _generate_validation_summary(self, score: float, passes: bool, issue_count: int) -> str:
        """Generate a summary of the validation results."""
        if passes:
            if score >= 0.9:
                return f"Excellent prompt quality (score: {score:.2f}) - ready for use"
            elif score >= 0.8:
                return f"Good prompt quality (score: {score:.2f}) - minor improvements possible"
            else:
                return f"Acceptable prompt quality (score: {score:.2f}) - passes validation"
        else:
            return f"Prompt needs improvement (score: {score:.2f}) - {issue_count} issues found"
"""
LLMValidatorAgent for comprehensive prompt validation using LLM reasoning.

This agent leverages Large Language Models with specialized system prompts
to perform sophisticated validation of prompt syntax, logical consistency,
completeness, and adherence to best practices using intelligent reasoning.
"""

from typing import Dict, Any, List, Optional
from agents.llm_agent import LLMAgent
from agents.base import AgentResult
from models import PromptIteration, UserFeedback


class LLMValidatorAgent(LLMAgent):
    """
    LLM-enhanced agent for comprehensive prompt validation.
    
    Uses specialized system prompts and reasoning frameworks to validate:
    - Syntax and formatting with intelligent error detection
    - Logical consistency and coherence with reasoning
    - Completeness against comprehensive criteria
    - Quality assurance with best practices validation
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the LLMValidatorAgent."""
        super().__init__("LLMValidatorAgent", config)
        
        # Validation-specific configuration
        self.validation_strictness = self.config.get('validation_strictness', 'moderate')
        self.validation_criteria = self.config.get('validation_criteria', 
            ['syntax', 'logic', 'completeness', 'quality', 'best_practices'])
        self.min_quality_threshold = self.config.get('min_quality_threshold', 0.6)
        self.comprehensive_analysis = self.config.get('comprehensive_analysis', True)
    
    def process(self, 
                prompt: str, 
                context: Optional[Dict[str, Any]] = None,
                history: Optional[List[PromptIteration]] = None,
                feedback: Optional[UserFeedback] = None) -> AgentResult:
        """
        Perform LLM-enhanced comprehensive validation of the prompt.
        
        Args:
            prompt: The prompt text to validate
            context: Optional context about the prompt's intended use
            history: Optional list of previous prompt iterations
            feedback: Optional user feedback from previous iterations
            
        Returns:
            AgentResult containing comprehensive validation results and recommendations
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
            # Prepare validation prompt for LLM
            validation_prompt = self._build_validation_prompt(prompt, context, history, feedback)
            
            # Call LLM for validation
            llm_response = self._call_llm(validation_prompt, context)
            
            if not llm_response['success']:
                return AgentResult(
                    agent_name=self.name,
                    success=False,
                    analysis={},
                    suggestions=[],
                    confidence_score=0.0,
                    error_message=f"LLM validation failed: {llm_response['error']}"
                )
            
            # Parse and structure the LLM response
            parsed_response = self._parse_llm_response(llm_response['response'])
            
            # Extract validation results
            validation_results = self._extract_validation_results(parsed_response, llm_response)
            
            # Determine overall validation status
            passes_validation = self._determine_validation_status(validation_results)
            
            # Generate validation suggestions
            suggestions = self._extract_validation_suggestions(parsed_response)
            
            # Calculate confidence in validation results
            confidence_score = self._calculate_validation_confidence(
                parsed_response, validation_results
            )
            
            # Compile comprehensive analysis
            analysis = {
                'llm_validation': {
                    'raw_response': parsed_response['raw_response'],
                    'model_used': llm_response['model'],
                    'tokens_used': llm_response['tokens_used'],
                    'reasoning': parsed_response.get('reasoning', ''),
                    'confidence': parsed_response.get('confidence', 0.8)
                },
                'validation_results': validation_results,
                'passes_validation': passes_validation,
                'validation_summary': self._generate_validation_summary(validation_results, passes_validation),
                'quality_assessment': self._assess_overall_quality(validation_results),
                'best_practices_compliance': self._assess_best_practices_compliance(validation_results)
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
                error_message=f"LLM validation failed: {str(e)}"
            )
    
    def _get_base_system_prompt(self) -> str:
        """Get the base system prompt for LLM validation."""
        return """You are an expert prompt validation specialist with deep knowledge of prompt engineering standards, best practices, and quality assurance. Your role is to comprehensively validate prompts for correctness, effectiveness, and adherence to established standards.

Your validation should be:
- Systematic and thorough across multiple dimensions
- Based on established prompt engineering principles and standards
- Focused on identifying both critical issues and improvement opportunities
- Contextually aware of the intended use case and requirements
- Balanced between strictness and practical usability

Provide detailed analysis with clear reasoning for all validation decisions and recommendations."""
    
    def _get_best_practices_prompt(self) -> str:
        """Get the best practices prompt for validation."""
        return """Validate against these comprehensive prompt engineering standards:

SYNTAX AND FORMATTING STANDARDS:
- Proper grammar, spelling, and punctuation
- Consistent formatting and structure
- Balanced brackets, quotes, and special characters
- Appropriate use of markdown and formatting elements
- Clear section organization and hierarchy

LOGICAL CONSISTENCY STANDARDS:
- No contradictory instructions or requirements
- Logical flow and sequence of information
- Coherent relationships between different parts
- Absence of circular references or dependencies
- Clear cause-and-effect relationships

COMPLETENESS STANDARDS:
- Clear task definition with specific objectives
- Sufficient context and background information
- Explicit success criteria and evaluation metrics
- Proper input/output specifications
- Comprehensive coverage of requirements and constraints

QUALITY STANDARDS:
- Appropriate complexity for target audience
- Clear and unambiguous language
- Actionable and specific instructions
- Optimal length and information density
- Effective use of examples and illustrations

BEST PRACTICES COMPLIANCE:
- Adherence to established prompt engineering patterns
- Proper use of prompting techniques (few-shot, chain-of-thought, etc.)
- Appropriate role definition and context setting
- Effective constraint specification and boundary setting
- Optimal structure for the intended LLM and use case"""
    
    def _get_reasoning_framework_prompt(self) -> str:
        """Get the reasoning framework for structured validation."""
        return """Use this comprehensive validation framework:

1. INITIAL ASSESSMENT
   - Overall prompt quality and readiness
   - Immediate red flags or critical issues
   - Alignment with stated objectives and context

2. SYSTEMATIC VALIDATION
   - Syntax: Grammar, formatting, structure, consistency
   - Logic: Coherence, flow, contradictions, dependencies
   - Completeness: Coverage, missing elements, sufficiency
   - Quality: Clarity, specificity, actionability, effectiveness

3. BEST PRACTICES EVALUATION
   - Adherence to prompt engineering standards
   - Use of established patterns and techniques
   - Optimization for target LLM and use case
   - Compliance with domain-specific guidelines

4. CONTEXTUAL VALIDATION
   - Appropriateness for intended use case
   - Suitability for target audience
   - Alignment with stated constraints and requirements
   - Consideration of deployment environment

5. COMPREHENSIVE SCORING
   - Individual dimension scores with justification
   - Overall quality score and pass/fail determination
   - Confidence assessment for validation results
   - Priority ranking of identified issues

6. RECOMMENDATIONS GENERATION
   - Specific, actionable improvement suggestions
   - Priority-based issue resolution guidance
   - Best practices implementation recommendations
   - Quality enhancement opportunities

Provide detailed reasoning for all assessments and clear justification for pass/fail decisions."""
    
    def _build_validation_prompt(self, prompt: str, context: Optional[Dict[str, Any]], 
                               history: Optional[List[PromptIteration]], 
                               feedback: Optional[UserFeedback]) -> str:
        """Build the validation prompt for the LLM."""
        validation_prompt_parts = [
            "Please perform a comprehensive validation of the following prompt using your expertise in prompt engineering standards and best practices:",
            f"\n--- PROMPT TO VALIDATE ---\n{prompt}\n--- END PROMPT ---\n"
        ]
        
        # Add context information
        if context:
            context_info = ["VALIDATION CONTEXT:"]
            if 'intended_use' in context:
                context_info.append(f"Intended Use: {context['intended_use']}")
            if 'target_audience' in context:
                context_info.append(f"Target Audience: {context['target_audience']}")
            if 'domain' in context:
                context_info.append(f"Domain: {context['domain']}")
            if 'quality_requirements' in context:
                context_info.append(f"Quality Requirements: {context['quality_requirements']}")
            if 'constraints' in context:
                context_info.append(f"Constraints: {context['constraints']}")
            
            validation_prompt_parts.extend(context_info)
        
        # Add validation criteria
        validation_prompt_parts.extend([
            f"\nVALIDATION CRITERIA:",
            f"Strictness Level: {self.validation_strictness}",
            f"Quality Threshold: {self.min_quality_threshold}",
            f"Validation Areas: {', '.join(self.validation_criteria)}"
        ])
        
        # Add history context if available
        if history and len(history) > 0:
            validation_prompt_parts.append(f"\nITERATION CONTEXT: This is iteration {len(history) + 1} in an improvement process")
            
            # Include previous validation results if available
            latest_iteration = history[-1]
            if latest_iteration.agent_analysis and 'ValidatorAgent' in latest_iteration.agent_analysis:
                validation_prompt_parts.append("Previous validation identified issues - check for resolution")
        
        # Add user feedback
        if feedback:
            feedback_info = [
                f"\nUSER FEEDBACK:",
                f"Satisfaction Rating: {feedback.satisfaction_rating}/5"
            ]
            if feedback.specific_issues:
                feedback_info.append(f"Reported Issues: {', '.join(feedback.specific_issues)}")
            if feedback.desired_improvements:
                feedback_info.append(f"Desired Improvements: {feedback.desired_improvements}")
            
            validation_prompt_parts.extend(feedback_info)
        
        # Add validation instructions
        validation_instructions = [
            "\nVALIDATION REQUIREMENTS:",
            "Perform comprehensive validation across all specified criteria:",
            "",
            "1. SYNTAX VALIDATION",
            "   - Grammar, spelling, punctuation accuracy",
            "   - Formatting consistency and structure",
            "   - Special character usage and balance",
            "",
            "2. LOGICAL CONSISTENCY",
            "   - Instruction coherence and flow",
            "   - Absence of contradictions",
            "   - Logical dependencies and relationships",
            "",
            "3. COMPLETENESS ASSESSMENT",
            "   - Task definition clarity and specificity",
            "   - Context and background sufficiency",
            "   - Success criteria and constraint specification",
            "",
            "4. QUALITY EVALUATION",
            "   - Language clarity and precision",
            "   - Actionability and specificity",
            "   - Appropriate complexity and scope",
            "",
            "5. BEST PRACTICES COMPLIANCE",
            "   - Adherence to prompt engineering standards",
            "   - Use of established techniques and patterns",
            "   - Optimization for effectiveness",
            "",
            "Provide your response in this format:",
            "",
            "VALIDATION RESULTS:",
            "Syntax: [PASS/FAIL] - [Score 0.0-1.0] - [Reasoning]",
            "Logic: [PASS/FAIL] - [Score 0.0-1.0] - [Reasoning]", 
            "Completeness: [PASS/FAIL] - [Score 0.0-1.0] - [Reasoning]",
            "Quality: [PASS/FAIL] - [Score 0.0-1.0] - [Reasoning]",
            "Best Practices: [PASS/FAIL] - [Score 0.0-1.0] - [Reasoning]",
            "",
            "OVERALL: [PASS/FAIL] - [Overall Score 0.0-1.0]",
            "",
            "CRITICAL ISSUES:",
            "[List any critical issues that must be addressed]",
            "",
            "RECOMMENDATIONS:",
            "[Specific, prioritized improvement suggestions]",
            "",
            "CONFIDENCE: [0.0-1.0 with justification]"
        ]
        
        validation_prompt_parts.extend(validation_instructions)
        
        return "\n".join(validation_prompt_parts)
    
    def _extract_validation_results(self, parsed_response: Dict[str, Any], 
                                  llm_response: Dict[str, Any]) -> Dict[str, Any]:
        """Extract structured validation results from LLM response."""
        import re
        
        raw_response = parsed_response['raw_response']
        
        # Extract validation scores and status for each criterion
        validation_results = {}
        
        criteria_patterns = {
            'syntax': r'syntax[:\s]*(\w+)[^\d]*(\d+(?:\.\d+)?)',
            'logic': r'logic[:\s]*(\w+)[^\d]*(\d+(?:\.\d+)?)',
            'completeness': r'completeness[:\s]*(\w+)[^\d]*(\d+(?:\.\d+)?)',
            'quality': r'quality[:\s]*(\w+)[^\d]*(\d+(?:\.\d+)?)',
            'best_practices': r'best practices[:\s]*(\w+)[^\d]*(\d+(?:\.\d+)?)'
        }
        
        for criterion, pattern in criteria_patterns.items():
            match = re.search(pattern, raw_response, re.IGNORECASE)
            if match:
                status = match.group(1).upper()
                score = float(match.group(2))
                # Normalize score to 0-1 range if needed
                if score > 1.0:
                    score = score / 10.0 if score <= 10.0 else score / 100.0
                
                validation_results[criterion] = {
                    'status': status,
                    'score': min(1.0, max(0.0, score)),
                    'passes': status == 'PASS'
                }
            else:
                # Default values if not found
                validation_results[criterion] = {
                    'status': 'UNKNOWN',
                    'score': 0.5,
                    'passes': False
                }
        
        # Extract overall validation result
        overall_match = re.search(r'overall[:\s]*(\w+)[^\d]*(\d+(?:\.\d+)?)', raw_response, re.IGNORECASE)
        if overall_match:
            overall_status = overall_match.group(1).upper()
            overall_score = float(overall_match.group(2))
            if overall_score > 1.0:
                overall_score = overall_score / 10.0 if overall_score <= 10.0 else overall_score / 100.0
        else:
            # Calculate overall from individual scores
            individual_scores = [result['score'] for result in validation_results.values()]
            overall_score = sum(individual_scores) / len(individual_scores) if individual_scores else 0.5
            overall_status = 'PASS' if overall_score >= self.min_quality_threshold else 'FAIL'
        
        validation_results['overall'] = {
            'status': overall_status,
            'score': min(1.0, max(0.0, overall_score)),
            'passes': overall_status == 'PASS'
        }
        
        # Extract critical issues
        critical_issues = self._extract_critical_issues(raw_response)
        validation_results['critical_issues'] = critical_issues
        
        # Extract detailed reasoning
        validation_results['detailed_reasoning'] = self._extract_detailed_reasoning(raw_response)
        
        return validation_results
    
    def _extract_critical_issues(self, response: str) -> List[str]:
        """Extract critical issues from LLM response."""
        import re
        
        # Look for critical issues section
        critical_section_match = re.search(
            r'CRITICAL ISSUES[:\s]*\n(.*?)(?=\n\n|\nRECOMMENDATIONS|\nCONFIDENCE|\Z)',
            response,
            re.DOTALL | re.IGNORECASE
        )
        
        if critical_section_match:
            issues_text = critical_section_match.group(1).strip()
            
            # Check for "None" indicators
            if any(word in issues_text.lower() for word in ['none', 'no issues', 'not identified']):
                return []
            
            # Extract bullet points or numbered items
            issues = re.findall(r'^\s*[\d\-\*]\.\s*(.+)$', issues_text, re.MULTILINE)
            if not issues:
                # Try to extract lines that look like issues
                lines = [line.strip() for line in issues_text.split('\n') if line.strip()]
                issues = [line for line in lines if len(line) > 10 and 'none' not in line.lower()]
            return issues
        
        # Fallback: look for mentions of critical problems
        critical_keywords = ['critical', 'severe', 'major issue', 'must fix', 'blocking']
        critical_issues = []
        
        for keyword in critical_keywords:
            pattern = rf'{keyword}[^.]*\.'
            matches = re.findall(pattern, response, re.IGNORECASE)
            critical_issues.extend(matches)
        
        return critical_issues[:5]  # Limit to top 5 critical issues
    
    def _extract_detailed_reasoning(self, response: str) -> Dict[str, str]:
        """Extract detailed reasoning for each validation criterion."""
        import re
        
        reasoning = {}
        
        # Extract reasoning for each criterion
        criteria = ['syntax', 'logic', 'completeness', 'quality', 'best practices']
        
        for criterion in criteria:
            # Look for reasoning after the criterion
            pattern = rf'{criterion}[:\s]*\w+[^\d]*\d+(?:\.\d+)?[^\n]*\n([^A-Z\n]+)'
            match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
            if match:
                reasoning[criterion] = match.group(1).strip()
            else:
                reasoning[criterion] = f"No detailed reasoning found for {criterion}"
        
        return reasoning
    
    def _determine_validation_status(self, validation_results: Dict[str, Any]) -> bool:
        """Determine overall validation pass/fail status."""
        overall_result = validation_results.get('overall', {})
        
        # Check overall score against threshold
        overall_score = overall_result.get('score', 0.0)
        if overall_score < self.min_quality_threshold:
            return False
        
        # Check for critical issues
        critical_issues = validation_results.get('critical_issues', [])
        if len(critical_issues) > 0 and any(issue.strip() for issue in critical_issues):
            return False
        
        # Check individual criteria based on strictness
        if self.validation_strictness == 'strict':
            # All criteria must pass
            for criterion in ['syntax', 'logic', 'completeness', 'quality', 'best_practices']:
                if criterion in validation_results:
                    if not validation_results[criterion].get('passes', False):
                        return False
        elif self.validation_strictness == 'moderate':
            # Most criteria must pass, allow one failure
            failed_count = 0
            for criterion in ['syntax', 'logic', 'completeness', 'quality', 'best_practices']:
                if criterion in validation_results:
                    if not validation_results[criterion].get('passes', False):
                        failed_count += 1
            if failed_count > 1:
                return False
        # For 'lenient', rely on overall score only
        
        return True
    
    def _extract_validation_suggestions(self, parsed_response: Dict[str, Any]) -> List[str]:
        """Extract validation suggestions from LLM response."""
        suggestions = parsed_response.get('recommendations', [])
        
        # Look for recommendations section
        import re
        raw_response = parsed_response['raw_response']
        
        recommendations_match = re.search(
            r'RECOMMENDATIONS[:\s]*\n(.*?)(?=\n\n|\nCONFIDENCE|\Z)',
            raw_response,
            re.DOTALL | re.IGNORECASE
        )
        
        if recommendations_match:
            recommendations_text = recommendations_match.group(1)
            # Extract bullet points or numbered items
            items = re.findall(r'^\s*[\d\-\*]\.\s*(.+)$', recommendations_text, re.MULTILINE)
            suggestions.extend(items)
        
        # Remove duplicates while preserving order
        unique_suggestions = []
        seen = set()
        for suggestion in suggestions:
            if suggestion not in seen:
                unique_suggestions.append(suggestion)
                seen.add(suggestion)
        
        return unique_suggestions[:10]  # Limit to top 10 suggestions
    
    def _calculate_validation_confidence(self, parsed_response: Dict[str, Any], 
                                       validation_results: Dict[str, Any]) -> float:
        """Calculate confidence in validation results."""
        base_confidence = parsed_response.get('confidence', 0.8)
        
        # Adjust based on validation completeness
        overall_score = validation_results.get('overall', {}).get('score', 0.5)
        
        # Higher confidence for clear pass/fail results
        if overall_score > 0.8 or overall_score < 0.3:
            base_confidence += 0.1
        
        # Lower confidence for borderline results
        if 0.4 <= overall_score <= 0.7:
            base_confidence -= 0.1
        
        # Adjust based on response quality
        response_text = parsed_response['raw_response']
        
        # Higher confidence for detailed reasoning
        detailed_reasoning = validation_results.get('detailed_reasoning', {})
        if sum(len(reasoning) for reasoning in detailed_reasoning.values()) > 500:
            base_confidence += 0.1
        
        # Higher confidence for specific critical issues identification
        critical_issues = validation_results.get('critical_issues', [])
        if len(critical_issues) > 0:
            base_confidence += 0.05
        
        return min(1.0, max(0.0, base_confidence))
    
    def _generate_validation_summary(self, validation_results: Dict[str, Any], 
                                   passes_validation: bool) -> str:
        """Generate a summary of validation results."""
        overall_score = validation_results.get('overall', {}).get('score', 0.0)
        critical_issues_count = len(validation_results.get('critical_issues', []))
        
        if passes_validation:
            if overall_score >= 0.9:
                return f"Excellent prompt quality (score: {overall_score:.2f}) - exceeds all validation criteria"
            elif overall_score >= 0.8:
                return f"High prompt quality (score: {overall_score:.2f}) - meets all validation standards"
            else:
                return f"Acceptable prompt quality (score: {overall_score:.2f}) - passes validation requirements"
        else:
            if critical_issues_count > 0:
                return f"Validation failed (score: {overall_score:.2f}) - {critical_issues_count} critical issues must be resolved"
            else:
                return f"Validation failed (score: {overall_score:.2f}) - quality improvements needed"
    
    def _assess_overall_quality(self, validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall quality across all validation dimensions."""
        scores = {}
        for criterion in ['syntax', 'logic', 'completeness', 'quality', 'best_practices']:
            if criterion in validation_results:
                scores[criterion] = validation_results[criterion].get('score', 0.0)
        
        overall_score = validation_results.get('overall', {}).get('score', 0.0)
        
        # Calculate quality metrics
        avg_score = sum(scores.values()) / len(scores) if scores else 0.0
        min_score = min(scores.values()) if scores else 0.0
        max_score = max(scores.values()) if scores else 0.0
        
        quality_level = "Excellent" if overall_score >= 0.9 else \
                       "Good" if overall_score >= 0.7 else \
                       "Fair" if overall_score >= 0.5 else \
                       "Poor"
        
        return {
            'overall_score': overall_score,
            'average_score': avg_score,
            'minimum_score': min_score,
            'maximum_score': max_score,
            'quality_level': quality_level,
            'individual_scores': scores,
            'score_consistency': max_score - min_score if scores else 0.0
        }
    
    def _assess_best_practices_compliance(self, validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Assess compliance with prompt engineering best practices."""
        best_practices_result = validation_results.get('best_practices', {})
        
        compliance_score = best_practices_result.get('score', 0.0)
        compliance_status = best_practices_result.get('status', 'UNKNOWN')
        
        compliance_level = "Full Compliance" if compliance_score >= 0.9 else \
                          "High Compliance" if compliance_score >= 0.7 else \
                          "Moderate Compliance" if compliance_score >= 0.5 else \
                          "Low Compliance"
        
        return {
            'compliance_score': compliance_score,
            'compliance_status': compliance_status,
            'compliance_level': compliance_level,
            'passes_best_practices': best_practices_result.get('passes', False)
        }
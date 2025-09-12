"""
LLMRefinerAgent for intelligent prompt refinement using LLM capabilities.

This agent leverages Large Language Models with specialized system prompts
to generate sophisticated prompt improvements based on analysis results,
user feedback, and embedded prompt engineering best practices.
"""

from typing import Dict, Any, List, Optional
from agents.llm_agent import LLMAgent
from agents.llm_agent_logger import LLMAgentLogger
from agents.base import AgentResult
from models import PromptIteration, UserFeedback


class LLMRefinerAgent(LLMAgent):
    """
    LLM-enhanced agent for intelligent prompt refinement.
    
    Uses specialized system prompts and reasoning frameworks to:
    - Generate improved prompt versions with sophisticated reasoning
    - Apply prompt engineering best practices systematically
    - Incorporate user feedback and analysis results intelligently
    - Provide detailed explanations for refinement decisions
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the LLMRefinerAgent."""
        super().__init__("LLMRefinerAgent", config)
        
        # Initialize specialized logger for LLM interactions
        self.llm_logger = LLMAgentLogger("LLMRefinerAgent")
        
        # Refinement-specific configuration
        self.refinement_style = self.config.get('refinement_style', 'comprehensive')
        self.preserve_intent = self.config.get('preserve_intent', True)
        self.max_refinement_iterations = self.config.get('max_refinement_iterations', 3)
        self.improvement_focus = self.config.get('improvement_focus', ['structure', 'clarity', 'completeness'])
    
    def process(self, 
                prompt: str, 
                context: Optional[Dict[str, Any]] = None,
                history: Optional[List[PromptIteration]] = None,
                feedback: Optional[UserFeedback] = None) -> AgentResult:
        """
        Generate an LLM-enhanced refined version of the prompt.
        
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
            # Extract session context for logging
            session_id = context.get('session_id') if context else None
            iteration = context.get('iteration') if context else None
            
            # Prepare refinement prompt for LLM
            refinement_prompt = self._build_refinement_prompt(prompt, context, history, feedback)
            
            # Log the LLM call with refinement context
            self.llm_logger.log_llm_call(
                prompt=refinement_prompt,
                context=context,
                session_id=session_id,
                iteration=iteration,
                model_config={'agent_type': 'refiner', 'task': 'prompt_refinement'}
            )
            
            # Call LLM for refinement
            llm_response = self._call_llm(refinement_prompt, context)
            
            # Log the LLM response
            self.llm_logger.log_llm_response(
                response=llm_response,
                session_id=session_id,
                iteration=iteration
            )
            
            # Check if fallback should be used
            # Check if LLM call failed and handle error
            if not llm_response['success']:
                self.llm_logger.log_error(
                    error_type="llm_refinement_failed",
                    error_message=llm_response.get('error', 'Unknown LLM error'),
                    context={'session_id': session_id, 'iteration': iteration}
                )
                self._handle_llm_failure(prompt, context, history, feedback, 
                                         llm_response.get('error', 'LLM service unavailable'))
            
            # Parse and extract the refined prompt
            parsed_response = self._parse_llm_response(llm_response['response'])
            
            # Log parsing results
            parsing_success = parsed_response.get('success', True)
            parsing_errors = parsed_response.get('errors', [])
            self.llm_logger.log_parsed_response(
                parsed_data=parsed_response,
                session_id=session_id,
                iteration=iteration,
                parsing_success=parsing_success,
                parsing_errors=parsing_errors
            )
            
            # Log raw LLM feedback for orchestration debugging
            self.llm_logger.log_orchestration_raw_feedback(
                agent_name='LLMRefinerAgent',
                raw_llm_response=llm_response['response'],
                parsed_data=parsed_response,
                session_id=session_id,
                iteration=iteration
            )
            
            # Extract the refined prompt and improvements
            refined_prompt = self._extract_refined_prompt(parsed_response)
            
            # Log refined prompt extraction
            extraction_success = refined_prompt != "No refined prompt could be extracted from LLM response."
            self.llm_logger.log_component_extraction(
                component_type="refined_prompt",
                extracted_data=refined_prompt,
                success=extraction_success,
                extraction_method="regex_pattern_matching"
            )
            
            # Analyze the improvements made
            improvements_analysis = self._analyze_refinement_quality(
                prompt, refined_prompt, parsed_response, llm_response
            )
            
            # Log refinement quality analysis
            self.llm_logger.log_component_extraction(
                component_type="improvement_analysis",
                extracted_data=improvements_analysis,
                success=improvements_analysis.get('extraction_successful', False),
                extraction_method="quality_assessment",
                confidence=improvements_analysis.get('quality_score', 0.0)
            )
            
            # Generate suggestions for further refinement
            suggestions = self._extract_refinement_suggestions(parsed_response)
            
            # Log refinement suggestions extraction
            self.llm_logger.log_component_extraction(
                component_type="refinement_suggestions",
                extracted_data=suggestions,
                success=len(suggestions) > 0,
                extraction_method="pattern_extraction"
            )
            
            # Extract and log best practices applied
            best_practices = self._extract_best_practices_applied(parsed_response)
            self.llm_logger.log_agent_reasoning(
                reasoning_type="best_practices_applied",
                reasoning_text=f"Applied {len(best_practices)} best practices: {', '.join(best_practices)}",
                metadata={
                    'practices_count': len(best_practices),
                    'practices_list': best_practices,
                    'session_id': session_id,
                    'iteration': iteration
                }
            )
            
            # Log refinement reasoning from LLM response
            improvements_explanation = improvements_analysis.get('improvements_explanation', '')
            if improvements_explanation:
                self.llm_logger.log_agent_reasoning(
                    reasoning_type="refinement_reasoning",
                    reasoning_text=improvements_explanation,
                    metadata={
                        'reasoning_source': 'llm_response',
                        'quality_score': improvements_analysis.get('quality_score', 0.0),
                        'session_id': session_id,
                        'iteration': iteration
                    }
                )
            
            # Calculate confidence in the refinement
            confidence_score = self._calculate_refinement_confidence(
                parsed_response, improvements_analysis
            )
            
            # Log confidence calculation with refinement quality factors
            confidence_factors = {
                'extraction_success': float(improvements_analysis.get('extraction_successful', False)),
                'quality_score': improvements_analysis.get('quality_score', 0.0),
                'best_practices_count': len(best_practices) / 10.0,  # Normalize to 0-1
                'structural_improvements': float(improvements_analysis.get('structural_improvements', False)),
                'clarity_improvements': float(improvements_analysis.get('clarity_improvements', False)),
                'completeness_improvements': float(improvements_analysis.get('completeness_improvements', False))
            }
            
            self.llm_logger.log_confidence_calculation(
                confidence_score=confidence_score,
                reasoning=f"Confidence based on refinement quality analysis: {improvements_analysis.get('quality_score', 0.0):.2f}, "
                         f"extraction success: {improvements_analysis.get('extraction_successful', False)}, "
                         f"best practices applied: {len(best_practices)}",
                factors=confidence_factors,
                calculation_method="refinement_quality_weighted"
            )
            
            # Compile analysis
            analysis = {
                'original_prompt': prompt,
                'refined_prompt': refined_prompt,
                'llm_refinement': {
                    'raw_response': parsed_response['raw_response'],
                    'model_used': llm_response['model'],
                    'tokens_used': llm_response['tokens_used'],
                    'reasoning': parsed_response.get('reasoning', ''),
                    'confidence': parsed_response.get('confidence', 0.8)
                },
                'improvements_analysis': improvements_analysis,
                'refinement_quality': self._assess_refinement_quality(improvements_analysis),
                'best_practices_applied': best_practices
            }
            
            # Log the complete refinement analysis
            self.llm_logger.log_agent_reasoning(
                reasoning_type="refinement_analysis_complete",
                reasoning_text=f"Refinement completed with quality: {analysis['refinement_quality']}. "
                              f"Applied {len(best_practices)} best practices. "
                              f"Confidence: {confidence_score:.3f}",
                metadata={
                    'analysis_summary': {
                        'original_length': len(prompt),
                        'refined_length': len(refined_prompt),
                        'length_change': improvements_analysis.get('length_change', 0),
                        'quality_score': improvements_analysis.get('quality_score', 0.0),
                        'extraction_successful': improvements_analysis.get('extraction_successful', False)
                    },
                    'session_id': session_id,
                    'iteration': iteration,
                    'final_confidence': confidence_score
                }
            )
            
            return AgentResult(
                agent_name=self.name,
                success=True,
                analysis=analysis,
                suggestions=suggestions,
                confidence_score=confidence_score
            )
            
        except Exception as e:
            # Log the exception
            session_id = context.get('session_id') if context else None
            iteration = context.get('iteration') if context else None
            
            self.llm_logger.log_error(
                error_type="refinement_processing_exception",
                error_message=str(e),
                context={
                    'session_id': session_id,
                    'iteration': iteration,
                    'prompt_length': len(prompt) if prompt else 0,
                    'has_context': context is not None,
                    'has_history': history is not None and len(history) > 0,
                    'has_feedback': feedback is not None
                },
                exception=e
            )
            
            # Handle the exception by raising it with detailed context
            self._handle_llm_failure(prompt, context, history, feedback, str(e))
    
    def _get_base_system_prompt(self) -> str:
        """Get the base system prompt for LLM refinement."""
        return """You are an expert prompt engineer specializing in prompt refinement and optimization. Your role is to take existing prompts and improve them using advanced prompt engineering techniques and best practices.

Your refinements should:
- Preserve the original intent and core objectives
- Apply systematic improvements based on prompt engineering principles
- Enhance clarity, structure, and effectiveness
- Consider the target audience and use case context
- Provide clear reasoning for all changes made

Focus on creating prompts that are more effective, clearer, and better structured while maintaining the original purpose."""
    
    def _get_best_practices_prompt(self) -> str:
        """Get the best practices prompt for refinement."""
        return """Apply these prompt engineering best practices in your refinements:

STRUCTURAL IMPROVEMENTS:
- Use clear section headers and organization
- Implement logical information hierarchy
- Separate context, task, and requirements clearly
- Add proper formatting and spacing

CLARITY ENHANCEMENTS:
- Replace vague terms with specific language
- Eliminate ambiguous instructions
- Use active voice and direct commands
- Define technical terms when necessary

COMPLETENESS OPTIMIZATIONS:
- Add missing context and background information
- Include specific examples and templates
- Define success criteria and constraints explicitly
- Specify desired output format and structure

EFFECTIVENESS TECHNIQUES:
- Use appropriate prompting techniques (few-shot, chain-of-thought, etc.)
- Optimize for the target LLM capabilities
- Balance specificity with flexibility
- Include error handling and edge case considerations

ADVANCED TECHNIQUES:
- Role-based prompting for expertise simulation
- Step-by-step reasoning frameworks
- Template structures for consistency
- Metacognitive prompting for self-reflection"""
    
    def _get_reasoning_framework_prompt(self) -> str:
        """Get the reasoning framework for structured refinement."""
        return """Use this systematic refinement framework:

1. ANALYSIS PHASE
   - Identify the core task and objectives
   - Assess current prompt strengths and weaknesses
   - Determine improvement priorities based on context

2. PLANNING PHASE
   - Select appropriate refinement techniques
   - Plan structural and content improvements
   - Consider trade-offs and constraints

3. REFINEMENT PHASE
   - Apply structural improvements systematically
   - Enhance clarity and specificity
   - Add missing elements and context
   - Implement advanced prompting techniques

4. VALIDATION PHASE
   - Verify intent preservation
   - Check for logical consistency
   - Ensure completeness and clarity
   - Validate against best practices

5. DOCUMENTATION PHASE
   - Explain all changes made and reasoning
   - Highlight key improvements and techniques applied
   - Provide confidence assessment
   - Suggest further optimization opportunities

Present your refined prompt clearly marked, followed by detailed explanation of improvements."""
    
    def _build_refinement_prompt(self, prompt: str, context: Optional[Dict[str, Any]], 
                               history: Optional[List[PromptIteration]], 
                               feedback: Optional[UserFeedback]) -> str:
        """Build the refinement prompt for the LLM."""
        refinement_prompt_parts = [
            "Please refine and improve the following prompt using your expertise in prompt engineering:",
            f"\n--- ORIGINAL PROMPT ---\n{prompt}\n--- END ORIGINAL PROMPT ---\n"
        ]
        
        # Add context information
        if context:
            context_info = ["CONTEXT INFORMATION:"]
            if 'intended_use' in context:
                context_info.append(f"Intended Use: {context['intended_use']}")
            if 'target_audience' in context:
                context_info.append(f"Target Audience: {context['target_audience']}")
            if 'domain' in context:
                context_info.append(f"Domain: {context['domain']}")
            if 'constraints' in context:
                context_info.append(f"Constraints: {context['constraints']}")
            
            refinement_prompt_parts.extend(context_info)
        
        # Add analysis from previous iterations if available
        if history and len(history) > 0:
            latest_iteration = history[-1]
            if latest_iteration.agent_analysis and 'LLMAnalyzerAgent' in latest_iteration.agent_analysis:
                analyzer_results = latest_iteration.agent_analysis['LLMAnalyzerAgent']
                if analyzer_results.get('success'):
                    refinement_prompt_parts.append("\nPREVIOUS ANALYSIS INSIGHTS:")
                    analysis = analyzer_results.get('analysis', {})
                    
                    # Include key analysis points
                    if 'structure_analysis' in analysis:
                        struct_analysis = analysis['structure_analysis']
                        if struct_analysis.get('needs_restructuring'):
                            refinement_prompt_parts.append("- Structure needs improvement")
                    
                    if 'clarity_analysis' in analysis:
                        clarity_analysis = analysis['clarity_analysis']
                        if clarity_analysis.get('has_ambiguity'):
                            refinement_prompt_parts.append("- Contains ambiguous language")
                    
                    if 'completeness_analysis' in analysis:
                        completeness_analysis = analysis['completeness_analysis']
                        if completeness_analysis.get('missing_elements'):
                            refinement_prompt_parts.append("- Missing essential elements")
        
        # Add user feedback
        if feedback:
            feedback_info = [
                f"\nUSER FEEDBACK:",
                f"Satisfaction Rating: {feedback.satisfaction_rating}/5"
            ]
            if feedback.specific_issues:
                feedback_info.append(f"Specific Issues to Address: {', '.join(feedback.specific_issues)}")
            if feedback.desired_improvements:
                feedback_info.append(f"Desired Improvements: {feedback.desired_improvements}")
            
            refinement_prompt_parts.extend(feedback_info)
        
        # Add refinement instructions
        refinement_instructions = [
            "\nREFINEMENT REQUIREMENTS:",
            "1. Preserve the original intent and core objectives",
            "2. Apply prompt engineering best practices systematically",
            "3. Improve structure, clarity, and completeness",
            "4. Add specific examples where helpful",
            "5. Include proper formatting and organization",
            "",
            "Provide your response in this format:",
            "--- REFINED PROMPT ---",
            "[Your improved prompt here]",
            "--- END REFINED PROMPT ---",
            "",
            "IMPROVEMENTS MADE:",
            "[Detailed explanation of changes and reasoning]",
            "",
            "BEST PRACTICES APPLIED:",
            "[List of specific techniques and principles used]",
            "",
            "CONFIDENCE: [0.0-1.0 score with justification]"
        ]
        
        refinement_prompt_parts.extend(refinement_instructions)
        
        return "\n".join(refinement_prompt_parts)
    
    def _extract_refined_prompt(self, parsed_response: Dict[str, Any]) -> str:
        """Extract the refined prompt from LLM response."""
        import re
        
        raw_response = parsed_response['raw_response']
        
        # Look for refined prompt section
        refined_prompt_match = re.search(
            r'---\s*REFINED PROMPT\s*---\s*\n(.*?)\n\s*---\s*END REFINED PROMPT\s*---',
            raw_response,
            re.DOTALL | re.IGNORECASE
        )
        
        if refined_prompt_match:
            return refined_prompt_match.group(1).strip()
        
        # Fallback: look for other prompt markers
        fallback_patterns = [
            r'improved prompt[:\s]*\n(.*?)(?=\nIMPROVEMENTS|\nBEST PRACTICES|\Z)',
            r'refined version[:\s]*\n(.*?)(?=\nIMPROVEMENTS|\nBEST PRACTICES|\Z)',
            r'optimized prompt[:\s]*\n(.*?)(?=\nIMPROVEMENTS|\nBEST PRACTICES|\Z)'
        ]
        
        for pattern in fallback_patterns:
            match = re.search(pattern, raw_response, re.DOTALL | re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        # If no clear refined prompt found, return a structured version of the original
        return "No refined prompt could be extracted from LLM response."
    
    def _analyze_refinement_quality(self, original: str, refined: str, 
                                  parsed_response: Dict[str, Any], 
                                  llm_response: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the quality of the refinement."""
        if refined == "No refined prompt could be extracted from LLM response.":
            # Log failed extraction
            self.llm_logger.log_component_extraction(
                component_type="refinement_quality_analysis",
                extracted_data=None,
                success=False,
                extraction_method="quality_assessment"
            )
            return {
                'extraction_successful': False,
                'length_change': 0,
                'structural_improvements': False,
                'clarity_improvements': False,
                'completeness_improvements': False,
                'quality_score': 0.0
            }
        
        # Calculate basic metrics
        length_change = len(refined) - len(original)
        length_change_ratio = length_change / len(original) if len(original) > 0 else 0
        
        # Analyze structural improvements
        original_sections = original.count('##') + original.count('---')
        refined_sections = refined.count('##') + refined.count('---')
        structural_improvements = refined_sections > original_sections
        
        # Analyze clarity improvements
        clarity_indicators = ['specific', 'clear', 'detailed', 'explicit']
        clarity_improvements = any(indicator in refined.lower() for indicator in clarity_indicators)
        
        # Analyze completeness improvements
        completeness_indicators = ['example', 'format', 'requirement', 'criteria', 'context']
        completeness_improvements = any(
            refined.lower().count(indicator) > original.lower().count(indicator)
            for indicator in completeness_indicators
        )
        
        # Extract improvements from LLM response
        improvements_text = self._extract_improvements_explanation(parsed_response['raw_response'])
        
        # Calculate quality score
        quality_factors = [
            structural_improvements,
            clarity_improvements,
            completeness_improvements,
            len(improvements_text) > 50,  # Detailed explanation provided
            length_change_ratio > 0.5  # Reasonable length increase
        ]
        
        quality_score = sum(quality_factors) / len(quality_factors)
        
        # Log the quality analysis process
        self.llm_logger.log_agent_reasoning(
            reasoning_type="improvement_quality_analysis",
            reasoning_text=f"Quality analysis completed: structural={structural_improvements}, "
                          f"clarity={clarity_improvements}, completeness={completeness_improvements}, "
                          f"detailed_explanation={len(improvements_text) > 50}, "
                          f"reasonable_length_change={length_change_ratio > 0.5}. "
                          f"Overall quality score: {quality_score:.3f}",
            metadata={
                'quality_factors': {
                    'structural_improvements': structural_improvements,
                    'clarity_improvements': clarity_improvements,
                    'completeness_improvements': completeness_improvements,
                    'has_detailed_explanation': len(improvements_text) > 50,
                    'reasonable_length_change': length_change_ratio > 0.5
                },
                'metrics': {
                    'length_change': length_change,
                    'length_change_ratio': length_change_ratio,
                    'original_sections': original_sections,
                    'refined_sections': refined_sections,
                    'explanation_length': len(improvements_text)
                }
            }
        )
        
        return {
            'extraction_successful': True,
            'length_change': length_change,
            'length_change_ratio': length_change_ratio,
            'structural_improvements': structural_improvements,
            'clarity_improvements': clarity_improvements,
            'completeness_improvements': completeness_improvements,
            'improvements_explanation': improvements_text,
            'quality_score': quality_score,
            'sections_added': refined_sections - original_sections,
            'word_count_change': len(refined.split()) - len(original.split())
        }
    
    def _extract_improvements_explanation(self, response: str) -> str:
        """Extract the improvements explanation from LLM response."""
        import re
        
        # Look for improvements section
        improvements_patterns = [
            r'IMPROVEMENTS MADE[:\s]*\n(.*?)(?=\nBEST PRACTICES|\nCONFIDENCE|\Z)',
            r'improvements[:\s]*\n(.*?)(?=\nBEST PRACTICES|\nCONFIDENCE|\Z)',
            r'changes made[:\s]*\n(.*?)(?=\nBEST PRACTICES|\nCONFIDENCE|\Z)'
        ]
        
        for pattern in improvements_patterns:
            match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return ""
    
    def _extract_best_practices_applied(self, parsed_response: Dict[str, Any]) -> List[str]:
        """Extract best practices applied from LLM response."""
        import re
        
        raw_response = parsed_response['raw_response']
        
        # Look for best practices section
        best_practices_match = re.search(
            r'BEST PRACTICES APPLIED[:\s]*\n(.*?)(?=\nCONFIDENCE|\Z)',
            raw_response,
            re.DOTALL | re.IGNORECASE
        )
        
        if best_practices_match:
            practices_text = best_practices_match.group(1)
            # Extract bullet points or numbered items
            practices = re.findall(r'^\s*[\d\-\*]\.\s*(.+)$', practices_text, re.MULTILINE)
            return practices
        
        # Fallback: look for mentions of specific practices
        practice_keywords = [
            'clear structure', 'specific language', 'examples added', 'context provided',
            'requirements defined', 'format specified', 'role-based prompting',
            'chain-of-thought', 'few-shot examples', 'step-by-step instructions'
        ]
        
        found_practices = []
        for keyword in practice_keywords:
            if keyword in raw_response.lower():
                found_practices.append(keyword.title())
        
        return found_practices
    
    def _extract_refinement_suggestions(self, parsed_response: Dict[str, Any]) -> List[str]:
        """Extract suggestions for further refinement."""
        suggestions = parsed_response.get('recommendations', [])
        
        # Look for additional suggestions in the response
        import re
        raw_response = parsed_response['raw_response']
        
        suggestion_patterns = [
            r'further improvements?[:\s]*\n(.*?)(?=\n\n|\Z)',
            r'additional suggestions?[:\s]*\n(.*?)(?=\n\n|\Z)',
            r'next steps?[:\s]*\n(.*?)(?=\n\n|\Z)'
        ]
        
        for pattern in suggestion_patterns:
            match = re.search(pattern, raw_response, re.DOTALL | re.IGNORECASE)
            if match:
                suggestion_text = match.group(1)
                items = re.findall(r'^\s*[\d\-\*]\.\s*(.+)$', suggestion_text, re.MULTILINE)
                suggestions.extend(items)
        
        # Remove duplicates
        unique_suggestions = []
        seen = set()
        for suggestion in suggestions:
            if suggestion not in seen:
                unique_suggestions.append(suggestion)
                seen.add(suggestion)
        
        return unique_suggestions[:8]  # Limit to top 8 suggestions
    
    def _calculate_refinement_confidence(self, parsed_response: Dict[str, Any], 
                                       improvements_analysis: Dict[str, Any]) -> float:
        """Calculate confidence in the refinement quality."""
        base_confidence = parsed_response.get('confidence', 0.8)
        original_confidence = base_confidence
        
        # Adjust based on refinement quality
        if improvements_analysis.get('extraction_successful', False):
            quality_score = improvements_analysis.get('quality_score', 0.5)
            base_confidence = (base_confidence + quality_score) / 2
        else:
            base_confidence = 0.3  # Low confidence if extraction failed
        
        # Adjust based on response quality
        response_text = parsed_response['raw_response']
        
        # Higher confidence for detailed explanations
        explanation_bonus = 0.0
        if len(improvements_analysis.get('improvements_explanation', '')) > 200:
            explanation_bonus = 0.1
            base_confidence += explanation_bonus
        
        # Higher confidence for multiple best practices applied
        best_practices_count = len(self._extract_best_practices_applied(parsed_response))
        practices_bonus = 0.0
        if best_practices_count >= 3:
            practices_bonus = 0.1
            base_confidence += practices_bonus
        
        # Lower confidence for minimal changes
        length_penalty = 0.0
        if improvements_analysis.get('length_change', 0) < 50:
            length_penalty = -0.1
            base_confidence += length_penalty
        
        final_confidence = min(1.0, max(0.0, base_confidence))
        
        # Log the confidence calculation details
        self.llm_logger.log_agent_reasoning(
            reasoning_type="confidence_calculation_details",
            reasoning_text=f"Confidence calculation: base={original_confidence:.3f}, "
                          f"quality_adjusted={(original_confidence + improvements_analysis.get('quality_score', 0.5))/2:.3f}, "
                          f"explanation_bonus={explanation_bonus:.3f}, "
                          f"practices_bonus={practices_bonus:.3f}, "
                          f"length_penalty={length_penalty:.3f}, "
                          f"final={final_confidence:.3f}",
            metadata={
                'confidence_components': {
                    'original_llm_confidence': original_confidence,
                    'quality_score': improvements_analysis.get('quality_score', 0.5),
                    'extraction_successful': improvements_analysis.get('extraction_successful', False),
                    'explanation_length': len(improvements_analysis.get('improvements_explanation', '')),
                    'best_practices_count': best_practices_count,
                    'length_change': improvements_analysis.get('length_change', 0)
                },
                'adjustments': {
                    'explanation_bonus': explanation_bonus,
                    'practices_bonus': practices_bonus,
                    'length_penalty': length_penalty
                }
            }
        )
        
        return final_confidence
    
    def _assess_refinement_quality(self, improvements_analysis: Dict[str, Any]) -> str:
        """Assess the overall quality of the refinement."""
        if not improvements_analysis.get('extraction_successful', False):
            return "Refinement extraction failed - unable to assess quality"
        
        quality_score = improvements_analysis.get('quality_score', 0.0)
        
        if quality_score >= 0.8:
            return "Excellent refinement with comprehensive improvements across multiple areas"
        elif quality_score >= 0.6:
            return "Good refinement with significant improvements in key areas"
        elif quality_score >= 0.4:
            return "Moderate refinement with some improvements made"
        elif quality_score >= 0.2:
            return "Minor refinement with limited improvements"
        else:
            return "Minimal refinement - consider more substantial changes"
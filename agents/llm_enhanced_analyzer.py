"""
LLMAnalyzerAgent for advanced prompt analysis using LLM capabilities.

This agent leverages Large Language Models with specialized system prompts
to perform sophisticated analysis of prompt structure, clarity, and effectiveness
using embedded prompt engineering best practices.
"""

from typing import Dict, Any, List, Optional
from agents.llm_agent import LLMAgent
from agents.base import AgentResult
from agents.llm_agent_logger import LLMAgentLogger
from models import PromptIteration, UserFeedback


class LLMAnalyzerAgent(LLMAgent):
    """
    LLM-enhanced agent for intelligent prompt analysis.
    
    Uses specialized system prompts and reasoning frameworks to analyze:
    - Prompt structure and organization
    - Clarity and readability with context awareness
    - Completeness against best practices
    - Potential improvements with reasoning
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the LLMAnalyzerAgent."""
        super().__init__("LLMAnalyzerAgent", config)
        
        # Analysis-specific configuration
        self.analysis_depth = self.config.get('analysis_depth', 'comprehensive')
        self.focus_areas = self.config.get('focus_areas', ['structure', 'clarity', 'completeness'])
        self.domain_expertise = self.config.get('domain_expertise', 'general')
        
        # Initialize LLM-specific logger for analyzer
        self.llm_logger = LLMAgentLogger("LLMAnalyzerAgent")
    
    def process(self, 
                prompt: str, 
                context: Optional[Dict[str, Any]] = None,
                history: Optional[List[PromptIteration]] = None,
                feedback: Optional[UserFeedback] = None) -> AgentResult:
        """
        Perform LLM-enhanced analysis of the prompt.
        
        Args:
            prompt: The prompt text to analyze
            context: Optional context about the prompt's intended use
            history: Optional list of previous prompt iterations
            feedback: Optional user feedback from previous iterations
            
        Returns:
            AgentResult containing LLM-powered analysis and suggestions
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
            # Get session context for logging
            session_id = context.get('session_id') if context else None
            iteration = context.get('iteration') if context else None
            
            # Prepare analysis prompt for LLM
            analysis_prompt = self._build_analysis_prompt(prompt, context, history, feedback)
            
            # Log the analysis reasoning approach
            self.llm_logger.log_agent_reasoning(
                reasoning_type="analysis_approach",
                reasoning_text=f"Starting comprehensive analysis with depth: {self.analysis_depth}, "
                              f"focus areas: {', '.join(self.focus_areas)}, "
                              f"domain expertise: {self.domain_expertise}",
                metadata={
                    'analysis_depth': self.analysis_depth,
                    'focus_areas': self.focus_areas,
                    'domain_expertise': self.domain_expertise,
                    'prompt_length': len(prompt),
                    'has_history': history is not None and len(history) > 0,
                    'has_feedback': feedback is not None
                }
            )
            
            # Call LLM for analysis
            llm_response = self._call_llm(analysis_prompt, context)
            
            # Check if LLM call failed and handle error
            if not llm_response['success']:
                self.llm_logger.log_error(
                    error_type="llm_analysis_failed",
                    error_message=f"LLM analysis failed: {llm_response['error']}",
                    context={'session_id': session_id, 'iteration': iteration}
                )
                self._handle_llm_failure(prompt, context, history, feedback, 
                                         llm_response.get('error', 'LLM service unavailable'))
            
            # Parse and structure the LLM response
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
                agent_name='LLMAnalyzerAgent',
                raw_llm_response=llm_response['response'],
                parsed_data=parsed_response,
                session_id=session_id,
                iteration=iteration
            )
            
            # Extract structured analysis with logging
            analysis = self._extract_analysis_components(parsed_response, llm_response)
            
            # Log analysis insights extraction
            self.llm_logger.log_agent_reasoning(
                reasoning_type="analysis_insights",
                reasoning_text=f"Extracted analysis insights: "
                              f"Structure score: {analysis.get('structure_analysis', {}).get('structure_score', 'N/A')}, "
                              f"Clarity score: {analysis.get('clarity_analysis', {}).get('clarity_score', 'N/A')}, "
                              f"Completeness score: {analysis.get('completeness_analysis', {}).get('completeness_score', 'N/A')}, "
                              f"Effectiveness score: {analysis.get('effectiveness_analysis', {}).get('effectiveness_score', 'N/A')}",
                metadata={
                    'analysis_components': list(analysis.keys()),
                    'structure_analysis': analysis.get('structure_analysis', {}),
                    'clarity_analysis': analysis.get('clarity_analysis', {}),
                    'completeness_analysis': analysis.get('completeness_analysis', {}),
                    'effectiveness_analysis': analysis.get('effectiveness_analysis', {}),
                    'best_practices_assessment': analysis.get('best_practices_assessment', {})
                }
            )
            
            # Generate actionable suggestions with logging
            suggestions = self._extract_suggestions(parsed_response)
            
            # Log suggestions extraction
            self.llm_logger.log_component_extraction(
                component_type="suggestions",
                extracted_data=suggestions,
                success=len(suggestions) > 0,
                extraction_method="pattern_matching_and_llm_parsing",
                confidence=0.8 if len(suggestions) > 0 else 0.3
            )
            
            # Calculate confidence based on LLM response and analysis depth
            confidence_score = self._calculate_llm_confidence(parsed_response, llm_response)
            
            # Log confidence calculation with detailed reasoning
            confidence_factors = {
                'response_quality': 0.8 if len(parsed_response.get('raw_response', '')) > 500 else 0.5,
                'analysis_completeness': 0.9 if len(analysis) >= 4 else 0.6,
                'suggestions_count': min(1.0, len(suggestions) / 5.0),
                'llm_confidence': parsed_response.get('confidence', 0.7)
            }
            
            self.llm_logger.log_confidence_calculation(
                confidence_score=confidence_score,
                reasoning=f"Confidence calculated based on response quality, analysis completeness, "
                         f"suggestions count ({len(suggestions)}), and LLM-reported confidence",
                factors=confidence_factors,
                calculation_method="weighted_average_with_quality_indicators"
            )
            
            return AgentResult(
                agent_name=self.name,
                success=True,
                analysis=analysis,
                suggestions=suggestions,
                confidence_score=confidence_score
            )
            
        except Exception as e:
            # Enhanced exception logging for analysis
            exception_context = {
                'analysis_depth': self.analysis_depth,
                'focus_areas': self.focus_areas,
                'prompt_length': len(prompt),
                'has_context': context is not None,
                'has_history': history is not None and len(history) > 0 if history else False,
                'has_feedback': feedback is not None
            }
            
            self.llm_logger.log_error(
                error_type='analysis_processing_exception',
                error_message=str(e),
                context=exception_context,
                exception=e
            )
            
            # Handle the exception by raising it with detailed context
            self._handle_llm_failure(prompt, context, history, feedback, str(e))
    
    def _get_base_system_prompt(self) -> str:
        """Get the base system prompt for LLM analysis."""
        return """You are an expert prompt engineering analyst with deep knowledge of best practices and optimization techniques. Your role is to analyze prompts for structure, clarity, completeness, and effectiveness.

Your analysis should be:
- Systematic and thorough
- Based on established prompt engineering principles
- Focused on actionable improvements
- Contextually aware of the intended use case

Analyze each prompt across multiple dimensions and provide specific, actionable recommendations for improvement."""
    
    def _get_best_practices_prompt(self) -> str:
        """Get the best practices prompt for analysis."""
        return """Apply these prompt engineering best practices in your analysis:

STRUCTURE BEST PRACTICES:
- Clear task definition with specific action verbs
- Logical organization with distinct sections
- Proper use of context and background information
- Explicit success criteria and constraints

CLARITY BEST PRACTICES:
- Unambiguous language and instructions
- Specific rather than vague terminology
- Appropriate complexity for target audience
- Clear input/output specifications

COMPLETENESS BEST PRACTICES:
- All necessary information provided
- Edge cases and constraints addressed
- Examples and formatting guidelines included
- Success metrics and evaluation criteria defined

EFFECTIVENESS BEST PRACTICES:
- Optimized for the intended LLM and use case
- Balanced specificity without over-constraint
- Consideration of token efficiency
- Alignment with user goals and expectations"""
    
    def _get_reasoning_framework_prompt(self) -> str:
        """Get the reasoning framework for structured analysis."""
        return """Use this structured reasoning framework for analysis:

1. INITIAL ASSESSMENT
   - Identify the core task and objectives
   - Assess overall prompt quality at first glance
   - Note immediate strengths and weaknesses

2. SYSTEMATIC EVALUATION
   - Structure: Organization, sections, logical flow
   - Clarity: Language precision, ambiguity detection
   - Completeness: Missing elements, information gaps
   - Context: Appropriateness for intended use

3. BEST PRACTICES ALIGNMENT
   - Compare against established prompt engineering standards
   - Identify deviations from optimal patterns
   - Assess adherence to domain-specific guidelines

4. IMPROVEMENT IDENTIFICATION
   - Prioritize issues by impact and feasibility
   - Generate specific, actionable recommendations
   - Consider trade-offs and implementation complexity

5. CONFIDENCE ASSESSMENT
   - Evaluate certainty of analysis conclusions
   - Identify areas requiring additional context
   - Provide confidence scores for recommendations

Format your response with clear sections and specific examples."""
    
    def _build_analysis_prompt(self, prompt: str, context: Optional[Dict[str, Any]], 
                             history: Optional[List[PromptIteration]], 
                             feedback: Optional[UserFeedback]) -> str:
        """Build the analysis prompt for the LLM."""
        analysis_prompt_parts = [
            "Please analyze the following prompt using your expertise in prompt engineering:",
            f"\n--- PROMPT TO ANALYZE ---\n{prompt}\n--- END PROMPT ---\n"
        ]
        
        # Add context if available
        if context:
            context_info = []
            if 'intended_use' in context:
                context_info.append(f"Intended Use: {context['intended_use']}")
            if 'target_audience' in context:
                context_info.append(f"Target Audience: {context['target_audience']}")
            if 'domain' in context:
                context_info.append(f"Domain: {context['domain']}")
            
            if context_info:
                analysis_prompt_parts.append("CONTEXT INFORMATION:")
                analysis_prompt_parts.extend(context_info)
        
        # Add history insights if available
        if history and len(history) > 0:
            analysis_prompt_parts.append(f"\nPREVIOUS ITERATIONS: {len(history)} iterations exist")
            if len(history) > 1:
                analysis_prompt_parts.append("Consider this as part of an iterative improvement process.")
        
        # Add feedback if available
        if feedback:
            feedback_info = [
                f"\nUSER FEEDBACK:",
                f"Satisfaction Rating: {feedback.satisfaction_rating}/5"
            ]
            if feedback.specific_issues:
                feedback_info.append(f"Specific Issues: {', '.join(feedback.specific_issues)}")
            if feedback.desired_improvements:
                feedback_info.append(f"Desired Improvements: {feedback.desired_improvements}")
            
            analysis_prompt_parts.extend(feedback_info)
        
        # Add analysis instructions
        analysis_instructions = [
            "\nProvide a comprehensive analysis covering:",
            "1. STRUCTURE ANALYSIS - Organization, sections, logical flow",
            "2. CLARITY ASSESSMENT - Language precision, ambiguity, readability", 
            "3. COMPLETENESS EVALUATION - Missing elements, information gaps",
            "4. EFFECTIVENESS REVIEW - Alignment with best practices and goals",
            "5. IMPROVEMENT RECOMMENDATIONS - Specific, prioritized suggestions",
            "",
            "Include confidence scores (0.0-1.0) for your assessments and provide reasoning for your conclusions."
        ]
        
        analysis_prompt_parts.extend(analysis_instructions)
        
        return "\n".join(analysis_prompt_parts)
    
    def _extract_analysis_components(self, parsed_response: Dict[str, Any], 
                                   llm_response: Dict[str, Any]) -> Dict[str, Any]:
        """Extract structured analysis components from LLM response."""
        raw_response = parsed_response['raw_response']
        
        # Extract different analysis sections using pattern matching
        import re
        
        # Log the start of component extraction
        self.llm_logger.log_agent_reasoning(
            reasoning_type="component_extraction_start",
            reasoning_text="Beginning extraction of analysis components from LLM response",
            metadata={
                'raw_response_length': len(raw_response),
                'model_used': llm_response.get('model', 'unknown'),
                'tokens_used': llm_response.get('tokens_used', 0)
            }
        )
        
        # Extract structure analysis with logging
        structure_analysis = self._extract_structure_analysis(raw_response)
        self.llm_logger.log_component_extraction(
            component_type="structure_analysis",
            extracted_data=structure_analysis,
            success=bool(structure_analysis.get('analysis_text')),
            extraction_method="regex_pattern_matching",
            confidence=structure_analysis.get('structure_score', 0.7)
        )
        
        # Extract clarity analysis with logging
        clarity_analysis = self._extract_clarity_analysis(raw_response)
        self.llm_logger.log_component_extraction(
            component_type="clarity_analysis",
            extracted_data=clarity_analysis,
            success=bool(clarity_analysis.get('analysis_text')),
            extraction_method="regex_pattern_matching",
            confidence=clarity_analysis.get('clarity_score', 0.7)
        )
        
        # Extract completeness analysis with logging
        completeness_analysis = self._extract_completeness_analysis(raw_response)
        self.llm_logger.log_component_extraction(
            component_type="completeness_analysis",
            extracted_data=completeness_analysis,
            success=bool(completeness_analysis.get('analysis_text')),
            extraction_method="regex_pattern_matching",
            confidence=completeness_analysis.get('completeness_score', 0.7)
        )
        
        # Extract effectiveness analysis with logging
        effectiveness_analysis = self._extract_effectiveness_analysis(raw_response)
        self.llm_logger.log_component_extraction(
            component_type="effectiveness_analysis",
            extracted_data=effectiveness_analysis,
            success=bool(effectiveness_analysis.get('analysis_text')),
            extraction_method="regex_pattern_matching",
            confidence=effectiveness_analysis.get('effectiveness_score', 0.7)
        )
        
        # Extract best practices assessment with logging
        best_practices_assessment = self._extract_best_practices_assessment(raw_response)
        self.llm_logger.log_component_extraction(
            component_type="best_practices_assessment",
            extracted_data=best_practices_assessment,
            success=bool(best_practices_assessment.get('follows_best_practices') is not None),
            extraction_method="pattern_matching_and_keyword_analysis",
            confidence=best_practices_assessment.get('best_practices_score', 0.7)
        )
        
        analysis = {
            'llm_analysis': {
                'raw_response': raw_response,
                'model_used': llm_response['model'],
                'tokens_used': llm_response['tokens_used'],
                'reasoning': parsed_response.get('reasoning', ''),
                'confidence': parsed_response.get('confidence', 0.8)
            },
            'structure_analysis': structure_analysis,
            'clarity_analysis': clarity_analysis,
            'completeness_analysis': completeness_analysis,
            'effectiveness_analysis': effectiveness_analysis,
            'best_practices_assessment': best_practices_assessment
        }
        
        # Log the completion of component extraction
        self.llm_logger.log_agent_reasoning(
            reasoning_type="component_extraction_complete",
            reasoning_text=f"Successfully extracted {len(analysis)} analysis components",
            metadata={
                'extracted_components': list(analysis.keys()),
                'total_components': len(analysis),
                'extraction_success_rate': sum(1 for comp in analysis.values() 
                                             if isinstance(comp, dict) and comp.get('analysis_text')) / len(analysis)
            }
        )
        
        return analysis
    
    def _extract_structure_analysis(self, response: str) -> Dict[str, Any]:
        """Extract structure analysis from LLM response."""
        import re
        
        # Look for structure-related content
        structure_section = re.search(r'structure[^:]*:?\s*(.+?)(?=\n\n|\n[A-Z]|\Z)', 
                                    response, re.IGNORECASE | re.DOTALL)
        
        structure_text = structure_section.group(1) if structure_section else ""
        
        # Extract key insights
        has_clear_sections = 'section' in structure_text.lower() or 'organize' in structure_text.lower()
        has_logical_flow = 'flow' in structure_text.lower() or 'logical' in structure_text.lower()
        needs_restructuring = 'restructur' in structure_text.lower() or 'reorganiz' in structure_text.lower()
        structure_score = self._extract_score(structure_text, default=0.7)
        
        # Log detailed structure analysis results
        self.llm_logger.log_agent_reasoning(
            reasoning_type="structure_analysis_results",
            reasoning_text=f"Structure analysis completed. Score: {structure_score:.2f}. "
                          f"Clear sections: {has_clear_sections}, Logical flow: {has_logical_flow}, "
                          f"Needs restructuring: {needs_restructuring}",
            metadata={
                'structure_score': structure_score,
                'has_clear_sections': has_clear_sections,
                'has_logical_flow': has_logical_flow,
                'needs_restructuring': needs_restructuring,
                'analysis_text_length': len(structure_text),
                'section_found': structure_section is not None
            }
        )
        
        return {
            'analysis_text': structure_text,
            'has_clear_sections': has_clear_sections,
            'has_logical_flow': has_logical_flow,
            'needs_restructuring': needs_restructuring,
            'structure_score': structure_score
        }
    
    def _extract_clarity_analysis(self, response: str) -> Dict[str, Any]:
        """Extract clarity analysis from LLM response."""
        import re
        
        clarity_section = re.search(r'clarity[^:]*:?\s*(.+?)(?=\n\n|\n[A-Z]|\Z)', 
                                  response, re.IGNORECASE | re.DOTALL)
        
        clarity_text = clarity_section.group(1) if clarity_section else ""
        
        # Extract key insights
        is_clear = 'clear' in clarity_text.lower() and 'unclear' not in clarity_text.lower()
        has_ambiguity = 'ambig' in clarity_text.lower() or 'confus' in clarity_text.lower()
        needs_simplification = 'simplif' in clarity_text.lower() or 'complex' in clarity_text.lower()
        clarity_score = self._extract_score(clarity_text, default=0.7)
        
        # Log detailed clarity analysis results
        self.llm_logger.log_agent_reasoning(
            reasoning_type="clarity_analysis_results",
            reasoning_text=f"Clarity analysis completed. Score: {clarity_score:.2f}. "
                          f"Is clear: {is_clear}, Has ambiguity: {has_ambiguity}, "
                          f"Needs simplification: {needs_simplification}",
            metadata={
                'clarity_score': clarity_score,
                'is_clear': is_clear,
                'has_ambiguity': has_ambiguity,
                'needs_simplification': needs_simplification,
                'analysis_text_length': len(clarity_text),
                'section_found': clarity_section is not None
            }
        )
        
        return {
            'analysis_text': clarity_text,
            'is_clear': is_clear,
            'has_ambiguity': has_ambiguity,
            'needs_simplification': needs_simplification,
            'clarity_score': clarity_score
        }
    
    def _extract_completeness_analysis(self, response: str) -> Dict[str, Any]:
        """Extract completeness analysis from LLM response."""
        import re
        
        completeness_section = re.search(r'completeness[^:]*:?\s*(.+?)(?=\n\n|\n[A-Z]|\Z)', 
                                       response, re.IGNORECASE | re.DOTALL)
        
        completeness_text = completeness_section.group(1) if completeness_section else ""
        
        # Extract key insights
        is_complete = 'complete' in completeness_text.lower() and 'incomplete' not in completeness_text.lower()
        missing_elements = 'missing' in completeness_text.lower() or 'lack' in completeness_text.lower()
        needs_examples = 'example' in completeness_text.lower()
        completeness_score = self._extract_score(completeness_text, default=0.7)
        
        # Log detailed completeness analysis results
        self.llm_logger.log_agent_reasoning(
            reasoning_type="completeness_analysis_results",
            reasoning_text=f"Completeness analysis completed. Score: {completeness_score:.2f}. "
                          f"Is complete: {is_complete}, Missing elements: {missing_elements}, "
                          f"Needs examples: {needs_examples}",
            metadata={
                'completeness_score': completeness_score,
                'is_complete': is_complete,
                'missing_elements': missing_elements,
                'needs_examples': needs_examples,
                'analysis_text_length': len(completeness_text),
                'section_found': completeness_section is not None
            }
        )
        
        return {
            'analysis_text': completeness_text,
            'is_complete': is_complete,
            'missing_elements': missing_elements,
            'needs_examples': needs_examples,
            'completeness_score': completeness_score
        }
    
    def _extract_effectiveness_analysis(self, response: str) -> Dict[str, Any]:
        """Extract effectiveness analysis from LLM response."""
        import re
        
        effectiveness_section = re.search(r'effectiveness[^:]*:?\s*(.+?)(?=\n\n|\n[A-Z]|\Z)', 
                                        response, re.IGNORECASE | re.DOTALL)
        
        effectiveness_text = effectiveness_section.group(1) if effectiveness_section else ""
        
        # Extract key insights
        is_effective = 'effective' in effectiveness_text.lower()
        needs_optimization = 'optim' in effectiveness_text.lower() or 'improv' in effectiveness_text.lower()
        aligns_with_goals = 'align' in effectiveness_text.lower() or 'goal' in effectiveness_text.lower()
        effectiveness_score = self._extract_score(effectiveness_text, default=0.7)
        
        # Log detailed effectiveness analysis results
        self.llm_logger.log_agent_reasoning(
            reasoning_type="effectiveness_analysis_results",
            reasoning_text=f"Effectiveness analysis completed. Score: {effectiveness_score:.2f}. "
                          f"Is effective: {is_effective}, Needs optimization: {needs_optimization}, "
                          f"Aligns with goals: {aligns_with_goals}",
            metadata={
                'effectiveness_score': effectiveness_score,
                'is_effective': is_effective,
                'needs_optimization': needs_optimization,
                'aligns_with_goals': aligns_with_goals,
                'analysis_text_length': len(effectiveness_text),
                'section_found': effectiveness_section is not None
            }
        )
        
        return {
            'analysis_text': effectiveness_text,
            'is_effective': is_effective,
            'needs_optimization': needs_optimization,
            'aligns_with_goals': aligns_with_goals,
            'effectiveness_score': effectiveness_score
        }
    
    def _extract_best_practices_assessment(self, response: str) -> Dict[str, Any]:
        """Extract best practices assessment from LLM response."""
        import re
        
        # Look for best practices mentions
        best_practices_mentions = re.findall(r'best practice[s]?[^.]*', response, re.IGNORECASE)
        
        follows_best_practices = len(best_practices_mentions) > 0 and any(
            'follow' in mention.lower() or 'adhere' in mention.lower() 
            for mention in best_practices_mentions
        )
        
        violations = re.findall(r'violat[^.]*|deviat[^.]*|not follow[^.]*', response, re.IGNORECASE)
        best_practices_score = 0.8 if follows_best_practices else 0.5
        
        # Log detailed best practices assessment results
        self.llm_logger.log_agent_reasoning(
            reasoning_type="best_practices_assessment_results",
            reasoning_text=f"Best practices assessment completed. Score: {best_practices_score:.2f}. "
                          f"Follows best practices: {follows_best_practices}, "
                          f"Mentions found: {len(best_practices_mentions)}, "
                          f"Violations found: {len(violations)}",
            metadata={
                'best_practices_score': best_practices_score,
                'follows_best_practices': follows_best_practices,
                'mentions_count': len(best_practices_mentions),
                'violations_count': len(violations),
                'best_practices_mentions': best_practices_mentions[:5],  # Limit for logging
                'violations': violations[:3]  # Limit for logging
            }
        )
        
        return {
            'follows_best_practices': follows_best_practices,
            'best_practices_mentions': best_practices_mentions,
            'violations': violations,
            'best_practices_score': best_practices_score
        }
    
    def _extract_suggestions(self, parsed_response: Dict[str, Any]) -> List[str]:
        """Extract actionable suggestions from LLM response."""
        suggestions = parsed_response.get('recommendations', [])
        
        # Also look for improvement suggestions in the raw response
        import re
        raw_response = parsed_response['raw_response']
        
        # Log the start of suggestions extraction
        self.llm_logger.log_agent_reasoning(
            reasoning_type="suggestions_extraction_start",
            reasoning_text="Starting extraction of actionable suggestions from LLM response",
            metadata={
                'initial_suggestions_count': len(suggestions),
                'raw_response_length': len(raw_response)
            }
        )
        
        # Find recommendation sections
        recommendation_patterns = [
            r'recommend[^:]*:?\s*(.+?)(?=\n\n|\n[A-Z]|\Z)',
            r'suggest[^:]*:?\s*(.+?)(?=\n\n|\n[A-Z]|\Z)',
            r'improv[^:]*:?\s*(.+?)(?=\n\n|\n[A-Z]|\Z)'
        ]
        
        extracted_from_patterns = 0
        for pattern in recommendation_patterns:
            matches = re.findall(pattern, raw_response, re.IGNORECASE | re.DOTALL)
            for match in matches:
                # Split by bullet points or numbers
                items = re.split(r'\n\s*[\d\-\*]\.\s*', match)
                new_suggestions = [item.strip() for item in items if item.strip()]
                suggestions.extend(new_suggestions)
                extracted_from_patterns += len(new_suggestions)
        
        # Remove duplicates while preserving order
        unique_suggestions = []
        seen = set()
        for suggestion in suggestions:
            if suggestion not in seen:
                unique_suggestions.append(suggestion)
                seen.add(suggestion)
        
        final_suggestions = unique_suggestions[:10]  # Limit to top 10 suggestions
        
        # Log the completion of suggestions extraction
        self.llm_logger.log_agent_reasoning(
            reasoning_type="suggestions_extraction_complete",
            reasoning_text=f"Suggestions extraction completed. Found {len(final_suggestions)} unique suggestions "
                          f"from {len(suggestions)} total extracted (including duplicates)",
            metadata={
                'final_suggestions_count': len(final_suggestions),
                'total_extracted': len(suggestions),
                'extracted_from_patterns': extracted_from_patterns,
                'duplicates_removed': len(suggestions) - len(unique_suggestions),
                'suggestions_preview': final_suggestions[:3] if final_suggestions else []
            }
        )
        
        return final_suggestions
    
    def _extract_score(self, text: str, default: float = 0.7) -> float:
        """Extract numerical score from text."""
        import re
        
        # Look for score patterns
        score_patterns = [
            r'score[:\s]+(\d+(?:\.\d+)?)',
            r'(\d+(?:\.\d+)?)\s*[/\\]\s*(?:10|100)',
            r'(\d+(?:\.\d+)?)%'
        ]
        
        for pattern in score_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                score = float(match.group(1))
                # Normalize to 0-1 range
                if score > 1.0:
                    score = score / 10.0 if score <= 10.0 else score / 100.0
                return min(1.0, max(0.0, score))
        
        return default
    
    def _calculate_llm_confidence(self, parsed_response: Dict[str, Any], 
                                llm_response: Dict[str, Any]) -> float:
        """Calculate confidence score based on LLM response quality."""
        base_confidence = parsed_response.get('confidence', 0.8)
        
        # Adjust based on response quality indicators
        response_text = parsed_response['raw_response']
        
        # Higher confidence for detailed responses
        if len(response_text) > 500:
            base_confidence += 0.1
        
        # Higher confidence for structured responses
        if response_text.count('\n') > 10:
            base_confidence += 0.05
        
        # Higher confidence for specific recommendations
        if len(parsed_response.get('recommendations', [])) > 3:
            base_confidence += 0.05
        
        # Lower confidence for short or vague responses
        if len(response_text) < 200:
            base_confidence -= 0.2
        
        return min(1.0, max(0.0, base_confidence))
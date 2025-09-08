"""
LLMAnalyzerAgent for advanced prompt analysis using LLM capabilities.

This agent leverages Large Language Models with specialized system prompts
to perform sophisticated analysis of prompt structure, clarity, and effectiveness
using embedded prompt engineering best practices.
"""

from typing import Dict, Any, List, Optional
from agents.llm_agent import LLMAgent
from agents.base import AgentResult
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
            # Prepare analysis prompt for LLM
            analysis_prompt = self._build_analysis_prompt(prompt, context, history, feedback)
            
            # Call LLM for analysis
            llm_response = self._call_llm(analysis_prompt, context)
            
            if not llm_response['success']:
                return AgentResult(
                    agent_name=self.name,
                    success=False,
                    analysis={},
                    suggestions=[],
                    confidence_score=0.0,
                    error_message=f"LLM analysis failed: {llm_response['error']}"
                )
            
            # Parse and structure the LLM response
            parsed_response = self._parse_llm_response(llm_response['response'])
            
            # Extract structured analysis
            analysis = self._extract_analysis_components(parsed_response, llm_response)
            
            # Generate actionable suggestions
            suggestions = self._extract_suggestions(parsed_response)
            
            # Calculate confidence based on LLM response and analysis depth
            confidence_score = self._calculate_llm_confidence(parsed_response, llm_response)
            
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
                error_message=f"LLM analysis failed: {str(e)}"
            )
    
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
        
        analysis = {
            'llm_analysis': {
                'raw_response': raw_response,
                'model_used': llm_response['model'],
                'tokens_used': llm_response['tokens_used'],
                'reasoning': parsed_response.get('reasoning', ''),
                'confidence': parsed_response.get('confidence', 0.8)
            },
            'structure_analysis': self._extract_structure_analysis(raw_response),
            'clarity_analysis': self._extract_clarity_analysis(raw_response),
            'completeness_analysis': self._extract_completeness_analysis(raw_response),
            'effectiveness_analysis': self._extract_effectiveness_analysis(raw_response),
            'best_practices_assessment': self._extract_best_practices_assessment(raw_response)
        }
        
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
        
        return {
            'analysis_text': structure_text,
            'has_clear_sections': has_clear_sections,
            'has_logical_flow': has_logical_flow,
            'needs_restructuring': needs_restructuring,
            'structure_score': self._extract_score(structure_text, default=0.7)
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
        
        return {
            'analysis_text': clarity_text,
            'is_clear': is_clear,
            'has_ambiguity': has_ambiguity,
            'needs_simplification': needs_simplification,
            'clarity_score': self._extract_score(clarity_text, default=0.7)
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
        
        return {
            'analysis_text': completeness_text,
            'is_complete': is_complete,
            'missing_elements': missing_elements,
            'needs_examples': needs_examples,
            'completeness_score': self._extract_score(completeness_text, default=0.7)
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
        
        return {
            'analysis_text': effectiveness_text,
            'is_effective': is_effective,
            'needs_optimization': needs_optimization,
            'aligns_with_goals': aligns_with_goals,
            'effectiveness_score': self._extract_score(effectiveness_text, default=0.7)
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
        
        return {
            'follows_best_practices': follows_best_practices,
            'best_practices_mentions': best_practices_mentions,
            'violations': violations,
            'best_practices_score': 0.8 if follows_best_practices else 0.5
        }
    
    def _extract_suggestions(self, parsed_response: Dict[str, Any]) -> List[str]:
        """Extract actionable suggestions from LLM response."""
        suggestions = parsed_response.get('recommendations', [])
        
        # Also look for improvement suggestions in the raw response
        import re
        raw_response = parsed_response['raw_response']
        
        # Find recommendation sections
        recommendation_patterns = [
            r'recommend[^:]*:?\s*(.+?)(?=\n\n|\n[A-Z]|\Z)',
            r'suggest[^:]*:?\s*(.+?)(?=\n\n|\n[A-Z]|\Z)',
            r'improv[^:]*:?\s*(.+?)(?=\n\n|\n[A-Z]|\Z)'
        ]
        
        for pattern in recommendation_patterns:
            matches = re.findall(pattern, raw_response, re.IGNORECASE | re.DOTALL)
            for match in matches:
                # Split by bullet points or numbers
                items = re.split(r'\n\s*[\d\-\*]\.\s*', match)
                suggestions.extend([item.strip() for item in items if item.strip()])
        
        # Remove duplicates while preserving order
        unique_suggestions = []
        seen = set()
        for suggestion in suggestions:
            if suggestion not in seen:
                unique_suggestions.append(suggestion)
                seen.add(suggestion)
        
        return unique_suggestions[:10]  # Limit to top 10 suggestions
    
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
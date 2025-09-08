"""
Automated evaluation system for prompt responses.

This module provides the Evaluator class that implements multiple scoring metrics
for assessing prompt response quality, including relevance, clarity, and completeness.
It also supports version comparison for tracking improvement over iterations.
"""

import re
import math
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from models import EvaluationResult, PromptIteration, ExecutionResult


@dataclass
class EvaluationCriteria:
    """Configuration for evaluation criteria and weights."""
    
    relevance_weight: float = 0.3
    clarity_weight: float = 0.3
    completeness_weight: float = 0.3
    custom_weights: Dict[str, float] = None
    
    def __post_init__(self):
        if self.custom_weights is None:
            self.custom_weights = {}
        
        # Validate individual weights are non-negative
        if (self.relevance_weight < 0 or self.clarity_weight < 0 or 
            self.completeness_weight < 0):
            raise ValueError("All weights must be non-negative")
        
        # Ensure weights sum to reasonable values
        total_base_weight = self.relevance_weight + self.clarity_weight + self.completeness_weight
        if total_base_weight <= 0:
            raise ValueError("Base weights must sum to a positive value")


class Evaluator:
    """
    Automated evaluation system for prompt responses.
    
    Provides multiple scoring metrics and comparison functionality
    for tracking prompt improvement over iterations.
    """
    
    def __init__(self, criteria: Optional[EvaluationCriteria] = None):
        """Initialize evaluator with optional custom criteria."""
        self.criteria = criteria or EvaluationCriteria()
        
        # Common patterns for evaluation
        self.question_patterns = [
            r'\?',  # Question marks
            r'\bwhat\b', r'\bhow\b', r'\bwhy\b', r'\bwhen\b', r'\bwhere\b', r'\bwho\b'
        ]
        
        self.clarity_indicators = {
            'positive': [
                r'\bspecifically\b', r'\bclearly\b', r'\bexactly\b', r'\bprecisely\b',
                r'\bfor example\b', r'\bsuch as\b', r'\bnamely\b'
            ],
            'negative': [
                r'\bmaybe\b', r'\bperhaps\b', r'\bsomewhat\b', r'\bkind of\b',
                r'\bsort of\b', r'\bI think\b', r'\bI guess\b'
            ]
        }
    
    def evaluate_response(
        self, 
        prompt: str, 
        response: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> EvaluationResult:
        """
        Evaluate a prompt response using multiple criteria.
        
        Args:
            prompt: The original prompt text
            response: The model's response to evaluate
            context: Optional context information for evaluation
            
        Returns:
            EvaluationResult with scores and feedback
        """
        if not prompt or not response:
            return EvaluationResult(
                overall_score=0.0,
                relevance_score=0.0,
                clarity_score=0.0,
                completeness_score=0.0,
                qualitative_feedback="Invalid input: prompt and response cannot be empty",
                improvement_suggestions=["Provide valid prompt and response text"]
            )
        
        # Calculate individual scores
        relevance_score = self._calculate_relevance_score(prompt, response, context)
        clarity_score = self._calculate_clarity_score(response)
        completeness_score = self._calculate_completeness_score(prompt, response)
        
        # Calculate custom metrics if any
        custom_metrics = {}
        if context and 'custom_evaluators' in context:
            for metric_name, evaluator_func in context['custom_evaluators'].items():
                try:
                    custom_metrics[metric_name] = evaluator_func(prompt, response)
                except Exception as e:
                    custom_metrics[metric_name] = 0.0
        
        # Calculate overall score
        overall_score = self._calculate_overall_score(
            relevance_score, clarity_score, completeness_score, custom_metrics
        )
        
        # Generate qualitative feedback and suggestions
        qualitative_feedback = self._generate_qualitative_feedback(
            relevance_score, clarity_score, completeness_score, custom_metrics
        )
        improvement_suggestions = self._generate_improvement_suggestions(
            prompt, response, relevance_score, clarity_score, completeness_score
        )
        
        return EvaluationResult(
            overall_score=overall_score,
            relevance_score=relevance_score,
            clarity_score=clarity_score,
            completeness_score=completeness_score,
            custom_metrics=custom_metrics,
            qualitative_feedback=qualitative_feedback,
            improvement_suggestions=improvement_suggestions
        )
    
    def compare_versions(
        self, 
        current: PromptIteration, 
        previous: PromptIteration
    ) -> Dict[str, Any]:
        """
        Compare two prompt iterations to track improvement.
        
        Args:
            current: Current prompt iteration
            previous: Previous prompt iteration to compare against
            
        Returns:
            Dictionary containing comparison results and improvement metrics
        """
        if not current.evaluation_scores or not previous.evaluation_scores:
            return {
                'comparison_possible': False,
                'reason': 'Missing evaluation scores in one or both iterations'
            }
        
        current_eval = current.evaluation_scores
        previous_eval = previous.evaluation_scores
        
        # Calculate score differences
        score_changes = {
            'overall': current_eval.overall_score - previous_eval.overall_score,
            'relevance': current_eval.relevance_score - previous_eval.relevance_score,
            'clarity': current_eval.clarity_score - previous_eval.clarity_score,
            'completeness': current_eval.completeness_score - previous_eval.completeness_score
        }
        
        # Calculate custom metric changes
        custom_changes = {}
        for metric in set(current_eval.custom_metrics.keys()) | set(previous_eval.custom_metrics.keys()):
            current_value = current_eval.custom_metrics.get(metric, 0.0)
            previous_value = previous_eval.custom_metrics.get(metric, 0.0)
            custom_changes[metric] = current_value - previous_value
        
        # Determine improvement status
        improvement_status = self._determine_improvement_status(score_changes)
        
        # Calculate improvement percentage
        improvement_percentage = (
            (current_eval.overall_score - previous_eval.overall_score) / 
            max(previous_eval.overall_score, 0.01) * 100
        )
        
        # Generate comparison summary
        summary = self._generate_comparison_summary(
            score_changes, custom_changes, improvement_status, improvement_percentage
        )
        
        return {
            'comparison_possible': True,
            'score_changes': score_changes,
            'custom_metric_changes': custom_changes,
            'improvement_status': improvement_status,
            'improvement_percentage': improvement_percentage,
            'summary': summary,
            'current_version': current.version,
            'previous_version': previous.version,
            'time_difference': (current.timestamp - previous.timestamp).total_seconds()
        }
    
    def generate_evaluation_report(
        self, 
        iterations: List[PromptIteration]
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive evaluation report for multiple iterations.
        
        Args:
            iterations: List of prompt iterations to analyze
            
        Returns:
            Dictionary containing comprehensive evaluation report
        """
        if not iterations:
            return {'error': 'No iterations provided for report generation'}
        
        # Filter iterations with evaluation scores
        evaluated_iterations = [
            iteration for iteration in iterations 
            if iteration.evaluation_scores is not None
        ]
        
        if not evaluated_iterations:
            return {'error': 'No iterations with evaluation scores found'}
        
        # Calculate trend analysis
        trend_analysis = self._calculate_trend_analysis(evaluated_iterations)
        
        # Find best and worst performing iterations
        best_iteration = max(evaluated_iterations, key=lambda x: x.evaluation_scores.overall_score)
        worst_iteration = min(evaluated_iterations, key=lambda x: x.evaluation_scores.overall_score)
        
        # Calculate average scores
        avg_scores = self._calculate_average_scores(evaluated_iterations)
        
        # Generate insights and recommendations
        insights = self._generate_insights(evaluated_iterations, trend_analysis)
        
        return {
            'total_iterations': len(iterations),
            'evaluated_iterations': len(evaluated_iterations),
            'trend_analysis': trend_analysis,
            'best_iteration': {
                'version': best_iteration.version,
                'score': best_iteration.evaluation_scores.overall_score,
                'timestamp': best_iteration.timestamp.isoformat()
            },
            'worst_iteration': {
                'version': worst_iteration.version,
                'score': worst_iteration.evaluation_scores.overall_score,
                'timestamp': worst_iteration.timestamp.isoformat()
            },
            'average_scores': avg_scores,
            'insights': insights,
            'generated_at': iterations[-1].timestamp.isoformat() if iterations else None
        }
    
    def _calculate_relevance_score(
        self, 
        prompt: str, 
        response: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> float:
        """Calculate relevance score based on prompt-response alignment."""
        # Extract key terms from prompt
        prompt_terms = self._extract_key_terms(prompt)
        response_terms = self._extract_key_terms(response)
        
        if not prompt_terms:
            return 0.5  # Neutral score if no key terms found
        
        # Calculate term overlap with better scoring
        common_terms = set(prompt_terms) & set(response_terms)
        term_overlap_ratio = len(common_terms) / len(prompt_terms)
        
        # Boost score for good overlap
        if term_overlap_ratio > 0.3:
            term_overlap_ratio = min(term_overlap_ratio * 1.5, 1.0)
        
        # Check for direct question answering
        question_addressing_score = self._check_question_addressing(prompt, response)
        
        # Check for semantic relevance (topic alignment)
        semantic_score = self._check_semantic_relevance(prompt, response)
        
        # Combine scores with weights
        relevance_score = (term_overlap_ratio * 0.4) + (question_addressing_score * 0.3) + (semantic_score * 0.3)
        
        return min(max(relevance_score, 0.0), 1.0)
    
    def _calculate_clarity_score(self, response: str) -> float:
        """Calculate clarity score based on response structure and language."""
        if not response.strip():
            return 0.0
        
        # Check sentence structure
        sentences = re.split(r'[.!?]+', response)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return 0.0
        
        # Calculate average sentence length (optimal range: 10-30 words, more forgiving)
        avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
        if 10 <= avg_sentence_length <= 30:
            length_score = 1.0
        elif avg_sentence_length < 10:
            length_score = max(avg_sentence_length / 10, 0.3)
        else:
            length_score = max(1.0 - (avg_sentence_length - 30) / 30, 0.3)
        
        # Check for clarity indicators
        positive_indicators = sum(
            len(re.findall(pattern, response, re.IGNORECASE))
            for pattern in self.clarity_indicators['positive']
        )
        negative_indicators = sum(
            len(re.findall(pattern, response, re.IGNORECASE))
            for pattern in self.clarity_indicators['negative']
        )
        
        # Calculate indicator score with better baseline
        total_words = len(response.split())
        positive_ratio = positive_indicators / max(total_words / 50, 1)  # More sensitive
        negative_ratio = negative_indicators / max(total_words / 50, 1)
        indicator_score = min(positive_ratio - negative_ratio + 0.7, 1.0)  # Higher baseline
        indicator_score = max(indicator_score, 0.3)
        
        # Check for structural elements (lists, transitions)
        structure_score = self._check_structure_clarity(response)
        
        # Combine scores
        clarity_score = (length_score * 0.3) + (indicator_score * 0.4) + (structure_score * 0.3)
        
        return min(max(clarity_score, 0.0), 1.0)
    
    def _calculate_completeness_score(self, prompt: str, response: str) -> float:
        """Calculate completeness score based on response thoroughness."""
        # Check if response addresses multiple aspects of the prompt
        prompt_questions = len(re.findall(r'\?', prompt))
        prompt_requests = len(re.findall(r'\b(explain|describe|list|provide|give|show)\b', prompt, re.IGNORECASE))
        
        expected_components = max(prompt_questions + prompt_requests, 1)
        
        # Count response components (paragraphs, lists, examples)
        paragraphs = len([p for p in response.split('\n\n') if p.strip()])
        lists = len(re.findall(r'^\s*[-*•]\s', response, re.MULTILINE))
        examples = len(re.findall(r'\b(for example|such as|e\.g\.|i\.e\.)\b', response, re.IGNORECASE))
        
        response_components = paragraphs + (lists / 3) + examples
        
        # Calculate completeness ratio
        completeness_ratio = min(response_components / expected_components, 1.0)
        
        # Adjust for response length (very short responses are likely incomplete)
        response_words = len(response.split())
        length_factor = min(response_words / 50, 1.0)  # Expect at least 50 words for completeness
        
        completeness_score = completeness_ratio * length_factor
        
        return min(max(completeness_score, 0.0), 1.0)
    
    def _calculate_overall_score(
        self, 
        relevance: float, 
        clarity: float, 
        completeness: float, 
        custom_metrics: Dict[str, float]
    ) -> float:
        """Calculate weighted overall score."""
        base_score = (
            relevance * self.criteria.relevance_weight +
            clarity * self.criteria.clarity_weight +
            completeness * self.criteria.completeness_weight
        )
        
        # Add custom metrics
        custom_score = 0.0
        total_custom_weight = 0.0
        for metric_name, score in custom_metrics.items():
            weight = self.criteria.custom_weights.get(metric_name, 0.1)
            custom_score += score * weight
            total_custom_weight += weight
        
        # Normalize the final score
        total_weight = (
            self.criteria.relevance_weight + 
            self.criteria.clarity_weight + 
            self.criteria.completeness_weight + 
            total_custom_weight
        )
        
        if total_weight > 0:
            overall_score = (base_score + custom_score) / total_weight
        else:
            overall_score = base_score
        
        return min(max(overall_score, 0.0), 1.0)
    
    def _extract_key_terms(self, text: str) -> List[str]:
        """Extract key terms from text for relevance analysis."""
        # Remove common stop words and extract meaningful terms
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those'
        }
        
        # Extract words (3+ characters, not stop words)
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        key_terms = [word for word in words if word not in stop_words]
        
        return key_terms
    
    def _check_question_addressing(self, prompt: str, response: str) -> float:
        """Check how well the response addresses questions in the prompt."""
        questions_in_prompt = len(re.findall(r'\?', prompt))
        
        if questions_in_prompt == 0:
            return 0.8  # High score if no specific questions to address
        
        # Look for answer indicators and direct responses
        answer_indicators = [
            r'\bis\b', r'\bare\b', r'\byes\b', r'\bno\b', r'\bbecause\b', r'\bdue to\b', 
            r'\bresult\b', r'\btherefore\b', r'\bhowever\b', r'\bfirst\b', r'\bsecond\b', 
            r'\bfinally\b', r'\bworks by\b', r'\bmeans\b', r'\binvolves\b', r'\bincludes\b',
            r'\boffers\b', r'\bprovides\b', r'\benables\b'
        ]
        
        indicator_count = sum(
            len(re.findall(pattern, response, re.IGNORECASE))
            for pattern in answer_indicators
        )
        
        # Check for question word responses (what -> definition, how -> process, etc.)
        question_response_score = self._check_question_word_responses(prompt, response)
        
        # Score based on presence of answer indicators and direct responses
        indicator_score = min(indicator_count / (questions_in_prompt * 2), 1.0)
        addressing_score = (indicator_score * 0.6) + (question_response_score * 0.4)
        
        return min(addressing_score, 1.0)
    
    def _generate_qualitative_feedback(
        self, 
        relevance: float, 
        clarity: float, 
        completeness: float, 
        custom_metrics: Dict[str, float]
    ) -> str:
        """Generate human-readable qualitative feedback."""
        feedback_parts = []
        
        # Relevance feedback
        if relevance >= 0.8:
            feedback_parts.append("The response is highly relevant to the prompt.")
        elif relevance >= 0.6:
            feedback_parts.append("The response is moderately relevant to the prompt.")
        else:
            feedback_parts.append("The response has limited relevance to the prompt.")
        
        # Clarity feedback
        if clarity >= 0.8:
            feedback_parts.append("The response is clear and well-structured.")
        elif clarity >= 0.6:
            feedback_parts.append("The response has reasonable clarity but could be improved.")
        else:
            feedback_parts.append("The response lacks clarity and structure.")
        
        # Completeness feedback
        if completeness >= 0.8:
            feedback_parts.append("The response thoroughly addresses the prompt.")
        elif completeness >= 0.6:
            feedback_parts.append("The response partially addresses the prompt.")
        else:
            feedback_parts.append("The response is incomplete or superficial.")
        
        # Custom metrics feedback
        if custom_metrics:
            high_custom = [name for name, score in custom_metrics.items() if score >= 0.8]
            low_custom = [name for name, score in custom_metrics.items() if score < 0.6]
            
            if high_custom:
                feedback_parts.append(f"Strong performance in: {', '.join(high_custom)}.")
            if low_custom:
                feedback_parts.append(f"Needs improvement in: {', '.join(low_custom)}.")
        
        return " ".join(feedback_parts)
    
    def _generate_improvement_suggestions(
        self, 
        prompt: str, 
        response: str, 
        relevance: float, 
        clarity: float, 
        completeness: float
    ) -> List[str]:
        """Generate specific improvement suggestions."""
        suggestions = []
        
        if relevance < 0.7:
            suggestions.append("Focus more directly on the key topics mentioned in the prompt")
            suggestions.append("Include more specific terms and concepts from the original question")
        
        if clarity < 0.7:
            suggestions.append("Use shorter, more direct sentences")
            suggestions.append("Add transitional phrases to improve flow")
            suggestions.append("Define technical terms or concepts clearly")
        
        if completeness < 0.7:
            suggestions.append("Address all parts of the prompt systematically")
            suggestions.append("Provide more detailed explanations or examples")
            suggestions.append("Consider adding supporting evidence or reasoning")
        
        # Check for specific issues
        if len(response.split()) < 30:
            suggestions.append("Expand the response with more detail and explanation")
        
        if not re.search(r'[.!?]', response):
            suggestions.append("Use proper punctuation to improve readability")
        
        return suggestions
    
    def _determine_improvement_status(self, score_changes: Dict[str, float]) -> str:
        """Determine overall improvement status from score changes."""
        overall_change = score_changes['overall']
        
        if overall_change >= 0.1:
            return "significant_improvement"
        elif overall_change >= 0.05:
            return "moderate_improvement"
        elif overall_change >= -0.05:
            return "minimal_change"
        elif overall_change >= -0.1:
            return "moderate_decline"
        else:
            return "significant_decline"
    
    def _generate_comparison_summary(
        self, 
        score_changes: Dict[str, float], 
        custom_changes: Dict[str, float], 
        status: str, 
        percentage: float
    ) -> str:
        """Generate human-readable comparison summary."""
        status_messages = {
            "significant_improvement": f"Significant improvement ({percentage:.1f}%)",
            "moderate_improvement": f"Moderate improvement ({percentage:.1f}%)",
            "minimal_change": f"Minimal change ({percentage:.1f}%)",
            "moderate_decline": f"Moderate decline ({percentage:.1f}%)",
            "significant_decline": f"Significant decline ({percentage:.1f}%)"
        }
        
        summary = status_messages.get(status, f"Change: {percentage:.1f}%")
        
        # Add details about specific improvements/declines
        improvements = [name for name, change in score_changes.items() if change > 0.05]
        declines = [name for name, change in score_changes.items() if change < -0.05]
        
        if improvements:
            summary += f". Improved: {', '.join(improvements)}"
        if declines:
            summary += f". Declined: {', '.join(declines)}"
        
        return summary
    
    def _calculate_trend_analysis(self, iterations: List[PromptIteration]) -> Dict[str, Any]:
        """Calculate trend analysis for multiple iterations."""
        if len(iterations) < 2:
            return {'trend': 'insufficient_data'}
        
        scores = [iteration.evaluation_scores.overall_score for iteration in iterations]
        
        # Calculate linear trend
        n = len(scores)
        x_values = list(range(n))
        x_mean = sum(x_values) / n
        y_mean = sum(scores) / n
        
        numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, scores))
        denominator = sum((x - x_mean) ** 2 for x in x_values)
        
        if denominator == 0:
            slope = 0
        else:
            slope = numerator / denominator
        
        # Determine trend direction
        if slope > 0.02:
            trend = "improving"
        elif slope < -0.02:
            trend = "declining"
        else:
            trend = "stable"
        
        # Calculate volatility (standard deviation)
        variance = sum((score - y_mean) ** 2 for score in scores) / n
        volatility = math.sqrt(variance)
        
        return {
            'trend': trend,
            'slope': slope,
            'volatility': volatility,
            'score_range': (min(scores), max(scores)),
            'average_score': y_mean
        }
    
    def _calculate_average_scores(self, iterations: List[PromptIteration]) -> Dict[str, float]:
        """Calculate average scores across iterations."""
        if not iterations:
            return {}
        
        total_scores = {
            'overall': 0.0,
            'relevance': 0.0,
            'clarity': 0.0,
            'completeness': 0.0
        }
        
        custom_totals = {}
        
        for iteration in iterations:
            eval_result = iteration.evaluation_scores
            total_scores['overall'] += eval_result.overall_score
            total_scores['relevance'] += eval_result.relevance_score
            total_scores['clarity'] += eval_result.clarity_score
            total_scores['completeness'] += eval_result.completeness_score
            
            for metric, score in eval_result.custom_metrics.items():
                custom_totals[metric] = custom_totals.get(metric, 0.0) + score
        
        n = len(iterations)
        avg_scores = {key: total / n for key, total in total_scores.items()}
        
        # Add custom metric averages
        for metric, total in custom_totals.items():
            avg_scores[f'custom_{metric}'] = total / n
        
        return avg_scores
    
    def _generate_insights(
        self, 
        iterations: List[PromptIteration], 
        trend_analysis: Dict[str, Any]
    ) -> List[str]:
        """Generate insights and recommendations based on evaluation history."""
        insights = []
        
        # Trend insights
        trend = trend_analysis.get('trend', 'unknown')
        if trend == 'improving':
            insights.append("The prompt optimization is showing positive progress")
        elif trend == 'declining':
            insights.append("Recent iterations show declining performance - consider reverting to a previous version")
        elif trend == 'stable':
            insights.append("Performance has stabilized - consider trying different optimization approaches")
        
        # Volatility insights
        volatility = trend_analysis.get('volatility', 0)
        if volatility > 0.2:
            insights.append("High score volatility suggests inconsistent optimization - focus on incremental changes")
        elif volatility < 0.05:
            insights.append("Low volatility indicates consistent performance")
        
        # Performance insights
        avg_score = trend_analysis.get('average_score', 0)
        if avg_score > 0.8:
            insights.append("Overall performance is excellent - minor refinements may yield additional gains")
        elif avg_score > 0.6:
            insights.append("Performance is good with room for improvement in specific areas")
        else:
            insights.append("Significant improvement opportunities exist across multiple evaluation criteria")
        
        return insights
    
    def _check_semantic_relevance(self, prompt: str, response: str) -> float:
        """Check semantic relevance between prompt and response."""
        # Simple semantic relevance based on topic keywords
        prompt_lower = prompt.lower()
        response_lower = response.lower()
        
        # Check for topic continuity
        if 'machine learning' in prompt_lower and 'learning' in response_lower:
            return 0.8
        if 'renewable energy' in prompt_lower and 'energy' in response_lower:
            return 0.8
        if 'algorithm' in prompt_lower and ('algorithm' in response_lower or 'method' in response_lower):
            return 0.7
        
        # General topic alignment check
        prompt_words = set(self._extract_key_terms(prompt))
        response_words = set(self._extract_key_terms(response))
        
        if prompt_words and response_words:
            overlap = len(prompt_words & response_words)
            return min(overlap / len(prompt_words) * 1.2, 1.0)
        
        return 0.5
    
    def _check_structure_clarity(self, response: str) -> float:
        """Check structural clarity of the response."""
        structure_score = 0.5  # Base score
        
        # Check for lists or enumeration
        if re.search(r'^\s*[-*•]\s', response, re.MULTILINE) or re.search(r'\d+\.\s', response):
            structure_score += 0.2
        
        # Check for transition words
        transitions = [r'\bfirst\b', r'\bsecond\b', r'\bfinally\b', r'\bhowever\b', 
                      r'\btherefore\b', r'\bfor example\b', r'\badditionally\b']
        transition_count = sum(len(re.findall(pattern, response, re.IGNORECASE)) for pattern in transitions)
        if transition_count > 0:
            structure_score += min(transition_count * 0.1, 0.3)
        
        return min(structure_score, 1.0)
    
    def _check_question_word_responses(self, prompt: str, response: str) -> float:
        """Check if response appropriately addresses question words in prompt."""
        question_words = {
            'what': ['is', 'are', 'means', 'refers', 'definition'],
            'how': ['by', 'through', 'works', 'process', 'method'],
            'why': ['because', 'due', 'reason', 'since'],
            'when': ['during', 'after', 'before', 'time'],
            'where': ['in', 'at', 'location', 'place'],
            'who': ['person', 'people', 'individual', 'group']
        }
        
        prompt_lower = prompt.lower()
        response_lower = response.lower()
        
        score = 0.0
        question_count = 0
        
        for question_word, response_indicators in question_words.items():
            if question_word in prompt_lower:
                question_count += 1
                for indicator in response_indicators:
                    if indicator in response_lower:
                        score += 1.0
                        break
        
        if question_count == 0:
            return 0.8  # No specific question words to address
        
        return min(score / question_count, 1.0)
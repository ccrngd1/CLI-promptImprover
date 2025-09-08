"""
AgentEnsemble for coordinating multiple agents and consensus mechanisms.

This module implements the coordination layer that manages multiple agents,
handles their collaboration, implements voting mechanisms for conflicting
recommendations, and provides timeout handling and error recovery.
"""

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from agents.base import Agent, AgentResult
from agents.analyzer import AnalyzerAgent
from agents.refiner import RefinerAgent
from agents.validator import ValidatorAgent
from models import PromptIteration, UserFeedback


@dataclass
class EnsembleResult:
    """Represents the result of ensemble processing."""
    
    success: bool
    agent_results: Dict[str, AgentResult]
    consensus_analysis: Dict[str, Any]
    final_recommendations: List[str]
    confidence_score: float
    processing_time: float
    errors: List[str] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []


@dataclass
class ConsensusConfig:
    """Configuration for consensus mechanisms."""
    
    voting_method: str = "weighted"  # "majority", "weighted", "unanimous"
    confidence_threshold: float = 0.5  # Lowered from 0.6 to be more permissive
    agreement_threshold: float = 0.7
    timeout_seconds: float = 30.0
    retry_attempts: int = 2
    require_validator_approval: bool = True


class AgentEnsemble:
    """
    Coordinates multiple agents for collaborative prompt improvement.
    
    Manages agent execution, implements consensus mechanisms, handles timeouts,
    and provides error recovery for robust multi-agent collaboration.
    """
    
    def __init__(self, config: Optional[ConsensusConfig] = None):
        """
        Initialize the AgentEnsemble.
        
        Args:
            config: Optional consensus configuration
        """
        self.config = config or ConsensusConfig()
        
        # Initialize agents
        self.agents = {
            'analyzer': AnalyzerAgent(),
            'refiner': RefinerAgent(),
            'validator': ValidatorAgent()
        }
        
        # Execution tracking
        self.execution_history = []
        self.performance_metrics = {}
    
    def process_prompt(self, 
                      prompt: str,
                      context: Optional[Dict[str, Any]] = None,
                      history: Optional[List[PromptIteration]] = None,
                      feedback: Optional[UserFeedback] = None,
                      agent_subset: Optional[List[str]] = None) -> EnsembleResult:
        """
        Process a prompt using multiple agents with consensus mechanisms.
        
        Args:
            prompt: The prompt text to process
            context: Optional context about the prompt's intended use
            history: Optional list of previous prompt iterations
            feedback: Optional user feedback from previous iterations
            agent_subset: Optional list of specific agents to use
            
        Returns:
            EnsembleResult containing consensus analysis and recommendations
        """
        start_time = time.time()
        
        # Determine which agents to use
        active_agents = self._select_agents(agent_subset)
        
        # Execute agents with timeout handling
        agent_results = self._execute_agents_with_timeout(
            active_agents, prompt, context, history, feedback
        )
        
        # Analyze consensus and conflicts
        consensus_analysis = self._analyze_consensus(agent_results)
        
        # Generate final recommendations
        final_recommendations = self._generate_consensus_recommendations(
            agent_results, consensus_analysis
        )
        
        # Calculate overall confidence
        confidence_score = self._calculate_ensemble_confidence(
            agent_results, consensus_analysis
        )
        
        # Determine overall success
        success = self._determine_success(agent_results, consensus_analysis)
        
        processing_time = time.time() - start_time
        
        # Update performance metrics
        self._update_performance_metrics(agent_results, processing_time)
        
        return EnsembleResult(
            success=success,
            agent_results=agent_results,
            consensus_analysis=consensus_analysis,
            final_recommendations=final_recommendations,
            confidence_score=confidence_score,
            processing_time=processing_time,
            errors=self._collect_errors(agent_results)
        )
    
    def _select_agents(self, agent_subset: Optional[List[str]]) -> Dict[str, Agent]:
        """Select which agents to use for processing."""
        if agent_subset:
            return {name: agent for name, agent in self.agents.items() 
                   if name in agent_subset}
        return self.agents.copy()
    
    def _execute_agents_with_timeout(self, 
                                   agents: Dict[str, Agent],
                                   prompt: str,
                                   context: Optional[Dict[str, Any]],
                                   history: Optional[List[PromptIteration]],
                                   feedback: Optional[UserFeedback]) -> Dict[str, AgentResult]:
        """Execute agents with timeout handling and error recovery."""
        results = {}
        
        with ThreadPoolExecutor(max_workers=len(agents)) as executor:
            # Submit all agent tasks
            future_to_agent = {}
            for agent_name, agent in agents.items():
                future = executor.submit(
                    self._execute_agent_with_retry,
                    agent, prompt, context, history, feedback
                )
                future_to_agent[future] = agent_name
            
            # Collect results with timeout
            for future in future_to_agent:
                agent_name = future_to_agent[future]
                try:
                    result = future.result(timeout=self.config.timeout_seconds)
                    results[agent_name] = result
                except FutureTimeoutError:
                    # Create timeout error result
                    results[agent_name] = AgentResult(
                        agent_name=agent_name,
                        success=False,
                        analysis={},
                        suggestions=[],
                        confidence_score=0.0,
                        error_message=f"Agent execution timed out after {self.config.timeout_seconds}s"
                    )
                except Exception as e:
                    # Create error result
                    results[agent_name] = AgentResult(
                        agent_name=agent_name,
                        success=False,
                        analysis={},
                        suggestions=[],
                        confidence_score=0.0,
                        error_message=f"Agent execution failed: {str(e)}"
                    )
        
        return results
    
    def _execute_agent_with_retry(self,
                                agent: Agent,
                                prompt: str,
                                context: Optional[Dict[str, Any]],
                                history: Optional[List[PromptIteration]],
                                feedback: Optional[UserFeedback]) -> AgentResult:
        """Execute an agent with retry logic."""
        last_error = None
        
        for attempt in range(self.config.retry_attempts + 1):
            try:
                result = agent.process(prompt, context, history, feedback)
                if result.success:
                    return result
                else:
                    last_error = result.error_message
            except Exception as e:
                last_error = str(e)
                if attempt < self.config.retry_attempts:
                    time.sleep(0.1 * (attempt + 1))  # Exponential backoff
        
        # All retries failed
        return AgentResult(
            agent_name=agent.get_name(),
            success=False,
            analysis={},
            suggestions=[],
            confidence_score=0.0,
            error_message=f"Agent failed after {self.config.retry_attempts + 1} attempts: {last_error}"
        )
    
    def _analyze_consensus(self, agent_results: Dict[str, AgentResult]) -> Dict[str, Any]:
        """Analyze consensus and conflicts between agent results."""
        successful_results = {name: result for name, result in agent_results.items() 
                            if result.success}
        
        if not successful_results:
            return {
                'consensus_level': 0.0,
                'agreement_areas': [],
                'conflict_areas': [],
                'voting_results': {},
                'consensus_type': 'none'
            }
        
        # Analyze suggestion overlap
        suggestion_analysis = self._analyze_suggestion_consensus(successful_results)
        
        # Analyze confidence agreement
        confidence_analysis = self._analyze_confidence_consensus(successful_results)
        
        # Perform voting on key decisions
        voting_results = self._perform_voting(successful_results)
        
        # Calculate overall consensus level
        consensus_level = self._calculate_consensus_level(
            suggestion_analysis, confidence_analysis, voting_results
        )
        
        # Determine consensus type
        consensus_type = self._determine_consensus_type(consensus_level)
        
        return {
            'consensus_level': consensus_level,
            'agreement_areas': suggestion_analysis['agreements'],
            'conflict_areas': suggestion_analysis['conflicts'],
            'confidence_analysis': confidence_analysis,
            'voting_results': voting_results,
            'consensus_type': consensus_type,
            'successful_agents': list(successful_results.keys()),
            'failed_agents': [name for name, result in agent_results.items() 
                            if not result.success]
        }
    
    def _analyze_suggestion_consensus(self, results: Dict[str, AgentResult]) -> Dict[str, Any]:
        """Analyze consensus in agent suggestions."""
        all_suggestions = []
        agent_suggestions = {}
        
        for agent_name, result in results.items():
            suggestions = result.suggestions
            all_suggestions.extend(suggestions)
            agent_suggestions[agent_name] = suggestions
        
        # Find common suggestions (simplified keyword matching)
        suggestion_counts = {}
        for suggestion in all_suggestions:
            # Normalize suggestion for comparison
            normalized = suggestion.lower().strip()
            suggestion_counts[normalized] = suggestion_counts.get(normalized, 0) + 1
        
        # Identify agreements (suggestions mentioned by multiple agents)
        agreements = []
        conflicts = []
        
        for suggestion, count in suggestion_counts.items():
            if count > 1:
                agreements.append({
                    'suggestion': suggestion,
                    'agent_count': count,
                    'agents': [name for name, suggestions in agent_suggestions.items()
                             if any(suggestion in s.lower() for s in suggestions)]
                })
            else:
                conflicts.append({
                    'suggestion': suggestion,
                    'agent': next(name for name, suggestions in agent_suggestions.items()
                                if any(suggestion in s.lower() for s in suggestions))
                })
        
        return {
            'agreements': agreements,
            'conflicts': conflicts,
            'total_suggestions': len(all_suggestions),
            'unique_suggestions': len(suggestion_counts),
            'agreement_ratio': len(agreements) / max(len(suggestion_counts), 1)
        }
    
    def _analyze_confidence_consensus(self, results: Dict[str, AgentResult]) -> Dict[str, Any]:
        """Analyze consensus in agent confidence scores."""
        confidence_scores = [result.confidence_score for result in results.values()]
        
        if not confidence_scores:
            return {'agreement': 0.0, 'variance': 0.0, 'mean': 0.0}
        
        mean_confidence = sum(confidence_scores) / len(confidence_scores)
        variance = sum((score - mean_confidence) ** 2 for score in confidence_scores) / len(confidence_scores)
        
        # High agreement if variance is low
        agreement = max(0.0, 1.0 - (variance * 4))  # Scale variance to 0-1
        
        return {
            'agreement': agreement,
            'variance': variance,
            'mean': mean_confidence,
            'scores': confidence_scores,
            'range': max(confidence_scores) - min(confidence_scores)
        }
    
    def _perform_voting(self, results: Dict[str, AgentResult]) -> Dict[str, Any]:
        """Perform voting on key decisions based on agent results."""
        voting_results = {}
        
        # Vote on overall prompt quality
        quality_votes = {}
        for agent_name, result in results.items():
            if result.confidence_score >= 0.8:
                quality_votes[agent_name] = 'high'
            elif result.confidence_score >= 0.6:
                quality_votes[agent_name] = 'medium'
            else:
                quality_votes[agent_name] = 'low'
        
        voting_results['prompt_quality'] = self._tally_votes(quality_votes)
        
        # Vote on need for refinement (based on suggestions count)
        refinement_votes = {}
        for agent_name, result in results.items():
            suggestion_count = len(result.suggestions)
            if suggestion_count >= 5:
                refinement_votes[agent_name] = 'major_refinement'
            elif suggestion_count >= 2:
                refinement_votes[agent_name] = 'minor_refinement'
            else:
                refinement_votes[agent_name] = 'no_refinement'
        
        voting_results['refinement_need'] = self._tally_votes(refinement_votes)
        
        # Special handling for validator approval
        if 'validator' in results:
            validator_result = results['validator']
            if validator_result.success and validator_result.analysis:
                passes_validation = validator_result.analysis.get('passes_validation', False)
                voting_results['validator_approval'] = passes_validation
            else:
                voting_results['validator_approval'] = False
        
        return voting_results
    
    def _tally_votes(self, votes: Dict[str, str]) -> Dict[str, Any]:
        """Tally votes and determine winner based on voting method."""
        if not votes:
            return {'winner': None, 'votes': {}, 'method': self.config.voting_method}
        
        vote_counts = {}
        for vote in votes.values():
            vote_counts[vote] = vote_counts.get(vote, 0) + 1
        
        if self.config.voting_method == "majority":
            winner = max(vote_counts, key=vote_counts.get)
            winner_count = vote_counts[winner]
            total_votes = sum(vote_counts.values())
            
            # Require actual majority (>50%)
            if winner_count > total_votes / 2:
                return {
                    'winner': winner,
                    'votes': vote_counts,
                    'method': 'majority',
                    'confidence': winner_count / total_votes
                }
            else:
                return {
                    'winner': None,
                    'votes': vote_counts,
                    'method': 'majority',
                    'confidence': 0.0
                }
        
        elif self.config.voting_method == "weighted":
            # Weight votes by agent confidence (simplified - could use actual confidence scores)
            winner = max(vote_counts, key=vote_counts.get)
            return {
                'winner': winner,
                'votes': vote_counts,
                'method': 'weighted',
                'confidence': vote_counts[winner] / sum(vote_counts.values())
            }
        
        elif self.config.voting_method == "unanimous":
            if len(vote_counts) == 1:
                winner = list(vote_counts.keys())[0]
                return {
                    'winner': winner,
                    'votes': vote_counts,
                    'method': 'unanimous',
                    'confidence': 1.0
                }
            else:
                return {
                    'winner': None,
                    'votes': vote_counts,
                    'method': 'unanimous',
                    'confidence': 0.0
                }
        
        return {'winner': None, 'votes': vote_counts, 'method': 'unknown'}
    
    def _calculate_consensus_level(self, 
                                 suggestion_analysis: Dict[str, Any],
                                 confidence_analysis: Dict[str, Any],
                                 voting_results: Dict[str, Any]) -> float:
        """Calculate overall consensus level."""
        # Weight different factors
        suggestion_weight = 0.4
        confidence_weight = 0.3
        voting_weight = 0.3
        
        # Suggestion consensus
        suggestion_score = suggestion_analysis.get('agreement_ratio', 0.0)
        
        # Confidence consensus
        confidence_score = confidence_analysis.get('agreement', 0.0)
        
        # Voting consensus (average confidence of voting results)
        voting_confidences = []
        for vote_result in voting_results.values():
            if isinstance(vote_result, dict) and 'confidence' in vote_result:
                voting_confidences.append(vote_result['confidence'])
            elif isinstance(vote_result, bool):
                voting_confidences.append(1.0 if vote_result else 0.0)
        
        voting_score = sum(voting_confidences) / max(len(voting_confidences), 1)
        
        # Calculate weighted consensus
        consensus_level = (
            suggestion_score * suggestion_weight +
            confidence_score * confidence_weight +
            voting_score * voting_weight
        )
        
        return min(1.0, max(0.0, consensus_level))
    
    def _determine_consensus_type(self, consensus_level: float) -> str:
        """Determine the type of consensus achieved."""
        if consensus_level >= 0.9:
            return 'strong_consensus'
        elif consensus_level >= 0.7:
            return 'moderate_consensus'
        elif consensus_level >= 0.5:
            return 'weak_consensus'
        else:
            return 'no_consensus'
    
    def _generate_consensus_recommendations(self, 
                                         agent_results: Dict[str, AgentResult],
                                         consensus_analysis: Dict[str, Any]) -> List[str]:
        """Generate final recommendations based on consensus analysis."""
        recommendations = []
        
        # Add high-consensus suggestions
        for agreement in consensus_analysis.get('agreement_areas', []):
            if agreement['agent_count'] >= 2:
                recommendations.append(
                    f"Consensus recommendation: {agreement['suggestion']} "
                    f"(supported by {agreement['agent_count']} agents)"
                )
        
        # Add validator-specific recommendations if validator approval is required
        if self.config.require_validator_approval:
            validator_approval = consensus_analysis.get('voting_results', {}).get('validator_approval', False)
            if not validator_approval:
                recommendations.append("Validation required: Address validation issues before proceeding")
        
        # Add refinement recommendations based on voting
        refinement_vote = consensus_analysis.get('voting_results', {}).get('refinement_need', {})
        if refinement_vote.get('winner') == 'major_refinement':
            recommendations.append("Major refinement recommended based on agent consensus")
        elif refinement_vote.get('winner') == 'minor_refinement':
            recommendations.append("Minor refinement recommended based on agent consensus")
        
        # Add conflict resolution recommendations
        conflicts = consensus_analysis.get('conflict_areas', [])
        if len(conflicts) > 3:
            recommendations.append("Multiple conflicting suggestions detected - consider iterative refinement")
        
        # If no consensus, provide fallback recommendations
        if consensus_analysis.get('consensus_type') == 'no_consensus':
            recommendations.append("No clear consensus reached - consider manual review or additional context")
        
        return recommendations
    
    def _calculate_ensemble_confidence(self, 
                                     agent_results: Dict[str, AgentResult],
                                     consensus_analysis: Dict[str, Any]) -> float:
        """Calculate overall ensemble confidence score."""
        successful_results = [result for result in agent_results.values() if result.success]
        
        if not successful_results:
            return 0.0
        
        # Base confidence from individual agents
        individual_confidences = [result.confidence_score for result in successful_results]
        base_confidence = sum(individual_confidences) / len(individual_confidences)
        
        # Consensus bonus/penalty
        consensus_level = consensus_analysis.get('consensus_level', 0.0)
        consensus_bonus = (consensus_level - 0.5) * 0.2  # +/- 0.1 based on consensus
        
        # Validator approval bonus
        validator_approval = consensus_analysis.get('voting_results', {}).get('validator_approval', False)
        validator_bonus = 0.1 if validator_approval else -0.1
        
        # Calculate final confidence
        final_confidence = base_confidence + consensus_bonus + validator_bonus
        
        return min(1.0, max(0.0, final_confidence))
    
    def _determine_success(self, 
                         agent_results: Dict[str, AgentResult],
                         consensus_analysis: Dict[str, Any]) -> bool:
        """Determine if the ensemble processing was successful."""
        # At least one agent must succeed
        successful_agents = [result for result in agent_results.values() if result.success]
        if not successful_agents:
            return False
        
        # Check consensus threshold
        consensus_level = consensus_analysis.get('consensus_level', 0.0)
        if consensus_level < self.config.confidence_threshold:
            return False
        
        # Check validator approval if required
        if self.config.require_validator_approval:
            validator_approval = consensus_analysis.get('voting_results', {}).get('validator_approval', False)
            if not validator_approval:
                return False
        
        return True
    
    def _collect_errors(self, agent_results: Dict[str, AgentResult]) -> List[str]:
        """Collect all errors from agent results."""
        errors = []
        for agent_name, result in agent_results.items():
            if not result.success and result.error_message:
                errors.append(f"{agent_name}: {result.error_message}")
        return errors
    
    def _update_performance_metrics(self, 
                                  agent_results: Dict[str, AgentResult],
                                  processing_time: float) -> None:
        """Update performance metrics for monitoring."""
        # Track success rates
        for agent_name, result in agent_results.items():
            if agent_name not in self.performance_metrics:
                self.performance_metrics[agent_name] = {
                    'total_runs': 0,
                    'successful_runs': 0,
                    'average_confidence': 0.0,
                    'average_processing_time': 0.0
                }
            
            metrics = self.performance_metrics[agent_name]
            metrics['total_runs'] += 1
            
            if result.success:
                metrics['successful_runs'] += 1
                # Update rolling average confidence
                old_avg = metrics['average_confidence']
                new_confidence = result.confidence_score
                metrics['average_confidence'] = (old_avg + new_confidence) / 2
        
        # Track overall processing time
        if 'ensemble' not in self.performance_metrics:
            self.performance_metrics['ensemble'] = {
                'total_runs': 0,
                'average_processing_time': 0.0
            }
        
        ensemble_metrics = self.performance_metrics['ensemble']
        ensemble_metrics['total_runs'] += 1
        old_avg_time = ensemble_metrics['average_processing_time']
        ensemble_metrics['average_processing_time'] = (old_avg_time + processing_time) / 2
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        return self.performance_metrics.copy()
    
    def reset_performance_metrics(self) -> None:
        """Reset performance metrics."""
        self.performance_metrics = {}
    
    def update_config(self, config: ConsensusConfig) -> None:
        """Update the consensus configuration."""
        self.config = config
    
    def add_agent(self, name: str, agent: Agent) -> None:
        """Add a new agent to the ensemble."""
        self.agents[name] = agent
    
    def remove_agent(self, name: str) -> bool:
        """Remove an agent from the ensemble."""
        if name in self.agents:
            del self.agents[name]
            return True
        return False
    
    def get_agent_names(self) -> List[str]:
        """Get list of available agent names."""
        return list(self.agents.keys())
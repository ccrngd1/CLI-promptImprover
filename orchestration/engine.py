"""
LLM Orchestration Engine for intelligent workflow coordination.

This module implements the LLMOrchestrationEngine class that coordinates the
improvement pipeline using LLM reasoning, handles conflict resolution, agent
output synthesis, and convergence detection with intelligent decision making.
"""

import time
import json
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

from agents.base import AgentResult
from agents.ensemble import AgentEnsemble, EnsembleResult
from agents.llm_enhanced_analyzer import LLMAnalyzerAgent
from agents.llm_enhanced_refiner import LLMRefinerAgent
from agents.llm_enhanced_validator import LLMValidatorAgent
from bedrock.executor import BedrockExecutor, ModelConfig
from evaluation.evaluator import Evaluator
from best_practices.system_prompts import SystemPromptManager, AgentType
from best_practices.repository import BestPracticesRepository
from models import PromptIteration, ExecutionResult, EvaluationResult, UserFeedback
from error_handling import (
    OrchestrationError, AgentError, TimeoutError, handle_orchestration_errors,
    with_retry, RetryConfig, global_error_handler, ErrorSeverity
)
from logging_config import get_logger, orchestration_logger, performance_logger, log_exception


@dataclass
class ConvergenceAnalysis:
    """Analysis of convergence status and criteria."""
    
    has_converged: bool
    convergence_score: float
    convergence_reasons: List[str]
    improvement_trend: str  # 'improving', 'stable', 'declining'
    iterations_analyzed: int
    confidence: float
    llm_reasoning: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'has_converged': self.has_converged,
            'convergence_score': self.convergence_score,
            'convergence_reasons': self.convergence_reasons,
            'improvement_trend': self.improvement_trend,
            'iterations_analyzed': self.iterations_analyzed,
            'confidence': self.confidence,
            'llm_reasoning': self.llm_reasoning
        }


@dataclass
class OrchestrationResult:
    """Result of LLM orchestration for a single iteration."""
    
    success: bool
    orchestrated_prompt: str
    agent_results: Dict[str, AgentResult]
    execution_result: Optional[ExecutionResult]
    evaluation_result: Optional[EvaluationResult]
    conflict_resolutions: List[Dict[str, Any]]
    synthesis_reasoning: str
    orchestration_decisions: List[str]
    convergence_analysis: Optional[ConvergenceAnalysis]
    processing_time: float
    llm_orchestrator_confidence: float
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'success': self.success,
            'orchestrated_prompt': self.orchestrated_prompt,
            'agent_results': {name: {
                'agent_name': result.agent_name,
                'success': result.success,
                'suggestions': result.suggestions,
                'confidence_score': result.confidence_score,
                'error_message': result.error_message
            } for name, result in self.agent_results.items()},
            'execution_result': self.execution_result.to_dict() if self.execution_result else None,
            'evaluation_result': self.evaluation_result.to_dict() if self.evaluation_result else None,
            'conflict_resolutions': self.conflict_resolutions,
            'synthesis_reasoning': self.synthesis_reasoning,
            'orchestration_decisions': self.orchestration_decisions,
            'convergence_analysis': self.convergence_analysis.to_dict() if self.convergence_analysis else None,
            'processing_time': self.processing_time,
            'llm_orchestrator_confidence': self.llm_orchestrator_confidence,
            'error_message': self.error_message
        }


class LLMOrchestrationEngine:
    """
    LLM-powered orchestration engine for intelligent workflow coordination.
    
    Coordinates multiple agents using LLM reasoning, resolves conflicts through
    intelligent analysis, synthesizes agent outputs, and makes strategic decisions
    about the optimization process including convergence detection.
    """
    
    def __init__(self, 
                 bedrock_executor: BedrockExecutor,
                 evaluator: Evaluator,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize the LLM Orchestration Engine.
        
        Args:
            bedrock_executor: Bedrock executor for prompt execution
            evaluator: Evaluator for response assessment
            config: Optional configuration dictionary
        """
        self.bedrock_executor = bedrock_executor
        self.evaluator = evaluator
        self.config = config or {}
        
        # Initialize components
        self.best_practices_repo = BestPracticesRepository()
        self.system_prompt_manager = SystemPromptManager(self.best_practices_repo)
        
        # Initialize agent ensemble
        self.agent_ensemble = AgentEnsemble()
        
        # Initialize LLM-enhanced agents
        self.llm_agents = {
            'analyzer': LLMAnalyzerAgent(),
            'refiner': LLMRefinerAgent(), 
            'validator': LLMValidatorAgent()
        }
        
        # Orchestration configuration
        self.orchestrator_model_config = ModelConfig(
            model_id=self.config.get('orchestrator_model', 'anthropic.claude-3-sonnet-20240229-v1:0'),
            temperature=self.config.get('orchestrator_temperature', 0.3),
            max_tokens=self.config.get('orchestrator_max_tokens', 2000)
        )
        
        # Convergence criteria
        self.convergence_config = {
            'min_iterations': self.config.get('min_iterations', 3),
            'max_iterations': self.config.get('max_iterations', 10),
            'score_improvement_threshold': self.config.get('score_improvement_threshold', 0.02),
            'stability_window': self.config.get('stability_window', 3),
            'convergence_confidence_threshold': self.config.get('convergence_confidence_threshold', 0.8)
        }
        
        # Execution tracking
        self.orchestration_history = []
        
        # Initialize logger
        self.logger = get_logger('orchestration')
        
        # Initialize error recovery strategies
        self._setup_error_recovery()
    
    @handle_orchestration_errors
    @with_retry(RetryConfig(max_attempts=2, base_delay=1.0))
    def run_llm_orchestrated_iteration(self,
                                     prompt: str,
                                     context: Optional[Dict[str, Any]] = None,
                                     history: Optional[List[PromptIteration]] = None,
                                     feedback: Optional[UserFeedback] = None,
                                     model_config: Optional[ModelConfig] = None) -> OrchestrationResult:
        """
        Run a single LLM-orchestrated optimization iteration.
        
        Args:
            prompt: The prompt to optimize
            context: Optional context about the prompt's intended use
            history: Optional list of previous iterations
            feedback: Optional user feedback
            model_config: Optional model configuration for execution
            
        Returns:
            OrchestrationResult containing the complete iteration results
        """
        session_id = context.get('session_id', 'unknown') if context else 'unknown'
        iteration = len(history) + 1 if history else 1
        
        # Start performance tracking
        performance_logger.start_timer(f'orchestration_iteration_{session_id}_{iteration}')
        
        self.logger.info(
            f"Starting LLM orchestration iteration {iteration}",
            extra={
                'session_id': session_id,
                'iteration': iteration,
                'prompt_length': len(prompt),
                'has_feedback': feedback is not None,
                'history_length': len(history) if history else 0
            }
        )
        
        start_time = time.time()
        
        try:
            # Step 1: Coordinate agent analysis using LLM orchestration
            self.logger.debug("Starting agent coordination phase")
            agent_coordination_result = self._coordinate_agent_analysis(
                prompt, context, history, feedback
            )
            
            if not agent_coordination_result['success']:
                error_msg = agent_coordination_result.get('error', 'Unknown coordination error')
                self.logger.error(
                    f"Agent coordination failed: {error_msg}",
                    extra={'session_id': session_id, 'iteration': iteration}
                )
                
                return OrchestrationResult(
                    success=False,
                    orchestrated_prompt=prompt,
                    agent_results={},
                    execution_result=None,
                    evaluation_result=None,
                    conflict_resolutions=[],
                    synthesis_reasoning="Agent coordination failed",
                    orchestration_decisions=[],
                    convergence_analysis=None,
                    processing_time=time.time() - start_time,
                    llm_orchestrator_confidence=0.0,
                    error_message=error_msg
                )
            
            agent_results = agent_coordination_result['agent_results']
            
            # Log agent coordination success
            orchestration_logger.log_agent_coordination(
                session_id=session_id,
                iteration=iteration,
                execution_order=agent_coordination_result.get('execution_order', []),
                strategy_type=agent_coordination_result.get('strategy_type', 'unknown'),
                reasoning=agent_coordination_result.get('reasoning', '')
            )
            
            # Step 2: Resolve conflicts and synthesize recommendations
            self.logger.debug("Starting synthesis phase")
            synthesis_result = self._synthesize_agent_outputs_with_llm(
                agent_results, prompt, context, history
            )
            
            # Log conflict resolution if any
            if synthesis_result.get('conflict_resolutions'):
                orchestration_logger.log_conflict_resolution(
                    session_id=session_id,
                    iteration=iteration,
                    conflicts=synthesis_result.get('conflicts', []),
                    resolution_method=synthesis_result.get('resolution_method', 'llm_synthesis'),
                    final_decision=synthesis_result.get('orchestrated_prompt', prompt)
                )
            
            # Step 3: Generate orchestrated prompt
            orchestrated_prompt = synthesis_result.get('orchestrated_prompt', prompt)
            
            # Step 4: Execute the orchestrated prompt
            execution_result = None
            if model_config:
                self.logger.debug("Executing orchestrated prompt")
                try:
                    execution_result = self.bedrock_executor.execute_prompt(
                        orchestrated_prompt, model_config
                    )
                except Exception as e:
                    self.logger.warning(
                        f"Prompt execution failed: {str(e)}",
                        extra={'session_id': session_id, 'iteration': iteration}
                    )
                    log_exception(self.logger, e, {'session_id': session_id, 'iteration': iteration})
            
            # Step 5: Evaluate the result
            evaluation_result = None
            if execution_result and execution_result.success:
                self.logger.debug("Evaluating execution result")
                try:
                    evaluation_result = self.evaluator.evaluate_response(
                        orchestrated_prompt, execution_result.response_text, context
                    )
                except Exception as e:
                    self.logger.warning(
                        f"Evaluation failed: {str(e)}",
                        extra={'session_id': session_id, 'iteration': iteration}
                    )
                    log_exception(self.logger, e, {'session_id': session_id, 'iteration': iteration})
            
            # Step 6: Analyze convergence if history is available
            convergence_analysis = None
            if history and len(history) >= self.convergence_config['min_iterations']:
                self.logger.debug("Analyzing convergence")
                try:
                    convergence_analysis = self._analyze_convergence_with_llm(
                        history, evaluation_result, synthesis_result
                    )
                    
                    # Log convergence analysis
                    if convergence_analysis:
                        orchestration_logger.log_convergence_analysis(
                            session_id=session_id,
                            iteration=iteration,
                            has_converged=convergence_analysis.has_converged,
                            convergence_score=convergence_analysis.convergence_score,
                            reasoning=convergence_analysis.llm_reasoning,
                            confidence=convergence_analysis.confidence
                        )
                        
                except Exception as e:
                    self.logger.warning(
                        f"Convergence analysis failed: {str(e)}",
                        extra={'session_id': session_id, 'iteration': iteration}
                    )
                    log_exception(self.logger, e, {'session_id': session_id, 'iteration': iteration})
            
            processing_time = time.time() - start_time
            
            # Log synthesis decision
            orchestration_logger.log_synthesis_decision(
                session_id=session_id,
                iteration=iteration,
                agent_results=agent_results,
                synthesis_reasoning=synthesis_result.get('reasoning', ''),
                final_prompt=orchestrated_prompt,
                confidence=synthesis_result.get('confidence', 0.8)
            )
            
            # Create orchestration result
            result = OrchestrationResult(
                success=True,
                orchestrated_prompt=orchestrated_prompt,
                agent_results=agent_results,
                execution_result=execution_result,
                evaluation_result=evaluation_result,
                conflict_resolutions=synthesis_result.get('conflict_resolutions', []),
                synthesis_reasoning=synthesis_result.get('reasoning', ''),
                orchestration_decisions=synthesis_result.get('decisions', []),
                convergence_analysis=convergence_analysis,
                processing_time=processing_time,
                llm_orchestrator_confidence=synthesis_result.get('confidence', 0.8)
            )
            
            # Track orchestration history
            self.orchestration_history.append(result.to_dict())
            
            # End performance tracking
            performance_logger.end_timer(
                f'orchestration_iteration_{session_id}_{iteration}',
                session_id=session_id,
                iteration=iteration,
                success=True,
                agent_count=len(agent_results),
                has_conflicts=len(synthesis_result.get('conflict_resolutions', [])) > 0
            )
            
            self.logger.info(
                f"LLM orchestration iteration {iteration} completed successfully",
                extra={
                    'session_id': session_id,
                    'iteration': iteration,
                    'processing_time': processing_time,
                    'confidence': result.llm_orchestrator_confidence,
                    'agent_count': len(agent_results),
                    'has_convergence': convergence_analysis is not None
                }
            )
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            
            # End performance tracking with error
            performance_logger.end_timer(
                f'orchestration_iteration_{session_id}_{iteration}',
                session_id=session_id,
                iteration=iteration,
                success=False,
                error_type=type(e).__name__
            )
            
            # Log the error
            log_exception(
                self.logger, 
                e, 
                {
                    'session_id': session_id,
                    'iteration': iteration,
                    'processing_time': processing_time,
                    'operation': 'llm_orchestrated_iteration'
                }
            )
            
            # Create error result
            error_result = OrchestrationResult(
                success=False,
                orchestrated_prompt=prompt,  # Fallback to original prompt
                agent_results={},
                execution_result=None,
                evaluation_result=None,
                conflict_resolutions=[],
                synthesis_reasoning=f"Orchestration failed: {str(e)}",
                orchestration_decisions=[],
                convergence_analysis=None,
                processing_time=processing_time,
                llm_orchestrator_confidence=0.0,
                error_message=f"LLM orchestration failed: {str(e)}"
            )
            
            # Track failed orchestration
            self.orchestration_history.append(error_result.to_dict())
            
            return error_result
            
        except Exception as e:
            return OrchestrationResult(
                success=False,
                orchestrated_prompt=prompt,
                agent_results={},
                execution_result=None,
                evaluation_result=None,
                conflict_resolutions=[],
                synthesis_reasoning="Orchestration failed due to exception",
                orchestration_decisions=[],
                convergence_analysis=None,
                processing_time=time.time() - start_time,
                llm_orchestrator_confidence=0.0,
                error_message=f"Orchestration error: {str(e)}"
            )
    
    def _coordinate_agent_analysis(self,
                                 prompt: str,
                                 context: Optional[Dict[str, Any]],
                                 history: Optional[List[PromptIteration]],
                                 feedback: Optional[UserFeedback]) -> Dict[str, Any]:
        """
        Coordinate agent analysis using LLM-based orchestration decisions.
        
        Args:
            prompt: The prompt to analyze
            context: Optional context information
            history: Optional iteration history
            feedback: Optional user feedback
            
        Returns:
            Dictionary containing coordination results
        """
        try:
            # Determine which agents to use based on LLM orchestration
            agent_selection = self._determine_agent_execution_order(
                prompt, context, history, feedback
            )
            
            if not agent_selection['success']:
                return {
                    'success': False,
                    'error': 'Failed to determine agent execution strategy'
                }
            
            # Execute agents in the determined order
            agent_results = {}
            execution_order = agent_selection['execution_order']
            
            for agent_name in execution_order:
                if agent_name in self.llm_agents:
                    agent = self.llm_agents[agent_name]
                    result = agent.process(prompt, context, history, feedback)
                    agent_results[agent_name] = result
                    
                    # Check if we should continue based on results
                    if not result.success and agent_selection.get('stop_on_failure', False):
                        break
            
            return {
                'success': True,
                'agent_results': agent_results,
                'execution_strategy': agent_selection
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Agent coordination failed: {str(e)}"
            }
    
    def _determine_agent_execution_order(self,
                                       prompt: str,
                                       context: Optional[Dict[str, Any]],
                                       history: Optional[List[PromptIteration]],
                                       feedback: Optional[UserFeedback]) -> Dict[str, Any]:
        """
        Use LLM reasoning to determine optimal agent execution order and strategy.
        
        Args:
            prompt: The prompt to analyze
            context: Optional context information
            history: Optional iteration history
            feedback: Optional user feedback
            
        Returns:
            Dictionary containing execution strategy
        """
        try:
            # Build orchestration prompt for agent coordination
            orchestration_prompt = self._build_agent_coordination_prompt(
                prompt, context, history, feedback
            )
            
            # Get LLM decision on agent coordination
            llm_response = self.bedrock_executor.execute_prompt(
                orchestration_prompt, self.orchestrator_model_config
            )
            
            if not llm_response.success:
                # Fallback to default strategy
                return {
                    'success': True,
                    'execution_order': ['analyzer', 'refiner', 'validator'],
                    'strategy': 'default_fallback',
                    'reasoning': 'LLM coordination failed, using default strategy'
                }
            
            # Parse LLM response for coordination strategy
            strategy = self._parse_coordination_strategy(llm_response.response_text)
            
            return {
                'success': True,
                'execution_order': strategy.get('execution_order', ['analyzer', 'refiner', 'validator']),
                'strategy': strategy.get('strategy_type', 'llm_coordinated'),
                'reasoning': strategy.get('reasoning', ''),
                'stop_on_failure': strategy.get('stop_on_failure', False),
                'parallel_execution': strategy.get('parallel_execution', False)
            }
            
        except Exception as e:
            # Fallback to default strategy
            return {
                'success': True,
                'execution_order': ['analyzer', 'refiner', 'validator'],
                'strategy': 'error_fallback',
                'reasoning': f'Error in coordination: {str(e)}'
            }
    
    def _build_agent_coordination_prompt(self,
                                       prompt: str,
                                       context: Optional[Dict[str, Any]],
                                       history: Optional[List[PromptIteration]],
                                       feedback: Optional[UserFeedback]) -> str:
        """Build the LLM prompt for agent coordination decisions."""
        
        system_prompt = self.system_prompt_manager.generate_system_prompt(
            AgentType.ORCHESTRATOR, context
        )
        
        coordination_prompt_parts = [
            f"System: {system_prompt}",
            "",
            "You are coordinating multiple AI agents to optimize a prompt. Determine the optimal execution strategy.",
            "",
            f"PROMPT TO OPTIMIZE:\n{prompt}",
            ""
        ]
        
        # Add context if available
        if context:
            coordination_prompt_parts.extend([
                "CONTEXT:",
                f"Intended Use: {context.get('intended_use', 'Not specified')}",
                f"Domain: {context.get('domain', 'General')}",
                f"Target Audience: {context.get('target_audience', 'General')}",
                ""
            ])
        
        # Add history insights
        if history:
            coordination_prompt_parts.extend([
                f"ITERATION HISTORY: {len(history)} previous iterations",
                "Consider the optimization progress when determining strategy.",
                ""
            ])
        
        # Add feedback if available
        if feedback:
            coordination_prompt_parts.extend([
                "USER FEEDBACK:",
                f"Satisfaction: {feedback.satisfaction_rating}/5",
                f"Issues: {', '.join(feedback.specific_issues) if feedback.specific_issues else 'None'}",
                f"Desired Improvements: {feedback.desired_improvements or 'None'}",
                ""
            ])
        
        # Add feedback analysis if available
        if context and 'feedback_analysis' in context:
            feedback_analysis = context['feedback_analysis']
            if feedback_analysis.get('success'):
                analysis = feedback_analysis['analysis']
                patterns = feedback_analysis.get('patterns', [])
                suggestions = feedback_analysis.get('suggestions', [])
                
                coordination_prompt_parts.extend([
                    "FEEDBACK PATTERN ANALYSIS:",
                    f"Average User Rating: {analysis.get('average_rating', 0):.1f}/5",
                    f"Rating Trend: {analysis.get('rating_trend', 'unknown').title()}",
                    f"Total Feedback Sessions: {feedback_analysis.get('feedback_count', 0)}",
                    ""
                ])
                
                if analysis.get('common_issues'):
                    coordination_prompt_parts.append("Most Common User Issues:")
                    for issue, count in analysis['common_issues'][:3]:
                        coordination_prompt_parts.append(f"  - {issue} (mentioned {count} times)")
                    coordination_prompt_parts.append("")
                
                if patterns:
                    coordination_prompt_parts.append("Identified Patterns:")
                    for pattern in patterns[:3]:
                        coordination_prompt_parts.append(f"  - {pattern}")
                    coordination_prompt_parts.append("")
                
                if suggestions:
                    coordination_prompt_parts.append("Strategic Suggestions:")
                    for suggestion in suggestions[:3]:
                        coordination_prompt_parts.append(f"  - {suggestion}")
                    coordination_prompt_parts.append("")
        
        # Add coordination instructions
        coordination_prompt_parts.extend([
            "AVAILABLE AGENTS:",
            "- analyzer: Analyzes prompt structure, clarity, and best practices compliance",
            "- refiner: Generates improved prompt versions based on analysis",
            "- validator: Validates refined prompts for quality and correctness",
            "",
            "COORDINATION TASK:",
            "Determine the optimal execution strategy considering:",
            "1. Current prompt quality and needs",
            "2. Iteration history and progress",
            "3. User feedback and requirements",
            "4. Efficiency and effectiveness trade-offs",
            "",
            "Respond in this JSON format:",
            "{",
            '  "execution_order": ["agent1", "agent2", "agent3"],',
            '  "strategy_type": "comprehensive|focused|validation_only|analysis_only",',
            '  "reasoning": "Detailed explanation of strategy choice",',
            '  "stop_on_failure": false,',
            '  "parallel_execution": false',
            "}"
        ])
        
        return "\n".join(coordination_prompt_parts)
    
    def _parse_coordination_strategy(self, llm_response: str) -> Dict[str, Any]:
        """Parse LLM response for coordination strategy."""
        try:
            # Try to extract JSON from response
            import re
            json_match = re.search(r'\{[^}]*\}', llm_response, re.DOTALL)
            if json_match:
                strategy_json = json_match.group(0)
                strategy = json.loads(strategy_json)
                return strategy
        except (json.JSONDecodeError, AttributeError):
            pass
        
        # Fallback parsing based on keywords
        strategy = {
            'execution_order': ['analyzer', 'refiner', 'validator'],
            'strategy_type': 'comprehensive',
            'reasoning': 'Parsed from LLM response using keyword extraction'
        }
        
        # Extract execution order if mentioned
        if 'analyzer' in llm_response and 'refiner' in llm_response and 'validator' in llm_response:
            # Try to determine order from text
            analyzer_pos = llm_response.find('analyzer')
            refiner_pos = llm_response.find('refiner')
            validator_pos = llm_response.find('validator')
            
            positions = [
                (analyzer_pos, 'analyzer'),
                (refiner_pos, 'refiner'),
                (validator_pos, 'validator')
            ]
            positions = [(pos, agent) for pos, agent in positions if pos != -1]
            positions.sort()
            
            if len(positions) == 3:
                strategy['execution_order'] = [agent for _, agent in positions]
        
        # Determine strategy type from keywords
        if 'focused' in llm_response.lower():
            strategy['strategy_type'] = 'focused'
        elif 'validation' in llm_response.lower() and 'only' in llm_response.lower():
            strategy['strategy_type'] = 'validation_only'
            strategy['execution_order'] = ['validator']
        elif 'analysis' in llm_response.lower() and 'only' in llm_response.lower():
            strategy['strategy_type'] = 'analysis_only'
            strategy['execution_order'] = ['analyzer']
        
        return strategy
    
    def _synthesize_agent_outputs_with_llm(self,
                                         agent_results: Dict[str, AgentResult],
                                         original_prompt: str,
                                         context: Optional[Dict[str, Any]],
                                         history: Optional[List[PromptIteration]]) -> Dict[str, Any]:
        """
        Use LLM reasoning to synthesize agent outputs and resolve conflicts.
        
        Args:
            agent_results: Results from individual agents
            original_prompt: The original prompt being optimized
            context: Optional context information
            history: Optional iteration history
            
        Returns:
            Dictionary containing synthesis results
        """
        try:
            # Build synthesis prompt
            synthesis_prompt = self._build_synthesis_prompt(
                agent_results, original_prompt, context, history
            )
            
            # Get LLM synthesis
            llm_response = self.bedrock_executor.execute_prompt(
                synthesis_prompt, self.orchestrator_model_config
            )
            
            if not llm_response.success:
                return {
                    'orchestrated_prompt': original_prompt,
                    'reasoning': 'LLM synthesis failed, using original prompt',
                    'confidence': 0.3,
                    'conflict_resolutions': [],
                    'decisions': ['Fallback to original prompt due to synthesis failure']
                }
            
            # Parse synthesis results
            synthesis_result = self._parse_synthesis_response(
                llm_response.response_text, original_prompt
            )
            
            return synthesis_result
            
        except Exception as e:
            return {
                'orchestrated_prompt': original_prompt,
                'reasoning': f'Synthesis failed due to error: {str(e)}',
                'confidence': 0.2,
                'conflict_resolutions': [],
                'decisions': ['Error fallback to original prompt']
            }
    
    def _build_synthesis_prompt(self,
                              agent_results: Dict[str, AgentResult],
                              original_prompt: str,
                              context: Optional[Dict[str, Any]],
                              history: Optional[List[PromptIteration]]) -> str:
        """Build the LLM prompt for agent output synthesis."""
        
        system_prompt = self.system_prompt_manager.generate_system_prompt(
            AgentType.ORCHESTRATOR, context
        )
        
        synthesis_prompt_parts = [
            f"System: {system_prompt}",
            "",
            "Synthesize the outputs from multiple AI agents to create an optimized prompt.",
            "",
            f"ORIGINAL PROMPT:\n{original_prompt}",
            "",
            "AGENT ANALYSIS RESULTS:"
        ]
        
        # Add agent results
        for agent_name, result in agent_results.items():
            synthesis_prompt_parts.extend([
                f"\n{agent_name.upper()} AGENT:",
                f"Success: {result.success}",
                f"Confidence: {result.confidence_score:.2f}"
            ])
            
            if result.success:
                if result.suggestions:
                    synthesis_prompt_parts.append("Suggestions:")
                    for i, suggestion in enumerate(result.suggestions[:5], 1):
                        synthesis_prompt_parts.append(f"  {i}. {suggestion}")
                
                # Add specific analysis for refiner
                if agent_name == 'refiner' and result.analysis:
                    refined_prompt = result.analysis.get('refined_prompt')
                    if refined_prompt and refined_prompt != "No refined prompt could be extracted from LLM response.":
                        synthesis_prompt_parts.extend([
                            "Refined Prompt:",
                            f"{refined_prompt}"
                        ])
            else:
                synthesis_prompt_parts.append(f"Error: {result.error_message}")
        
        # Add context if available
        if context:
            synthesis_prompt_parts.extend([
                "",
                "CONTEXT REQUIREMENTS:",
                f"Intended Use: {context.get('intended_use', 'Not specified')}",
                f"Domain: {context.get('domain', 'General')}",
                f"Target Audience: {context.get('target_audience', 'General')}"
            ])
        
        # Add synthesis instructions
        synthesis_prompt_parts.extend([
            "",
            "SYNTHESIS TASK:",
            "1. Analyze all agent recommendations and identify conflicts",
            "2. Resolve conflicts using evidence-based reasoning",
            "3. Synthesize the best elements into an optimized prompt",
            "4. Ensure the result preserves the original intent",
            "5. Apply prompt engineering best practices",
            "",
            "Respond in this format:",
            "",
            "CONFLICT ANALYSIS:",
            "[Identify and analyze any conflicting recommendations]",
            "",
            "CONFLICT RESOLUTIONS:",
            "[Explain how conflicts were resolved with reasoning]",
            "",
            "SYNTHESIS REASONING:",
            "[Detailed explanation of synthesis decisions]",
            "",
            "ORCHESTRATION DECISIONS:",
            "[Key strategic decisions made during synthesis]",
            "",
            "OPTIMIZED PROMPT:",
            "[The final synthesized and optimized prompt]",
            "",
            "CONFIDENCE: [0.0-1.0 score with justification]"
        ])
        
        return "\n".join(synthesis_prompt_parts)
    
    def _parse_synthesis_response(self, llm_response: str, fallback_prompt: str) -> Dict[str, Any]:
        """Parse LLM synthesis response into structured results."""
        import re
        
        result = {
            'orchestrated_prompt': fallback_prompt,
            'reasoning': '',
            'confidence': 0.7,
            'conflict_resolutions': [],
            'decisions': []
        }
        
        try:
            # Extract optimized prompt
            prompt_match = re.search(
                r'OPTIMIZED PROMPT[:\s]*\n(.*?)(?=\n\n|\nCONFIDENCE|\Z)',
                llm_response,
                re.DOTALL | re.IGNORECASE
            )
            if prompt_match:
                optimized_prompt = prompt_match.group(1).strip()
                if optimized_prompt and len(optimized_prompt) > 10:
                    result['orchestrated_prompt'] = optimized_prompt
            
            # Extract synthesis reasoning
            reasoning_match = re.search(
                r'SYNTHESIS REASONING[:\s]*\n(.*?)(?=\n\n|\nORCHESTRATION|\nOPTIMIZED|\Z)',
                llm_response,
                re.DOTALL | re.IGNORECASE
            )
            if reasoning_match:
                result['reasoning'] = reasoning_match.group(1).strip()
            
            # Extract orchestration decisions
            decisions_match = re.search(
                r'ORCHESTRATION DECISIONS[:\s]*\n(.*?)(?=\n\n|\nOPTIMIZED|\nCONFIDENCE|\Z)',
                llm_response,
                re.DOTALL | re.IGNORECASE
            )
            if decisions_match:
                decisions_text = decisions_match.group(1).strip()
                # Extract bullet points or numbered items
                decisions = re.findall(r'^\s*[\d\-\*]\.\s*(.+)$', decisions_text, re.MULTILINE)
                if decisions:
                    result['decisions'] = decisions
                else:
                    # Split by lines if no bullet points
                    result['decisions'] = [line.strip() for line in decisions_text.split('\n') 
                                         if line.strip() and len(line.strip()) > 5]
            
            # Extract conflict resolutions
            conflicts_match = re.search(
                r'CONFLICT RESOLUTIONS[:\s]*\n(.*?)(?=\n\n|\nSYNTHESIS|\nORCHESTRATION|\Z)',
                llm_response,
                re.DOTALL | re.IGNORECASE
            )
            if conflicts_match:
                conflicts_text = conflicts_match.group(1).strip()
                if conflicts_text and 'no conflict' not in conflicts_text.lower():
                    result['conflict_resolutions'] = [
                        {'description': conflicts_text, 'resolution_method': 'llm_reasoning'}
                    ]
            
            # Extract confidence
            confidence_match = re.search(
                r'CONFIDENCE[:\s]+(\d+(?:\.\d+)?)',
                llm_response,
                re.IGNORECASE
            )
            if confidence_match:
                confidence = float(confidence_match.group(1))
                if confidence > 1.0:
                    confidence = confidence / 100.0  # Handle percentage format
                result['confidence'] = min(1.0, max(0.0, confidence))
            
        except Exception as e:
            result['reasoning'] = f'Error parsing synthesis response: {str(e)}'
            result['confidence'] = 0.5
        
        return result
    
    def _analyze_convergence_with_llm(self,
                                    history: List[PromptIteration],
                                    current_evaluation: Optional[EvaluationResult],
                                    synthesis_result: Dict[str, Any]) -> ConvergenceAnalysis:
        """
        Use LLM reasoning to analyze convergence status and criteria.
        
        Args:
            history: List of previous iterations
            current_evaluation: Current iteration's evaluation result
            synthesis_result: Results from agent synthesis
            
        Returns:
            ConvergenceAnalysis with LLM-based convergence assessment
        """
        try:
            # Build convergence analysis prompt
            convergence_prompt = self._build_convergence_analysis_prompt(
                history, current_evaluation, synthesis_result
            )
            
            # Get LLM analysis
            llm_response = self.bedrock_executor.execute_prompt(
                convergence_prompt, self.orchestrator_model_config
            )
            
            if not llm_response.success:
                return self._fallback_convergence_analysis(history, current_evaluation)
            
            # Parse convergence analysis
            convergence_result = self._parse_convergence_response(
                llm_response.response_text, history, current_evaluation
            )
            
            return convergence_result
            
        except Exception as e:
            return ConvergenceAnalysis(
                has_converged=False,
                convergence_score=0.5,
                convergence_reasons=[f'Analysis failed: {str(e)}'],
                improvement_trend='unknown',
                iterations_analyzed=len(history),
                confidence=0.3,
                llm_reasoning=f'Error in convergence analysis: {str(e)}'
            )
    
    def _build_convergence_analysis_prompt(self,
                                         history: List[PromptIteration],
                                         current_evaluation: Optional[EvaluationResult],
                                         synthesis_result: Dict[str, Any]) -> str:
        """Build the LLM prompt for convergence analysis."""
        
        system_prompt = self.system_prompt_manager.generate_system_prompt(
            AgentType.ORCHESTRATOR
        )
        
        convergence_prompt_parts = [
            f"System: {system_prompt}",
            "",
            "Analyze the convergence status of the prompt optimization process.",
            "",
            "ITERATION HISTORY:"
        ]
        
        # Add recent iterations with scores
        recent_iterations = history[-5:]  # Last 5 iterations
        for i, iteration in enumerate(recent_iterations):
            iteration_num = len(history) - len(recent_iterations) + i + 1
            convergence_prompt_parts.append(f"Iteration {iteration_num}:")
            
            if iteration.evaluation_scores:
                eval_scores = iteration.evaluation_scores
                convergence_prompt_parts.extend([
                    f"  Overall Score: {eval_scores.overall_score:.3f}",
                    f"  Relevance: {eval_scores.relevance_score:.3f}",
                    f"  Clarity: {eval_scores.clarity_score:.3f}",
                    f"  Completeness: {eval_scores.completeness_score:.3f}"
                ])
            else:
                convergence_prompt_parts.append("  No evaluation scores available")
        
        # Add current evaluation if available
        if current_evaluation:
            convergence_prompt_parts.extend([
                "",
                f"CURRENT ITERATION (#{len(history) + 1}):",
                f"Overall Score: {current_evaluation.overall_score:.3f}",
                f"Relevance: {current_evaluation.relevance_score:.3f}",
                f"Clarity: {current_evaluation.clarity_score:.3f}",
                f"Completeness: {current_evaluation.completeness_score:.3f}"
            ])
        
        # Add synthesis insights
        convergence_prompt_parts.extend([
            "",
            "SYNTHESIS INSIGHTS:",
            f"Orchestrator Confidence: {synthesis_result.get('confidence', 0.7):.2f}",
            f"Conflicts Resolved: {len(synthesis_result.get('conflict_resolutions', []))}",
            f"Strategic Decisions: {len(synthesis_result.get('decisions', []))}"
        ])
        
        # Add convergence criteria
        convergence_prompt_parts.extend([
            "",
            "CONVERGENCE CRITERIA:",
            f"Minimum Iterations: {self.convergence_config['min_iterations']}",
            f"Maximum Iterations: {self.convergence_config['max_iterations']}",
            f"Score Improvement Threshold: {self.convergence_config['score_improvement_threshold']}",
            f"Stability Window: {self.convergence_config['stability_window']} iterations",
            "",
            "ANALYSIS TASK:",
            "Determine if the optimization process has converged by analyzing:",
            "1. Score improvement trends over recent iterations",
            "2. Stability of performance metrics",
            "3. Diminishing returns in optimization efforts",
            "4. Quality plateau indicators",
            "5. Synthesis complexity and conflict patterns",
            "",
            "Respond in this format:",
            "",
            "TREND ANALYSIS:",
            "[Analyze the improvement trend: improving/stable/declining]",
            "",
            "CONVERGENCE ASSESSMENT:",
            "[Detailed analysis of convergence indicators]",
            "",
            "CONVERGENCE DECISION: [YES/NO]",
            "",
            "CONVERGENCE REASONS:",
            "[List specific reasons supporting the convergence decision]",
            "",
            "CONVERGENCE SCORE: [0.0-1.0]",
            "",
            "CONFIDENCE: [0.0-1.0 with justification]"
        ])
        
        return "\n".join(convergence_prompt_parts)
    
    def _parse_convergence_response(self,
                                  llm_response: str,
                                  history: List[PromptIteration],
                                  current_evaluation: Optional[EvaluationResult]) -> ConvergenceAnalysis:
        """Parse LLM convergence analysis response."""
        import re
        
        # Default values
        has_converged = False
        convergence_score = 0.5
        convergence_reasons = []
        improvement_trend = 'stable'
        confidence = 0.7
        
        try:
            # Extract convergence decision
            decision_match = re.search(
                r'CONVERGENCE DECISION[:\s]*(\w+)',
                llm_response,
                re.IGNORECASE
            )
            if decision_match:
                decision = decision_match.group(1).upper()
                has_converged = decision in ['YES', 'TRUE', 'CONVERGED']
            
            # Extract trend analysis
            trend_match = re.search(
                r'TREND ANALYSIS[:\s]*\n(.*?)(?=\n\n|\nCONVERGENCE|\Z)',
                llm_response,
                re.DOTALL | re.IGNORECASE
            )
            if trend_match:
                trend_text = trend_match.group(1).lower()
                if 'improving' in trend_text:
                    improvement_trend = 'improving'
                elif 'declining' in trend_text:
                    improvement_trend = 'declining'
                else:
                    improvement_trend = 'stable'
            
            # Extract convergence reasons
            reasons_match = re.search(
                r'CONVERGENCE REASONS[:\s]*\n(.*?)(?=\n\n|\nCONVERGENCE SCORE|\nCONFIDENCE|\Z)',
                llm_response,
                re.DOTALL | re.IGNORECASE
            )
            if reasons_match:
                reasons_text = reasons_match.group(1).strip()
                # Extract bullet points or numbered items
                reasons = re.findall(r'^\s*[\d\-\*]\.\s*(.+)$', reasons_text, re.MULTILINE)
                if reasons:
                    convergence_reasons = reasons
                else:
                    # Split by lines if no bullet points
                    convergence_reasons = [line.strip() for line in reasons_text.split('\n') 
                                         if line.strip() and len(line.strip()) > 5]
            
            # Extract convergence score
            score_match = re.search(
                r'CONVERGENCE SCORE[:\s]+(\d+(?:\.\d+)?)',
                llm_response,
                re.IGNORECASE
            )
            if score_match:
                score = float(score_match.group(1))
                if score > 1.0:
                    score = score / 100.0  # Handle percentage format
                convergence_score = min(1.0, max(0.0, score))
            
            # Extract confidence
            confidence_match = re.search(
                r'CONFIDENCE[:\s]+(\d+(?:\.\d+)?)',
                llm_response,
                re.IGNORECASE
            )
            if confidence_match:
                conf = float(confidence_match.group(1))
                if conf > 1.0:
                    conf = conf / 100.0  # Handle percentage format
                confidence = min(1.0, max(0.0, conf))
            
        except Exception as e:
            convergence_reasons.append(f'Error parsing convergence response: {str(e)}')
            confidence = 0.5
        
        return ConvergenceAnalysis(
            has_converged=has_converged,
            convergence_score=convergence_score,
            convergence_reasons=convergence_reasons,
            improvement_trend=improvement_trend,
            iterations_analyzed=len(history),
            confidence=confidence,
            llm_reasoning=llm_response
        )
    
    def _fallback_convergence_analysis(self,
                                     history: List[PromptIteration],
                                     current_evaluation: Optional[EvaluationResult]) -> ConvergenceAnalysis:
        """Provide fallback convergence analysis when LLM analysis fails."""
        
        # Simple rule-based convergence analysis
        if len(history) < self.convergence_config['min_iterations']:
            return ConvergenceAnalysis(
                has_converged=False,
                convergence_score=0.2,
                convergence_reasons=['Insufficient iterations for convergence analysis'],
                improvement_trend='unknown',
                iterations_analyzed=len(history),
                confidence=0.8,
                llm_reasoning='Fallback analysis: insufficient data'
            )
        
        # Analyze recent score trends
        recent_scores = []
        for iteration in history[-self.convergence_config['stability_window']:]:
            if iteration.evaluation_scores:
                recent_scores.append(iteration.evaluation_scores.overall_score)
        
        if current_evaluation:
            recent_scores.append(current_evaluation.overall_score)
        
        if len(recent_scores) < 2:
            return ConvergenceAnalysis(
                has_converged=False,
                convergence_score=0.3,
                convergence_reasons=['Insufficient evaluation data'],
                improvement_trend='unknown',
                iterations_analyzed=len(history),
                confidence=0.6,
                llm_reasoning='Fallback analysis: insufficient evaluation data'
            )
        
        # Calculate improvement trend
        score_changes = [recent_scores[i] - recent_scores[i-1] 
                        for i in range(1, len(recent_scores))]
        avg_change = sum(score_changes) / len(score_changes)
        
        if avg_change > self.convergence_config['score_improvement_threshold']:
            improvement_trend = 'improving'
        elif avg_change < -self.convergence_config['score_improvement_threshold']:
            improvement_trend = 'declining'
        else:
            improvement_trend = 'stable'
        
        # Check for convergence
        max_score_change = max(abs(change) for change in score_changes)
        has_converged = (
            improvement_trend == 'stable' and 
            max_score_change < self.convergence_config['score_improvement_threshold']
        )
        
        convergence_score = 1.0 - max_score_change if has_converged else 0.5
        
        return ConvergenceAnalysis(
            has_converged=has_converged,
            convergence_score=convergence_score,
            convergence_reasons=[
                f'Score trend: {improvement_trend}',
                f'Max recent change: {max_score_change:.3f}',
                f'Stability threshold: {self.convergence_config["score_improvement_threshold"]}'
            ],
            improvement_trend=improvement_trend,
            iterations_analyzed=len(history),
            confidence=0.7,
            llm_reasoning='Fallback rule-based convergence analysis'
        )
    
    def determine_convergence_with_reasoning(self, history: List[PromptIteration]) -> ConvergenceAnalysis:
        """
        Public method to determine convergence with LLM reasoning.
        
        Args:
            history: List of prompt iterations to analyze
            
        Returns:
            ConvergenceAnalysis with convergence assessment
        """
        if not history:
            return ConvergenceAnalysis(
                has_converged=False,
                convergence_score=0.0,
                convergence_reasons=['No iteration history provided'],
                improvement_trend='unknown',
                iterations_analyzed=0,
                confidence=1.0,
                llm_reasoning='No data available for analysis'
            )
        
        return self._analyze_convergence_with_llm(history, None, {})
    
    def resolve_agent_conflicts_with_llm(self, conflicting_recommendations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Public method to resolve conflicts between agent recommendations using LLM reasoning.
        
        Args:
            conflicting_recommendations: List of conflicting agent recommendations
            
        Returns:
            Dictionary containing conflict resolution results
        """
        if not conflicting_recommendations:
            return {
                'resolution': 'No conflicts to resolve',
                'method': 'none_required',
                'confidence': 1.0,
                'reasoning': 'No conflicting recommendations provided'
            }
        
        # This would be implemented as part of the synthesis process
        # For now, return a placeholder implementation
        return {
            'resolution': 'Conflicts resolved through LLM reasoning',
            'method': 'llm_synthesis',
            'confidence': 0.8,
            'reasoning': 'LLM-based conflict resolution using evidence and best practices'
        }
    
    def get_orchestration_history(self) -> List[Dict[str, Any]]:
        """Get the orchestration history for analysis."""
        return [result.to_dict() for result in self.orchestration_history]
    
    def clear_orchestration_history(self) -> None:
        """Clear the orchestration history."""
        self.orchestration_history = []
    
    def update_convergence_config(self, config: Dict[str, Any]) -> None:
        """Update convergence configuration."""
        self.convergence_config.update(config)
    
    def get_convergence_config(self) -> Dict[str, Any]:
        """Get current convergence configuration."""
        return self.convergence_config.copy()
    
    def _setup_error_recovery(self) -> None:
        """Set up error recovery strategies for orchestration."""
        from error_handling import FallbackStrategy, ErrorCategory
        
        # Register fallback strategies for orchestration errors
        global_error_handler.register_recovery_strategy(
            ErrorCategory.ORCHESTRATION_ERROR,
            FallbackStrategy(
                fallback_function=self._orchestration_fallback
            )
        )
        
        global_error_handler.register_recovery_strategy(
            ErrorCategory.AGENT_ERROR,
            FallbackStrategy(
                fallback_function=self._agent_failure_fallback
            )
        )
        
        global_error_handler.register_recovery_strategy(
            ErrorCategory.TIMEOUT_ERROR,
            FallbackStrategy(
                fallback_function=self._timeout_fallback
            )
        )
    
    def _orchestration_fallback(self, error: Exception, context: Dict[str, Any]) -> OrchestrationResult:
        """Fallback strategy for orchestration errors."""
        self.logger.warning(
            f"Using orchestration fallback due to error: {str(error)}",
            extra=context
        )
        
        # Return minimal orchestration result
        return OrchestrationResult(
            success=False,
            orchestrated_prompt=context.get('original_prompt', ''),
            agent_results={},
            execution_result=None,
            evaluation_result=None,
            conflict_resolutions=[],
            synthesis_reasoning=f"Fallback due to orchestration error: {str(error)}",
            orchestration_decisions=[],
            convergence_analysis=None,
            processing_time=0.0,
            llm_orchestrator_confidence=0.0,
            error_message=f"Orchestration fallback: {str(error)}"
        )
    
    def _agent_failure_fallback(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback strategy for agent failures."""
        self.logger.warning(
            f"Using agent failure fallback due to error: {str(error)}",
            extra=context
        )
        
        return {
            'success': False,
            'agent_results': {},
            'error': f"Agent failure fallback: {str(error)}"
        }
    
    def _timeout_fallback(self, error: Exception, context: Dict[str, Any]) -> OrchestrationResult:
        """Fallback strategy for timeout errors."""
        self.logger.warning(
            f"Using timeout fallback due to error: {str(error)}",
            extra=context
        )
        
        return OrchestrationResult(
            success=False,
            orchestrated_prompt=context.get('original_prompt', ''),
            agent_results={},
            execution_result=None,
            evaluation_result=None,
            conflict_resolutions=[],
            synthesis_reasoning=f"Timeout fallback: {str(error)}",
            orchestration_decisions=[],
            convergence_analysis=None,
            processing_time=0.0,
            llm_orchestrator_confidence=0.0,
            error_message=f"Timeout fallback: {str(error)}"
        )
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics for monitoring."""
        return global_error_handler.get_error_statistics()
    
    def get_orchestration_history(self) -> List[Dict[str, Any]]:
        """Get orchestration history for analysis."""
        return self.orchestration_history.copy()
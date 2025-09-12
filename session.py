"""
Session management system with orchestration integration.

This module provides the SessionManager class that handles optimization workflow
state and coordination, integrating with the LLM orchestration engine to manage
the complete prompt optimization lifecycle.
"""

import time
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field

from models import PromptIteration, ExecutionResult, EvaluationResult, UserFeedback
from storage.history import HistoryManager
from orchestration.engine import LLMOrchestrationEngine, OrchestrationResult
from bedrock.executor import BedrockExecutor, ModelConfig
from evaluation.evaluator import Evaluator


@dataclass
class SessionConfig:
    """Configuration for optimization sessions."""
    
    max_iterations: int = 10
    min_iterations: int = 3
    convergence_threshold: float = 0.02
    auto_finalize_on_convergence: bool = False
    collect_feedback_after_each_iteration: bool = True
    model_config: Optional[ModelConfig] = None
    orchestration_config: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'max_iterations': self.max_iterations,
            'min_iterations': self.min_iterations,
            'convergence_threshold': self.convergence_threshold,
            'auto_finalize_on_convergence': self.auto_finalize_on_convergence,
            'collect_feedback_after_each_iteration': self.collect_feedback_after_each_iteration,
            'model_config': self.model_config.to_dict() if self.model_config else None,
            'orchestration_config': self.orchestration_config
        }


@dataclass
class SessionState:
    """Current state of an optimization session."""
    
    session_id: str
    status: str  # 'active', 'paused', 'converged', 'finalized', 'error'
    current_iteration: int
    current_prompt: str
    initial_prompt: str
    context: Optional[Dict[str, Any]]
    config: SessionConfig
    created_at: datetime
    last_updated: datetime
    convergence_detected: bool = False
    convergence_reason: Optional[str] = None
    error_message: Optional[str] = None
    orchestration_summary: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'session_id': self.session_id,
            'status': self.status,
            'current_iteration': self.current_iteration,
            'current_prompt': self.current_prompt,
            'initial_prompt': self.initial_prompt,
            'context': self.context,
            'config': self.config.to_dict(),
            'created_at': self.created_at.isoformat(),
            'last_updated': self.last_updated.isoformat(),
            'convergence_detected': self.convergence_detected,
            'convergence_reason': self.convergence_reason,
            'error_message': self.error_message,
            'orchestration_summary': self.orchestration_summary
        }


@dataclass
class SessionResult:
    """Result of a session operation."""
    
    success: bool
    session_state: Optional[SessionState]
    iteration_result: Optional[OrchestrationResult]
    message: str
    requires_user_input: bool = False
    suggested_actions: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'success': self.success,
            'session_state': self.session_state.to_dict() if self.session_state else None,
            'iteration_result': self.iteration_result.to_dict() if self.iteration_result else None,
            'message': self.message,
            'requires_user_input': self.requires_user_input,
            'suggested_actions': self.suggested_actions
        }


class SessionManager:
    """
    Manages optimization workflow state and coordination with orchestration integration.
    
    Handles session creation, iteration tracking, user feedback collection, and
    integrates orchestration results and agent recommendations into session history.
    """
    
    def __init__(self,
                 bedrock_executor: BedrockExecutor,
                 evaluator: Evaluator,
                 history_manager: Optional[HistoryManager] = None,
                 orchestration_config: Optional[Dict[str, Any]] = None,
                 full_config: Optional[Dict[str, Any]] = None):
        """
        Initialize the SessionManager.
        
        Args:
            bedrock_executor: Bedrock executor for prompt execution
            evaluator: Evaluator for response assessment
            history_manager: Optional history manager (creates default if None)
            orchestration_config: Optional orchestration configuration (deprecated, use full_config)
            full_config: Full configuration including optimization settings
        """
        self.bedrock_executor = bedrock_executor
        self.evaluator = evaluator
        self.history_manager = history_manager or HistoryManager()
        
        # Use full_config if provided, otherwise fall back to orchestration_config for backward compatibility
        engine_config = full_config or orchestration_config or {}
        
        # Initialize orchestration engine
        self.orchestration_engine = LLMOrchestrationEngine(
            bedrock_executor=bedrock_executor,
            evaluator=evaluator,
            config=engine_config
        )
        
        # Active sessions tracking
        self.active_sessions: Dict[str, SessionState] = {}
        
        # Default configuration
        self.default_config = SessionConfig()
    
    def create_session(self,
                      initial_prompt: str,
                      context: Optional[Dict[str, Any]] = None,
                      config: Optional[SessionConfig] = None) -> SessionResult:
        """
        Create a new optimization session.
        
        Args:
            initial_prompt: The initial prompt to optimize
            context: Optional context about the prompt's intended use
            config: Optional session configuration
            
        Returns:
            SessionResult containing the new session state
        """
        try:
            # Generate unique session ID
            session_id = str(uuid.uuid4())
            
            # Use provided config or default
            session_config = config or self.default_config
            
            # Create session state
            session_state = SessionState(
                session_id=session_id,
                status='active',
                current_iteration=0,
                current_prompt=initial_prompt,
                initial_prompt=initial_prompt,
                context=context,
                config=session_config,
                created_at=datetime.now(),
                last_updated=datetime.now()
            )
            
            # Store in active sessions
            self.active_sessions[session_id] = session_state
            
            # Create session in history manager
            history_session_id = self.history_manager.create_session(
                initial_prompt=initial_prompt,
                context=str(context) if context else None
            )
            
            # Verify session creation
            if history_session_id != session_id:
                # Update session state with actual history session ID
                session_state.session_id = history_session_id
                self.active_sessions[history_session_id] = session_state
                del self.active_sessions[session_id]
            
            return SessionResult(
                success=True,
                session_state=session_state,
                iteration_result=None,
                message=f"Session {session_state.session_id} created successfully",
                suggested_actions=["Run first optimization iteration", "Configure session settings"]
            )
            
        except Exception as e:
            return SessionResult(
                success=False,
                session_state=None,
                iteration_result=None,
                message=f"Failed to create session: {str(e)}"
            )
    
    def run_optimization_iteration(self,
                                 session_id: str,
                                 user_feedback: Optional[UserFeedback] = None) -> SessionResult:
        """
        Run a single optimization iteration for a session.
        
        Args:
            session_id: The session identifier
            user_feedback: Optional user feedback from previous iteration
            
        Returns:
            SessionResult containing the iteration results
        """
        try:
            # Get session state
            session_state = self.active_sessions.get(session_id)
            if not session_state:
                return SessionResult(
                    success=False,
                    session_state=None,
                    iteration_result=None,
                    message=f"Session {session_id} not found"
                )
            
            # Check if session can continue
            if session_state.status not in ['active', 'paused']:
                return SessionResult(
                    success=False,
                    session_state=session_state,
                    iteration_result=None,
                    message=f"Session is {session_state.status} and cannot continue optimization"
                )
            
            # Check iteration limits
            if session_state.current_iteration >= session_state.config.max_iterations:
                session_state.status = 'finalized'
                session_state.convergence_reason = 'Maximum iterations reached'
                return SessionResult(
                    success=True,
                    session_state=session_state,
                    iteration_result=None,
                    message="Maximum iterations reached. Session finalized.",
                    suggested_actions=["Export final prompt", "Review session history"]
                )
            
            # Load session history for orchestration
            history = self.history_manager.load_session_history(session_id)
            
            # Analyze feedback patterns if we have enough history
            feedback_analysis = None
            if len(history) >= 2:
                feedback_analysis = self.analyze_feedback_patterns(session_id)
            
            # Enhance context with feedback analysis
            enhanced_context = session_state.context.copy() if session_state.context else {}
            if feedback_analysis and feedback_analysis.get('success'):
                enhanced_context['feedback_analysis'] = feedback_analysis
            
            # Run orchestrated iteration
            orchestration_result = self.orchestration_engine.run_llm_orchestrated_iteration(
                prompt=session_state.current_prompt,
                context=enhanced_context,
                history=history,
                feedback=user_feedback,
                model_config=session_state.config.model_config
            )
            
            # Update session state
            session_state.current_iteration += 1
            session_state.last_updated = datetime.now()
            
            if orchestration_result.success:
                # Update current prompt with orchestrated result
                session_state.current_prompt = orchestration_result.orchestrated_prompt
                
                # Create and save prompt iteration
                iteration = self._create_prompt_iteration(
                    session_state, orchestration_result, user_feedback
                )
                
                # Save iteration to history
                save_success = self.history_manager.save_iteration(iteration)
                if not save_success:
                    return SessionResult(
                        success=False,
                        session_state=session_state,
                        iteration_result=orchestration_result,
                        message="Failed to save iteration to history"
                    )
                
                # Update orchestration summary
                self._update_orchestration_summary(session_state, orchestration_result)
                
                # Check for convergence
                convergence_result = self._check_convergence(session_state, orchestration_result)
                
                # Determine next actions
                suggested_actions = self._determine_next_actions(session_state, orchestration_result)
                
                return SessionResult(
                    success=True,
                    session_state=session_state,
                    iteration_result=orchestration_result,
                    message=f"Iteration {session_state.current_iteration} completed successfully",
                    requires_user_input=session_state.config.collect_feedback_after_each_iteration,
                    suggested_actions=suggested_actions
                )
            else:
                # Handle orchestration failure
                session_state.status = 'error'
                session_state.error_message = orchestration_result.error_message
                
                return SessionResult(
                    success=False,
                    session_state=session_state,
                    iteration_result=orchestration_result,
                    message=f"Iteration failed: {orchestration_result.error_message}",
                    suggested_actions=["Retry iteration", "Check configuration", "Review session state"]
                )
                
        except Exception as e:
            # Update session state on error
            if session_id in self.active_sessions:
                self.active_sessions[session_id].status = 'error'
                self.active_sessions[session_id].error_message = str(e)
            
            return SessionResult(
                success=False,
                session_state=self.active_sessions.get(session_id),
                iteration_result=None,
                message=f"Iteration failed with exception: {str(e)}"
            )
    
    def collect_user_feedback(self,
                            session_id: str,
                            satisfaction_rating: int,
                            specific_issues: Optional[List[str]] = None,
                            desired_improvements: Optional[str] = None,
                            continue_optimization: bool = True) -> SessionResult:
        """
        Collect and process user feedback for a session.
        
        Args:
            session_id: The session identifier
            satisfaction_rating: User satisfaction rating (1-5)
            specific_issues: Optional list of specific issues
            desired_improvements: Optional description of desired improvements
            continue_optimization: Whether to continue optimization
            
        Returns:
            SessionResult containing the updated session state
        """
        try:
            # Get session state
            session_state = self.active_sessions.get(session_id)
            if not session_state:
                return SessionResult(
                    success=False,
                    session_state=None,
                    iteration_result=None,
                    message=f"Session {session_id} not found"
                )
            
            # Create user feedback object
            user_feedback = UserFeedback(
                satisfaction_rating=satisfaction_rating,
                specific_issues=specific_issues or [],
                desired_improvements=desired_improvements or "",
                continue_optimization=continue_optimization
            )
            
            # Validate feedback
            if not user_feedback.validate():
                return SessionResult(
                    success=False,
                    session_state=session_state,
                    iteration_result=None,
                    message="Invalid user feedback provided"
                )
            
            # Update latest iteration with feedback
            latest_iteration = self.history_manager.get_latest_iteration(session_id)
            if latest_iteration:
                latest_iteration.user_feedback = user_feedback
                self.history_manager.save_iteration(latest_iteration)
            
            # Update session state based on feedback
            if not continue_optimization:
                session_state.status = 'finalized'
                session_state.convergence_reason = 'User requested finalization'
            elif satisfaction_rating >= 4:
                # High satisfaction might indicate convergence
                session_state.convergence_detected = True
                session_state.convergence_reason = 'High user satisfaction'
            
            session_state.last_updated = datetime.now()
            
            # Determine suggested actions based on feedback
            suggested_actions = []
            if continue_optimization:
                if satisfaction_rating < 3:
                    suggested_actions.extend([
                        "Run another iteration with focus on specific issues",
                        "Review agent recommendations",
                        "Consider adjusting optimization strategy"
                    ])
                elif satisfaction_rating >= 4:
                    suggested_actions.extend([
                        "Consider finalizing the session",
                        "Run one more iteration for refinement",
                        "Export current prompt"
                    ])
                else:
                    suggested_actions.append("Continue with next iteration")
            else:
                suggested_actions.extend([
                    "Finalize session",
                    "Export optimized prompt",
                    "Review session history"
                ])
            
            return SessionResult(
                success=True,
                session_state=session_state,
                iteration_result=None,
                message="User feedback collected successfully",
                requires_user_input=False,
                suggested_actions=suggested_actions
            )
            
        except Exception as e:
            return SessionResult(
                success=False,
                session_state=self.active_sessions.get(session_id),
                iteration_result=None,
                message=f"Failed to collect user feedback: {str(e)}"
            )
    
    def finalize_session(self,
                        session_id: str,
                        export_reasoning: bool = True) -> SessionResult:
        """
        Finalize a session and export the optimized prompt with reasoning.
        
        Args:
            session_id: The session identifier
            export_reasoning: Whether to include reasoning explanations
            
        Returns:
            SessionResult containing finalization results
        """
        try:
            # Get session state
            session_state = self.active_sessions.get(session_id)
            if not session_state:
                return SessionResult(
                    success=False,
                    session_state=None,
                    iteration_result=None,
                    message=f"Session {session_id} not found"
                )
            
            # Get final prompt (current prompt or from latest iteration)
            final_prompt = session_state.current_prompt
            
            # Generate finalization summary with reasoning
            finalization_summary = self._generate_finalization_summary(
                session_state, export_reasoning
            )
            
            # Update session state
            session_state.status = 'finalized'
            session_state.last_updated = datetime.now()
            
            # Finalize in history manager
            finalize_success = self.history_manager.finalize_session(
                session_id, final_prompt
            )
            
            if not finalize_success:
                return SessionResult(
                    success=False,
                    session_state=session_state,
                    iteration_result=None,
                    message="Failed to finalize session in history"
                )
            
            # Remove from active sessions
            if session_id in self.active_sessions:
                del self.active_sessions[session_id]
            
            return SessionResult(
                success=True,
                session_state=session_state,
                iteration_result=None,
                message="Session finalized successfully",
                suggested_actions=[
                    "Export session data",
                    "Review optimization summary",
                    "Use optimized prompt in production"
                ]
            )
            
        except Exception as e:
            return SessionResult(
                success=False,
                session_state=self.active_sessions.get(session_id),
                iteration_result=None,
                message=f"Failed to finalize session: {str(e)}"
            )
    
    def get_session_state(self, session_id: str) -> Optional[SessionState]:
        """
        Get the current state of a session.
        
        Args:
            session_id: The session identifier
            
        Returns:
            SessionState or None if not found
        """
        return self.active_sessions.get(session_id)
    
    def list_active_sessions(self) -> List[SessionState]:
        """
        List all active sessions.
        
        Returns:
            List of active SessionState objects
        """
        return list(self.active_sessions.values())
    
    def pause_session(self, session_id: str) -> SessionResult:
        """
        Pause an active session.
        
        Args:
            session_id: The session identifier
            
        Returns:
            SessionResult indicating success or failure
        """
        session_state = self.active_sessions.get(session_id)
        if not session_state:
            return SessionResult(
                success=False,
                session_state=None,
                iteration_result=None,
                message=f"Session {session_id} not found"
            )
        
        if session_state.status == 'active':
            session_state.status = 'paused'
            session_state.last_updated = datetime.now()
            
            return SessionResult(
                success=True,
                session_state=session_state,
                iteration_result=None,
                message="Session paused successfully",
                suggested_actions=["Resume session", "Review current progress"]
            )
        else:
            return SessionResult(
                success=False,
                session_state=session_state,
                iteration_result=None,
                message=f"Cannot pause session with status: {session_state.status}"
            )
    
    def resume_session(self, session_id: str) -> SessionResult:
        """
        Resume a paused session.
        
        Args:
            session_id: The session identifier
            
        Returns:
            SessionResult indicating success or failure
        """
        session_state = self.active_sessions.get(session_id)
        if not session_state:
            return SessionResult(
                success=False,
                session_state=None,
                iteration_result=None,
                message=f"Session {session_id} not found"
            )
        
        if session_state.status == 'paused':
            session_state.status = 'active'
            session_state.last_updated = datetime.now()
            
            return SessionResult(
                success=True,
                session_state=session_state,
                iteration_result=None,
                message="Session resumed successfully",
                suggested_actions=["Continue optimization", "Run next iteration"]
            )
        else:
            return SessionResult(
                success=False,
                session_state=session_state,
                iteration_result=None,
                message=f"Cannot resume session with status: {session_state.status}"
            )
    
    def export_session_with_reasoning(self,
                                    session_id: str,
                                    export_path: str,
                                    include_orchestration_details: bool = True) -> SessionResult:
        """
        Export session data with orchestration reasoning and explanations.
        
        Args:
            session_id: The session identifier
            export_path: Path where to save the exported data
            include_orchestration_details: Whether to include detailed orchestration info
            
        Returns:
            SessionResult indicating success or failure
        """
        try:
            # Get session state
            session_state = self.active_sessions.get(session_id)
            if not session_state:
                # Try to load from history if not in active sessions
                session_info = self.history_manager.get_session_info(session_id)
                if not session_info:
                    return SessionResult(
                        success=False,
                        session_state=None,
                        iteration_result=None,
                        message=f"Session {session_id} not found"
                    )
            
            # Export basic session data
            export_success = self.history_manager.export_session(session_id, export_path)
            if not export_success:
                return SessionResult(
                    success=False,
                    session_state=session_state,
                    iteration_result=None,
                    message="Failed to export session data"
                )
            
            # Add orchestration reasoning if requested
            if include_orchestration_details and session_state:
                reasoning_data = {
                    'session_state': session_state.to_dict(),
                    'orchestration_summary': session_state.orchestration_summary,
                    'orchestration_history': [
                        result.to_dict() for result in self.orchestration_engine.orchestration_history
                    ]
                }
                
                # Append reasoning data to export file
                import json
                with open(export_path, 'r') as f:
                    export_data = json.load(f)
                
                export_data['orchestration_reasoning'] = reasoning_data
                
                with open(export_path, 'w') as f:
                    json.dump(export_data, f, indent=2)
            
            return SessionResult(
                success=True,
                session_state=session_state,
                iteration_result=None,
                message=f"Session exported successfully to {export_path}",
                suggested_actions=["Review exported data", "Share optimization results"]
            )
            
        except Exception as e:
            return SessionResult(
                success=False,
                session_state=self.active_sessions.get(session_id),
                iteration_result=None,
                message=f"Failed to export session: {str(e)}"
            )
    
    def _create_prompt_iteration(self,
                               session_state: SessionState,
                               orchestration_result: OrchestrationResult,
                               user_feedback: Optional[UserFeedback]) -> PromptIteration:
        """Create a PromptIteration from orchestration results."""
        
        # Extract agent analysis from orchestration result
        agent_analysis = {}
        for agent_name, agent_result in orchestration_result.agent_results.items():
            agent_analysis[agent_name] = {
                'success': agent_result.success,
                'suggestions': agent_result.suggestions,
                'confidence_score': agent_result.confidence_score,
                'analysis': getattr(agent_result, 'analysis', {}),
                'error_message': agent_result.error_message
            }
        
        # Add orchestration metadata
        agent_analysis['orchestration'] = {
            'synthesis_reasoning': orchestration_result.synthesis_reasoning,
            'orchestration_decisions': orchestration_result.orchestration_decisions,
            'conflict_resolutions': orchestration_result.conflict_resolutions,
            'llm_orchestrator_confidence': orchestration_result.llm_orchestrator_confidence,
            'processing_time': orchestration_result.processing_time
        }
        
        return PromptIteration(
            session_id=session_state.session_id,
            version=session_state.current_iteration,
            prompt_text=orchestration_result.orchestrated_prompt,
            timestamp=datetime.now(),
            agent_analysis=agent_analysis,
            execution_result=orchestration_result.execution_result,
            evaluation_scores=orchestration_result.evaluation_result,
            user_feedback=user_feedback
        )
    
    def analyze_feedback_patterns(self, session_id: str) -> Dict[str, Any]:
        """
        Analyze user feedback patterns to suggest optimization strategies.
        
        Args:
            session_id: The session identifier
            
        Returns:
            Dictionary containing feedback analysis and suggestions
        """
        try:
            # Load session history
            history = self.history_manager.load_session_history(session_id)
            if not history:
                return {
                    'success': False,
                    'message': 'No history available for analysis'
                }
            
            # Extract feedback data
            feedback_data = []
            for iteration in history:
                if iteration.user_feedback:
                    feedback_data.append({
                        'iteration': iteration.version,
                        'rating': iteration.user_feedback.satisfaction_rating,
                        'issues': iteration.user_feedback.specific_issues,
                        'improvements': iteration.user_feedback.desired_improvements,
                        'continue': iteration.user_feedback.continue_optimization,
                        'evaluation_score': getattr(iteration.evaluation_scores, 'overall_score', 0.0)
                    })
            
            if not feedback_data:
                return {
                    'success': False,
                    'message': 'No feedback data available for analysis'
                }
            
            # Analyze patterns
            analysis = self._analyze_feedback_trends(feedback_data)
            
            # Use orchestration to generate strategic suggestions
            suggestions = self._generate_feedback_based_suggestions(feedback_data, analysis)
            
            return {
                'success': True,
                'feedback_count': len(feedback_data),
                'analysis': analysis,
                'suggestions': suggestions,
                'patterns': self._identify_feedback_patterns(feedback_data)
            }
            
        except Exception as e:
            return {
                'success': False,
                'message': f'Failed to analyze feedback patterns: {str(e)}'
            }
    
    def _analyze_feedback_trends(self, feedback_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze trends in user feedback."""
        if not feedback_data:
            return {}
        
        ratings = [f['rating'] for f in feedback_data]
        eval_scores = [f['evaluation_score'] for f in feedback_data]
        
        # Calculate trends
        rating_trend = 'stable'
        if len(ratings) >= 2:
            if ratings[-1] > ratings[0]:
                rating_trend = 'improving'
            elif ratings[-1] < ratings[0]:
                rating_trend = 'declining'
        
        # Analyze common issues
        all_issues = []
        for f in feedback_data:
            all_issues.extend(f['issues'])
        
        issue_frequency = {}
        for issue in all_issues:
            issue_lower = issue.lower()
            issue_frequency[issue_lower] = issue_frequency.get(issue_lower, 0) + 1
        
        common_issues = sorted(issue_frequency.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            'average_rating': sum(ratings) / len(ratings),
            'rating_trend': rating_trend,
            'latest_rating': ratings[-1],
            'rating_variance': max(ratings) - min(ratings),
            'common_issues': common_issues,
            'total_issues': len(all_issues),
            'evaluation_correlation': self._calculate_correlation(ratings, eval_scores) if eval_scores else 0.0
        }
    
    def _identify_feedback_patterns(self, feedback_data: List[Dict[str, Any]]) -> List[str]:
        """Identify patterns in user feedback."""
        patterns = []
        
        if len(feedback_data) < 2:
            return patterns
        
        # Check for consistent low ratings
        recent_ratings = [f['rating'] for f in feedback_data[-3:]]
        if all(r <= 2 for r in recent_ratings):
            patterns.append("Consistently low satisfaction ratings")
        
        # Check for repeated issues
        issue_sets = [set(f['issues']) for f in feedback_data]
        if len(issue_sets) >= 2:
            common_across_iterations = set.intersection(*issue_sets)
            if common_across_iterations:
                patterns.append(f"Recurring issues: {', '.join(common_across_iterations)}")
        
        # Check for improvement requests
        improvement_keywords = ['more', 'better', 'clearer', 'specific', 'detailed']
        improvement_mentions = []
        for f in feedback_data:
            improvements = f['improvements'].lower()
            for keyword in improvement_keywords:
                if keyword in improvements:
                    improvement_mentions.append(keyword)
        
        if improvement_mentions:
            from collections import Counter
            common_requests = Counter(improvement_mentions).most_common(3)
            patterns.append(f"Common improvement requests: {', '.join([req[0] for req in common_requests])}")
        
        return patterns
    
    def _generate_feedback_based_suggestions(self, 
                                           feedback_data: List[Dict[str, Any]], 
                                           analysis: Dict[str, Any]) -> List[str]:
        """Generate optimization suggestions based on feedback analysis."""
        suggestions = []
        
        # Rating-based suggestions
        avg_rating = analysis.get('average_rating', 3.0)
        if avg_rating < 2.5:
            suggestions.append("Consider major prompt restructuring due to low satisfaction")
        elif avg_rating < 3.5:
            suggestions.append("Focus on addressing specific user issues")
        
        # Trend-based suggestions
        trend = analysis.get('rating_trend', 'stable')
        if trend == 'declining':
            suggestions.append("Review recent changes - satisfaction is declining")
        elif trend == 'improving':
            suggestions.append("Continue current optimization approach - satisfaction is improving")
        
        # Issue-based suggestions
        common_issues = analysis.get('common_issues', [])
        if common_issues:
            top_issue = common_issues[0][0]
            if 'clear' in top_issue or 'vague' in top_issue:
                suggestions.append("Focus on clarity and specificity improvements")
            elif 'structure' in top_issue or 'format' in top_issue:
                suggestions.append("Improve prompt structure and formatting")
            elif 'example' in top_issue:
                suggestions.append("Add more concrete examples to the prompt")
        
        # Correlation-based suggestions
        correlation = analysis.get('evaluation_correlation', 0.0)
        if correlation < 0.3:
            suggestions.append("User satisfaction doesn't align with evaluation scores - review criteria")
        
        return suggestions
    
    def _calculate_correlation(self, x: List[float], y: List[float]) -> float:
        """Calculate simple correlation coefficient."""
        if len(x) != len(y) or len(x) < 2:
            return 0.0
        
        n = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(x[i] * y[i] for i in range(n))
        sum_x2 = sum(xi * xi for xi in x)
        sum_y2 = sum(yi * yi for yi in y)
        
        numerator = n * sum_xy - sum_x * sum_y
        denominator = ((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y)) ** 0.5
        
        if denominator == 0:
            return 0.0
        
        return numerator / denominator
    
    def _update_orchestration_summary(self,
                                    session_state: SessionState,
                                    orchestration_result: OrchestrationResult) -> None:
        """Update the orchestration summary in session state."""
        
        if 'iterations' not in session_state.orchestration_summary:
            session_state.orchestration_summary['iterations'] = []
        
        iteration_summary = {
            'iteration': session_state.current_iteration,
            'success': orchestration_result.success,
            'confidence': orchestration_result.llm_orchestrator_confidence,
            'processing_time': orchestration_result.processing_time,
            'agent_count': len(orchestration_result.agent_results),
            'conflicts_resolved': len(orchestration_result.conflict_resolutions),
            'has_convergence_analysis': orchestration_result.convergence_analysis is not None
        }
        
        session_state.orchestration_summary['iterations'].append(iteration_summary)
        
        # Update overall statistics
        session_state.orchestration_summary['total_iterations'] = session_state.current_iteration
        session_state.orchestration_summary['average_confidence'] = sum(
            iter_sum['confidence'] for iter_sum in session_state.orchestration_summary['iterations']
        ) / len(session_state.orchestration_summary['iterations'])
        session_state.orchestration_summary['total_processing_time'] = sum(
            iter_sum['processing_time'] for iter_sum in session_state.orchestration_summary['iterations']
        )
    
    def _check_convergence(self,
                          session_state: SessionState,
                          orchestration_result: OrchestrationResult) -> bool:
        """Check if the session has converged based on orchestration analysis."""
        
        # Check if orchestration detected convergence
        if orchestration_result.convergence_analysis:
            convergence = orchestration_result.convergence_analysis
            if convergence.has_converged and convergence.confidence >= session_state.config.convergence_threshold:
                session_state.convergence_detected = True
                session_state.convergence_reason = f"LLM orchestration detected convergence: {convergence.llm_reasoning}"
                
                # Auto-finalize if configured
                if session_state.config.auto_finalize_on_convergence:
                    session_state.status = 'converged'
                
                return True
        
        # Check minimum iterations requirement
        if session_state.current_iteration < session_state.config.min_iterations:
            return False
        
        # Additional convergence checks based on evaluation scores
        history = self.history_manager.load_session_history(session_state.session_id)
        if len(history) >= 3:
            recent_scores = []
            for iteration in history[-3:]:
                if iteration.evaluation_scores:
                    recent_scores.append(iteration.evaluation_scores.overall_score)
            
            if len(recent_scores) >= 3:
                # Check if scores are stable (low variance)
                import statistics
                if statistics.stdev(recent_scores) < session_state.config.convergence_threshold:
                    session_state.convergence_detected = True
                    session_state.convergence_reason = "Evaluation scores have stabilized"
                    return True
        
        return False
    
    def _determine_next_actions(self,
                              session_state: SessionState,
                              orchestration_result: OrchestrationResult) -> List[str]:
        """Determine suggested next actions based on session state and results."""
        
        actions = []
        
        # Check convergence status
        if session_state.convergence_detected:
            actions.extend([
                "Consider finalizing the session",
                "Review convergence analysis",
                "Collect final user feedback"
            ])
        
        # Check iteration progress
        remaining_iterations = session_state.config.max_iterations - session_state.current_iteration
        if remaining_iterations <= 2:
            actions.append("Approaching maximum iterations - consider finalizing soon")
        
        # Check orchestration confidence
        if orchestration_result.llm_orchestrator_confidence < 0.5:
            actions.extend([
                "Review orchestration decisions",
                "Consider adjusting optimization strategy",
                "Check agent configuration"
            ])
        
        # Check evaluation scores
        if orchestration_result.evaluation_result:
            overall_score = orchestration_result.evaluation_result.overall_score
            if overall_score < 0.6:
                actions.extend([
                    "Continue optimization - scores can be improved",
                    "Review specific evaluation feedback",
                    "Consider providing more specific context"
                ])
            elif overall_score > 0.8:
                actions.extend([
                    "High quality achieved - consider finalizing",
                    "Run one more iteration for refinement"
                ])
        
        # Default actions
        if not actions:
            actions.extend([
                "Continue with next iteration",
                "Collect user feedback",
                "Review current progress"
            ])
        
        return actions
    
    def _generate_finalization_summary(self,
                                     session_state: SessionState,
                                     include_reasoning: bool) -> Dict[str, Any]:
        """Generate a comprehensive finalization summary with reasoning."""
        
        summary = {
            'session_id': session_state.session_id,
            'initial_prompt': session_state.initial_prompt,
            'final_prompt': session_state.current_prompt,
            'total_iterations': session_state.current_iteration,
            'convergence_detected': session_state.convergence_detected,
            'convergence_reason': session_state.convergence_reason,
            'finalized_at': datetime.now().isoformat()
        }
        
        if include_reasoning:
            # Add orchestration summary
            summary['orchestration_summary'] = session_state.orchestration_summary
            
            # Add improvement analysis
            try:
                history = self.history_manager.load_session_history(session_state.session_id)
                if history:
                    initial_score = history[0].evaluation_scores.overall_score if history[0].evaluation_scores else 0.0
                    final_score = history[-1].evaluation_scores.overall_score if history[-1].evaluation_scores else 0.0
                    
                    summary['improvement_analysis'] = {
                        'initial_evaluation_score': initial_score,
                        'final_evaluation_score': final_score,
                        'improvement': final_score - initial_score,
                        'improvement_percentage': ((final_score - initial_score) / max(initial_score, 0.1)) * 100
                    }
            except Exception:
                # Handle case where history loading fails (e.g., in tests with mocks)
                summary['improvement_analysis'] = {
                    'initial_evaluation_score': 0.0,
                    'final_evaluation_score': 0.0,
                    'improvement': 0.0,
                    'improvement_percentage': 0.0
                }
            
            # Add key decisions and reasoning
            summary['key_orchestration_decisions'] = []
            for iteration_summary in session_state.orchestration_summary.get('iterations', []):
                if iteration_summary.get('conflicts_resolved', 0) > 0:
                    summary['key_orchestration_decisions'].append({
                        'iteration': iteration_summary['iteration'],
                        'conflicts_resolved': iteration_summary['conflicts_resolved'],
                        'confidence': iteration_summary['confidence']
                    })
        
        return summary
"""
Tests for orchestration edge cases, conflict resolution, and consensus building scenarios.

Comprehensive tests for LLM orchestration edge cases including agent conflicts,
consensus building failures, synthesis challenges, and recovery mechanisms.
"""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any, Optional

from orchestration.engine import LLMOrchestrationEngine, OrchestrationResult, ConvergenceAnalysis
from bedrock.executor import BedrockExecutor, ModelConfig
from evaluation.evaluator import Evaluator
from models import PromptIteration, ExecutionResult, EvaluationResult, UserFeedback
from agents.base import AgentResult
from error_handling import (
    OrchestrationError, AgentError, TimeoutError, BedrockOptimizerException,
    ErrorCategory, ErrorSeverity, global_error_handler
)
from logging_config import setup_logging, orchestration_logger


class TestOrchestrationEdgeCases:
    """Test suite for orchestration edge cases and error scenarios."""
    
    def setup_method(self):
        """Set up test environment."""
        # Set up logging
        self.loggers = setup_logging(log_level='DEBUG', enable_structured_logging=True)
        
        # Create mock dependencies
        self.mock_bedrock_executor = Mock(spec=BedrockExecutor)
        self.mock_evaluator = Mock(spec=Evaluator)
        
        # Create orchestration engine
        self.orchestration_engine = LLMOrchestrationEngine(
            bedrock_executor=self.mock_bedrock_executor,
            evaluator=self.mock_evaluator,
            config={
                'orchestrator_model': 'test-model',
                'orchestrator_temperature': 0.3,
                'min_iterations': 2,
                'max_iterations': 5,
                'score_improvement_threshold': 0.02
            }
        )
        
        # Test data
        self.test_prompt = "Explain quantum computing to beginners"
        self.test_context = {'domain': 'technology', 'audience': 'beginners'}
        self.model_config = ModelConfig(model_id="test-model")
    
    def test_agent_conflict_resolution_high_disagreement(self):
        """Test conflict resolution when agents strongly disagree."""
        # Setup conflicting agent results
        conflicting_results = {
            'analyzer': AgentResult(
                agent_name='analyzer',
                success=True,
                analysis={'recommendation': 'highly_technical', 'confidence': 0.9},
                suggestions=['Use advanced terminology', 'Include mathematical formulas'],
                confidence_score=0.9
            ),
            'refiner': AgentResult(
                agent_name='refiner',
                success=True,
                analysis={'recommendation': 'extremely_simple', 'confidence': 0.95},
                suggestions=['Use only basic words', 'Avoid all technical terms'],
                confidence_score=0.95
            ),
            'validator': AgentResult(
                agent_name='validator',
                success=True,
                analysis={'recommendation': 'moderate_approach', 'confidence': 0.7},
                suggestions=['Balance technical accuracy with accessibility'],
                confidence_score=0.7
            )
        }
        
        # Mock LLM responses for conflict resolution
        coordination_response = ExecutionResult(
            model_name="test-model",
            response_text='{"execution_order": ["analyzer", "refiner", "validator"], "strategy_type": "conflict_resolution"}',
            execution_time=1.0,
            token_usage={'input': 50, 'output': 30},
            success=True
        )
        
        synthesis_response = ExecutionResult(
            model_name="test-model",
            response_text="""
            CONFLICT ANALYSIS:
            Major disagreement detected between analyzer (highly technical) and refiner (extremely simple).
            Validator suggests moderate approach.
            
            CONFLICT RESOLUTIONS:
            1. Prioritize accessibility for beginner audience
            2. Include technical accuracy but with explanations
            3. Use progressive complexity approach
            
            SYNTHESIS REASONING:
            Resolved conflict by adopting progressive complexity strategy that starts simple but builds understanding.
            
            ORCHESTRATION DECISIONS:
            1. Start with simple analogies (refiner input)
            2. Gradually introduce technical concepts (analyzer input)
            3. Ensure clarity at each step (validator input)
            
            OPTIMIZED PROMPT:
            Explain quantum computing to beginners using simple analogies first, then gradually introduce technical concepts with clear explanations at each step.
            
            CONFIDENCE: 0.85
            """,
            execution_time=2.0,
            token_usage={'input': 200, 'output': 150},
            success=True
        )
        
        execution_response = ExecutionResult(
            model_name="test-model",
            response_text="Quantum computing is like having a magical computer...",
            execution_time=1.5,
            token_usage={'input': 80, 'output': 120},
            success=True
        )
        
        self.mock_bedrock_executor.execute_prompt.side_effect = [
            coordination_response, synthesis_response, execution_response
        ]
        
        # Mock evaluator
        self.mock_evaluator.evaluate_response.return_value = EvaluationResult(
            overall_score=0.85,
            relevance_score=0.9,
            clarity_score=0.8,
            completeness_score=0.85,
            qualitative_feedback="Good balance of technical accuracy and accessibility"
        )
        
        # Mock agents
        with patch.object(self.orchestration_engine, 'llm_agents') as mock_agents:
            mock_agents.__getitem__.side_effect = lambda key: Mock(
                process=Mock(return_value=conflicting_results[key])
            )
            mock_agents.__contains__.return_value = True
            
            # Execute orchestration
            result = self.orchestration_engine.run_llm_orchestrated_iteration(
                prompt=self.test_prompt,
                context=self.test_context,
                model_config=self.model_config
            )
        
        # Verify conflict resolution
        assert result.success is True
        assert len(result.conflict_resolutions) > 0
        assert 'progressive complexity' in result.synthesis_reasoning.lower()
        assert result.llm_orchestrator_confidence >= 0.8
        
        # Verify all conflicting agents were considered
        assert len(result.agent_results) == 3
        assert 'analyzer' in result.agent_results
        assert 'refiner' in result.agent_results
        assert 'validator' in result.agent_results
    
    def test_agent_failure_cascade_recovery(self):
        """Test recovery when multiple agents fail in sequence."""
        # Setup cascading agent failures
        def failing_agent_process(*args, **kwargs):
            raise AgentError(
                "Agent processing failed due to internal error",
                agent_name="failing_agent",
                agent_operation="process"
            )
        
        # Mock agents with failures
        with patch.object(self.orchestration_engine, 'llm_agents') as mock_agents:
            # First agent fails
            mock_analyzer = Mock()
            mock_analyzer.process.side_effect = failing_agent_process
            
            # Second agent also fails
            mock_refiner = Mock()
            mock_refiner.process.side_effect = failing_agent_process
            
            # Third agent succeeds
            mock_validator = Mock()
            mock_validator.process.return_value = AgentResult(
                agent_name='validator',
                success=True,
                analysis={'fallback_analysis': True},
                suggestions=['Fallback validation successful'],
                confidence_score=0.6
            )
            
            mock_agents.__getitem__.side_effect = lambda key: {
                'analyzer': mock_analyzer,
                'refiner': mock_refiner,
                'validator': mock_validator
            }[key]
            mock_agents.__contains__.return_value = True
            
            # Mock LLM responses for fallback orchestration
            coordination_response = ExecutionResult(
                model_name="test-model",
                response_text='{"execution_order": ["validator"], "strategy_type": "fallback"}',
                execution_time=1.0,
                token_usage={'input': 50, 'output': 30},
                success=True
            )
            
            synthesis_response = ExecutionResult(
                model_name="test-model",
                response_text="""
                SYNTHESIS REASONING:
                Multiple agent failures detected. Using fallback strategy with available validator results.
                
                OPTIMIZED PROMPT:
                Explain quantum computing to beginners (fallback optimization based on validation only).
                
                CONFIDENCE: 0.6
                """,
                execution_time=1.5,
                token_usage={'input': 100, 'output': 80},
                success=True
            )
            
            self.mock_bedrock_executor.execute_prompt.side_effect = [
                coordination_response, synthesis_response
            ]
            
            # Execute orchestration
            result = self.orchestration_engine.run_llm_orchestrated_iteration(
                prompt=self.test_prompt,
                context=self.test_context,
                model_config=self.model_config
            )
        
        # Verify graceful degradation
        assert result.success is True  # Should succeed despite agent failures
        assert result.llm_orchestrator_confidence < 0.8  # Lower confidence due to failures
        assert 'fallback' in result.synthesis_reasoning.lower()
        assert len(result.agent_results) == 1  # Only validator succeeded
        assert 'validator' in result.agent_results
    
    def test_llm_orchestrator_failure_recovery(self):
        """Test recovery when the LLM orchestrator itself fails."""
        # Mock agents with successful results
        successful_results = {
            'analyzer': AgentResult('analyzer', True, {'score': 0.8}, ['good'], 0.8),
            'refiner': AgentResult('refiner', True, {'refined': True}, ['improved'], 0.85),
            'validator': AgentResult('validator', True, {'valid': True}, ['validated'], 0.9)
        }
        
        with patch.object(self.orchestration_engine, 'llm_agents') as mock_agents:
            mock_agents.__getitem__.side_effect = lambda key: Mock(
                process=Mock(return_value=successful_results[key])
            )
            mock_agents.__contains__.return_value = True
            
            # Mock LLM orchestrator failure
            self.mock_bedrock_executor.execute_prompt.side_effect = [
                ExecutionResult(  # Coordination fails
                    model_name="test-model",
                    response_text="",
                    execution_time=1.0,
                    token_usage={'input': 0, 'output': 0},
                    success=False,
                    error_message="LLM orchestrator failed"
                )
            ]
            
            # Execute orchestration
            result = self.orchestration_engine.run_llm_orchestrated_iteration(
                prompt=self.test_prompt,
                context=self.test_context,
                model_config=self.model_config
            )
        
        # Verify fallback behavior
        assert result.success is False
        assert result.error_message is not None
        assert "orchestrator" in result.error_message.lower() or "coordination" in result.error_message.lower()
        assert result.orchestrated_prompt == self.test_prompt  # Should fallback to original
    
    def test_consensus_building_with_low_confidence_agents(self):
        """Test consensus building when all agents have low confidence."""
        # Setup low-confidence agent results
        low_confidence_results = {
            'analyzer': AgentResult(
                agent_name='analyzer',
                success=True,
                analysis={'uncertain_analysis': True},
                suggestions=['Possibly improve structure'],
                confidence_score=0.4
            ),
            'refiner': AgentResult(
                agent_name='refiner',
                success=True,
                analysis={'tentative_refinement': True},
                suggestions=['Maybe simplify language'],
                confidence_score=0.3
            ),
            'validator': AgentResult(
                agent_name='validator',
                success=True,
                analysis={'weak_validation': True},
                suggestions=['Uncertain about validity'],
                confidence_score=0.35
            )
        }
        
        # Mock LLM responses for low-confidence consensus
        coordination_response = ExecutionResult(
            model_name="test-model",
            response_text='{"execution_order": ["analyzer", "refiner", "validator"], "strategy_type": "cautious"}',
            execution_time=1.0,
            token_usage={'input': 50, 'output': 30},
            success=True
        )
        
        synthesis_response = ExecutionResult(
            model_name="test-model",
            response_text="""
            SYNTHESIS REASONING:
            All agents report low confidence in their recommendations. 
            Adopting conservative approach with minimal changes.
            
            ORCHESTRATION DECISIONS:
            1. Make only minor adjustments due to low agent confidence
            2. Preserve original prompt structure
            3. Flag for potential human review
            
            OPTIMIZED PROMPT:
            Explain quantum computing to beginners with minor clarity improvements.
            
            CONFIDENCE: 0.4
            """,
            execution_time=2.0,
            token_usage={'input': 200, 'output': 100},
            success=True
        )
        
        execution_response = ExecutionResult(
            model_name="test-model",
            response_text="Conservative quantum computing explanation...",
            execution_time=1.5,
            token_usage={'input': 80, 'output': 100},
            success=True
        )
        
        self.mock_bedrock_executor.execute_prompt.side_effect = [
            coordination_response, synthesis_response, execution_response
        ]
        
        self.mock_evaluator.evaluate_response.return_value = EvaluationResult(
            overall_score=0.7,
            relevance_score=0.75,
            clarity_score=0.7,
            completeness_score=0.65,
            qualitative_feedback="Conservative approach with minimal changes"
        )
        
        # Mock agents
        with patch.object(self.orchestration_engine, 'llm_agents') as mock_agents:
            mock_agents.__getitem__.side_effect = lambda key: Mock(
                process=Mock(return_value=low_confidence_results[key])
            )
            mock_agents.__contains__.return_value = True
            
            # Execute orchestration
            result = self.orchestration_engine.run_llm_orchestrated_iteration(
                prompt=self.test_prompt,
                context=self.test_context,
                model_config=self.model_config
            )
        
        # Verify conservative consensus
        assert result.success is True
        assert result.llm_orchestrator_confidence <= 0.5  # Low confidence due to agent uncertainty
        assert 'conservative' in result.synthesis_reasoning.lower() or 'cautious' in result.synthesis_reasoning.lower()
        assert 'minor' in result.synthesis_reasoning.lower()
    
    def test_convergence_analysis_with_conflicting_trends(self):
        """Test convergence analysis with conflicting evaluation trends."""
        # Create history with conflicting trends
        conflicting_history = []
        scores = [0.6, 0.8, 0.7, 0.85, 0.75, 0.9]  # Oscillating with overall improvement
        
        for i, score in enumerate(scores):
            iteration = PromptIteration(
                session_id="conflict-trend-session",
                version=i + 1,
                prompt_text=f"Iteration {i + 1} prompt",
                timestamp=None,
                evaluation_scores=EvaluationResult(
                    overall_score=score,
                    relevance_score=score,
                    clarity_score=score,
                    completeness_score=score,
                    qualitative_feedback=f"Score: {score}"
                )
            )
            conflicting_history.append(iteration)
        
        # Mock LLM convergence analysis
        convergence_response = ExecutionResult(
            model_name="test-model",
            response_text="""
            TREND ANALYSIS:
            The evaluation scores show an oscillating pattern with overall upward trend.
            Recent iterations show instability despite improvement.
            
            CONVERGENCE ASSESSMENT:
            Mixed signals - overall improvement but recent instability suggests continued optimization potential.
            
            CONVERGENCE DECISION: NO
            
            CONVERGENCE REASONS:
            1. Oscillating pattern indicates instability
            2. Recent variations exceed stability threshold
            3. Overall trend is positive but inconsistent
            
            CONVERGENCE SCORE: 0.6
            
            CONFIDENCE: 0.75
            """,
            execution_time=1.5,
            token_usage={'input': 150, 'output': 100},
            success=True
        )
        
        self.mock_bedrock_executor.execute_prompt.return_value = convergence_response
        
        # Execute convergence analysis
        convergence = self.orchestration_engine.determine_convergence_with_reasoning(conflicting_history)
        
        # Verify convergence analysis handles conflicting trends
        assert convergence.has_converged is False
        assert convergence.improvement_trend in ['improving', 'oscillating', 'mixed']
        assert convergence.convergence_score == 0.6
        assert convergence.confidence == 0.75
        assert len(convergence.convergence_reasons) >= 3
        assert any('oscillating' in reason.lower() or 'instability' in reason.lower() 
                  for reason in convergence.convergence_reasons)
    
    def test_synthesis_with_contradictory_agent_outputs(self):
        """Test synthesis when agents provide contradictory outputs."""
        # Setup contradictory agent results
        contradictory_results = {
            'analyzer': AgentResult(
                agent_name='analyzer',
                success=True,
                analysis={
                    'recommendation': 'add_complexity',
                    'reasoning': 'Prompt needs more technical depth'
                },
                suggestions=['Add technical details', 'Include advanced concepts'],
                confidence_score=0.8
            ),
            'refiner': AgentResult(
                agent_name='refiner',
                success=True,
                analysis={
                    'recommendation': 'reduce_complexity',
                    'reasoning': 'Prompt is too complex for beginners'
                },
                suggestions=['Simplify language', 'Remove technical jargon'],
                confidence_score=0.85
            ),
            'validator': AgentResult(
                agent_name='validator',
                success=True,
                analysis={
                    'recommendation': 'restructure_completely',
                    'reasoning': 'Current structure is fundamentally flawed'
                },
                suggestions=['Complete restructure needed', 'Change approach entirely'],
                confidence_score=0.7
            )
        }
        
        # Mock LLM responses for contradiction resolution
        coordination_response = ExecutionResult(
            model_name="test-model",
            response_text='{"execution_order": ["analyzer", "refiner", "validator"], "strategy_type": "contradiction_resolution"}',
            execution_time=1.0,
            token_usage={'input': 50, 'output': 30},
            success=True
        )
        
        synthesis_response = ExecutionResult(
            model_name="test-model",
            response_text="""
            CONFLICT ANALYSIS:
            Severe contradictions detected:
            - Analyzer wants more complexity
            - Refiner wants less complexity  
            - Validator wants complete restructure
            
            CONFLICT RESOLUTIONS:
            1. Prioritize target audience (beginners) over technical completeness
            2. Use structured approach to balance complexity
            3. Implement progressive disclosure strategy
            
            SYNTHESIS REASONING:
            Resolved contradictions by adopting layered approach that satisfies all agents:
            - Start simple (refiner)
            - Build systematically (validator structure)
            - Include depth where appropriate (analyzer)
            
            ORCHESTRATION DECISIONS:
            1. Use clear structure with progressive complexity
            2. Start with simple concepts, build to more advanced
            3. Maintain beginner accessibility throughout
            
            OPTIMIZED PROMPT:
            Explain quantum computing to beginners using a structured, layered approach that starts with simple analogies and progressively builds to more technical concepts while maintaining clarity.
            
            CONFIDENCE: 0.75
            """,
            execution_time=3.0,
            token_usage={'input': 300, 'output': 200},
            success=True
        )
        
        execution_response = ExecutionResult(
            model_name="test-model",
            response_text="Structured quantum computing explanation with progressive complexity...",
            execution_time=2.0,
            token_usage={'input': 120, 'output': 150},
            success=True
        )
        
        self.mock_bedrock_executor.execute_prompt.side_effect = [
            coordination_response, synthesis_response, execution_response
        ]
        
        self.mock_evaluator.evaluate_response.return_value = EvaluationResult(
            overall_score=0.82,
            relevance_score=0.85,
            clarity_score=0.8,
            completeness_score=0.8,
            qualitative_feedback="Good synthesis of contradictory requirements"
        )
        
        # Mock agents
        with patch.object(self.orchestration_engine, 'llm_agents') as mock_agents:
            mock_agents.__getitem__.side_effect = lambda key: Mock(
                process=Mock(return_value=contradictory_results[key])
            )
            mock_agents.__contains__.return_value = True
            
            # Execute orchestration
            result = self.orchestration_engine.run_llm_orchestrated_iteration(
                prompt=self.test_prompt,
                context=self.test_context,
                model_config=self.model_config
            )
        
        # Verify contradiction resolution
        assert result.success is True
        assert len(result.conflict_resolutions) > 0
        assert 'contradiction' in result.synthesis_reasoning.lower() or 'conflict' in result.synthesis_reasoning.lower()
        assert 'layered' in result.synthesis_reasoning.lower() or 'progressive' in result.synthesis_reasoning.lower()
        assert result.llm_orchestrator_confidence >= 0.7
        
        # Verify all contradictory agents were considered
        assert len(result.agent_results) == 3
    
    def test_timeout_during_orchestration_phases(self):
        """Test timeout handling during different orchestration phases."""
        # Test timeout during agent coordination
        with patch.object(self.orchestration_engine, 'llm_agents') as mock_agents:
            mock_agents.__getitem__.side_effect = lambda key: Mock(
                process=Mock(return_value=AgentResult('test', True, {}, [], 0.8))
            )
            mock_agents.__contains__.return_value = True
            
            # Mock slow LLM response that times out
            def slow_llm_response(*args, **kwargs):
                time.sleep(2)  # Simulate slow response
                return ExecutionResult("test", "response", 2.0, {}, True)
            
            self.mock_bedrock_executor.execute_prompt.side_effect = slow_llm_response
            
            # Execute with timeout handling
            with patch('error_handling.global_error_handler.handle_error') as mock_error_handler:
                mock_error_handler.side_effect = lambda error, context: None  # Don't recover
                
                result = self.orchestration_engine.run_llm_orchestrated_iteration(
                    prompt=self.test_prompt,
                    context=self.test_context,
                    model_config=self.model_config
                )
                
                # Should handle timeout gracefully
                assert result.success is False
                assert result.error_message is not None
    
    def test_orchestration_with_partial_agent_success(self):
        """Test orchestration when some agents succeed and others fail."""
        # Setup mixed agent results
        mixed_results = {
            'analyzer': AgentResult('analyzer', True, {'analysis': 'good'}, ['suggestion'], 0.8),
            'refiner': None,  # This agent will fail
            'validator': AgentResult('validator', True, {'validation': 'passed'}, ['validated'], 0.9)
        }
        
        with patch.object(self.orchestration_engine, 'llm_agents') as mock_agents:
            def get_agent_result(key):
                if key == 'refiner':
                    mock_agent = Mock()
                    mock_agent.process.side_effect = AgentError(
                        "Refiner agent failed",
                        agent_name="refiner",
                        agent_operation="process"
                    )
                    return mock_agent
                else:
                    return Mock(process=Mock(return_value=mixed_results[key]))
            
            mock_agents.__getitem__.side_effect = get_agent_result
            mock_agents.__contains__.return_value = True
            
            # Mock LLM responses for partial success handling
            coordination_response = ExecutionResult(
                model_name="test-model",
                response_text='{"execution_order": ["analyzer", "validator"], "strategy_type": "partial_success"}',
                execution_time=1.0,
                token_usage={'input': 50, 'output': 30},
                success=True
            )
            
            synthesis_response = ExecutionResult(
                model_name="test-model",
                response_text="""
                SYNTHESIS REASONING:
                Working with partial agent results due to refiner failure.
                Using analyzer and validator outputs for optimization.
                
                OPTIMIZED PROMPT:
                Explain quantum computing to beginners (optimized with available agent feedback).
                
                CONFIDENCE: 0.7
                """,
                execution_time=1.5,
                token_usage={'input': 100, 'output': 80},
                success=True
            )
            
            execution_response = ExecutionResult(
                model_name="test-model",
                response_text="Partial optimization quantum computing explanation...",
                execution_time=1.5,
                token_usage={'input': 80, 'output': 100},
                success=True
            )
            
            self.mock_bedrock_executor.execute_prompt.side_effect = [
                coordination_response, synthesis_response, execution_response
            ]
            
            self.mock_evaluator.evaluate_response.return_value = EvaluationResult(
                overall_score=0.75,
                relevance_score=0.8,
                clarity_score=0.7,
                completeness_score=0.75,
                qualitative_feedback="Good result despite partial agent failure"
            )
            
            # Execute orchestration
            result = self.orchestration_engine.run_llm_orchestrated_iteration(
                prompt=self.test_prompt,
                context=self.test_context,
                model_config=self.model_config
            )
        
        # Verify partial success handling
        assert result.success is True
        assert len(result.agent_results) == 2  # Only successful agents
        assert 'analyzer' in result.agent_results
        assert 'validator' in result.agent_results
        assert 'refiner' not in result.agent_results
        assert result.llm_orchestrator_confidence < 0.8  # Lower confidence due to missing agent
        assert 'partial' in result.synthesis_reasoning.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
"""
Integration tests for LLM Orchestration Engine.

This module provides comprehensive tests for the LLMOrchestrationEngine,
including complete optimization cycles, conflict resolution, convergence
detection, and agent coordination scenarios.
"""

import pytest
import time
from datetime import datetime
from typing import Dict, Any, List, Optional
from unittest.mock import Mock, patch, MagicMock

from orchestration.engine import LLMOrchestrationEngine, OrchestrationResult, ConvergenceAnalysis
from bedrock.executor import BedrockExecutor, ModelConfig
from evaluation.evaluator import Evaluator
from models import PromptIteration, ExecutionResult, EvaluationResult, UserFeedback
from agents.base import AgentResult


class TestLLMOrchestrationEngine:
    """Test suite for LLM Orchestration Engine."""
    
    @pytest.fixture
    def mock_bedrock_executor(self):
        """Create a mock Bedrock executor."""
        executor = Mock(spec=BedrockExecutor)
        
        # Mock successful execution
        executor.execute_prompt.return_value = ExecutionResult(
            model_name="anthropic.claude-3-sonnet-20240229-v1:0",
            response_text="This is a test response from the orchestrator LLM.",
            execution_time=1.5,
            token_usage={'input_tokens': 100, 'output_tokens': 50},
            success=True
        )
        
        return executor
    
    @pytest.fixture
    def mock_evaluator(self):
        """Create a mock evaluator."""
        evaluator = Mock(spec=Evaluator)
        
        # Mock evaluation result
        evaluator.evaluate_response.return_value = EvaluationResult(
            overall_score=0.85,
            relevance_score=0.8,
            clarity_score=0.9,
            completeness_score=0.85,
            qualitative_feedback="Good response quality",
            improvement_suggestions=["Add more examples", "Improve structure"]
        )
        
        return evaluator
    
    @pytest.fixture
    def orchestration_engine(self, mock_bedrock_executor, mock_evaluator):
        """Create an orchestration engine with mocked dependencies."""
        config = {
            'orchestrator_model': 'anthropic.claude-3-sonnet-20240229-v1:0',
            'orchestrator_temperature': 0.3,
            'min_iterations': 1,  # Set to 1 so convergence analysis runs from iteration 2
            'max_iterations': 5,
            'score_improvement_threshold': 0.02
        }
        
        return LLMOrchestrationEngine(
            bedrock_executor=mock_bedrock_executor,
            evaluator=mock_evaluator,
            config=config
        )
    
    @pytest.fixture
    def sample_prompt(self):
        """Sample prompt for testing."""
        return "Explain the concept of machine learning in simple terms."
    
    @pytest.fixture
    def sample_context(self):
        """Sample context for testing."""
        return {
            'intended_use': 'Educational content',
            'target_audience': 'Beginners',
            'domain': 'Technology'
        }
    
    @pytest.fixture
    def sample_history(self):
        """Sample iteration history for testing."""
        history = []
        
        for i in range(3):
            iteration = PromptIteration(
                session_id="test_session",
                version=i + 1,
                prompt_text=f"Iteration {i + 1} prompt text",
                timestamp=datetime.now(),
                evaluation_scores=EvaluationResult(
                    overall_score=0.7 + (i * 0.05),  # Improving trend
                    relevance_score=0.75,
                    clarity_score=0.7,
                    completeness_score=0.65
                )
            )
            history.append(iteration)
        
        return history
    
    @pytest.fixture
    def sample_feedback(self):
        """Sample user feedback for testing."""
        return UserFeedback(
            satisfaction_rating=3,
            specific_issues=["Too technical", "Needs examples"],
            desired_improvements="Make it more beginner-friendly",
            continue_optimization=True
        )
    
    def test_orchestration_engine_initialization(self, mock_bedrock_executor, mock_evaluator):
        """Test orchestration engine initialization."""
        config = {'min_iterations': 3, 'max_iterations': 8}
        
        engine = LLMOrchestrationEngine(
            bedrock_executor=mock_bedrock_executor,
            evaluator=mock_evaluator,
            config=config
        )
        
        assert engine.bedrock_executor == mock_bedrock_executor
        assert engine.evaluator == mock_evaluator
        assert engine.convergence_config['min_iterations'] == 3
        assert engine.convergence_config['max_iterations'] == 8
        assert len(engine.llm_agents) == 3  # analyzer, refiner, validator
    
    def test_single_iteration_execution_success(self, orchestration_engine, sample_prompt, 
                                              sample_context, mock_bedrock_executor):
        """Test successful single iteration execution."""
        # Mock LLM responses for coordination and synthesis
        mock_responses = [
            # Agent coordination response
            ExecutionResult(
                model_name="test_model",
                response_text='{"execution_order": ["analyzer", "refiner", "validator"], "strategy_type": "comprehensive", "reasoning": "Full analysis needed"}',
                execution_time=1.0,
                token_usage={'input_tokens': 50, 'output_tokens': 30},
                success=True
            ),
            # Synthesis response
            ExecutionResult(
                model_name="test_model",
                response_text="""
                CONFLICT ANALYSIS:
                No major conflicts detected between agent recommendations.
                
                SYNTHESIS REASONING:
                Combined analyzer insights with refiner improvements and validator approval.
                
                ORCHESTRATION DECISIONS:
                1. Apply structural improvements from analyzer
                2. Incorporate clarity enhancements from refiner
                3. Ensure validation standards are met
                
                OPTIMIZED PROMPT:
                Explain machine learning in simple terms with examples and clear structure for beginners.
                
                CONFIDENCE: 0.85
                """,
                execution_time=2.0,
                token_usage={'input_tokens': 200, 'output_tokens': 100},
                success=True
            ),
            # Prompt execution response
            ExecutionResult(
                model_name="test_model",
                response_text="Machine learning is like teaching computers to learn patterns...",
                execution_time=1.5,
                token_usage={'input_tokens': 80, 'output_tokens': 120},
                success=True
            )
        ]
        
        mock_bedrock_executor.execute_prompt.side_effect = mock_responses
        
        # Mock agent results
        with patch.object(orchestration_engine, 'llm_agents') as mock_agents:
            mock_analyzer = Mock()
            mock_analyzer.process.return_value = AgentResult(
                agent_name="LLMAnalyzerAgent",
                success=True,
                analysis={'structure_score': 0.8},
                suggestions=["Add examples", "Improve structure"],
                confidence_score=0.8
            )
            
            mock_refiner = Mock()
            mock_refiner.process.return_value = AgentResult(
                agent_name="LLMRefinerAgent",
                success=True,
                analysis={'refined_prompt': 'Improved prompt with better structure'},
                suggestions=["Use simpler language", "Add context"],
                confidence_score=0.85
            )
            
            mock_validator = Mock()
            mock_validator.process.return_value = AgentResult(
                agent_name="LLMValidatorAgent",
                success=True,
                analysis={'passes_validation': True},
                suggestions=["Minor formatting improvements"],
                confidence_score=0.9
            )
            
            mock_agents.__getitem__.side_effect = lambda key: {
                'analyzer': mock_analyzer,
                'refiner': mock_refiner,
                'validator': mock_validator
            }[key]
            
            mock_agents.__contains__.return_value = True
            
            # Execute iteration
            model_config = ModelConfig(model_id="anthropic.claude-3-sonnet-20240229-v1:0")
            result = orchestration_engine.run_llm_orchestrated_iteration(
                prompt=sample_prompt,
                context=sample_context,
                model_config=model_config
            )
        
        # Verify results
        assert result.success is True
        assert result.orchestrated_prompt != sample_prompt  # Should be optimized
        assert len(result.agent_results) == 3
        assert result.execution_result is not None
        assert result.execution_result.success is True
        assert result.evaluation_result is not None
        assert result.llm_orchestrator_confidence > 0.8
        assert result.processing_time > 0
    
    def test_agent_coordination_strategy_parsing(self, orchestration_engine, sample_prompt):
        """Test agent coordination strategy parsing."""
        # Test JSON response parsing
        json_response = '{"execution_order": ["validator", "analyzer"], "strategy_type": "validation_focused", "reasoning": "Focus on validation first"}'
        
        strategy = orchestration_engine._parse_coordination_strategy(json_response)
        
        assert strategy['execution_order'] == ['validator', 'analyzer']
        assert strategy['strategy_type'] == 'validation_focused'
        assert 'reasoning' in strategy
    
    def test_agent_coordination_fallback_parsing(self, orchestration_engine, sample_prompt):
        """Test fallback parsing when JSON parsing fails."""
        text_response = "I recommend running the analyzer first, then the refiner, and finally the validator for comprehensive analysis."
        
        strategy = orchestration_engine._parse_coordination_strategy(text_response)
        
        assert 'analyzer' in strategy['execution_order']
        assert 'refiner' in strategy['execution_order']
        assert 'validator' in strategy['execution_order']
        assert strategy['strategy_type'] == 'comprehensive'
    
    def test_synthesis_response_parsing(self, orchestration_engine):
        """Test synthesis response parsing."""
        synthesis_response = """
        CONFLICT ANALYSIS:
        Agent recommendations show minor conflicts in approach.
        
        CONFLICT RESOLUTIONS:
        Resolved by prioritizing clarity over brevity based on target audience.
        
        SYNTHESIS REASONING:
        Combined the best elements from all agents while maintaining focus on beginner-friendly language.
        
        ORCHESTRATION DECISIONS:
        1. Use analyzer's structural recommendations
        2. Apply refiner's language simplifications
        3. Incorporate validator's formatting suggestions
        
        OPTIMIZED PROMPT:
        Explain machine learning concepts using simple language, concrete examples, and clear structure suitable for beginners learning about technology.
        
        CONFIDENCE: 0.88
        """
        
        result = orchestration_engine._parse_synthesis_response(synthesis_response, "fallback prompt")
        
        assert result['orchestrated_prompt'] != "fallback prompt"
        assert "machine learning" in result['orchestrated_prompt'].lower()
        assert len(result['decisions']) == 3
        assert result['confidence'] == 0.88
        assert len(result['conflict_resolutions']) > 0
        assert result['reasoning'] != ''
    
    def test_convergence_analysis_with_history(self, orchestration_engine, sample_history):
        """Test convergence analysis with iteration history."""
        # Mock LLM response for convergence analysis
        convergence_response = """
        TREND ANALYSIS:
        The scores show a steady improving trend over the last 3 iterations.
        
        CONVERGENCE ASSESSMENT:
        While improvement is evident, the rate of change suggests continued optimization potential.
        
        CONVERGENCE DECISION: NO
        
        CONVERGENCE REASONS:
        1. Improvement trend is still active
        2. Score changes exceed stability threshold
        3. Recent iterations show meaningful gains
        
        CONVERGENCE SCORE: 0.65
        
        CONFIDENCE: 0.85
        """
        
        with patch.object(orchestration_engine.bedrock_executor, 'execute_prompt') as mock_execute:
            mock_execute.return_value = ExecutionResult(
                model_name="test_model",
                response_text=convergence_response,
                execution_time=1.0,
                token_usage={'input_tokens': 100, 'output_tokens': 80},
                success=True
            )
            
            convergence = orchestration_engine.determine_convergence_with_reasoning(sample_history)
        
        assert convergence.has_converged is False
        assert convergence.improvement_trend == 'improving'
        assert convergence.convergence_score == 0.65
        assert convergence.confidence == 0.85
        assert len(convergence.convergence_reasons) == 3
        assert convergence.iterations_analyzed == 3
    
    def test_convergence_analysis_converged_case(self, orchestration_engine, sample_history):
        """Test convergence analysis when optimization has converged."""
        # Create stable history
        stable_history = []
        for i in range(4):
            iteration = PromptIteration(
                session_id="test_session",
                version=i + 1,
                prompt_text=f"Stable iteration {i + 1}",
                timestamp=datetime.now(),
                evaluation_scores=EvaluationResult(
                    overall_score=0.85 + (i * 0.005),  # Very small improvements
                    relevance_score=0.85,
                    clarity_score=0.85,
                    completeness_score=0.85
                )
            )
            stable_history.append(iteration)
        
        convergence_response = """
        TREND ANALYSIS:
        The scores have stabilized with minimal changes over recent iterations.
        
        CONVERGENCE ASSESSMENT:
        Performance has reached a stable plateau with diminishing returns.
        
        CONVERGENCE DECISION: YES
        
        CONVERGENCE REASONS:
        1. Score changes are below improvement threshold
        2. Performance has stabilized at high quality level
        3. Diminishing returns observed in recent iterations
        
        CONVERGENCE SCORE: 0.92
        
        CONFIDENCE: 0.90
        """
        
        with patch.object(orchestration_engine.bedrock_executor, 'execute_prompt') as mock_execute:
            mock_execute.return_value = ExecutionResult(
                model_name="test_model",
                response_text=convergence_response,
                execution_time=1.0,
                token_usage={'input_tokens': 100, 'output_tokens': 80},
                success=True
            )
            
            convergence = orchestration_engine.determine_convergence_with_reasoning(stable_history)
        
        assert convergence.has_converged is True
        assert convergence.improvement_trend == 'stable'
        assert convergence.convergence_score == 0.92
        assert convergence.confidence == 0.90
    
    def test_fallback_convergence_analysis(self, orchestration_engine, sample_history):
        """Test fallback convergence analysis when LLM analysis fails."""
        with patch.object(orchestration_engine.bedrock_executor, 'execute_prompt') as mock_execute:
            mock_execute.return_value = ExecutionResult(
                model_name="test_model",
                response_text="",
                execution_time=1.0,
                token_usage={'input_tokens': 0, 'output_tokens': 0},
                success=False,
                error_message="LLM execution failed"
            )
            
            convergence = orchestration_engine.determine_convergence_with_reasoning(sample_history)
        
        # Should use fallback analysis
        assert convergence.improvement_trend in ['improving', 'stable', 'declining']
        assert convergence.iterations_analyzed == 3
        assert convergence.confidence > 0.5  # Fallback should have reasonable confidence
        assert 'fallback' in convergence.llm_reasoning.lower()
    
    def test_insufficient_history_convergence(self, orchestration_engine):
        """Test convergence analysis with insufficient history."""
        short_history = [
            PromptIteration(
                session_id="test_session",
                version=1,
                prompt_text="Single iteration",
                timestamp=datetime.now(),
                evaluation_scores=EvaluationResult(
                    overall_score=0.7,
                    relevance_score=0.7,
                    clarity_score=0.7,
                    completeness_score=0.7
                )
            )
        ]
        
        # Test the fallback method directly since we want to test insufficient history
        convergence = orchestration_engine._fallback_convergence_analysis(short_history, None)
        
        assert convergence.has_converged is False
        assert len(convergence.convergence_reasons) > 0
        assert 'insufficient' in convergence.convergence_reasons[0].lower()
        assert convergence.iterations_analyzed == 1
    
    def test_error_handling_in_orchestration(self, orchestration_engine, sample_prompt):
        """Test error handling during orchestration."""
        # Mock agent failure
        with patch.object(orchestration_engine, 'llm_agents') as mock_agents:
            mock_analyzer = Mock()
            mock_analyzer.process.side_effect = Exception("Agent processing failed")
            
            mock_agents.__getitem__.return_value = mock_analyzer
            mock_agents.__contains__.return_value = True
            
            result = orchestration_engine.run_llm_orchestrated_iteration(
                prompt=sample_prompt
            )
        
        assert result.success is False
        assert result.error_message is not None
        assert "agent coordination failed" in result.error_message.lower()
        assert result.orchestrated_prompt == sample_prompt  # Should fallback to original
    
    def test_conflict_resolution_integration(self, orchestration_engine):
        """Test conflict resolution functionality."""
        conflicting_recommendations = [
            {
                'agent': 'analyzer',
                'recommendation': 'Make the prompt more detailed',
                'confidence': 0.8
            },
            {
                'agent': 'refiner', 
                'recommendation': 'Simplify the prompt for clarity',
                'confidence': 0.85
            }
        ]
        
        resolution = orchestration_engine.resolve_agent_conflicts_with_llm(conflicting_recommendations)
        
        assert 'resolution' in resolution
        assert 'method' in resolution
        assert 'confidence' in resolution
        assert resolution['confidence'] > 0.0
    
    def test_orchestration_history_tracking(self, orchestration_engine, sample_prompt, mock_bedrock_executor):
        """Test orchestration history tracking."""
        initial_history_length = len(orchestration_engine.get_orchestration_history())
        
        # Mock successful responses
        mock_bedrock_executor.execute_prompt.side_effect = [
            ExecutionResult(
                model_name="test_model",
                response_text='{"execution_order": ["analyzer"], "strategy_type": "focused"}',
                execution_time=1.0,
                token_usage={'input_tokens': 50, 'output_tokens': 30},
                success=True
            ),
            ExecutionResult(
                model_name="test_model",
                response_text="OPTIMIZED PROMPT:\nImproved prompt\nCONFIDENCE: 0.8",
                execution_time=1.0,
                token_usage={'input_tokens': 100, 'output_tokens': 50},
                success=True
            )
        ]
        
        # Mock single agent
        with patch.object(orchestration_engine, 'llm_agents') as mock_agents:
            mock_analyzer = Mock()
            mock_analyzer.process.return_value = AgentResult(
                agent_name="LLMAnalyzerAgent",
                success=True,
                analysis={},
                suggestions=["Test suggestion"],
                confidence_score=0.8
            )
            
            mock_agents.__getitem__.return_value = mock_analyzer
            mock_agents.__contains__.return_value = True
            
            # Execute iteration
            result = orchestration_engine.run_llm_orchestrated_iteration(sample_prompt)
        
        # Check history was updated
        new_history_length = len(orchestration_engine.get_orchestration_history())
        assert new_history_length == initial_history_length + 1
        
        # Check history content
        history = orchestration_engine.get_orchestration_history()
        latest_entry = history[-1]
        assert latest_entry['success'] == result.success
        assert latest_entry['orchestrated_prompt'] == result.orchestrated_prompt
    
    def test_configuration_updates(self, orchestration_engine):
        """Test configuration updates."""
        original_config = orchestration_engine.get_convergence_config()
        
        new_config = {
            'min_iterations': 5,
            'score_improvement_threshold': 0.01
        }
        
        orchestration_engine.update_convergence_config(new_config)
        updated_config = orchestration_engine.get_convergence_config()
        
        assert updated_config['min_iterations'] == 5
        assert updated_config['score_improvement_threshold'] == 0.01
        # Other values should remain unchanged
        assert updated_config['max_iterations'] == original_config['max_iterations']
    
    def test_complete_optimization_cycle(self, orchestration_engine, sample_prompt, 
                                       sample_context, sample_feedback, mock_bedrock_executor):
        """Test a complete optimization cycle with multiple iterations."""
        # Mock responses for multiple iterations
        mock_responses = []
        
        # Responses for 3 iterations (coordination + synthesis for each)
        for i in range(3):
            # Coordination response
            mock_responses.append(ExecutionResult(
                model_name="test_model",
                response_text=f'{{"execution_order": ["analyzer", "refiner", "validator"], "strategy_type": "comprehensive", "reasoning": "Iteration {i+1} strategy"}}',
                execution_time=1.0,
                token_usage={'input_tokens': 50, 'output_tokens': 30},
                success=True
            ))
            
            # Synthesis response
            mock_responses.append(ExecutionResult(
                model_name="test_model",
                response_text=f"""
                SYNTHESIS REASONING:
                Iteration {i+1} synthesis with improvements.
                
                OPTIMIZED PROMPT:
                Explain machine learning with improved clarity and structure for iteration {i+1}.
                
                CONFIDENCE: {0.8 + i*0.05}
                """,
                execution_time=2.0,
                token_usage={'input_tokens': 200, 'output_tokens': 100},
                success=True
            ))
            
            # Execution response
            mock_responses.append(ExecutionResult(
                model_name="test_model",
                response_text=f"Iteration {i+1} response with machine learning explanation...",
                execution_time=1.5,
                token_usage={'input_tokens': 80, 'output_tokens': 120},
                success=True
            ))
            
            # Convergence analysis response (for iterations 2+)
            if i >= 1:
                mock_responses.append(ExecutionResult(
                    model_name="test_model",
                    response_text=f"""
                    TREND ANALYSIS:
                    Iteration {i+1} shows {'improving' if i < 2 else 'stable'} trend.
                    
                    CONVERGENCE DECISION: {'NO' if i < 2 else 'YES'}
                    
                    CONVERGENCE REASONS:
                    1. {'Continued improvement' if i < 2 else 'Stable performance achieved'}
                    
                    CONVERGENCE SCORE: {0.6 + i*0.15}
                    
                    CONFIDENCE: 0.85
                    """,
                    execution_time=1.0,
                    token_usage={'input_tokens': 100, 'output_tokens': 80},
                    success=True
                ))
        
        mock_bedrock_executor.execute_prompt.side_effect = mock_responses
        
        # Mock agents for all iterations
        with patch.object(orchestration_engine, 'llm_agents') as mock_agents:
            def create_mock_agent(name, confidence_base=0.8):
                mock_agent = Mock()
                mock_agent.process.return_value = AgentResult(
                    agent_name=name,
                    success=True,
                    analysis={'test_analysis': True},
                    suggestions=[f"{name} suggestion"],
                    confidence_score=confidence_base
                )
                return mock_agent
            
            mock_agents.__getitem__.side_effect = lambda key: {
                'analyzer': create_mock_agent("LLMAnalyzerAgent", 0.8),
                'refiner': create_mock_agent("LLMRefinerAgent", 0.85),
                'validator': create_mock_agent("LLMValidatorAgent", 0.9)
            }[key]
            
            mock_agents.__contains__.return_value = True
            
            # Run multiple iterations
            results = []
            history = []
            model_config = ModelConfig(model_id="anthropic.claude-3-sonnet-20240229-v1:0")
            
            for i in range(3):
                result = orchestration_engine.run_llm_orchestrated_iteration(
                    prompt=sample_prompt if i == 0 else results[-1].orchestrated_prompt,
                    context=sample_context,
                    history=history.copy(),
                    feedback=sample_feedback if i == 1 else None,
                    model_config=model_config
                )
                
                results.append(result)
                
                # Create iteration for history
                if result.evaluation_result:
                    iteration = PromptIteration(
                        session_id="test_cycle",
                        version=i + 1,
                        prompt_text=result.orchestrated_prompt,
                        timestamp=datetime.now(),
                        evaluation_scores=result.evaluation_result
                    )
                    history.append(iteration)
        
        # Verify cycle results
        assert len(results) == 3
        assert all(result.success for result in results)
        
        # Check progression
        for i, result in enumerate(results):
            assert result.orchestrated_prompt != sample_prompt
            # The orchestrated prompt should contain iteration content
            assert "machine learning" in result.orchestrated_prompt.lower()
            assert result.llm_orchestrator_confidence >= 0.7
        
        # Check convergence analysis was performed for later iterations
        assert results[1].convergence_analysis is not None
        assert results[2].convergence_analysis is not None
        
        # Check orchestration history
        orchestration_history = orchestration_engine.get_orchestration_history()
        assert len(orchestration_history) >= 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
#!/usr/bin/env python3
"""
Demonstration of the LLM Orchestration Engine.

This script shows how to use the LLMOrchestrationEngine to coordinate
multiple agents for intelligent prompt optimization with LLM reasoning.
"""

import os
from datetime import datetime
from orchestration.engine import LLMOrchestrationEngine
from bedrock.executor import BedrockExecutor, ModelConfig
from evaluation.evaluator import Evaluator
from models import PromptIteration, EvaluationResult, UserFeedback


def demo_orchestration_engine():
    """Demonstrate the LLM Orchestration Engine capabilities."""
    
    print("🤖 LLM Orchestration Engine Demo")
    print("=" * 50)
    
    # Note: This demo uses mock components since it requires AWS credentials
    print("\n📋 Setting up components...")
    
    # In a real implementation, you would use actual AWS credentials
    try:
        bedrock_executor = BedrockExecutor(region_name="us-east-1")
        print("✅ Bedrock executor initialized")
    except Exception as e:
        print(f"⚠️  Using mock Bedrock executor (AWS credentials not available): {e}")
        # Create a mock executor for demo purposes
        from unittest.mock import Mock
        from models import ExecutionResult
        
        bedrock_executor = Mock()
        bedrock_executor.execute_prompt.return_value = ExecutionResult(
            model_name="anthropic.claude-3-sonnet-20240229-v1:0",
            response_text="Mock LLM response for orchestration",
            execution_time=1.5,
            token_usage={'input_tokens': 100, 'output_tokens': 80},
            success=True
        )
    
    # Initialize evaluator
    evaluator = Evaluator()
    print("✅ Evaluator initialized")
    
    # Initialize orchestration engine
    config = {
        'orchestrator_model': 'anthropic.claude-3-sonnet-20240229-v1:0',
        'orchestrator_temperature': 0.3,
        'min_iterations': 2,
        'max_iterations': 8,
        'score_improvement_threshold': 0.02
    }
    
    orchestration_engine = LLMOrchestrationEngine(
        bedrock_executor=bedrock_executor,
        evaluator=evaluator,
        config=config
    )
    print("✅ LLM Orchestration Engine initialized")
    
    # Demo prompt and context
    sample_prompt = "Explain artificial intelligence to a beginner."
    sample_context = {
        'intended_use': 'Educational content',
        'target_audience': 'Beginners',
        'domain': 'Technology'
    }
    
    print(f"\n🎯 Original prompt: '{sample_prompt}'")
    print(f"📝 Context: {sample_context}")
    
    # Demo 1: Single iteration orchestration
    print("\n" + "=" * 50)
    print("🔄 Demo 1: Single Iteration Orchestration")
    print("=" * 50)
    
    model_config = ModelConfig(
        model_id="anthropic.claude-3-sonnet-20240229-v1:0",
        temperature=0.7,
        max_tokens=1000
    )
    
    try:
        result = orchestration_engine.run_llm_orchestrated_iteration(
            prompt=sample_prompt,
            context=sample_context,
            model_config=model_config
        )
        
        print(f"✅ Orchestration successful: {result.success}")
        print(f"🎯 Orchestrated prompt: '{result.orchestrated_prompt[:100]}...'")
        print(f"🤝 Agent results: {len(result.agent_results)} agents processed")
        print(f"🎯 Orchestrator confidence: {result.llm_orchestrator_confidence:.2f}")
        print(f"⏱️  Processing time: {result.processing_time:.2f}s")
        
        if result.conflict_resolutions:
            print(f"⚖️  Conflicts resolved: {len(result.conflict_resolutions)}")
        
        if result.orchestration_decisions:
            print(f"🧠 Strategic decisions: {len(result.orchestration_decisions)}")
            for i, decision in enumerate(result.orchestration_decisions[:3], 1):
                print(f"   {i}. {decision}")
        
    except Exception as e:
        print(f"❌ Orchestration failed: {e}")
    
    # Demo 2: Convergence analysis
    print("\n" + "=" * 50)
    print("🎯 Demo 2: Convergence Analysis")
    print("=" * 50)
    
    # Create sample iteration history
    sample_history = []
    for i in range(4):
        iteration = PromptIteration(
            session_id="demo_session",
            version=i + 1,
            prompt_text=f"Iteration {i + 1} prompt",
            timestamp=datetime.now(),
            evaluation_scores=EvaluationResult(
                overall_score=0.7 + (i * 0.05),  # Improving trend
                relevance_score=0.75,
                clarity_score=0.7 + (i * 0.03),
                completeness_score=0.65 + (i * 0.04)
            )
        )
        sample_history.append(iteration)
    
    print(f"📊 Analyzing {len(sample_history)} iterations...")
    
    try:
        convergence = orchestration_engine.determine_convergence_with_reasoning(sample_history)
        
        print(f"🎯 Has converged: {convergence.has_converged}")
        print(f"📈 Improvement trend: {convergence.improvement_trend}")
        print(f"📊 Convergence score: {convergence.convergence_score:.2f}")
        print(f"🎯 Analysis confidence: {convergence.confidence:.2f}")
        print(f"📝 Iterations analyzed: {convergence.iterations_analyzed}")
        
        if convergence.convergence_reasons:
            print("📋 Convergence reasons:")
            for i, reason in enumerate(convergence.convergence_reasons[:3], 1):
                print(f"   {i}. {reason}")
        
    except Exception as e:
        print(f"❌ Convergence analysis failed: {e}")
    
    # Demo 3: Configuration management
    print("\n" + "=" * 50)
    print("⚙️  Demo 3: Configuration Management")
    print("=" * 50)
    
    print("📋 Current convergence configuration:")
    current_config = orchestration_engine.get_convergence_config()
    for key, value in current_config.items():
        print(f"   {key}: {value}")
    
    # Update configuration
    new_config = {
        'min_iterations': 3,
        'score_improvement_threshold': 0.01
    }
    
    print(f"\n🔧 Updating configuration: {new_config}")
    orchestration_engine.update_convergence_config(new_config)
    
    updated_config = orchestration_engine.get_convergence_config()
    print("📋 Updated configuration:")
    for key, value in updated_config.items():
        if key in new_config:
            print(f"   {key}: {value} ✅")
        else:
            print(f"   {key}: {value}")
    
    # Demo 4: Orchestration history
    print("\n" + "=" * 50)
    print("📚 Demo 4: Orchestration History")
    print("=" * 50)
    
    history = orchestration_engine.get_orchestration_history()
    print(f"📊 Total orchestration runs: {len(history)}")
    
    if history:
        latest = history[-1]
        print(f"🕒 Latest run:")
        print(f"   Success: {latest['success']}")
        print(f"   Processing time: {latest['processing_time']:.2f}s")
        print(f"   Orchestrator confidence: {latest['llm_orchestrator_confidence']:.2f}")
    
    print("\n🎉 Demo completed successfully!")
    print("\n💡 Key Features Demonstrated:")
    print("   ✅ LLM-based agent coordination")
    print("   ✅ Intelligent conflict resolution")
    print("   ✅ Agent output synthesis")
    print("   ✅ Convergence detection with reasoning")
    print("   ✅ Configuration management")
    print("   ✅ Orchestration history tracking")


if __name__ == "__main__":
    demo_orchestration_engine()
#!/usr/bin/env python3
"""
Demo script for the SessionManager functionality.

This script demonstrates the key features of the SessionManager class
including session creation, iteration management, user feedback collection,
and session finalization with orchestration integration.
"""

import tempfile
import shutil
from unittest.mock import Mock
from datetime import datetime

from session import SessionManager, SessionConfig
from models import UserFeedback, ExecutionResult, EvaluationResult
from storage.history import HistoryManager
from orchestration.engine import OrchestrationResult, ConvergenceAnalysis
from agents.base import AgentResult


def create_mock_dependencies():
    """Create mock dependencies for demonstration."""
    
    # Create mock Bedrock executor
    mock_bedrock_executor = Mock()
    mock_bedrock_executor.execute_prompt.return_value = ExecutionResult(
        model_name="claude-3-sonnet",
        response_text="This is a sample response from the model.",
        execution_time=1.5,
        token_usage={"input": 100, "output": 50},
        success=True
    )
    
    # Create mock evaluator
    mock_evaluator = Mock()
    mock_evaluator.evaluate_response.return_value = EvaluationResult(
        overall_score=0.8,
        relevance_score=0.85,
        clarity_score=0.75,
        completeness_score=0.8,
        qualitative_feedback="Good quality response with clear structure."
    )
    
    return mock_bedrock_executor, mock_evaluator


def create_mock_orchestration_result():
    """Create a mock orchestration result for demonstration."""
    
    agent_results = {
        'analyzer': AgentResult(
            agent_name='analyzer',
            success=True,
            analysis={'structure_score': 0.8, 'clarity_issues': ['Minor formatting']},
            suggestions=['Improve clarity', 'Add examples'],
            confidence_score=0.8
        ),
        'refiner': AgentResult(
            agent_name='refiner',
            success=True,
            analysis={'refined_prompt': 'Write a comprehensive guide about machine learning fundamentals, including key concepts, algorithms, and practical applications. Structure the content with clear headings and provide concrete examples for each concept.'},
            suggestions=['Added structure and examples'],
            confidence_score=0.85
        ),
        'validator': AgentResult(
            agent_name='validator',
            success=True,
            analysis={'validation_passed': True, 'issues_found': []},
            suggestions=['Validation passed - prompt is well-structured'],
            confidence_score=0.9
        )
    }
    
    execution_result = ExecutionResult(
        model_name="claude-3-sonnet",
        response_text="Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed...",
        execution_time=2.1,
        token_usage={"input": 150, "output": 300},
        success=True
    )
    
    evaluation_result = EvaluationResult(
        overall_score=0.85,
        relevance_score=0.9,
        clarity_score=0.8,
        completeness_score=0.85,
        qualitative_feedback="Excellent improvement in structure and clarity. The prompt now provides clear guidance for comprehensive content creation."
    )
    
    convergence_analysis = ConvergenceAnalysis(
        has_converged=False,
        convergence_score=0.7,
        convergence_reasons=["Steady improvement", "Good agent consensus"],
        improvement_trend="improving",
        iterations_analyzed=2,
        confidence=0.75,
        llm_reasoning="The prompt shows consistent improvement but could benefit from one more iteration to reach optimal quality."
    )
    
    return OrchestrationResult(
        success=True,
        orchestrated_prompt="Write a comprehensive guide about machine learning fundamentals, including key concepts, algorithms, and practical applications. Structure the content with clear headings and provide concrete examples for each concept.",
        agent_results=agent_results,
        execution_result=execution_result,
        evaluation_result=evaluation_result,
        conflict_resolutions=[],
        synthesis_reasoning="Successfully synthesized agent recommendations to improve prompt structure and clarity while maintaining the original intent.",
        orchestration_decisions=[
            "Applied analyzer suggestions for better structure",
            "Incorporated refiner improvements for clarity",
            "Validated final prompt meets quality standards"
        ],
        convergence_analysis=convergence_analysis,
        processing_time=3.2,
        llm_orchestrator_confidence=0.85
    )


def demo_session_lifecycle():
    """Demonstrate the complete session lifecycle."""
    
    print("🚀 SessionManager Demo - Complete Lifecycle")
    print("=" * 50)
    
    # Create temporary directory for storage
    temp_dir = tempfile.mkdtemp()
    print(f"📁 Created temporary storage: {temp_dir}")
    
    try:
        # Initialize components
        mock_bedrock_executor, mock_evaluator = create_mock_dependencies()
        history_manager = HistoryManager(temp_dir)
        
        # Create SessionManager
        session_manager = SessionManager(
            bedrock_executor=mock_bedrock_executor,
            evaluator=mock_evaluator,
            history_manager=history_manager
        )
        
        # Mock the orchestration engine
        session_manager.orchestration_engine.run_llm_orchestrated_iteration = Mock()
        session_manager.orchestration_engine.run_llm_orchestrated_iteration.return_value = create_mock_orchestration_result()
        
        print("✅ SessionManager initialized successfully")
        
        # Step 1: Create a new session
        print("\n📝 Step 1: Creating new optimization session")
        initial_prompt = "Write about machine learning"
        context = {
            "intended_use": "Educational content creation",
            "domain": "Technology",
            "target_audience": "Beginners"
        }
        
        session_config = SessionConfig(
            max_iterations=5,
            min_iterations=2,
            convergence_threshold=0.05,
            collect_feedback_after_each_iteration=True
        )
        
        create_result = session_manager.create_session(
            initial_prompt=initial_prompt,
            context=context,
            config=session_config
        )
        
        if create_result.success:
            session_id = create_result.session_state.session_id
            print(f"✅ Session created: {session_id}")
            print(f"   Initial prompt: '{initial_prompt}'")
            print(f"   Context: {context}")
        else:
            print(f"❌ Failed to create session: {create_result.message}")
            return
        
        # Step 2: Run first optimization iteration
        print("\n🔄 Step 2: Running first optimization iteration")
        iteration_result = session_manager.run_optimization_iteration(session_id)
        
        if iteration_result.success:
            print("✅ First iteration completed successfully")
            print(f"   Iteration: {iteration_result.session_state.current_iteration}")
            print(f"   Optimized prompt: '{iteration_result.session_state.current_prompt[:100]}...'")
            print(f"   Orchestration confidence: {iteration_result.iteration_result.llm_orchestrator_confidence:.2f}")
            print(f"   Processing time: {iteration_result.iteration_result.processing_time:.1f}s")
            
            # Show agent results
            print("   Agent Results:")
            for agent_name, agent_result in iteration_result.iteration_result.agent_results.items():
                print(f"     - {agent_name}: {agent_result.confidence_score:.2f} confidence")
                print(f"       Suggestions: {', '.join(agent_result.suggestions[:2])}")
        else:
            print(f"❌ First iteration failed: {iteration_result.message}")
            return
        
        # Step 3: Collect user feedback
        print("\n💬 Step 3: Collecting user feedback")
        feedback_result = session_manager.collect_user_feedback(
            session_id=session_id,
            satisfaction_rating=4,
            specific_issues=["Could be more detailed"],
            desired_improvements="Add more specific examples and practical applications",
            continue_optimization=True
        )
        
        if feedback_result.success:
            print("✅ User feedback collected")
            print(f"   Satisfaction rating: 4/5")
            print(f"   Convergence detected: {feedback_result.session_state.convergence_detected}")
            print(f"   Suggested actions: {', '.join(feedback_result.suggested_actions[:2])}")
        else:
            print(f"❌ Failed to collect feedback: {feedback_result.message}")
        
        # Step 4: Run second iteration with feedback
        print("\n🔄 Step 4: Running second iteration with user feedback")
        user_feedback = UserFeedback(
            satisfaction_rating=4,
            specific_issues=["Could be more detailed"],
            desired_improvements="Add more specific examples and practical applications",
            continue_optimization=True
        )
        
        iteration2_result = session_manager.run_optimization_iteration(session_id, user_feedback)
        
        if iteration2_result.success:
            print("✅ Second iteration completed")
            print(f"   Current iteration: {iteration2_result.session_state.current_iteration}")
            print(f"   Evaluation score: {iteration2_result.iteration_result.evaluation_result.overall_score:.2f}")
            
            # Show orchestration summary
            orchestration_summary = iteration2_result.session_state.orchestration_summary
            print(f"   Total iterations: {orchestration_summary.get('total_iterations', 0)}")
            print(f"   Average confidence: {orchestration_summary.get('average_confidence', 0):.2f}")
        else:
            print(f"❌ Second iteration failed: {iteration2_result.message}")
        
        # Step 5: Session management operations
        print("\n⚙️  Step 5: Session management operations")
        
        # Pause session
        pause_result = session_manager.pause_session(session_id)
        print(f"   Pause session: {'✅' if pause_result.success else '❌'}")
        
        # Resume session
        resume_result = session_manager.resume_session(session_id)
        print(f"   Resume session: {'✅' if resume_result.success else '❌'}")
        
        # List active sessions
        active_sessions = session_manager.list_active_sessions()
        print(f"   Active sessions: {len(active_sessions)}")
        
        # Step 6: Finalize session
        print("\n🏁 Step 6: Finalizing session")
        finalize_result = session_manager.finalize_session(session_id, export_reasoning=True)
        
        if finalize_result.success:
            print("✅ Session finalized successfully")
            print(f"   Final status: {finalize_result.session_state.status}")
            print(f"   Suggested actions: {', '.join(finalize_result.suggested_actions)}")
        else:
            print(f"❌ Failed to finalize session: {finalize_result.message}")
        
        # Step 7: Export session data
        print("\n📤 Step 7: Exporting session with reasoning")
        export_path = f"{temp_dir}/session_export.json"
        export_result = session_manager.export_session_with_reasoning(
            session_id=session_id,
            export_path=export_path,
            include_orchestration_details=True
        )
        
        if export_result.success:
            print(f"✅ Session exported to: {export_path}")
        else:
            print(f"❌ Export failed: {export_result.message}")
        
        print("\n🎉 Demo completed successfully!")
        print("\nKey Features Demonstrated:")
        print("  ✓ Session creation with custom configuration")
        print("  ✓ LLM orchestration integration")
        print("  ✓ User feedback collection and processing")
        print("  ✓ Convergence detection and analysis")
        print("  ✓ Session state management (pause/resume)")
        print("  ✓ Session finalization with reasoning")
        print("  ✓ Export with orchestration details")
        
    finally:
        # Clean up
        shutil.rmtree(temp_dir, ignore_errors=True)
        print(f"\n🧹 Cleaned up temporary directory: {temp_dir}")


if __name__ == "__main__":
    demo_session_lifecycle()
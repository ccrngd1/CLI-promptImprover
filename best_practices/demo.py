"""Demonstration of best practices repository and system prompt management."""

from best_practices.repository import BestPracticesRepository, BestPracticeCategory
from best_practices.system_prompts import SystemPromptManager, AgentType
from best_practices.reasoning_frameworks import ReasoningFramework, FrameworkType


def demonstrate_best_practices_repository():
    """Demonstrate the best practices repository functionality."""
    print("=== Best Practices Repository Demo ===\n")
    
    # Initialize repository
    repo = BestPracticesRepository()
    
    # Show default rules
    print(f"Loaded {len(repo.rules)} default best practice rules")
    
    # Get rules by category
    clarity_rules = repo.get_rules_by_category(BestPracticeCategory.CLARITY)
    print(f"\nClarity rules: {len(clarity_rules)}")
    for rule in clarity_rules:
        print(f"  - {rule.title}: {rule.description}")
    
    # Get applicable rules for specific conditions
    applicable_rules = repo.get_applicable_rules(["complex_prompts", "analysis_tasks"])
    print(f"\nRules applicable to complex analysis tasks: {len(applicable_rules)}")
    for rule in applicable_rules[:3]:  # Show first 3
        print(f"  - {rule.title} (Priority: {rule.priority})")
    
    # Get system prompt fragments
    fragments = repo.get_system_prompt_fragments(["format_specific"])
    print(f"\nSystem prompt fragments for format-specific tasks: {len(fragments)}")
    for fragment in fragments[:2]:  # Show first 2
        print(f"  - {fragment[:100]}...")
    
    print("\n" + "="*50 + "\n")


def demonstrate_system_prompt_manager():
    """Demonstrate the system prompt manager functionality."""
    print("=== System Prompt Manager Demo ===\n")
    
    # Initialize components
    repo = BestPracticesRepository()
    manager = SystemPromptManager(repo)
    
    # Show available agent types
    agent_types = manager.get_all_agent_types()
    print(f"Available agent types: {[at.value for at in agent_types]}")
    
    # Generate basic system prompt
    analyzer_prompt = manager.generate_system_prompt(AgentType.ANALYZER)
    print(f"\nAnalyzer system prompt length: {len(analyzer_prompt)} characters")
    print("Sample from analyzer prompt:")
    print(analyzer_prompt[:300] + "...")
    
    # Generate context-aware prompt
    context = {
        "prompt_text": "Analyze this marketing copy for clarity and effectiveness",
        "domain": "marketing",
        "task_type": "analysis",
        "complexity_level": "high"
    }
    
    context_prompt = manager.generate_context_aware_prompt(
        AgentType.ANALYZER,
        context,
        domain_knowledge="marketing"
    )
    
    print(f"\nContext-aware prompt length: {len(context_prompt)} characters")
    print("Context-specific additions found:", 
          "High Complexity Handling" in context_prompt)
    
    print("\n" + "="*50 + "\n")


def demonstrate_reasoning_frameworks():
    """Demonstrate the reasoning frameworks functionality."""
    print("=== Reasoning Frameworks Demo ===\n")
    
    # Initialize framework
    framework = ReasoningFramework()
    
    # Show available frameworks
    available = framework.get_available_frameworks()
    print(f"Available frameworks: {[f.value for f in available]}")
    
    # Get specific framework
    analysis_framework = framework.get_framework(FrameworkType.SYSTEMATIC_ANALYSIS)
    print(f"\nSystematic Analysis Framework:")
    print(f"  - Name: {analysis_framework.name}")
    print(f"  - Steps: {len(analysis_framework.steps)}")
    print(f"  - Meta-instructions: {len(analysis_framework.meta_instructions)}")
    
    # Generate framework prompt
    framework_prompt = framework.get_framework_prompt(FrameworkType.ITERATIVE_REFINEMENT)
    print(f"\nIterative Refinement framework prompt length: {len(framework_prompt)} characters")
    
    # Show framework steps
    refinement_framework = framework.get_framework(FrameworkType.ITERATIVE_REFINEMENT)
    print(f"\nRefinement framework steps:")
    for step in refinement_framework.steps:
        print(f"  {step.step_number}. {step.step_name}: {step.description}")
    
    print("\n" + "="*50 + "\n")


def demonstrate_integration():
    """Demonstrate how all components work together."""
    print("=== Integration Demo ===\n")
    
    # Initialize all components
    repo = BestPracticesRepository()
    manager = SystemPromptManager(repo)
    framework = ReasoningFramework()
    
    # Scenario: Generate a complete system prompt for a refiner agent
    # working on a complex healthcare prompt
    
    context = {
        "original_prompt": "Write about patient care",
        "domain": "healthcare",
        "task_type": "refinement",
        "complexity_level": "high",
        "user_experience_level": "beginner"
    }
    
    # Generate system prompt
    system_prompt = manager.generate_context_aware_prompt(
        AgentType.REFINER,
        context,
        domain_knowledge="healthcare"
    )
    
    # Add reasoning framework
    reasoning_prompt = framework.get_framework_prompt(FrameworkType.ITERATIVE_REFINEMENT)
    
    # Combine for complete agent prompt
    complete_prompt = f"{system_prompt}\n\n{reasoning_prompt}"
    
    print("Generated complete agent prompt for healthcare prompt refinement:")
    print(f"  - Total length: {len(complete_prompt)} characters")
    print(f"  - Includes best practices: {'Best Practices to Follow' in complete_prompt}")
    print(f"  - Includes reasoning framework: {'Iterative Refinement Framework' in complete_prompt}")
    print(f"  - Context-aware adaptations: {'High Complexity Handling' in complete_prompt}")
    print(f"  - Beginner-friendly: {'Beginner-Friendly Approach' in complete_prompt}")
    
    # Show sample of the complete prompt
    print(f"\nSample from complete prompt:")
    print(complete_prompt[:400] + "...")
    
    print("\n" + "="*50 + "\n")


if __name__ == "__main__":
    """Run all demonstrations."""
    print("Best Practices Repository and System Prompt Management Demo")
    print("=" * 60)
    
    demonstrate_best_practices_repository()
    demonstrate_system_prompt_manager()
    demonstrate_reasoning_frameworks()
    demonstrate_integration()
    
    print("Demo completed successfully!")
    print("\nKey features demonstrated:")
    print("✓ Curated prompt engineering best practices")
    print("✓ Specialized system prompt templates for each agent type")
    print("✓ Reasoning frameworks for structured analysis")
    print("✓ Dynamic system prompt generation based on context")
    print("✓ Integration of all components for complete agent prompts")
# Best Practices Repository and System Prompt Management

This module provides a comprehensive system for managing prompt engineering best practices and generating specialized system prompts for different agent types in the Bedrock Prompt Optimizer.

## Overview

The best practices system consists of three main components:

1. **BestPracticesRepository**: Manages curated prompt engineering techniques and rules
2. **SystemPromptManager**: Generates specialized system prompts for different agent types
3. **ReasoningFramework**: Provides structured reasoning frameworks for LLM-based analysis

## Components

### BestPracticesRepository

The repository manages a curated collection of prompt engineering best practices organized by categories:

- **Clarity**: Rules for clear and specific language
- **Structure**: Guidelines for prompt organization and flow
- **Examples**: Best practices for providing examples and demonstrations
- **Reasoning**: Techniques for requesting step-by-step reasoning
- **Context**: Guidelines for providing sufficient context
- **Instructions**: Rules for effective instruction formatting
- **Output Format**: Best practices for specifying output requirements
- **Role Definition**: Guidelines for establishing clear AI roles

#### Key Features

- **Rule Management**: Add, update, and remove best practice rules
- **Conditional Application**: Apply rules based on specific conditions (domain, task type, etc.)
- **Priority Sorting**: Rules are prioritized to ensure most important practices are applied first
- **Persistence**: Save and load rules from JSON files
- **System Prompt Integration**: Generate prompt fragments for inclusion in system prompts

#### Usage Example

```python
from best_practices.repository import BestPracticesRepository, BestPracticeCategory

# Initialize repository
repo = BestPracticesRepository()

# Get rules for specific category
clarity_rules = repo.get_rules_by_category(BestPracticeCategory.CLARITY)

# Get applicable rules for conditions
applicable_rules = repo.get_applicable_rules(["complex_prompts", "analysis_tasks"])

# Get system prompt fragments
fragments = repo.get_system_prompt_fragments(["format_specific"])
```

### SystemPromptManager

The system prompt manager generates specialized prompts for different agent types, incorporating best practices and context-specific adaptations.

#### Supported Agent Types

- **Analyzer**: Specializes in prompt structure and clarity analysis
- **Refiner**: Focuses on improving prompts based on analysis and feedback
- **Validator**: Ensures refined prompts meet quality standards
- **Evaluator**: Assesses prompt performance using multiple criteria
- **Orchestrator**: Coordinates multiple agents and synthesizes outputs

#### Key Features

- **Template-Based Generation**: Each agent type has a specialized template
- **Best Practice Integration**: Automatically includes relevant best practices
- **Context Awareness**: Adapts prompts based on domain and task context
- **Dynamic Adaptation**: Adds specific instructions for complexity levels and user experience
- **Reasoning Framework Integration**: Incorporates structured reasoning approaches

#### Usage Example

```python
from best_practices.system_prompts import SystemPromptManager, AgentType
from best_practices.repository import BestPracticesRepository

# Initialize components
repo = BestPracticesRepository()
manager = SystemPromptManager(repo)

# Generate basic system prompt
prompt = manager.generate_system_prompt(AgentType.ANALYZER)

# Generate context-aware prompt
context = {
    "domain": "healthcare",
    "task_type": "analysis",
    "complexity_level": "high"
}
context_prompt = manager.generate_context_aware_prompt(
    AgentType.ANALYZER, 
    context, 
    domain_knowledge="healthcare"
)
```

### ReasoningFramework

The reasoning framework provides structured approaches for LLM-based analysis and decision making.

#### Available Frameworks

- **Systematic Analysis**: Comprehensive prompt analysis with structured steps
- **Iterative Refinement**: Structured approach for improving prompts
- **Multi-Criteria Evaluation**: Framework for evaluating prompts using multiple criteria
- **Collaborative Synthesis**: Approach for synthesizing multiple agent outputs

#### Key Features

- **Step-by-Step Structure**: Each framework defines clear steps with guiding questions
- **Validation Criteria**: Built-in criteria for validating framework execution
- **Prompt Generation**: Generate complete prompts that incorporate the framework
- **Execution Validation**: Validate that frameworks were properly followed
- **Extensibility**: Add custom frameworks for specific use cases

#### Usage Example

```python
from best_practices.reasoning_frameworks import ReasoningFramework, FrameworkType

# Initialize framework
framework = ReasoningFramework()

# Get specific framework
analysis_framework = framework.get_framework(FrameworkType.SYSTEMATIC_ANALYSIS)

# Generate framework prompt
framework_prompt = framework.get_framework_prompt(FrameworkType.ITERATIVE_REFINEMENT)

# Validate framework execution
execution_result = {"step_1_completed": True, "step_2_completed": True}
validation = framework.validate_framework_execution(
    FrameworkType.SYSTEMATIC_ANALYSIS, 
    execution_result
)
```

## Integration Example

Here's how all components work together to create a complete agent system prompt:

```python
from best_practices.repository import BestPracticesRepository
from best_practices.system_prompts import SystemPromptManager, AgentType
from best_practices.reasoning_frameworks import ReasoningFramework, FrameworkType

# Initialize all components
repo = BestPracticesRepository()
manager = SystemPromptManager(repo)
framework = ReasoningFramework()

# Define context
context = {
    "domain": "healthcare",
    "task_type": "refinement",
    "complexity_level": "high",
    "user_experience_level": "beginner"
}

# Generate complete system prompt
system_prompt = manager.generate_context_aware_prompt(
    AgentType.REFINER,
    context,
    domain_knowledge="healthcare"
)

# Add reasoning framework
reasoning_prompt = framework.get_framework_prompt(FrameworkType.ITERATIVE_REFINEMENT)

# Combine for complete agent prompt
complete_prompt = f"{system_prompt}\n\n{reasoning_prompt}"
```

## Default Best Practices

The repository comes with 8 default best practice rules covering essential prompt engineering techniques:

1. **Use Clear and Specific Language** - Avoid ambiguous terms and use precise language
2. **Use Clear Structure with Headers** - Organize prompts with clear sections
3. **Provide Few-Shot Examples** - Include 1-3 examples of desired format
4. **Request Step-by-Step Reasoning** - Ask for explicit reasoning steps
5. **Provide Sufficient Context** - Include relevant background information
6. **Use Imperative Voice** - Use direct commands rather than questions
7. **Specify Output Format** - Clearly define expected output structure
8. **Define Clear Roles** - Establish clear role and expertise for the AI

## Extensibility

The system is designed to be easily extensible:

- **Add New Rules**: Create custom best practice rules for specific domains
- **Custom Agent Types**: Define new agent types with specialized templates
- **Custom Frameworks**: Add domain-specific reasoning frameworks
- **Dynamic Adaptation**: Extend context-aware adaptations for new scenarios

## Testing

Comprehensive unit tests are provided in `test_best_practices.py` covering:

- Repository functionality (CRUD operations, rule filtering, persistence)
- System prompt generation (basic, context-aware, template management)
- Reasoning framework operations (framework retrieval, prompt generation, validation)
- Data model serialization and deserialization
- Integration scenarios

Run tests with:
```bash
python -m pytest test_best_practices.py -v
```

## Demonstration

Run the demonstration script to see all components in action:
```bash
python -m best_practices.demo
```

This will show:
- Best practices repository functionality
- System prompt generation for different agent types
- Reasoning framework integration
- Complete integration example

## File Structure

```
best_practices/
├── __init__.py                 # Module initialization
├── repository.py              # Best practices repository
├── system_prompts.py          # System prompt management
├── reasoning_frameworks.py    # Reasoning frameworks
├── demo.py                   # Demonstration script
├── README.md                 # This documentation
└── data/                     # Storage directory for saved rules
```

## Requirements Satisfied

This implementation satisfies the following requirements from the specification:

- **6.1**: LLM agents use specialized system prompts with embedded best practices
- **6.4**: Best practices are incorporated into agent system prompts and reasoning
- **6.6**: System supports updating best practices and system prompt customization

The system provides a robust foundation for managing prompt engineering knowledge and generating intelligent, context-aware system prompts for the multi-agent optimization system.
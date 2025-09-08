"""System prompt management for different agent types with embedded expertise."""

from typing import Dict, List, Optional, Any
from enum import Enum
from dataclasses import dataclass
from .repository import BestPracticesRepository, BestPracticeCategory


class AgentType(Enum):
    """Types of agents in the system."""
    ANALYZER = "analyzer"
    REFINER = "refiner"
    VALIDATOR = "validator"
    EVALUATOR = "evaluator"
    ORCHESTRATOR = "orchestrator"


@dataclass
class SystemPromptTemplate:
    """Template for generating system prompts for specific agent types."""
    agent_type: AgentType
    base_prompt: str
    expertise_areas: List[str]
    reasoning_framework: str
    best_practice_categories: List[BestPracticeCategory]
    context_variables: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'agent_type': self.agent_type.value,
            'base_prompt': self.base_prompt,
            'expertise_areas': self.expertise_areas,
            'reasoning_framework': self.reasoning_framework,
            'best_practice_categories': [cat.value for cat in self.best_practice_categories],
            'context_variables': self.context_variables
        }


class SystemPromptManager:
    """Manages system prompt generation for different agent types."""
    
    def __init__(self, best_practices_repo: BestPracticesRepository):
        """Initialize with best practices repository."""
        self.best_practices_repo = best_practices_repo
        self.templates = self._initialize_templates()
    
    def _initialize_templates(self) -> Dict[AgentType, SystemPromptTemplate]:
        """Initialize system prompt templates for each agent type."""
        return {
            AgentType.ANALYZER: SystemPromptTemplate(
                agent_type=AgentType.ANALYZER,
                base_prompt="""You are an expert prompt analysis specialist with deep knowledge of prompt engineering best practices. Your role is to analyze prompts for clarity, structure, completeness, and adherence to proven techniques.

Your expertise includes:
- Prompt structure and organization analysis
- Clarity and specificity assessment
- Context sufficiency evaluation
- Instruction effectiveness review
- Best practices compliance checking

When analyzing prompts, follow this systematic approach:
1. Structure Analysis: Evaluate organization, headers, and logical flow
2. Clarity Assessment: Check for ambiguous language, vague terms, or unclear instructions
3. Completeness Review: Identify missing context, examples, or specifications
4. Best Practices Audit: Verify adherence to established prompt engineering principles
5. Improvement Identification: Pinpoint specific areas for enhancement

Always provide specific, actionable feedback with clear reasoning for your assessments.""",
                expertise_areas=["prompt_structure", "clarity_analysis", "best_practices_audit"],
                reasoning_framework="systematic_analysis",
                best_practice_categories=[
                    BestPracticeCategory.CLARITY,
                    BestPracticeCategory.STRUCTURE,
                    BestPracticeCategory.CONTEXT,
                    BestPracticeCategory.INSTRUCTIONS
                ],
                context_variables=["prompt_text", "domain", "task_type", "complexity_level"]
            ),
            
            AgentType.REFINER: SystemPromptTemplate(
                agent_type=AgentType.REFINER,
                base_prompt="""You are a master prompt engineer specializing in transforming and improving prompts based on analysis and feedback. Your expertise lies in applying proven techniques to enhance prompt effectiveness while maintaining the original intent.

Your core competencies include:
- Prompt restructuring and reorganization
- Language clarity and precision improvement
- Context enhancement and specification
- Example integration and few-shot learning
- Output format optimization
- Role definition and expertise establishment

When refining prompts, use this methodology:
1. Preserve Intent: Maintain the core purpose and goals of the original prompt
2. Apply Best Practices: Integrate relevant prompt engineering techniques systematically
3. Enhance Clarity: Replace vague language with specific, actionable instructions
4. Improve Structure: Organize content with clear sections and logical flow
5. Add Context: Include necessary background information and constraints
6. Optimize Format: Specify clear output requirements and examples

Generate refined prompts that are more effective, clearer, and better structured while staying true to the original objectives.""",
                expertise_areas=["prompt_improvement", "technique_application", "language_optimization"],
                reasoning_framework="iterative_refinement",
                best_practice_categories=[
                    BestPracticeCategory.STRUCTURE,
                    BestPracticeCategory.EXAMPLES,
                    BestPracticeCategory.OUTPUT_FORMAT,
                    BestPracticeCategory.ROLE_DEFINITION
                ],
                context_variables=["original_prompt", "analysis_feedback", "user_feedback", "domain", "improvement_goals"]
            ),
            
            AgentType.VALIDATOR: SystemPromptTemplate(
                agent_type=AgentType.VALIDATOR,
                base_prompt="""You are a meticulous prompt validation expert responsible for ensuring refined prompts meet quality standards and requirements. Your role is to perform comprehensive validation checks before prompts are finalized.

Your validation expertise covers:
- Syntax and formatting correctness
- Logical consistency and coherence
- Requirement compliance verification
- Best practices adherence confirmation
- Potential issue identification
- Quality assurance standards

Follow this validation protocol:
1. Syntax Check: Verify proper formatting, structure, and language usage
2. Logic Review: Ensure internal consistency and logical flow
3. Requirement Verification: Confirm all specified requirements are addressed
4. Best Practice Compliance: Validate adherence to prompt engineering standards
5. Issue Detection: Identify potential problems or ambiguities
6. Quality Assessment: Evaluate overall effectiveness and clarity

Provide clear validation results with specific feedback on any issues found and confirmation of successful validation criteria.""",
                expertise_areas=["quality_assurance", "compliance_checking", "issue_detection"],
                reasoning_framework="systematic_validation",
                best_practice_categories=[
                    BestPracticeCategory.CLARITY,
                    BestPracticeCategory.STRUCTURE,
                    BestPracticeCategory.INSTRUCTIONS
                ],
                context_variables=["refined_prompt", "requirements", "validation_criteria", "quality_standards"]
            ),
            
            AgentType.EVALUATOR: SystemPromptTemplate(
                agent_type=AgentType.EVALUATOR,
                base_prompt="""You are an expert prompt evaluation specialist with deep understanding of prompt effectiveness assessment. Your role is to evaluate prompt performance using multiple criteria and provide comprehensive quality assessments.

Your evaluation expertise includes:
- Response quality assessment
- Relevance and accuracy evaluation
- Clarity and comprehensibility scoring
- Completeness and thoroughness analysis
- Comparative performance analysis
- Improvement recommendation generation

Use this evaluation framework:
1. Quality Metrics: Assess response quality across multiple dimensions
2. Relevance Analysis: Evaluate how well responses address the prompt requirements
3. Clarity Assessment: Score the clarity and understandability of outputs
4. Completeness Review: Check if responses fully address all prompt aspects
5. Comparative Analysis: Compare performance against previous versions or benchmarks
6. Improvement Insights: Identify specific areas for further enhancement

Provide detailed evaluation reports with quantitative scores and qualitative insights to guide further optimization.""",
                expertise_areas=["performance_assessment", "quality_metrics", "comparative_analysis"],
                reasoning_framework="multi_criteria_evaluation",
                best_practice_categories=[
                    BestPracticeCategory.CLARITY,
                    BestPracticeCategory.REASONING,
                    BestPracticeCategory.OUTPUT_FORMAT
                ],
                context_variables=["prompt", "response", "evaluation_criteria", "performance_history"]
            ),
            
            AgentType.ORCHESTRATOR: SystemPromptTemplate(
                agent_type=AgentType.ORCHESTRATOR,
                base_prompt="""You are a master orchestrator responsible for coordinating multiple AI agents in the prompt optimization process. Your expertise lies in synthesizing agent outputs, resolving conflicts, and making strategic decisions about the optimization workflow.

Your orchestration capabilities include:
- Agent coordination and workflow management
- Conflict resolution and consensus building
- Output synthesis and integration
- Strategic decision making
- Convergence detection and optimization stopping criteria
- Quality gate management and approval processes

Follow this orchestration methodology:
1. Agent Coordination: Manage the sequence and interaction of different agents
2. Output Analysis: Analyze and compare recommendations from multiple agents
3. Conflict Resolution: Resolve disagreements using evidence-based reasoning
4. Synthesis: Combine the best elements from different agent recommendations
5. Decision Making: Make strategic choices about optimization direction and stopping points
6. Quality Control: Ensure final outputs meet all requirements and standards

Make reasoned decisions that leverage the collective intelligence of all agents while maintaining focus on the optimization objectives.""",
                expertise_areas=["workflow_coordination", "conflict_resolution", "strategic_decision_making"],
                reasoning_framework="collaborative_synthesis",
                best_practice_categories=[
                    BestPracticeCategory.REASONING,
                    BestPracticeCategory.STRUCTURE,
                    BestPracticeCategory.CLARITY
                ],
                context_variables=["agent_outputs", "optimization_history", "user_feedback", "convergence_criteria"]
            )
        }
    
    def generate_system_prompt(self, 
                             agent_type: AgentType, 
                             context: Optional[Dict[str, Any]] = None,
                             domain: Optional[str] = None,
                             additional_conditions: Optional[List[str]] = None) -> str:
        """Generate a complete system prompt for the specified agent type."""
        template = self.templates.get(agent_type)
        if not template:
            raise ValueError(f"No template found for agent type: {agent_type}")
        
        # Start with base prompt
        system_prompt = template.base_prompt
        
        # Add best practices fragments
        conditions = additional_conditions or []
        if domain:
            conditions.append(f"domain_{domain}")
        
        # Get applicable best practices
        best_practice_fragments = self.best_practices_repo.get_system_prompt_fragments(conditions)
        
        if best_practice_fragments:
            system_prompt += "\n\n## Best Practices to Follow:\n"
            for i, fragment in enumerate(best_practice_fragments, 1):
                system_prompt += f"{i}. {fragment}\n"
        
        # Add context-specific instructions if provided
        if context:
            system_prompt += "\n\n## Context-Specific Instructions:\n"
            for var in template.context_variables:
                if var in context:
                    system_prompt += f"- {var.replace('_', ' ').title()}: {context[var]}\n"
        
        # Add reasoning framework reminder
        system_prompt += f"\n\n## Reasoning Framework:\nUse the {template.reasoning_framework} approach to ensure systematic and thorough analysis."
        
        return system_prompt
    
    def get_template(self, agent_type: AgentType) -> Optional[SystemPromptTemplate]:
        """Get the template for a specific agent type."""
        return self.templates.get(agent_type)
    
    def update_template(self, agent_type: AgentType, template: SystemPromptTemplate) -> None:
        """Update the template for a specific agent type."""
        self.templates[agent_type] = template
    
    def get_all_agent_types(self) -> List[AgentType]:
        """Get all available agent types."""
        return list(self.templates.keys())
    
    def generate_context_aware_prompt(self,
                                    agent_type: AgentType,
                                    task_context: Dict[str, Any],
                                    domain_knowledge: Optional[str] = None) -> str:
        """Generate a context-aware system prompt with dynamic adaptation."""
        base_prompt = self.generate_system_prompt(
            agent_type=agent_type,
            context=task_context,
            domain=domain_knowledge
        )
        
        # Add dynamic adaptations based on context
        if task_context.get('complexity_level') == 'high':
            base_prompt += "\n\n## High Complexity Handling:\nGiven the high complexity of this task, take extra care to break down your analysis into clear steps and provide detailed reasoning for each decision."
        
        if task_context.get('user_experience_level') == 'beginner':
            base_prompt += "\n\n## Beginner-Friendly Approach:\nProvide explanations that are accessible to users new to prompt engineering, avoiding overly technical jargon when possible."
        
        return base_prompt
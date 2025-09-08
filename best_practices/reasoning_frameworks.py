"""Reasoning frameworks for LLM-based analysis and decision making."""

from typing import Dict, List, Optional, Any
from enum import Enum
from dataclasses import dataclass


class FrameworkType(Enum):
    """Types of reasoning frameworks available."""
    SYSTEMATIC_ANALYSIS = "systematic_analysis"
    ITERATIVE_REFINEMENT = "iterative_refinement"
    SYSTEMATIC_VALIDATION = "systematic_validation"
    MULTI_CRITERIA_EVALUATION = "multi_criteria_evaluation"
    COLLABORATIVE_SYNTHESIS = "collaborative_synthesis"
    PROBLEM_SOLVING = "problem_solving"
    DECISION_MAKING = "decision_making"


@dataclass
class ReasoningStep:
    """Represents a single step in a reasoning framework."""
    step_number: int
    step_name: str
    description: str
    guiding_questions: List[str]
    expected_outputs: List[str]
    validation_criteria: List[str]


@dataclass
class ReasoningFrameworkDefinition:
    """Complete definition of a reasoning framework."""
    framework_type: FrameworkType
    name: str
    description: str
    steps: List[ReasoningStep]
    meta_instructions: List[str]
    quality_checks: List[str]


class ReasoningFramework:
    """Manages reasoning frameworks for structured LLM analysis and decision making."""
    
    def __init__(self):
        """Initialize with default reasoning frameworks."""
        self.frameworks = self._initialize_frameworks()
    
    def _initialize_frameworks(self) -> Dict[FrameworkType, ReasoningFrameworkDefinition]:
        """Initialize the default set of reasoning frameworks."""
        return {
            FrameworkType.SYSTEMATIC_ANALYSIS: ReasoningFrameworkDefinition(
                framework_type=FrameworkType.SYSTEMATIC_ANALYSIS,
                name="Systematic Analysis Framework",
                description="Structured approach for comprehensive prompt analysis",
                steps=[
                    ReasoningStep(
                        step_number=1,
                        step_name="Initial Assessment",
                        description="Conduct high-level evaluation of the prompt",
                        guiding_questions=[
                            "What is the primary objective of this prompt?",
                            "Who is the intended audience?",
                            "What type of response is expected?"
                        ],
                        expected_outputs=["Objective summary", "Audience identification", "Response type classification"],
                        validation_criteria=["Clear objective stated", "Audience clearly identified", "Response type specified"]
                    ),
                    ReasoningStep(
                        step_number=2,
                        step_name="Structure Analysis",
                        description="Evaluate the organizational structure and flow",
                        guiding_questions=[
                            "Is the prompt well-organized with clear sections?",
                            "Does the information flow logically?",
                            "Are there appropriate headers or separators?"
                        ],
                        expected_outputs=["Structure assessment", "Flow evaluation", "Organization recommendations"],
                        validation_criteria=["Structure clearly assessed", "Flow issues identified", "Specific improvements suggested"]
                    ),
                    ReasoningStep(
                        step_number=3,
                        step_name="Clarity Evaluation",
                        description="Assess language clarity and specificity",
                        guiding_questions=[
                            "Are the instructions clear and unambiguous?",
                            "Is the language specific enough?",
                            "Are there any vague or confusing terms?"
                        ],
                        expected_outputs=["Clarity score", "Ambiguity identification", "Language improvement suggestions"],
                        validation_criteria=["Clarity issues identified", "Specific examples provided", "Improvements suggested"]
                    ),
                    ReasoningStep(
                        step_number=4,
                        step_name="Completeness Review",
                        description="Check for missing elements and context",
                        guiding_questions=[
                            "Is sufficient context provided?",
                            "Are all necessary constraints specified?",
                            "Are examples provided where helpful?"
                        ],
                        expected_outputs=["Completeness assessment", "Missing elements list", "Context enhancement suggestions"],
                        validation_criteria=["Missing elements identified", "Context gaps noted", "Enhancement suggestions provided"]
                    ),
                    ReasoningStep(
                        step_number=5,
                        step_name="Best Practices Audit",
                        description="Verify adherence to prompt engineering best practices",
                        guiding_questions=[
                            "Which best practices are already applied?",
                            "Which best practices are missing?",
                            "How can best practices be better integrated?"
                        ],
                        expected_outputs=["Best practices compliance report", "Missing practices identification", "Integration recommendations"],
                        validation_criteria=["Compliance clearly assessed", "Missing practices identified", "Integration plan provided"]
                    )
                ],
                meta_instructions=[
                    "Be systematic and thorough in your analysis",
                    "Provide specific examples and evidence for your assessments",
                    "Focus on actionable improvements",
                    "Consider the prompt's intended use case throughout your analysis"
                ],
                quality_checks=[
                    "All steps completed systematically",
                    "Specific evidence provided for assessments",
                    "Actionable recommendations given",
                    "Analysis aligned with prompt objectives"
                ]
            ),
            
            FrameworkType.ITERATIVE_REFINEMENT: ReasoningFrameworkDefinition(
                framework_type=FrameworkType.ITERATIVE_REFINEMENT,
                name="Iterative Refinement Framework",
                description="Structured approach for improving prompts through iterative enhancement",
                steps=[
                    ReasoningStep(
                        step_number=1,
                        step_name="Intent Preservation",
                        description="Ensure the core intent and objectives are maintained",
                        guiding_questions=[
                            "What is the core intent of the original prompt?",
                            "What are the key objectives that must be preserved?",
                            "What constraints must be maintained?"
                        ],
                        expected_outputs=["Intent statement", "Objective list", "Constraint identification"],
                        validation_criteria=["Intent clearly stated", "Objectives preserved", "Constraints respected"]
                    ),
                    ReasoningStep(
                        step_number=2,
                        step_name="Improvement Planning",
                        description="Plan specific improvements based on analysis and feedback",
                        guiding_questions=[
                            "What are the priority areas for improvement?",
                            "Which best practices should be applied?",
                            "How can structure be enhanced?"
                        ],
                        expected_outputs=["Improvement priority list", "Best practices application plan", "Structure enhancement strategy"],
                        validation_criteria=["Priorities clearly defined", "Best practices identified", "Strategy is actionable"]
                    ),
                    ReasoningStep(
                        step_number=3,
                        step_name="Language Enhancement",
                        description="Improve clarity, specificity, and precision of language",
                        guiding_questions=[
                            "Which terms need to be more specific?",
                            "How can instructions be clearer?",
                            "What ambiguities need resolution?"
                        ],
                        expected_outputs=["Language improvements", "Specificity enhancements", "Ambiguity resolutions"],
                        validation_criteria=["Language clearly improved", "Specificity increased", "Ambiguities resolved"]
                    ),
                    ReasoningStep(
                        step_number=4,
                        step_name="Structure Optimization",
                        description="Reorganize and structure the prompt for better flow and clarity",
                        guiding_questions=[
                            "How should the prompt be organized?",
                            "What sections or headers are needed?",
                            "How can the flow be improved?"
                        ],
                        expected_outputs=["Organizational structure", "Section definitions", "Flow improvements"],
                        validation_criteria=["Structure is logical", "Sections are clear", "Flow is improved"]
                    ),
                    ReasoningStep(
                        step_number=5,
                        step_name="Enhancement Integration",
                        description="Integrate all improvements into a cohesive refined prompt",
                        guiding_questions=[
                            "How do all improvements work together?",
                            "Is the refined prompt cohesive?",
                            "Does it maintain the original intent?"
                        ],
                        expected_outputs=["Integrated refined prompt", "Cohesion assessment", "Intent verification"],
                        validation_criteria=["Prompt is cohesive", "Improvements integrated", "Intent preserved"]
                    )
                ],
                meta_instructions=[
                    "Maintain the original intent throughout refinement",
                    "Apply improvements systematically and thoughtfully",
                    "Ensure all changes enhance rather than complicate",
                    "Test refinements against the original objectives"
                ],
                quality_checks=[
                    "Original intent preserved",
                    "Improvements systematically applied",
                    "Refined prompt is cohesive",
                    "Enhancement goals achieved"
                ]
            ),
            
            FrameworkType.MULTI_CRITERIA_EVALUATION: ReasoningFrameworkDefinition(
                framework_type=FrameworkType.MULTI_CRITERIA_EVALUATION,
                name="Multi-Criteria Evaluation Framework",
                description="Comprehensive evaluation approach using multiple assessment criteria",
                steps=[
                    ReasoningStep(
                        step_number=1,
                        step_name="Criteria Definition",
                        description="Define and prioritize evaluation criteria",
                        guiding_questions=[
                            "What are the key quality dimensions to evaluate?",
                            "How should criteria be weighted?",
                            "What are the success thresholds?"
                        ],
                        expected_outputs=["Evaluation criteria list", "Criteria weights", "Success thresholds"],
                        validation_criteria=["Criteria clearly defined", "Weights assigned", "Thresholds set"]
                    ),
                    ReasoningStep(
                        step_number=2,
                        step_name="Individual Assessment",
                        description="Evaluate performance on each criterion independently",
                        guiding_questions=[
                            "How does the prompt/response perform on each criterion?",
                            "What evidence supports each assessment?",
                            "What are the strengths and weaknesses?"
                        ],
                        expected_outputs=["Individual criterion scores", "Supporting evidence", "Strength/weakness analysis"],
                        validation_criteria=["All criteria assessed", "Evidence provided", "Analysis is thorough"]
                    ),
                    ReasoningStep(
                        step_number=3,
                        step_name="Comparative Analysis",
                        description="Compare performance against benchmarks or previous versions",
                        guiding_questions=[
                            "How does this compare to previous versions?",
                            "What improvements or regressions are evident?",
                            "How does it compare to benchmarks?"
                        ],
                        expected_outputs=["Comparison results", "Improvement identification", "Benchmark analysis"],
                        validation_criteria=["Comparisons made", "Changes identified", "Benchmarks considered"]
                    ),
                    ReasoningStep(
                        step_number=4,
                        step_name="Synthesis and Scoring",
                        description="Synthesize individual assessments into overall evaluation",
                        guiding_questions=[
                            "What is the weighted overall score?",
                            "Which criteria drive the overall assessment?",
                            "What is the confidence level in the evaluation?"
                        ],
                        expected_outputs=["Overall score", "Key drivers analysis", "Confidence assessment"],
                        validation_criteria=["Score calculated correctly", "Drivers identified", "Confidence stated"]
                    ),
                    ReasoningStep(
                        step_number=5,
                        step_name="Improvement Recommendations",
                        description="Generate specific recommendations for improvement",
                        guiding_questions=[
                            "Which areas need the most improvement?",
                            "What specific changes would have the highest impact?",
                            "What are the next steps for optimization?"
                        ],
                        expected_outputs=["Priority improvement areas", "Specific recommendations", "Next steps"],
                        validation_criteria=["Priorities clear", "Recommendations specific", "Steps actionable"]
                    )
                ],
                meta_instructions=[
                    "Be objective and evidence-based in all assessments",
                    "Consider multiple perspectives and use cases",
                    "Provide specific, actionable feedback",
                    "Balance individual criteria with overall effectiveness"
                ],
                quality_checks=[
                    "All criteria systematically evaluated",
                    "Evidence supports all assessments",
                    "Recommendations are actionable",
                    "Evaluation is comprehensive and fair"
                ]
            )
        }
    
    def get_framework(self, framework_type: FrameworkType) -> Optional[ReasoningFrameworkDefinition]:
        """Get a specific reasoning framework."""
        return self.frameworks.get(framework_type)
    
    def get_framework_prompt(self, framework_type: FrameworkType) -> str:
        """Generate a prompt that incorporates the reasoning framework."""
        framework = self.get_framework(framework_type)
        if not framework:
            raise ValueError(f"Framework {framework_type} not found")
        
        prompt = f"## {framework.name}\n\n{framework.description}\n\n"
        prompt += "Follow these steps systematically:\n\n"
        
        for step in framework.steps:
            prompt += f"### Step {step.step_number}: {step.step_name}\n"
            prompt += f"{step.description}\n\n"
            prompt += "Consider these guiding questions:\n"
            for question in step.guiding_questions:
                prompt += f"- {question}\n"
            prompt += "\nExpected outputs:\n"
            for output in step.expected_outputs:
                prompt += f"- {output}\n"
            prompt += "\n"
        
        prompt += "## Meta-Instructions:\n"
        for instruction in framework.meta_instructions:
            prompt += f"- {instruction}\n"
        
        prompt += "\n## Quality Checks:\n"
        for check in framework.quality_checks:
            prompt += f"- {check}\n"
        
        return prompt
    
    def validate_framework_execution(self, 
                                   framework_type: FrameworkType, 
                                   execution_result: Dict[str, Any]) -> Dict[str, bool]:
        """Validate that a framework was executed properly."""
        framework = self.get_framework(framework_type)
        if not framework:
            return {"valid": False, "error": f"Framework {framework_type} not found"}
        
        validation_results = {}
        
        # Check if all steps were addressed
        for step in framework.steps:
            step_key = f"step_{step.step_number}_completed"
            validation_results[step_key] = step_key in execution_result
        
        # Check quality criteria
        for i, check in enumerate(framework.quality_checks):
            check_key = f"quality_check_{i+1}_passed"
            validation_results[check_key] = check_key in execution_result
        
        validation_results["overall_valid"] = all(validation_results.values())
        
        return validation_results
    
    def get_available_frameworks(self) -> List[FrameworkType]:
        """Get list of all available framework types."""
        return list(self.frameworks.keys())
    
    def add_custom_framework(self, framework: ReasoningFrameworkDefinition) -> None:
        """Add a custom reasoning framework."""
        self.frameworks[framework.framework_type] = framework
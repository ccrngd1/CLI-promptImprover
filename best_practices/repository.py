"""Best practices repository for curated prompt engineering techniques."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from enum import Enum
import json
import os


class BestPracticeCategory(Enum):
    """Categories for organizing best practices."""
    CLARITY = "clarity"
    STRUCTURE = "structure"
    EXAMPLES = "examples"
    REASONING = "reasoning"
    CONTEXT = "context"
    INSTRUCTIONS = "instructions"
    OUTPUT_FORMAT = "output_format"
    ROLE_DEFINITION = "role_definition"


@dataclass
class BestPracticeRule:
    """Represents a single best practice rule for prompt engineering."""
    rule_id: str
    category: BestPracticeCategory
    title: str
    description: str
    system_prompt_fragment: str
    applicability_conditions: List[str]
    priority: int
    examples: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'rule_id': self.rule_id,
            'category': self.category.value,
            'title': self.title,
            'description': self.description,
            'system_prompt_fragment': self.system_prompt_fragment,
            'applicability_conditions': self.applicability_conditions,
            'priority': self.priority,
            'examples': self.examples
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BestPracticeRule':
        """Create from dictionary."""
        return cls(
            rule_id=data['rule_id'],
            category=BestPracticeCategory(data['category']),
            title=data['title'],
            description=data['description'],
            system_prompt_fragment=data['system_prompt_fragment'],
            applicability_conditions=data['applicability_conditions'],
            priority=data['priority'],
            examples=data['examples']
        )


class BestPracticesRepository:
    """Repository for managing curated prompt engineering best practices."""
    
    def __init__(self, storage_path: str = "best_practices/data"):
        """Initialize the repository with storage path."""
        self.storage_path = storage_path
        self.rules: Dict[str, BestPracticeRule] = {}
        self._ensure_storage_directory()
        self._load_default_rules()
    
    def _ensure_storage_directory(self) -> None:
        """Ensure storage directory exists."""
        os.makedirs(self.storage_path, exist_ok=True)
    
    def _load_default_rules(self) -> None:
        """Load default best practice rules."""
        default_rules = self._get_default_rules()
        for rule in default_rules:
            self.rules[rule.rule_id] = rule
    
    def _get_default_rules(self) -> List[BestPracticeRule]:
        """Get the default set of best practice rules."""
        return [
            BestPracticeRule(
                rule_id="clarity_001",
                category=BestPracticeCategory.CLARITY,
                title="Use Clear and Specific Language",
                description="Avoid ambiguous terms and use precise, specific language",
                system_prompt_fragment="Use clear, specific language. Avoid ambiguous terms like 'good', 'bad', 'some', 'many'. Be precise in your instructions and requirements.",
                applicability_conditions=["all_prompts"],
                priority=1,
                examples=["Instead of 'Write something good about X', use 'Write a 200-word analysis highlighting three key benefits of X'"]
            ),
            BestPracticeRule(
                rule_id="structure_001",
                category=BestPracticeCategory.STRUCTURE,
                title="Use Clear Structure with Headers",
                description="Organize prompts with clear sections and headers",
                system_prompt_fragment="Structure your prompts with clear sections using headers like 'Task:', 'Context:', 'Requirements:', 'Output Format:'. This improves readability and comprehension.",
                applicability_conditions=["complex_prompts", "multi_part_tasks"],
                priority=2,
                examples=["Task: Analyze the data\nContext: Sales data from Q1\nRequirements: Focus on trends\nOutput Format: Bullet points"]
            ),
            BestPracticeRule(
                rule_id="examples_001",
                category=BestPracticeCategory.EXAMPLES,
                title="Provide Few-Shot Examples",
                description="Include 1-3 examples of desired input/output format",
                system_prompt_fragment="When possible, include 1-3 examples showing the desired input/output format. Examples should be diverse and representative of the expected use cases.",
                applicability_conditions=["format_specific", "complex_outputs"],
                priority=3,
                examples=["Example 1: Input: 'apple' → Output: 'fruit, red/green, grows on trees'"]
            ),
            BestPracticeRule(
                rule_id="reasoning_001",
                category=BestPracticeCategory.REASONING,
                title="Request Step-by-Step Reasoning",
                description="Ask for explicit reasoning steps for complex tasks",
                system_prompt_fragment="For complex analysis or decision-making tasks, explicitly request step-by-step reasoning. Use phrases like 'Think through this step by step' or 'Explain your reasoning process'.",
                applicability_conditions=["analysis_tasks", "decision_making"],
                priority=2,
                examples=["Think through this step by step: 1) First analyze X, 2) Then consider Y, 3) Finally conclude Z"]
            ),
            BestPracticeRule(
                rule_id="context_001",
                category=BestPracticeCategory.CONTEXT,
                title="Provide Sufficient Context",
                description="Include relevant background information and constraints",
                system_prompt_fragment="Provide sufficient context including background information, constraints, target audience, and any relevant domain knowledge needed for the task.",
                applicability_conditions=["domain_specific", "contextual_tasks"],
                priority=1,
                examples=["Context: You are analyzing financial data for a startup in the healthcare sector targeting elderly patients"]
            ),
            BestPracticeRule(
                rule_id="instructions_001",
                category=BestPracticeCategory.INSTRUCTIONS,
                title="Use Imperative Voice",
                description="Use direct, imperative commands rather than questions",
                system_prompt_fragment="Use imperative voice with direct commands. Instead of asking 'Can you analyze this?', use 'Analyze this data and identify three key trends'.",
                applicability_conditions=["all_prompts"],
                priority=2,
                examples=["Good: 'Summarize the key points' vs Poor: 'Could you please summarize the key points?'"]
            ),
            BestPracticeRule(
                rule_id="output_format_001",
                category=BestPracticeCategory.OUTPUT_FORMAT,
                title="Specify Output Format",
                description="Clearly define the expected output format and structure",
                system_prompt_fragment="Clearly specify the expected output format including structure, length, style, and any formatting requirements. Use examples when the format is complex.",
                applicability_conditions=["structured_outputs", "formatted_responses"],
                priority=2,
                examples=["Output Format: JSON with keys 'summary', 'recommendations', 'confidence_score'"]
            ),
            BestPracticeRule(
                rule_id="role_definition_001",
                category=BestPracticeCategory.ROLE_DEFINITION,
                title="Define Clear Roles",
                description="Establish clear role and expertise for the AI",
                system_prompt_fragment="Define a clear role and area of expertise for the AI. Use phrases like 'You are an expert in X' or 'Act as a Y specialist' to establish context and expected knowledge level.",
                applicability_conditions=["expert_tasks", "domain_specific"],
                priority=3,
                examples=["You are a senior data scientist with expertise in machine learning and statistical analysis"]
            )
        ]
    
    def get_rules_by_category(self, category: BestPracticeCategory) -> List[BestPracticeRule]:
        """Get all rules for a specific category."""
        return [rule for rule in self.rules.values() if rule.category == category]
    
    def get_applicable_rules(self, conditions: List[str]) -> List[BestPracticeRule]:
        """Get rules applicable to given conditions, sorted by priority."""
        applicable_rules = []
        for rule in self.rules.values():
            if any(condition in rule.applicability_conditions or 
                   "all_prompts" in rule.applicability_conditions 
                   for condition in conditions):
                applicable_rules.append(rule)
        
        return sorted(applicable_rules, key=lambda r: r.priority)
    
    def get_system_prompt_fragments(self, conditions: List[str]) -> List[str]:
        """Get system prompt fragments for applicable rules."""
        applicable_rules = self.get_applicable_rules(conditions)
        return [rule.system_prompt_fragment for rule in applicable_rules]
    
    def add_rule(self, rule: BestPracticeRule) -> None:
        """Add a new best practice rule."""
        self.rules[rule.rule_id] = rule
    
    def update_rule(self, rule_id: str, updated_rule: BestPracticeRule) -> None:
        """Update an existing rule."""
        if rule_id in self.rules:
            self.rules[rule_id] = updated_rule
        else:
            raise ValueError(f"Rule {rule_id} not found")
    
    def remove_rule(self, rule_id: str) -> None:
        """Remove a rule."""
        if rule_id in self.rules:
            del self.rules[rule_id]
        else:
            raise ValueError(f"Rule {rule_id} not found")
    
    def save_to_file(self, filename: str = "best_practices.json") -> None:
        """Save rules to a JSON file."""
        filepath = os.path.join(self.storage_path, filename)
        data = {rule_id: rule.to_dict() for rule_id, rule in self.rules.items()}
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_from_file(self, filename: str = "best_practices.json") -> None:
        """Load rules from a JSON file."""
        filepath = os.path.join(self.storage_path, filename)
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            self.rules = {}
            for rule_id, rule_data in data.items():
                self.rules[rule_id] = BestPracticeRule.from_dict(rule_data)
    
    def get_rule_by_id(self, rule_id: str) -> Optional[BestPracticeRule]:
        """Get a specific rule by ID."""
        return self.rules.get(rule_id)
    
    def list_all_rules(self) -> List[BestPracticeRule]:
        """Get all rules sorted by priority."""
        return sorted(self.rules.values(), key=lambda r: r.priority)
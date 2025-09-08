"""Unit tests for best practices repository and system prompt management."""

import unittest
import tempfile
import shutil
import os
import json
from unittest.mock import Mock, patch

from best_practices.repository import (
    BestPracticesRepository, 
    BestPracticeRule, 
    BestPracticeCategory
)
from best_practices.system_prompts import (
    SystemPromptManager, 
    SystemPromptTemplate, 
    AgentType
)
from best_practices.reasoning_frameworks import (
    ReasoningFramework, 
    FrameworkType, 
    ReasoningStep,
    ReasoningFrameworkDefinition
)


class TestBestPracticesRepository(unittest.TestCase):
    """Test cases for BestPracticesRepository."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.repo = BestPracticesRepository(storage_path=self.temp_dir)
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_initialization_loads_default_rules(self):
        """Test that repository initializes with default rules."""
        self.assertGreater(len(self.repo.rules), 0)
        self.assertIn("clarity_001", self.repo.rules)
        self.assertIn("structure_001", self.repo.rules)
    
    def test_get_rules_by_category(self):
        """Test filtering rules by category."""
        clarity_rules = self.repo.get_rules_by_category(BestPracticeCategory.CLARITY)
        self.assertGreater(len(clarity_rules), 0)
        
        for rule in clarity_rules:
            self.assertEqual(rule.category, BestPracticeCategory.CLARITY)
    
    def test_get_applicable_rules(self):
        """Test getting rules applicable to specific conditions."""
        # Test with general conditions
        rules = self.repo.get_applicable_rules(["complex_prompts"])
        self.assertGreater(len(rules), 0)
        
        # Test with all_prompts condition
        all_rules = self.repo.get_applicable_rules(["any_condition"])
        self.assertGreater(len(all_rules), 0)
        
        # Verify rules are sorted by priority
        priorities = [rule.priority for rule in all_rules]
        self.assertEqual(priorities, sorted(priorities))
    
    def test_get_system_prompt_fragments(self):
        """Test getting system prompt fragments for conditions."""
        fragments = self.repo.get_system_prompt_fragments(["complex_prompts"])
        self.assertIsInstance(fragments, list)
        self.assertGreater(len(fragments), 0)
        
        for fragment in fragments:
            self.assertIsInstance(fragment, str)
            self.assertGreater(len(fragment), 0)
    
    def test_add_rule(self):
        """Test adding a new rule."""
        new_rule = BestPracticeRule(
            rule_id="test_001",
            category=BestPracticeCategory.CLARITY,
            title="Test Rule",
            description="A test rule",
            system_prompt_fragment="Test fragment",
            applicability_conditions=["test_condition"],
            priority=5,
            examples=["Test example"]
        )
        
        initial_count = len(self.repo.rules)
        self.repo.add_rule(new_rule)
        
        self.assertEqual(len(self.repo.rules), initial_count + 1)
        self.assertIn("test_001", self.repo.rules)
        self.assertEqual(self.repo.rules["test_001"], new_rule)
    
    def test_update_rule(self):
        """Test updating an existing rule."""
        # Get an existing rule
        rule_id = list(self.repo.rules.keys())[0]
        original_rule = self.repo.rules[rule_id]
        
        # Create updated rule
        updated_rule = BestPracticeRule(
            rule_id=rule_id,
            category=original_rule.category,
            title="Updated Title",
            description="Updated description",
            system_prompt_fragment="Updated fragment",
            applicability_conditions=original_rule.applicability_conditions,
            priority=original_rule.priority,
            examples=original_rule.examples
        )
        
        self.repo.update_rule(rule_id, updated_rule)
        
        self.assertEqual(self.repo.rules[rule_id].title, "Updated Title")
        self.assertEqual(self.repo.rules[rule_id].description, "Updated description")
    
    def test_update_nonexistent_rule_raises_error(self):
        """Test that updating a nonexistent rule raises an error."""
        fake_rule = BestPracticeRule(
            rule_id="fake_001",
            category=BestPracticeCategory.CLARITY,
            title="Fake Rule",
            description="A fake rule",
            system_prompt_fragment="Fake fragment",
            applicability_conditions=["fake_condition"],
            priority=5,
            examples=["Fake example"]
        )
        
        with self.assertRaises(ValueError):
            self.repo.update_rule("nonexistent_id", fake_rule)
    
    def test_remove_rule(self):
        """Test removing a rule."""
        # Add a test rule first
        test_rule = BestPracticeRule(
            rule_id="remove_test",
            category=BestPracticeCategory.CLARITY,
            title="Remove Test",
            description="Rule to be removed",
            system_prompt_fragment="Remove fragment",
            applicability_conditions=["remove_condition"],
            priority=5,
            examples=["Remove example"]
        )
        self.repo.add_rule(test_rule)
        
        initial_count = len(self.repo.rules)
        self.repo.remove_rule("remove_test")
        
        self.assertEqual(len(self.repo.rules), initial_count - 1)
        self.assertNotIn("remove_test", self.repo.rules)
    
    def test_remove_nonexistent_rule_raises_error(self):
        """Test that removing a nonexistent rule raises an error."""
        with self.assertRaises(ValueError):
            self.repo.remove_rule("nonexistent_id")
    
    def test_save_and_load_from_file(self):
        """Test saving and loading rules from file."""
        # Save current rules
        self.repo.save_to_file("test_rules.json")
        
        # Verify file exists
        filepath = os.path.join(self.temp_dir, "test_rules.json")
        self.assertTrue(os.path.exists(filepath))
        
        # Create new repository and load rules
        new_repo = BestPracticesRepository(storage_path=self.temp_dir)
        new_repo.rules = {}  # Clear default rules
        new_repo.load_from_file("test_rules.json")
        
        # Verify rules match
        self.assertEqual(len(new_repo.rules), len(self.repo.rules))
        for rule_id in self.repo.rules:
            self.assertIn(rule_id, new_repo.rules)
            self.assertEqual(new_repo.rules[rule_id].title, self.repo.rules[rule_id].title)
    
    def test_get_rule_by_id(self):
        """Test getting a specific rule by ID."""
        rule_id = list(self.repo.rules.keys())[0]
        rule = self.repo.get_rule_by_id(rule_id)
        
        self.assertIsNotNone(rule)
        self.assertEqual(rule.rule_id, rule_id)
        
        # Test nonexistent rule
        nonexistent_rule = self.repo.get_rule_by_id("nonexistent")
        self.assertIsNone(nonexistent_rule)
    
    def test_list_all_rules(self):
        """Test listing all rules sorted by priority."""
        all_rules = self.repo.list_all_rules()
        
        self.assertEqual(len(all_rules), len(self.repo.rules))
        
        # Verify sorting by priority
        priorities = [rule.priority for rule in all_rules]
        self.assertEqual(priorities, sorted(priorities))


class TestSystemPromptManager(unittest.TestCase):
    """Test cases for SystemPromptManager."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.repo = BestPracticesRepository(storage_path=self.temp_dir)
        self.manager = SystemPromptManager(self.repo)
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_initialization_creates_templates(self):
        """Test that manager initializes with templates for all agent types."""
        expected_agent_types = [
            AgentType.ANALYZER,
            AgentType.REFINER,
            AgentType.VALIDATOR,
            AgentType.EVALUATOR,
            AgentType.ORCHESTRATOR
        ]
        
        for agent_type in expected_agent_types:
            self.assertIn(agent_type, self.manager.templates)
            template = self.manager.templates[agent_type]
            self.assertIsInstance(template, SystemPromptTemplate)
            self.assertEqual(template.agent_type, agent_type)
    
    def test_generate_system_prompt_basic(self):
        """Test basic system prompt generation."""
        prompt = self.manager.generate_system_prompt(AgentType.ANALYZER)
        
        self.assertIsInstance(prompt, str)
        self.assertGreater(len(prompt), 0)
        self.assertIn("expert prompt analysis specialist", prompt.lower())
        self.assertIn("best practices", prompt.lower())
    
    def test_generate_system_prompt_with_context(self):
        """Test system prompt generation with context."""
        context = {
            "prompt_text": "Test prompt",
            "domain": "healthcare",
            "task_type": "analysis"
        }
        
        prompt = self.manager.generate_system_prompt(
            AgentType.ANALYZER, 
            context=context
        )
        
        self.assertIn("Test prompt", prompt)
        self.assertIn("healthcare", prompt)
        self.assertIn("analysis", prompt)
    
    def test_generate_system_prompt_with_domain(self):
        """Test system prompt generation with domain-specific conditions."""
        prompt = self.manager.generate_system_prompt(
            AgentType.REFINER,
            domain="finance"
        )
        
        self.assertIsInstance(prompt, str)
        self.assertGreater(len(prompt), 0)
    
    def test_generate_system_prompt_with_additional_conditions(self):
        """Test system prompt generation with additional conditions."""
        prompt = self.manager.generate_system_prompt(
            AgentType.VALIDATOR,
            additional_conditions=["complex_prompts", "format_specific"]
        )
        
        self.assertIsInstance(prompt, str)
        self.assertIn("Best Practices to Follow:", prompt)
    
    def test_generate_system_prompt_invalid_agent_type(self):
        """Test that invalid agent type raises error."""
        # Create a mock agent type that doesn't exist
        with patch('best_practices.system_prompts.AgentType') as mock_agent_type:
            mock_agent_type.INVALID = "invalid"
            
            with self.assertRaises(ValueError):
                self.manager.generate_system_prompt("invalid_type")
    
    def test_get_template(self):
        """Test getting a template for a specific agent type."""
        template = self.manager.get_template(AgentType.ANALYZER)
        
        self.assertIsNotNone(template)
        self.assertIsInstance(template, SystemPromptTemplate)
        self.assertEqual(template.agent_type, AgentType.ANALYZER)
    
    def test_update_template(self):
        """Test updating a template."""
        new_template = SystemPromptTemplate(
            agent_type=AgentType.ANALYZER,
            base_prompt="Updated base prompt",
            expertise_areas=["updated_area"],
            reasoning_framework="updated_framework",
            best_practice_categories=[BestPracticeCategory.CLARITY],
            context_variables=["updated_var"]
        )
        
        self.manager.update_template(AgentType.ANALYZER, new_template)
        
        updated_template = self.manager.get_template(AgentType.ANALYZER)
        self.assertEqual(updated_template.base_prompt, "Updated base prompt")
        self.assertEqual(updated_template.expertise_areas, ["updated_area"])
    
    def test_get_all_agent_types(self):
        """Test getting all available agent types."""
        agent_types = self.manager.get_all_agent_types()
        
        self.assertIsInstance(agent_types, list)
        self.assertGreater(len(agent_types), 0)
        
        expected_types = [
            AgentType.ANALYZER,
            AgentType.REFINER,
            AgentType.VALIDATOR,
            AgentType.EVALUATOR,
            AgentType.ORCHESTRATOR
        ]
        
        for expected_type in expected_types:
            self.assertIn(expected_type, agent_types)
    
    def test_generate_context_aware_prompt(self):
        """Test generating context-aware prompts with dynamic adaptations."""
        task_context = {
            "complexity_level": "high",
            "user_experience_level": "beginner",
            "domain": "machine_learning"
        }
        
        prompt = self.manager.generate_context_aware_prompt(
            AgentType.ANALYZER,
            task_context,
            domain_knowledge="machine_learning"
        )
        
        self.assertIn("High Complexity Handling:", prompt)
        self.assertIn("Beginner-Friendly Approach:", prompt)
        self.assertIn("machine_learning", prompt)


class TestReasoningFramework(unittest.TestCase):
    """Test cases for ReasoningFramework."""
    
    def setUp(self):
        """Set up test environment."""
        self.framework = ReasoningFramework()
    
    def test_initialization_creates_frameworks(self):
        """Test that framework initializes with default frameworks."""
        expected_frameworks = [
            FrameworkType.SYSTEMATIC_ANALYSIS,
            FrameworkType.ITERATIVE_REFINEMENT,
            FrameworkType.MULTI_CRITERIA_EVALUATION
        ]
        
        for framework_type in expected_frameworks:
            self.assertIn(framework_type, self.framework.frameworks)
    
    def test_get_framework(self):
        """Test getting a specific framework."""
        framework = self.framework.get_framework(FrameworkType.SYSTEMATIC_ANALYSIS)
        
        self.assertIsNotNone(framework)
        self.assertIsInstance(framework, ReasoningFrameworkDefinition)
        self.assertEqual(framework.framework_type, FrameworkType.SYSTEMATIC_ANALYSIS)
        self.assertGreater(len(framework.steps), 0)
    
    def test_get_nonexistent_framework(self):
        """Test getting a nonexistent framework returns None."""
        # Create a mock framework type that doesn't exist
        with patch('best_practices.reasoning_frameworks.FrameworkType') as mock_framework_type:
            mock_framework_type.NONEXISTENT = "nonexistent"
            
            framework = self.framework.get_framework("nonexistent")
            self.assertIsNone(framework)
    
    def test_get_framework_prompt(self):
        """Test generating a framework prompt."""
        prompt = self.framework.get_framework_prompt(FrameworkType.SYSTEMATIC_ANALYSIS)
        
        self.assertIsInstance(prompt, str)
        self.assertGreater(len(prompt), 0)
        self.assertIn("Systematic Analysis Framework", prompt)
        self.assertIn("Step 1:", prompt)
        self.assertIn("Meta-Instructions:", prompt)
        self.assertIn("Quality Checks:", prompt)
    
    def test_get_framework_prompt_invalid_type(self):
        """Test that invalid framework type raises error."""
        with patch('best_practices.reasoning_frameworks.FrameworkType') as mock_framework_type:
            mock_framework_type.INVALID = "invalid"
            
            with self.assertRaises(ValueError):
                self.framework.get_framework_prompt("invalid")
    
    def test_validate_framework_execution(self):
        """Test validating framework execution."""
        # Test with complete execution
        complete_execution = {
            "step_1_completed": True,
            "step_2_completed": True,
            "step_3_completed": True,
            "step_4_completed": True,
            "step_5_completed": True,
            "quality_check_1_passed": True,
            "quality_check_2_passed": True,
            "quality_check_3_passed": True,
            "quality_check_4_passed": True
        }
        
        result = self.framework.validate_framework_execution(
            FrameworkType.SYSTEMATIC_ANALYSIS,
            complete_execution
        )
        
        self.assertIn("overall_valid", result)
        
        # Test with incomplete execution
        incomplete_execution = {
            "step_1_completed": True,
            "step_2_completed": False
        }
        
        result = self.framework.validate_framework_execution(
            FrameworkType.SYSTEMATIC_ANALYSIS,
            incomplete_execution
        )
        
        self.assertIn("overall_valid", result)
        self.assertFalse(result["overall_valid"])
    
    def test_get_available_frameworks(self):
        """Test getting list of available frameworks."""
        frameworks = self.framework.get_available_frameworks()
        
        self.assertIsInstance(frameworks, list)
        self.assertGreater(len(frameworks), 0)
        
        expected_frameworks = [
            FrameworkType.SYSTEMATIC_ANALYSIS,
            FrameworkType.ITERATIVE_REFINEMENT,
            FrameworkType.MULTI_CRITERIA_EVALUATION
        ]
        
        for expected_framework in expected_frameworks:
            self.assertIn(expected_framework, frameworks)
    
    def test_add_custom_framework(self):
        """Test adding a custom framework."""
        custom_steps = [
            ReasoningStep(
                step_number=1,
                step_name="Custom Step",
                description="A custom step",
                guiding_questions=["Custom question?"],
                expected_outputs=["Custom output"],
                validation_criteria=["Custom criteria"]
            )
        ]
        
        custom_framework = ReasoningFrameworkDefinition(
            framework_type=FrameworkType.PROBLEM_SOLVING,
            name="Custom Framework",
            description="A custom framework for testing",
            steps=custom_steps,
            meta_instructions=["Custom instruction"],
            quality_checks=["Custom check"]
        )
        
        initial_count = len(self.framework.frameworks)
        self.framework.add_custom_framework(custom_framework)
        
        self.assertEqual(len(self.framework.frameworks), initial_count + 1)
        self.assertIn(FrameworkType.PROBLEM_SOLVING, self.framework.frameworks)
        
        retrieved_framework = self.framework.get_framework(FrameworkType.PROBLEM_SOLVING)
        self.assertEqual(retrieved_framework.name, "Custom Framework")


class TestBestPracticeRule(unittest.TestCase):
    """Test cases for BestPracticeRule data class."""
    
    def test_to_dict_conversion(self):
        """Test converting rule to dictionary."""
        rule = BestPracticeRule(
            rule_id="test_001",
            category=BestPracticeCategory.CLARITY,
            title="Test Rule",
            description="A test rule",
            system_prompt_fragment="Test fragment",
            applicability_conditions=["test_condition"],
            priority=5,
            examples=["Test example"]
        )
        
        rule_dict = rule.to_dict()
        
        self.assertEqual(rule_dict["rule_id"], "test_001")
        self.assertEqual(rule_dict["category"], "clarity")
        self.assertEqual(rule_dict["title"], "Test Rule")
        self.assertEqual(rule_dict["priority"], 5)
    
    def test_from_dict_conversion(self):
        """Test creating rule from dictionary."""
        rule_dict = {
            "rule_id": "test_001",
            "category": "clarity",
            "title": "Test Rule",
            "description": "A test rule",
            "system_prompt_fragment": "Test fragment",
            "applicability_conditions": ["test_condition"],
            "priority": 5,
            "examples": ["Test example"]
        }
        
        rule = BestPracticeRule.from_dict(rule_dict)
        
        self.assertEqual(rule.rule_id, "test_001")
        self.assertEqual(rule.category, BestPracticeCategory.CLARITY)
        self.assertEqual(rule.title, "Test Rule")
        self.assertEqual(rule.priority, 5)


if __name__ == '__main__':
    unittest.main()
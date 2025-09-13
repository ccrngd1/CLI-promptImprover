#!/usr/bin/env python3
"""
Command-line interface for the Bedrock Prompt Optimizer.

This module provides a comprehensive CLI with argument parsing, command structure,
progress indicators, and formatted output for agent results and orchestration decisions.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

# Rich for enhanced CLI output
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
    from rich.prompt import Prompt, Confirm
    from rich.syntax import Syntax
    from rich.tree import Tree
    from rich.text import Text
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("Warning: Rich library not available. Install with 'pip install rich' for enhanced output.")

from session import SessionManager, SessionConfig, UserFeedback
from bedrock.executor import BedrockExecutor, ModelConfig
from evaluation.evaluator import Evaluator
from storage.history import HistoryManager
from cli.config import ConfigManager
from config_loader import ConfigurationLoader


class CLIFormatter:
    """Handles formatted output for CLI with or without Rich."""
    
    def __init__(self):
        self.console = Console() if RICH_AVAILABLE else None
        self.use_rich = RICH_AVAILABLE
    
    def escape_rich_markup(self, text: str) -> str:
        """
        Escape Rich markup characters in text to prevent parsing errors.
        
        Args:
            text: Text that may contain Rich markup characters
            
        Returns:
            Text with Rich markup characters escaped
        """
        if not text:
            return text
        return text.replace('[', '\\[').replace(']', '\\]')
    
    def print(self, text: str, style: Optional[str] = None):
        """Print text with optional styling."""
        if self.use_rich and self.console:
            self.console.print(text, style=style)
        else:
            print(text)
    
    def print_panel(self, content: str, title: str, style: str = "blue"):
        """Print content in a panel."""
        if self.use_rich and self.console:
            # Escape Rich markup in content and title to prevent parsing errors
            escaped_content = self.escape_rich_markup(content)
            escaped_title = self.escape_rich_markup(title)
            self.console.print(Panel(escaped_content, title=escaped_title, border_style=style))
        else:
            print(f"\n=== {title} ===")
            print(content)
            print("=" * (len(title) + 8))
    
    def print_table(self, data: List[Dict[str, Any]], title: str):
        """Print data in a table format."""
        if not data:
            self.print(f"No data to display for {title}")
            return
        
        if self.use_rich and self.console:
            table = Table(title=title)
            
            # Add columns based on first row
            for key in data[0].keys():
                table.add_column(key.replace('_', ' ').title())
            
            # Add rows
            for row in data:
                table.add_row(*[str(value) for value in row.values()])
            
            self.console.print(table)
        else:
            print(f"\n{title}")
            print("-" * len(title))
            for item in data:
                for key, value in item.items():
                    print(f"{key.replace('_', ' ').title()}: {value}")
                print()
    
    def print_json(self, data: Dict[str, Any], title: str = "JSON Data"):
        """Print JSON data with syntax highlighting."""
        json_str = json.dumps(data, indent=2, default=str)
        
        if self.use_rich and self.console:
            syntax = Syntax(json_str, "json", theme="monokai", line_numbers=True)
            # Escape Rich markup in title to prevent parsing errors
            escaped_title = self.escape_rich_markup(title)
            self.console.print(Panel(syntax, title=escaped_title))
        else:
            print(f"\n{title}:")
            print(json_str)
    
    def print_orchestration_tree(self, orchestration_result: Dict[str, Any]):
        """Print orchestration results in a tree format."""
        if not self.use_rich or not self.console:
            self._print_orchestration_simple(orchestration_result)
            return
        
        tree = Tree("🎭 Orchestration Results")
        
        # Agent results with raw LLM feedback
        if 'agent_results' in orchestration_result:
            agents_branch = tree.add("🤖 Agent Results")
            for agent_name, result in orchestration_result['agent_results'].items():
                # Escape Rich markup in agent name to prevent parsing errors
                escaped_agent_name = self.escape_rich_markup(agent_name.title())
                agent_branch = agents_branch.add(f"[bold]{escaped_agent_name}[/bold]")
                agent_branch.add(f"✅ Success: {result.get('success', False)}")
                agent_branch.add(f"🎯 Confidence: {result.get('confidence_score', 0):.2f}")
                
                # Show raw LLM feedback if available
                analysis = result.get('analysis', {})
                if isinstance(analysis, dict):
                    # Check for LLM validation/analysis/refinement data
                    llm_data = (analysis.get('llm_validation') or 
                               analysis.get('llm_analysis') or 
                               analysis.get('llm_refinement'))
                    
                    if llm_data and isinstance(llm_data, dict):
                        raw_response = llm_data.get('raw_response', '')
                        if raw_response:
                            llm_branch = agent_branch.add("🧠 Raw LLM Feedback")
                            # Show full response without truncation
                            # Escape Rich markup in the response to prevent parsing errors
                            escaped_response = self.escape_rich_markup(raw_response)
                            llm_branch.add(f"[dim]{escaped_response}[/dim]")
                            
                            # Show model and token usage
                            if llm_data.get('model_used'):
                                llm_branch.add(f"🤖 Model: {llm_data['model_used']}")
                            if llm_data.get('tokens_used'):
                                llm_branch.add(f"🔢 Tokens: {llm_data['tokens_used']}")
                
                if result.get('suggestions'):
                    suggestions_branch = agent_branch.add("💡 Suggestions")
                    for suggestion in result['suggestions'][:3]:
                        # Escape Rich markup in suggestions to prevent parsing errors
                        escaped_suggestion = self.escape_rich_markup(suggestion)
                        suggestions_branch.add(f"• {escaped_suggestion}")
        
        # Orchestration decisions
        if 'orchestration_decisions' in orchestration_result:
            decisions_branch = tree.add("🎯 Orchestration Decisions")
            for decision in orchestration_result['orchestration_decisions']:
                # Escape Rich markup in decisions to prevent parsing errors
                escaped_decision = self.escape_rich_markup(decision)
                decisions_branch.add(f"• {escaped_decision}")
        
        # Conflict resolutions
        if 'conflict_resolutions' in orchestration_result:
            conflicts_branch = tree.add("⚖️ Conflict Resolutions")
            for conflict in orchestration_result['conflict_resolutions']:
                # Escape Rich markup in conflict descriptions to prevent parsing errors
                conflict_desc = conflict.get('description', str(conflict))
                escaped_conflict = self.escape_rich_markup(conflict_desc)
                conflicts_branch.add(f"• {escaped_conflict}")
        
        # Performance metrics
        metrics_branch = tree.add("📊 Performance")
        metrics_branch.add(f"⏱️ Processing Time: {orchestration_result.get('processing_time', 0):.2f}s")
        metrics_branch.add(f"🎯 Confidence: {orchestration_result.get('llm_orchestrator_confidence', 0):.2f}")
        
        self.console.print(tree)
    
    def _print_orchestration_simple(self, orchestration_result: Dict[str, Any]):
        """Print orchestration results in simple text format."""
        print("\n=== Orchestration Results ===")
        
        if 'agent_results' in orchestration_result:
            print("\nAgent Results:")
            for agent_name, result in orchestration_result['agent_results'].items():
                print(f"  {agent_name.title()}:")
                print(f"    Success: {result.get('success', False)}")
                print(f"    Confidence: {result.get('confidence_score', 0):.2f}")
                
                # Show raw LLM feedback if available
                analysis = result.get('analysis', {})
                if isinstance(analysis, dict):
                    # Check for LLM validation/analysis/refinement data
                    llm_data = (analysis.get('llm_validation') or 
                               analysis.get('llm_analysis') or 
                               analysis.get('llm_refinement'))
                    
                    if llm_data and isinstance(llm_data, dict):
                        raw_response = llm_data.get('raw_response', '')
                        if raw_response:
                            print("    Raw LLM Feedback:")
                            # Show full response without truncation
                            # Indent the response for readability
                            for line in raw_response.split('\n'):
                                print(f"      {line}")
                            
                            # Show model and token usage
                            if llm_data.get('model_used'):
                                print(f"      Model: {llm_data['model_used']}")
                            if llm_data.get('tokens_used'):
                                print(f"      Tokens: {llm_data['tokens_used']}")
                
                if result.get('suggestions'):
                    print("    Suggestions:")
                    for suggestion in result['suggestions'][:3]:
                        print(f"      - {suggestion}")
        
        if 'orchestration_decisions' in orchestration_result:
            print("\nOrchestration Decisions:")
            for decision in orchestration_result['orchestration_decisions']:
                print(f"  - {decision}")
        
        print(f"\nProcessing Time: {orchestration_result.get('processing_time', 0):.2f}s")
        print(f"Confidence: {orchestration_result.get('llm_orchestrator_confidence', 0):.2f}")


class PromptOptimizerCLI:
    """Main CLI application for the Bedrock Prompt Optimizer."""
    
    def __init__(self):
        self.formatter = CLIFormatter()
        self.config_manager = ConfigManager()
        self.config_loader = None  # Will be set when config path is provided
        self.session_manager = None
        self.bedrock_executor = None
        self.evaluator = None
        self.history_manager = None
        self._components_initialized = False
        
        # Don't initialize components in constructor - wait for config to be set
        # Components will be initialized on demand when needed
    
    def _initialize_components(self):
        """Initialize the core components based on configuration."""
        # Load configuration - use ConfigurationLoader if available for proper logging setup
        if self.config_loader:
            config = self.config_loader.get_config()
        else:
            config = self.config_manager.load_config()
        
        # Initialize Bedrock executor
        bedrock_config = config.get('bedrock', {})
        self.bedrock_executor = BedrockExecutor(
            region_name=bedrock_config.get('region', 'us-east-1')
        )
        
        # Initialize evaluator
        evaluator_config = config.get('evaluation', {})
        self.evaluator = Evaluator()
        
        # Initialize history manager
        storage_config = config.get('storage', {})
        self.history_manager = HistoryManager(
            storage_dir=storage_config.get('path', './prompt_history')
        )
        
        # Initialize session manager
        orchestration_config = config.get('orchestration', {})
        self.session_manager = SessionManager(
            bedrock_executor=self.bedrock_executor,
            evaluator=self.evaluator,
            history_manager=self.history_manager,
            orchestration_config=orchestration_config,
            full_config=config  # Pass full config to ensure optimization settings are available
        )
    
    def _apply_early_logging_suppression(self):
        """Apply early logging suppression based on config to prevent INFO messages during initialization."""
        try:
            import logging
            
            # Get the configured log level
            if self.config_loader:
                config = self.config_loader.get_config()
                log_level = config.get('logging', {}).get('level', 'INFO')
            else:
                log_level = 'INFO'
            
            # Convert to numeric level
            numeric_level = getattr(logging, log_level.upper(), logging.INFO)
            
            # If the configured level is ERROR, suppress INFO messages immediately
            if numeric_level >= logging.ERROR:
                # Set root logger to ERROR
                root_logger = logging.getLogger()
                root_logger.setLevel(logging.ERROR)
                
                # Suppress all application loggers
                app_loggers = [
                    'bedrock.executor', 'orchestration', 'agent_factory', 'agents',
                    'session', 'evaluation', 'storage', 'cli', 'llm_agents',
                    'llm_agents.llmanalyzeragent', 'llm_agents.llmrefineragent', 
                    'llm_agents.llmvalidatoragent'
                ]
                
                for logger_name in app_loggers:
                    logger = logging.getLogger(logger_name)
                    logger.setLevel(logging.ERROR)
                    # Prevent propagation to avoid duplicate suppression
                    logger.propagate = False
                    
        except Exception as e:
            # If early suppression fails, continue without it
            pass
    
    def _ensure_components_initialized(self):
        """Ensure components are initialized, initializing them if needed."""
        if not self._components_initialized:
            try:
                self._initialize_components()
                self._components_initialized = True
            except Exception as e:
                self.formatter.print(f"Error initializing components: {str(e)}", style="red")
                self.formatter.print("Make sure AWS credentials are configured and Bedrock is accessible.", style="yellow")
                sys.exit(1)
    
    def create_parser(self) -> argparse.ArgumentParser:
        """Create the argument parser with all commands and options."""
        parser = argparse.ArgumentParser(
            description="Bedrock Prompt Optimizer - Intelligent prompt optimization with multi-agent collaboration",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  %(prog)s optimize "Explain quantum computing" --context "Educational content for beginners"
  %(prog)s history --session-id abc123
  %(prog)s config --set bedrock.region us-west-2
  %(prog)s status --all
            """
        )
        
        # Global options
        parser.add_argument('--config', type=str, help='Path to configuration file')
        parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose output')
        parser.add_argument('--no-color', action='store_true', help='Disable colored output')
        
        # Subcommands
        subparsers = parser.add_subparsers(dest='command', help='Available commands')
        
        # Optimize command
        optimize_parser = subparsers.add_parser('optimize', help='Start a new optimization session')
        optimize_parser.add_argument('prompt', help='Initial prompt to optimize')
        optimize_parser.add_argument('--context', help='Context about the prompt\'s intended use')
        optimize_parser.add_argument('--domain', help='Domain or subject area')
        optimize_parser.add_argument('--audience', help='Target audience')
        optimize_parser.add_argument('--model', help='Bedrock model to use for execution')
        optimize_parser.add_argument('--max-iterations', type=int, default=10, help='Maximum optimization iterations')
        optimize_parser.add_argument('--auto-finalize', action='store_true', help='Auto-finalize on convergence')
        optimize_parser.add_argument('--interactive', action='store_true', help='Interactive mode with feedback prompts')
        
        # Continue command
        continue_parser = subparsers.add_parser('continue', help='Continue an existing optimization session')
        continue_parser.add_argument('session_id', help='Session ID to continue')
        continue_parser.add_argument('--feedback', help='Provide feedback for the last iteration')
        continue_parser.add_argument('--rating', type=int, choices=[1,2,3,4,5], help='Satisfaction rating (1-5)')
        
        # Feedback command
        feedback_parser = subparsers.add_parser('feedback', help='Provide feedback on a session iteration')
        feedback_parser.add_argument('session_id', help='Session ID to provide feedback for')
        feedback_parser.add_argument('--rating', type=int, choices=[1,2,3,4,5], required=True, help='Satisfaction rating (1-5)')
        feedback_parser.add_argument('--issues', nargs='*', help='List of specific issues')
        feedback_parser.add_argument('--improvements', help='Desired improvements description')
        feedback_parser.add_argument('--continue', dest='continue_opt', action='store_true', help='Continue optimization after feedback')
        feedback_parser.add_argument('--interactive', action='store_true', help='Interactive feedback collection')
        
        # History command
        history_parser = subparsers.add_parser('history', help='View optimization history')
        history_parser.add_argument('--session-id', help='Show history for specific session')
        history_parser.add_argument('--list', action='store_true', help='List all sessions')
        history_parser.add_argument('--export', help='Export session data to file')
        history_parser.add_argument('--format', choices=['json', 'text'], default='json', help='Export format')
        
        # Status command
        status_parser = subparsers.add_parser('status', help='Show session status')
        status_parser.add_argument('--session-id', help='Show status for specific session')
        status_parser.add_argument('--all', action='store_true', help='Show all active sessions')
        
        # Config command
        config_parser = subparsers.add_parser('config', help='Manage configuration')
        config_parser.add_argument('--show', action='store_true', help='Show current configuration')
        config_parser.add_argument('--set', help='Set configuration value (key=value)')
        config_parser.add_argument('--get', help='Get configuration value')
        config_parser.add_argument('--init', action='store_true', help='Initialize default configuration')
        
        # Models command
        models_parser = subparsers.add_parser('models', help='List available Bedrock models')
        models_parser.add_argument('--test', help='Test a specific model with a prompt')
        
        # Analyze command
        analyze_parser = subparsers.add_parser('analyze', help='Analyze session feedback patterns')
        analyze_parser.add_argument('session_id', help='Session ID to analyze')
        analyze_parser.add_argument('--export', help='Export analysis to file')
        analyze_parser.add_argument('--format', choices=['json', 'text'], default='text', help='Analysis format')
        
        return parser
    
    def run(self, args: Optional[List[str]] = None):
        """Run the CLI application."""
        parser = self.create_parser()
        parsed_args = parser.parse_args(args)
        
        # Handle global options
        if parsed_args.no_color:
            self.formatter.use_rich = False
        
        # Handle config path if provided - do this FIRST to set up logging properly
        if hasattr(parsed_args, 'config') and parsed_args.config:
            # Use ConfigurationLoader for proper logging setup when config file is provided
            self.config_loader = ConfigurationLoader(parsed_args.config)
            self.config_manager = ConfigManager(parsed_args.config)
            # Reset components initialization flag to force reinitialization with new config
            self._components_initialized = False
            
            # Apply logging configuration immediately to suppress early messages
            self._apply_early_logging_suppression()
        
        # Route to appropriate command handler
        if not parsed_args.command:
            parser.print_help()
            return
        
        try:
            command_method = getattr(self, f'cmd_{parsed_args.command}')
            command_method(parsed_args)
        except AttributeError:
            self.formatter.print(f"Unknown command: {parsed_args.command}", style="red")
            parser.print_help()
        except KeyboardInterrupt:
            self.formatter.print("\nOperation cancelled by user.", style="yellow")
        except Exception as e:
            self.formatter.print(f"Error executing command: {str(e)}", style="red")
            if parsed_args.verbose:
                import traceback
                traceback.print_exc()
    
    def cmd_optimize(self, args):
        """Handle the optimize command."""
        self._ensure_components_initialized()
        self.formatter.print("🚀 Starting prompt optimization session...", style="bold blue")
        
        # Build context
        context = {}
        if args.context:
            context['intended_use'] = args.context
        if args.domain:
            context['domain'] = args.domain
        if args.audience:
            context['target_audience'] = args.audience
        
        # Create session configuration
        model_config = None
        if args.model:
            model_config = ModelConfig(model_id=args.model)
        
        session_config = SessionConfig(
            max_iterations=args.max_iterations,
            auto_finalize_on_convergence=args.auto_finalize,
            collect_feedback_after_each_iteration=args.interactive,
            model_config=model_config
        )
        
        # Create session
        result = self.session_manager.create_session(
            initial_prompt=args.prompt,
            context=context if context else None,
            config=session_config
        )
        
        if not result.success:
            self.formatter.print(f"❌ Failed to create session: {result.message}", style="red")
            return
        
        session_id = result.session_state.session_id
        self.formatter.print(f"✅ Session created: {session_id}", style="green")
        
        # Display initial prompt
        self.formatter.print_panel(args.prompt, "Initial Prompt", "blue")
        
        # Run optimization iterations
        iteration_count = 0
        while iteration_count < args.max_iterations:
            self.formatter.print(f"\n🔄 Running iteration {iteration_count + 1}...", style="bold")
            
            # Show progress
            with self._create_progress() as progress:
                task = progress.add_task("Optimizing prompt...", total=100)
                
                # Run iteration
                iteration_result = self.session_manager.run_optimization_iteration(session_id)
                progress.update(task, completed=100)
            
            if not iteration_result.success:
                self.formatter.print(f"❌ Iteration failed: {iteration_result.message}", style="red")
                break
            
            # Display results
            self._display_iteration_results(iteration_result, session_id)
            
            iteration_count += 1
            
            # Check for convergence
            if iteration_result.session_state.convergence_detected:
                self.formatter.print("🎯 Convergence detected!", style="green")
                if args.auto_finalize:
                    self._finalize_session(session_id)
                    break
            
            # Interactive feedback
            if args.interactive and iteration_result.requires_user_input:
                if not self._collect_interactive_feedback(session_id):
                    break
        
        # Final status
        final_state = self.session_manager.get_session_state(session_id)
        if final_state:
            self._display_session_summary(final_state)
    
    def cmd_continue(self, args):
        """Handle the continue command."""
        self._ensure_components_initialized()
        session_id = args.session_id
        
        # Get session state
        session_state = self.session_manager.get_session_state(session_id)
        if not session_state:
            self.formatter.print(f"❌ Session {session_id} not found", style="red")
            return
        
        self.formatter.print(f"🔄 Continuing session {session_id}...", style="bold blue")
        
        # Prepare feedback if provided
        user_feedback = None
        if args.feedback or args.rating:
            user_feedback = UserFeedback(
                satisfaction_rating=args.rating or 3,
                specific_issues=[],
                desired_improvements=args.feedback or "",
                continue_optimization=True
            )
        
        # Run iteration
        with self._create_progress() as progress:
            task = progress.add_task("Running iteration...", total=100)
            result = self.session_manager.run_optimization_iteration(session_id, user_feedback)
            progress.update(task, completed=100)
        
        if result.success:
            self._display_iteration_results(result, session_id)
        else:
            self.formatter.print(f"❌ Iteration failed: {result.message}", style="red")
    
    def cmd_history(self, args):
        """Handle the history command."""
        if args.list:
            self._list_all_sessions()
        elif args.session_id:
            self._show_session_history(args.session_id, args.export, args.format)
        else:
            self.formatter.print("Please specify --list or --session-id", style="yellow")
    
    def cmd_status(self, args):
        """Handle the status command."""
        if args.all:
            self._show_all_session_status()
        elif args.session_id:
            self._show_session_status(args.session_id)
        else:
            self._show_all_session_status()
    
    def cmd_config(self, args):
        """Handle the config command."""
        if args.show:
            config = self.config_manager.load_config()
            self.formatter.print_json(config, "Current Configuration")
        elif args.set:
            self._set_config_value(args.set)
        elif args.get:
            self._get_config_value(args.get)
        elif args.init:
            self._initialize_config()
        else:
            self.formatter.print("Please specify an action: --show, --set, --get, or --init", style="yellow")
    
    def cmd_models(self, args):
        """Handle the models command."""
        self._ensure_components_initialized()
        if args.test:
            self._test_model(args.test)
        else:
            self._list_available_models()
    
    def cmd_feedback(self, args):
        """Handle the feedback command."""
        self._ensure_components_initialized()
        session_id = args.session_id
        
        # Get session state
        session_state = self.session_manager.get_session_state(session_id)
        if not session_state:
            self.formatter.print(f"❌ Session {session_id} not found", style="red")
            return
        
        self.formatter.print(f"💬 Collecting feedback for session {session_id}...", style="bold blue")
        
        if args.interactive:
            # Use interactive feedback collection
            self._collect_interactive_feedback(session_id)
        else:
            # Use command-line arguments
            issues = args.issues or []
            improvements = args.improvements or ""
            continue_opt = args.continue_opt
            
            result = self.session_manager.collect_user_feedback(
                session_id=session_id,
                satisfaction_rating=args.rating,
                specific_issues=issues,
                desired_improvements=improvements,
                continue_optimization=continue_opt
            )
            
            if result.success:
                self.formatter.print("✅ Feedback collected successfully", style="green")
                
                # Display feedback analysis if available
                if result.suggested_actions:
                    self.formatter.print("\n📋 Suggested Actions:", style="bold")
                    for action in result.suggested_actions:
                        self.formatter.print(f"  • {action}")
            else:
                self.formatter.print(f"❌ Failed to collect feedback: {result.message}", style="red")
    
    def cmd_analyze(self, args):
        """Handle the analyze command."""
        self._ensure_components_initialized()
        session_id = args.session_id
        
        self.formatter.print(f"📊 Analyzing feedback patterns for session {session_id}...", style="bold blue")
        
        # Get feedback analysis
        analysis_result = self.session_manager.analyze_feedback_patterns(session_id)
        
        if not analysis_result.get('success'):
            self.formatter.print(f"❌ Analysis failed: {analysis_result.get('message')}", style="red")
            return
        
        # Display analysis results
        analysis = analysis_result['analysis']
        suggestions = analysis_result['suggestions']
        patterns = analysis_result['patterns']
        
        # Summary panel
        summary_text = f"Feedback Count: {analysis_result['feedback_count']}\n"
        summary_text += f"Average Rating: {analysis['average_rating']:.1f}/5\n"
        summary_text += f"Rating Trend: {analysis['rating_trend'].title()}\n"
        summary_text += f"Latest Rating: {analysis['latest_rating']}/5"
        
        self.formatter.print_panel(summary_text, "Feedback Analysis Summary", "blue")
        
        # Common issues
        if analysis.get('common_issues'):
            issues_text = "\n".join([f"• {issue[0]} ({issue[1]} times)" for issue in analysis['common_issues']])
            self.formatter.print_panel(issues_text, "Most Common Issues", "yellow")
        
        # Patterns
        if patterns:
            patterns_text = "\n".join([f"• {pattern}" for pattern in patterns])
            self.formatter.print_panel(patterns_text, "Identified Patterns", "cyan")
        
        # Suggestions
        if suggestions:
            suggestions_text = "\n".join([f"• {suggestion}" for suggestion in suggestions])
            self.formatter.print_panel(suggestions_text, "Optimization Suggestions", "green")
        
        # Export if requested
        if args.export:
            try:
                import json
                export_data = {
                    'session_id': session_id,
                    'analysis_timestamp': datetime.now().isoformat(),
                    'feedback_analysis': analysis_result
                }
                
                if args.format == 'json':
                    with open(args.export, 'w') as f:
                        json.dump(export_data, f, indent=2)
                else:
                    with open(args.export, 'w') as f:
                        f.write(f"Feedback Analysis for Session {session_id}\n")
                        f.write("=" * 50 + "\n\n")
                        f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                        f.write(f"Feedback Count: {analysis_result['feedback_count']}\n")
                        f.write(f"Average Rating: {analysis['average_rating']:.1f}/5\n")
                        f.write(f"Rating Trend: {analysis['rating_trend'].title()}\n\n")
                        
                        if analysis.get('common_issues'):
                            f.write("Common Issues:\n")
                            for issue, count in analysis['common_issues']:
                                f.write(f"  - {issue} ({count} times)\n")
                            f.write("\n")
                        
                        if patterns:
                            f.write("Identified Patterns:\n")
                            for pattern in patterns:
                                f.write(f"  - {pattern}\n")
                            f.write("\n")
                        
                        if suggestions:
                            f.write("Optimization Suggestions:\n")
                            for suggestion in suggestions:
                                f.write(f"  - {suggestion}\n")
                
                self.formatter.print(f"✅ Analysis exported to {args.export}", style="green")
                
            except Exception as e:
                self.formatter.print(f"❌ Export failed: {str(e)}", style="red")
    
    def _display_iteration_results(self, result, session_id=None):
        """Display the results of an optimization iteration."""
        if not result.iteration_result:
            return
        
        orchestration_result = result.iteration_result.to_dict()
        
        # Display initial prompt if session_id is provided
        if session_id:
            session_state = self.session_manager.get_session_state(session_id)
            if session_state and session_state.initial_prompt:
                self.formatter.print_panel(
                    session_state.initial_prompt,
                    "Initial Prompt",
                    "blue"
                )
        
        # Display orchestration tree
        self.formatter.print_orchestration_tree(orchestration_result)
        
        # Display optimized prompt
        if orchestration_result.get('orchestrated_prompt'):
            self.formatter.print_panel(
                orchestration_result['orchestrated_prompt'],
                "Optimized Prompt",
                "green"
            )
        
        # Display synthesis reasoning
        if orchestration_result.get('synthesis_reasoning'):
            self.formatter.print_panel(
                orchestration_result['synthesis_reasoning'],
                "Orchestration Reasoning",
                "cyan"
            )
        
        # Display evaluation results
        if orchestration_result.get('evaluation_result'):
            eval_result = orchestration_result['evaluation_result']
            eval_text = f"Overall Score: {eval_result.get('overall_score', 0):.2f}\n"
            eval_text += f"Relevance: {eval_result.get('relevance_score', 0):.2f}\n"
            eval_text += f"Clarity: {eval_result.get('clarity_score', 0):.2f}\n"
            eval_text += f"Completeness: {eval_result.get('completeness_score', 0):.2f}"
            
            self.formatter.print_panel(eval_text, "Evaluation Scores", "yellow")
    
    def _collect_interactive_feedback(self, session_id: str) -> bool:
        """Collect interactive feedback from the user."""
        if not RICH_AVAILABLE:
            # Simple text-based feedback
            print("\nPlease provide feedback on the current iteration:")
            rating = input("Satisfaction rating (1-5): ")
            feedback = input("Any specific feedback or improvements needed: ")
            
            # Collect specific issues
            issues = []
            print("Identify specific issues (press Enter when done):")
            while True:
                issue = input("Issue (or Enter to finish): ").strip()
                if not issue:
                    break
                issues.append(issue)
            
            continue_opt = input("Continue optimization? (y/n): ").lower().startswith('y')
            
            try:
                rating = int(rating) if rating.isdigit() else 3
            except ValueError:
                rating = 3
        else:
            # Rich-based interactive feedback
            self.formatter.print("\n💬 Please provide feedback on this iteration:", style="bold")
            
            rating = Prompt.ask(
                "Satisfaction rating",
                choices=["1", "2", "3", "4", "5"],
                default="3"
            )
            rating = int(rating)
            
            feedback = Prompt.ask("Any specific feedback or improvements needed", default="")
            
            # Collect specific issues
            issues = []
            self.formatter.print("Identify specific issues (press Enter when done):")
            while True:
                issue = Prompt.ask("Issue (or Enter to finish)", default="")
                if not issue:
                    break
                issues.append(issue)
            
            continue_opt = Confirm.ask("Continue optimization?", default=True)
        
        # Collect feedback
        result = self.session_manager.collect_user_feedback(
            session_id=session_id,
            satisfaction_rating=rating,
            specific_issues=issues,
            desired_improvements=feedback,
            continue_optimization=continue_opt
        )
        
        if result.success:
            self.formatter.print("✅ Feedback collected", style="green")
            return continue_opt
        else:
            self.formatter.print(f"❌ Failed to collect feedback: {result.message}", style="red")
            return False
    
    def _create_progress(self):
        """Create a progress indicator."""
        if RICH_AVAILABLE:
            return Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TimeElapsedColumn(),
                console=self.formatter.console
            )
        else:
            # Simple progress placeholder
            class SimpleProgress:
                def __enter__(self):
                    return self
                def __exit__(self, *args):
                    pass
                def add_task(self, description, total=100):
                    print(f"Starting: {description}")
                    return 0
                def update(self, task_id, completed=100):
                    print("Completed.")
            
            return SimpleProgress()
    
    def _finalize_session(self, session_id: str):
        """Finalize a session."""
        result = self.session_manager.finalize_session(session_id)
        if result.success:
            self.formatter.print("✅ Session finalized successfully", style="green")
        else:
            self.formatter.print(f"❌ Failed to finalize session: {result.message}", style="red")
    
    def _display_session_summary(self, session_state):
        """Display a summary of the session."""
        summary_data = {
            'Session ID': session_state.session_id,
            'Status': session_state.status,
            'Iterations': session_state.current_iteration,
            'Created': session_state.created_at.strftime('%Y-%m-%d %H:%M:%S'),
            'Last Updated': session_state.last_updated.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        if session_state.convergence_detected:
            summary_data['Convergence'] = session_state.convergence_reason or 'Detected'
        
        self.formatter.print_table([summary_data], "Session Summary")
    
    def _list_all_sessions(self):
        """List all sessions."""
        # This would need to be implemented in the history manager
        self.formatter.print("📋 Listing all sessions...", style="bold")
        # Placeholder implementation
        sessions = []  # self.history_manager.list_all_sessions()
        if sessions:
            self.formatter.print_table(sessions, "All Sessions")
        else:
            self.formatter.print("No sessions found.", style="yellow")
    
    def _show_session_history(self, session_id: str, export_path: Optional[str], format_type: str):
        """Show history for a specific session."""
        self.formatter.print(f"📖 Loading history for session {session_id}...", style="bold")
        
        try:
            history = self.history_manager.load_session_history(session_id)
            if not history:
                self.formatter.print("No history found for this session.", style="yellow")
                return
            
            # Display history
            history_data = []
            for iteration in history:
                feedback_summary = 'No'
                if iteration.user_feedback:
                    rating = iteration.user_feedback.satisfaction_rating
                    feedback_summary = f"Rating: {rating}/5"
                    if iteration.user_feedback.specific_issues:
                        feedback_summary += f" ({len(iteration.user_feedback.specific_issues)} issues)"
                
                history_data.append({
                    'Version': iteration.version,
                    'Timestamp': iteration.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                    'Evaluation Score': getattr(iteration.evaluation_scores, 'overall_score', 'N/A'),
                    'User Feedback': feedback_summary
                })
            
            self.formatter.print_table(history_data, f"History for Session {session_id}")
            
            # Display detailed feedback if available
            feedback_iterations = [it for it in history if it.user_feedback]
            if feedback_iterations:
                self.formatter.print("\n💬 Detailed Feedback History:", style="bold")
                for iteration in feedback_iterations[-3:]:  # Show last 3 with feedback
                    feedback = iteration.user_feedback
                    self.formatter.print_panel(
                        f"Rating: {feedback.satisfaction_rating}/5\n"
                        f"Issues: {', '.join(feedback.specific_issues) if feedback.specific_issues else 'None'}\n"
                        f"Improvements: {feedback.desired_improvements or 'None'}\n"
                        f"Continue: {'Yes' if feedback.continue_optimization else 'No'}",
                        f"Iteration {iteration.version} Feedback",
                        "cyan"
                    )
            
            # Export if requested
            if export_path:
                result = self.session_manager.export_session_with_reasoning(
                    session_id, export_path, include_orchestration_details=True
                )
                if result.success:
                    self.formatter.print(f"✅ Session exported to {export_path}", style="green")
                else:
                    self.formatter.print(f"❌ Export failed: {result.message}", style="red")
        
        except Exception as e:
            self.formatter.print(f"❌ Error loading history: {str(e)}", style="red")
    
    def _show_session_status(self, session_id: str):
        """Show status for a specific session."""
        session_state = self.session_manager.get_session_state(session_id)
        if session_state:
            self._display_session_summary(session_state)
        else:
            self.formatter.print(f"Session {session_id} not found.", style="yellow")
    
    def _show_all_session_status(self):
        """Show status for all active sessions."""
        active_sessions = self.session_manager.list_active_sessions()
        if active_sessions:
            session_data = []
            for session in active_sessions:
                session_data.append({
                    'Session ID': session.session_id[:8] + '...',
                    'Status': session.status,
                    'Iterations': session.current_iteration,
                    'Last Updated': session.last_updated.strftime('%H:%M:%S')
                })
            self.formatter.print_table(session_data, "Active Sessions")
        else:
            self.formatter.print("No active sessions.", style="yellow")
    
    def _set_config_value(self, key_value: str):
        """Set a configuration value."""
        try:
            key, value = key_value.split('=', 1)
            self.config_manager.set_config_value(key, value)
            self.formatter.print(f"✅ Configuration updated: {key} = {value}", style="green")
        except ValueError:
            self.formatter.print("Invalid format. Use key=value", style="red")
        except Exception as e:
            self.formatter.print(f"❌ Failed to set configuration: {str(e)}", style="red")
    
    def _get_config_value(self, key: str):
        """Get a configuration value."""
        try:
            value = self.config_manager.get_config_value(key)
            self.formatter.print(f"{key}: {value}")
        except Exception as e:
            self.formatter.print(f"❌ Failed to get configuration: {str(e)}", style="red")
    
    def _initialize_config(self):
        """Initialize default configuration."""
        try:
            self.config_manager.create_default_config()
            self.formatter.print("✅ Default configuration created", style="green")
        except Exception as e:
            self.formatter.print(f"❌ Failed to initialize configuration: {str(e)}", style="red")
    
    def _list_available_models(self):
        """List available Bedrock models."""
        try:
            models = self.bedrock_executor.get_available_models()
            if models:
                model_data = [{'Model ID': model} for model in models]
                self.formatter.print_table(model_data, "Available Bedrock Models")
            else:
                self.formatter.print("No models available or unable to retrieve model list.", style="yellow")
        except Exception as e:
            self.formatter.print(f"❌ Error retrieving models: {str(e)}", style="red")
    
    def _test_model(self, test_prompt: str):
        """Test a model with a prompt."""
        self.formatter.print(f"🧪 Testing model with prompt: {test_prompt}", style="bold")
        
        # Use default model configuration
        model_config = ModelConfig()
        
        try:
            result = self.bedrock_executor.execute_prompt(test_prompt, model_config)
            if result.success:
                self.formatter.print_panel(result.response_text, "Model Response", "green")
                self.formatter.print(f"Execution time: {result.execution_time:.2f}s")
                self.formatter.print(f"Token usage: {result.token_usage}")
            else:
                self.formatter.print(f"❌ Model test failed: {result.error_message}", style="red")
        except Exception as e:
            self.formatter.print(f"❌ Error testing model: {str(e)}", style="red")


def main():
    """Main entry point for the CLI."""
    # Check for config argument early and set up logging before creating CLI
    import sys
    config_path = None
    for i, arg in enumerate(sys.argv):
        if arg == '--config' and i + 1 < len(sys.argv):
            config_path = sys.argv[i + 1]
            break
    
    # Set up logging early if config is provided
    if config_path:
        try:
            from config_loader import ConfigurationLoader
            ConfigurationLoader(config_path)  # This sets up logging
        except Exception:
            pass  # Fall back to default logging
    
    cli = PromptOptimizerCLI()
    cli.run()


if __name__ == '__main__':
    main()
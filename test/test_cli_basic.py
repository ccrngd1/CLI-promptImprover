#!/usr/bin/env python3
"""
Basic test for CLI functionality.
Tests the CLI components without requiring AWS credentials.
"""

import sys
import os
from pathlib import Path
from unittest.mock import Mock, patch

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_cli_import():
    """Test that CLI modules can be imported."""
    try:
        from cli.main import PromptOptimizerCLI, CLIFormatter
        from cli.config import ConfigManager
        print("✅ CLI modules imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Failed to import CLI modules: {e}")
        return False

def test_config_manager():
    """Test configuration manager basic functionality."""
    try:
        from cli.config import ConfigManager
        
        # Test with temporary config path
        config_manager = ConfigManager('./test_config.yaml')
        
        # Test default config generation
        default_config = config_manager._get_default_config()
        assert 'bedrock' in default_config
        assert 'orchestration' in default_config
        assert 'agents' in default_config
        
        print("✅ ConfigManager basic functionality works")
        return True
    except Exception as e:
        print(f"❌ ConfigManager test failed: {e}")
        return False

def test_cli_formatter():
    """Test CLI formatter functionality."""
    try:
        from cli.main import CLIFormatter
        
        formatter = CLIFormatter()
        
        # Test basic print (should not raise exceptions)
        formatter.print("Test message")
        formatter.print("Test styled message", style="green")
        
        # Test panel printing
        formatter.print_panel("Test content", "Test Title")
        
        # Test table printing
        test_data = [
            {'Name': 'Test', 'Value': 123},
            {'Name': 'Another', 'Value': 456}
        ]
        formatter.print_table(test_data, "Test Table")
        
        # Test JSON printing
        test_json = {'key': 'value', 'number': 42}
        formatter.print_json(test_json, "Test JSON")
        
        print("✅ CLIFormatter functionality works")
        return True
    except Exception as e:
        print(f"❌ CLIFormatter test failed: {e}")
        return False

def test_cli_parser():
    """Test CLI argument parser."""
    try:
        from cli.main import PromptOptimizerCLI
        
        # Mock the components that require AWS
        with patch('cli.main.BedrockExecutor'), \
             patch('cli.main.Evaluator'), \
             patch('cli.main.HistoryManager'), \
             patch('cli.main.SessionManager'):
            
            cli = PromptOptimizerCLI()
            parser = cli.create_parser()
            
            # Test help doesn't crash
            try:
                parser.parse_args(['--help'])
            except SystemExit:
                pass  # Expected for --help
            
            # Test basic argument parsing
            args = parser.parse_args(['optimize', 'test prompt'])
            assert args.command == 'optimize'
            assert args.prompt == 'test prompt'
            
            # Test config command parsing
            args = parser.parse_args(['config', '--show'])
            assert args.command == 'config'
            assert args.show == True
            
            print("✅ CLI argument parser works")
            return True
    except Exception as e:
        print(f"❌ CLI parser test failed: {e}")
        return False

def test_executable_script():
    """Test that the executable script can be run."""
    try:
        import subprocess
        
        # Test help command (should not require AWS)
        result = subprocess.run([
            sys.executable, './bedrock-optimizer', '--help'
        ], capture_output=True, text=True, timeout=10)
        
        # Should exit with code 0 for help
        if result.returncode == 0:
            print("✅ Executable script works")
            return True
        else:
            print(f"❌ Executable script failed with code {result.returncode}")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Executable script timed out")
        return False
    except Exception as e:
        print(f"❌ Executable script test failed: {e}")
        return False

def main():
    """Run all basic CLI tests."""
    print("🧪 Running basic CLI tests...\n")
    
    tests = [
        ("Import Test", test_cli_import),
        ("Config Manager Test", test_config_manager),
        ("CLI Formatter Test", test_cli_formatter),
        ("CLI Parser Test", test_cli_parser),
        ("Executable Script Test", test_executable_script),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        try:
            if test_func():
                passed += 1
            else:
                print(f"❌ {test_name} failed")
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All basic CLI tests passed!")
        return True
    else:
        print("⚠️  Some tests failed. Check the output above.")
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
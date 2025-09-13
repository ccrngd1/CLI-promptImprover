#!/usr/bin/env python3
"""
CLI tool for testing configuration loading and runtime changes.

This script provides a simple command-line interface to test
the configuration system functionality.
"""

import argparse
import json
from config_loader import (
    get_config_loader, load_config, get_optimization_config,
    is_llm_only_mode, update_config_runtime
)


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Test configuration loading and runtime changes"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Show config command
    show_parser = subparsers.add_parser('show', help='Show current configuration')
    show_parser.add_argument('--section', help='Show specific section only')
    
    # Update config command
    update_parser = subparsers.add_parser('update', help='Update configuration')
    update_parser.add_argument('key', help='Configuration key (e.g., optimization.llm_only_mode)')
    update_parser.add_argument('value', help='New value')
    update_parser.add_argument('--type', choices=['str', 'int', 'float', 'bool'], default='str',
                              help='Value type')
    
    # Toggle LLM-only mode
    toggle_parser = subparsers.add_parser('toggle-llm-only', help='Toggle LLM-only mode')
    
    # Reload config command
    reload_parser = subparsers.add_parser('reload', help='Reload configuration from file')
    
    # Validate config command
    validate_parser = subparsers.add_parser('validate', help='Validate current configuration')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == 'show':
            show_config(args.section)
        elif args.command == 'update':
            update_config(args.key, args.value, args.type)
        elif args.command == 'toggle-llm-only':
            toggle_llm_only_mode()
        elif args.command == 'reload':
            reload_config()
        elif args.command == 'validate':
            validate_config()
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return 1
    
    return 0


def show_config(section=None):
    """Show current configuration."""
    config = load_config()
    
    if section:
        if section in config:
            print(f"📋 Configuration section '{section}':")
            print(json.dumps(config[section], indent=2))
        else:
            print(f"❌ Section '{section}' not found")
            print(f"Available sections: {', '.join(config.keys())}")
    else:
        print("📋 Current configuration:")
        print(json.dumps(config, indent=2))
    
    # Show optimization status
    opt_config = get_optimization_config()
    print(f"\n🎯 Optimization status:")
    print(f"   - LLM-only mode: {is_llm_only_mode()}")
    print(f"   - Fallback enabled: {opt_config.get('fallback_to_heuristic', True)}")


def update_config(key, value, value_type):
    """Update configuration value."""
    # Convert value to appropriate type
    if value_type == 'bool':
        value = value.lower() in ('true', '1', 'yes', 'on')
    elif value_type == 'int':
        value = int(value)
    elif value_type == 'float':
        value = float(value)
    # str is default, no conversion needed
    
    print(f"🔧 Updating {key} = {value} ({value_type})")
    
    result = update_config_runtime({key: value})
    
    if result['success']:
        print(f"✅ Configuration updated successfully")
        print(f"   - Applied changes: {len(result['applied_changes'])}")
        for change in result['applied_changes']:
            print(f"     • {change['key']}: {change['old_value']} → {change['new_value']}")
    else:
        print(f"❌ Configuration update failed")
        for error in result.get('failed_changes', []):
            print(f"   - {error}")


def toggle_llm_only_mode():
    """Toggle LLM-only mode."""
    current_mode = is_llm_only_mode()
    new_mode = not current_mode
    
    print(f"🔄 Toggling LLM-only mode: {current_mode} → {new_mode}")
    
    result = update_config_runtime({'optimization.llm_only_mode': new_mode})
    
    if result['success']:
        print(f"✅ LLM-only mode {'enabled' if new_mode else 'disabled'}")
    else:
        print(f"❌ Failed to toggle LLM-only mode")


def reload_config():
    """Reload configuration from file."""
    print("🔄 Reloading configuration from file...")
    
    config_loader = get_config_loader()
    result = config_loader.reload_from_file()
    
    if result['success']:
        if result['reloaded']:
            print(f"✅ Configuration reloaded successfully")
            print(f"   - Changes detected: {len(result['changes_detected'])}")
            for change in result['changes_detected']:
                print(f"     • {change['type']}: {change['key']}")
        else:
            print("ℹ️  Configuration file unchanged, no reload needed")
    else:
        print(f"❌ Configuration reload failed")
        for error in result.get('errors', []):
            print(f"   - {error}")


def validate_config():
    """Validate current configuration."""
    print("✅ Validating configuration...")
    
    config_loader = get_config_loader()
    validation = config_loader.validate_current_config()
    
    if validation['valid']:
        print("✅ Configuration is valid")
    else:
        print("❌ Configuration validation failed")
        for error in validation['errors']:
            print(f"   - Error: {error}")
    
    if validation['warnings']:
        print("⚠️  Configuration warnings:")
        for warning in validation['warnings']:
            print(f"   - Warning: {warning}")
    
    if validation['info']:
        print("ℹ️  Configuration info:")
        for info in validation['info']:
            print(f"   - {info}")


if __name__ == '__main__':
    exit(main())
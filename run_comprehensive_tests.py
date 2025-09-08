#!/usr/bin/env python3
"""
Comprehensive test runner for the Bedrock Prompt Optimizer.

Runs all test suites including integration tests, performance tests,
error handling tests, and orchestration edge case tests.
"""

import sys
import subprocess
import time
import json
from pathlib import Path
from typing import Dict, List, Any
import argparse

from logging_config import setup_logging, performance_logger


class TestRunner:
    """Comprehensive test runner with reporting and analysis."""
    
    def __init__(self, log_dir: str = "test_logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Set up logging
        self.loggers = setup_logging(
            log_level='INFO',
            log_dir=str(self.log_dir),
            enable_structured_logging=True,
            enable_performance_logging=True
        )
        
        self.test_results = {}
        self.start_time = None
        self.end_time = None
    
    def run_test_suite(self, test_file: str, test_args: List[str] = None) -> Dict[str, Any]:
        """Run a specific test suite and capture results."""
        test_args = test_args or []
        
        print(f"\n{'='*60}")
        print(f"Running test suite: {test_file}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        # Build pytest command
        cmd = [
            sys.executable, "-m", "pytest",
            test_file,
            "-v",
            "--tb=short",
            "--json-report",
            f"--json-report-file={self.log_dir}/{test_file.replace('.py', '_report.json')}"
        ] + test_args
        
        try:
            # Run the test
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout per test suite
            )
            
            end_time = time.time()
            duration = end_time - start_time
            
            # Parse results
            test_result = {
                'test_file': test_file,
                'duration': duration,
                'return_code': result.returncode,
                'success': result.returncode == 0,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'command': ' '.join(cmd)
            }
            
            # Try to load JSON report if available
            json_report_path = self.log_dir / f"{test_file.replace('.py', '_report.json')}"
            if json_report_path.exists():
                try:
                    with open(json_report_path, 'r') as f:
                        json_report = json.load(f)
                        test_result['json_report'] = json_report
                        test_result['test_count'] = json_report.get('summary', {}).get('total', 0)
                        test_result['passed'] = json_report.get('summary', {}).get('passed', 0)
                        test_result['failed'] = json_report.get('summary', {}).get('failed', 0)
                        test_result['errors'] = json_report.get('summary', {}).get('error', 0)
                except Exception as e:
                    print(f"Warning: Could not parse JSON report: {e}")
            
            # Log performance
            performance_logger.log_metric(
                f'test_suite_duration',
                duration,
                test_file=test_file,
                success=test_result['success']
            )
            
            # Print summary
            if test_result['success']:
                print(f"✅ {test_file} PASSED ({duration:.2f}s)")
                if 'test_count' in test_result:
                    print(f"   Tests: {test_result['test_count']}, "
                          f"Passed: {test_result['passed']}, "
                          f"Failed: {test_result['failed']}")
            else:
                print(f"❌ {test_file} FAILED ({duration:.2f}s)")
                if 'test_count' in test_result:
                    print(f"   Tests: {test_result['test_count']}, "
                          f"Passed: {test_result['passed']}, "
                          f"Failed: {test_result['failed']}, "
                          f"Errors: {test_result['errors']}")
                
                # Print error details
                if result.stderr:
                    print(f"   Error output: {result.stderr[:200]}...")
            
            return test_result
            
        except subprocess.TimeoutExpired:
            end_time = time.time()
            duration = end_time - start_time
            
            print(f"⏰ {test_file} TIMEOUT ({duration:.2f}s)")
            
            return {
                'test_file': test_file,
                'duration': duration,
                'return_code': -1,
                'success': False,
                'timeout': True,
                'command': ' '.join(cmd)
            }
            
        except Exception as e:
            end_time = time.time()
            duration = end_time - start_time
            
            print(f"💥 {test_file} ERROR ({duration:.2f}s): {str(e)}")
            
            return {
                'test_file': test_file,
                'duration': duration,
                'return_code': -2,
                'success': False,
                'error': str(e),
                'command': ' '.join(cmd)
            }
    
    def run_all_tests(self, include_performance: bool = True, 
                     include_integration: bool = True,
                     include_edge_cases: bool = True) -> Dict[str, Any]:
        """Run all test suites."""
        self.start_time = time.time()
        
        print("🚀 Starting comprehensive test suite")
        print(f"Log directory: {self.log_dir}")
        
        # Define test suites
        test_suites = []
        
        # Basic unit tests
        basic_tests = [
            "test_models.py",
            "test_agents.py", 
            "test_bedrock_executor.py",
            "test_evaluator.py",
            "test_best_practices.py",
            "test_history_manager.py"
        ]
        
        for test in basic_tests:
            if Path(test).exists():
                test_suites.append((test, []))
        
        # Integration tests
        if include_integration:
            integration_tests = [
                "test_integration.py",
                "test_integration_comprehensive.py",
                "test_llm_orchestration.py",
                "test_session_manager.py",
                "test_feedback_integration.py",
                "test_evaluation_integration.py"
            ]
            
            for test in integration_tests:
                if Path(test).exists():
                    test_suites.append((test, []))
        
        # Performance tests
        if include_performance:
            if Path("test_performance.py").exists():
                test_suites.append(("test_performance.py", ["-m", "performance"]))
        
        # Edge case tests
        if include_edge_cases:
            if Path("test_orchestration_edge_cases.py").exists():
                test_suites.append(("test_orchestration_edge_cases.py", []))
        
        # CLI tests
        if Path("test_cli_basic.py").exists():
            test_suites.append(("test_cli_basic.py", []))
        
        # Run all test suites
        for test_file, test_args in test_suites:
            result = self.run_test_suite(test_file, test_args)
            self.test_results[test_file] = result
        
        self.end_time = time.time()
        
        # Generate summary report
        return self.generate_summary_report()
    
    def generate_summary_report(self) -> Dict[str, Any]:
        """Generate comprehensive test summary report."""
        total_duration = self.end_time - self.start_time if self.end_time and self.start_time else 0
        
        # Calculate statistics
        total_suites = len(self.test_results)
        passed_suites = sum(1 for result in self.test_results.values() if result['success'])
        failed_suites = total_suites - passed_suites
        
        total_tests = sum(result.get('test_count', 0) for result in self.test_results.values())
        total_passed = sum(result.get('passed', 0) for result in self.test_results.values())
        total_failed = sum(result.get('failed', 0) for result in self.test_results.values())
        total_errors = sum(result.get('errors', 0) for result in self.test_results.values())
        
        # Create summary
        summary = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_duration': total_duration,
            'test_suites': {
                'total': total_suites,
                'passed': passed_suites,
                'failed': failed_suites,
                'success_rate': passed_suites / total_suites if total_suites > 0 else 0
            },
            'individual_tests': {
                'total': total_tests,
                'passed': total_passed,
                'failed': total_failed,
                'errors': total_errors,
                'success_rate': total_passed / total_tests if total_tests > 0 else 0
            },
            'suite_results': self.test_results,
            'performance_metrics': {
                'average_suite_duration': total_duration / total_suites if total_suites > 0 else 0,
                'slowest_suite': max(
                    self.test_results.values(), 
                    key=lambda x: x['duration']
                )['test_file'] if self.test_results else None,
                'fastest_suite': min(
                    self.test_results.values(), 
                    key=lambda x: x['duration']
                )['test_file'] if self.test_results else None
            }
        }
        
        # Save summary report
        summary_path = self.log_dir / "test_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Print summary
        self.print_summary_report(summary)
        
        return summary
    
    def print_summary_report(self, summary: Dict[str, Any]) -> None:
        """Print formatted summary report."""
        print(f"\n{'='*80}")
        print("COMPREHENSIVE TEST SUMMARY")
        print(f"{'='*80}")
        
        print(f"Timestamp: {summary['timestamp']}")
        print(f"Total Duration: {summary['total_duration']:.2f} seconds")
        
        print(f"\nTest Suites:")
        print(f"  Total: {summary['test_suites']['total']}")
        print(f"  Passed: {summary['test_suites']['passed']} ✅")
        print(f"  Failed: {summary['test_suites']['failed']} ❌")
        print(f"  Success Rate: {summary['test_suites']['success_rate']:.1%}")
        
        print(f"\nIndividual Tests:")
        print(f"  Total: {summary['individual_tests']['total']}")
        print(f"  Passed: {summary['individual_tests']['passed']} ✅")
        print(f"  Failed: {summary['individual_tests']['failed']} ❌")
        print(f"  Errors: {summary['individual_tests']['errors']} 💥")
        print(f"  Success Rate: {summary['individual_tests']['success_rate']:.1%}")
        
        print(f"\nPerformance:")
        print(f"  Average Suite Duration: {summary['performance_metrics']['average_suite_duration']:.2f}s")
        if summary['performance_metrics']['slowest_suite']:
            print(f"  Slowest Suite: {summary['performance_metrics']['slowest_suite']}")
        if summary['performance_metrics']['fastest_suite']:
            print(f"  Fastest Suite: {summary['performance_metrics']['fastest_suite']}")
        
        # Print failed suites
        failed_suites = [
            name for name, result in summary['suite_results'].items() 
            if not result['success']
        ]
        
        if failed_suites:
            print(f"\nFailed Test Suites:")
            for suite in failed_suites:
                result = summary['suite_results'][suite]
                print(f"  ❌ {suite} ({result['duration']:.2f}s)")
                if 'timeout' in result:
                    print(f"     Reason: Timeout")
                elif 'error' in result:
                    print(f"     Reason: {result['error']}")
                elif result.get('failed', 0) > 0:
                    print(f"     Failed Tests: {result['failed']}")
        
        print(f"\n{'='*80}")
        
        # Overall result
        if summary['test_suites']['success_rate'] >= 0.9:
            print("🎉 OVERALL RESULT: EXCELLENT")
        elif summary['test_suites']['success_rate'] >= 0.8:
            print("✅ OVERALL RESULT: GOOD")
        elif summary['test_suites']['success_rate'] >= 0.7:
            print("⚠️  OVERALL RESULT: NEEDS ATTENTION")
        else:
            print("❌ OVERALL RESULT: CRITICAL ISSUES")
        
        print(f"{'='*80}")


def main():
    """Main entry point for test runner."""
    parser = argparse.ArgumentParser(description="Run comprehensive test suite")
    parser.add_argument("--no-performance", action="store_true", 
                       help="Skip performance tests")
    parser.add_argument("--no-integration", action="store_true",
                       help="Skip integration tests")
    parser.add_argument("--no-edge-cases", action="store_true",
                       help="Skip edge case tests")
    parser.add_argument("--log-dir", default="test_logs",
                       help="Directory for test logs")
    
    args = parser.parse_args()
    
    # Create test runner
    runner = TestRunner(log_dir=args.log_dir)
    
    # Run tests
    summary = runner.run_all_tests(
        include_performance=not args.no_performance,
        include_integration=not args.no_integration,
        include_edge_cases=not args.no_edge_cases
    )
    
    # Exit with appropriate code
    if summary['test_suites']['success_rate'] >= 0.8:
        sys.exit(0)  # Success
    else:
        sys.exit(1)  # Failure


if __name__ == "__main__":
    main()
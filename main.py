"""
Agent-R Implementation: Enhancing LLM Agents with Self-Reflection Capabilities

This implementation is based on the Agent-R paper, which proposes an iterative 
self-training framework to enable language model agents to reflect and recover 
from errors.

Usage:
    python main.py --mode analyze  # Analyze logs in data_by_challenges
    python main.py --mode demo     # Run a demonstration of Agent-R
"""

import argparse
import os
import json
import sys
from pprint import pprint

from failure_types import FailureType, FailureDetector
from recovery_decorator import RecoveryStrategy, recover_from_failure, with_failure_recovery
from log_analyzer import AgentLogAnalyzer
from agent_r import AgentR, BaseAgent
from comprehensive_example import (
    load_avatar_logs,
    agent_r_demonstration,
    analyze_agent_behaviors,
    compare_success_vs_failure,
    visualize_comparison
)


def analyze_logs(data_dir="data_by_challenges"):
    """Analyze agent logs in the data directory."""
    print(f"Analyzing logs in {data_dir}...")
    
    if not os.path.exists(data_dir):
        print(f"Error: Directory {data_dir} not found.")
        return
    
    analyzer = AgentLogAnalyzer(data_dir)
    
    print("Scanning all logs...")
    overview = analyzer.scan_logs()
    
    print("\nLog Analysis Overview:")
    pprint(overview)
    
    print("\nFinding common failure patterns...")
    patterns = analyzer.find_common_failure_patterns()
    
    print("\nCommon Failure Patterns:")
    pprint(patterns)
    
    # Analyze a few challenge directories
    challenge_dirs = []
    for item in os.listdir(data_dir):
        if os.path.isdir(os.path.join(data_dir, item)) and not item.startswith('.'):
            challenge_dirs.append(item)
    
    if challenge_dirs:
        # Analyze the first few challenge directories
        for challenge_dir in challenge_dirs[:3]:
            print(f"\nAnalyzing challenge directory: {challenge_dir}")
            analysis = analyzer.analyze_challenge_directory(challenge_dir)
            print(f"Total logs: {analysis['total_logs']}")
            print(f"Success rate: {analysis['success_rate']:.2%}")
    else:
        print("\nNo challenge directories found.")


def demo():
    """Run a demonstration of Agent-R."""
    print("Running Agent-R demonstration...")
    
    agent_r_demonstration()
    
    # Check if the avatar logs are available
    success_log_path = "data_by_challenges/glacier_avatar/o1-preview-2024-09-12_success_subtask_avatar_run_log.json"
    fail_log_path = "data_by_challenges/glacier_avatar/o1-preview-2024-09-12_fail_avatar_run_log.json"
    
    if os.path.exists(success_log_path) and os.path.exists(fail_log_path):
        # Analyze the avatar logs
        load_avatar_logs()
    else:
        print(f"\nAvatar log files not found. Please ensure they are in the current directory.")
        print(f"Expected files: {success_log_path} and {fail_log_path}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Agent-R: LLM Agent Enhancement with Self-Reflection")
    parser.add_argument("--mode", choices=["analyze", "demo"], default="demo",
                        help="Mode of operation: 'analyze' to analyze logs, 'demo' to run a demonstration")
    parser.add_argument("--data-dir", default="data_by_challenges",
                        help="Directory containing agent logs (for 'analyze' mode)")
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    if args.mode == "analyze":
        analyze_logs(args.data_dir)
    elif args.mode == "demo":
        demo()
    else:
        print(f"Unknown mode: {args.mode}")
        sys.exit(1)


if __name__ == "__main__":
    main()
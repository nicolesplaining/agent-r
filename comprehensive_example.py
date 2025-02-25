import os
import json
import glob
from pprint import pprint
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter, defaultdict
import pandas as pd
from failure_types import FailureDetector, FailureType
from agent_r import AgentR, BaseAgent
from recovery_decorator import recover_from_failure

def load_avatar_logs():
    """
    Load and analyze the avatar challenge logs from the samples provided.
    This function demonstrates how to analyze real logs using our Agent-R implementation.
    """
    # Path patterns to our sample logs
    success_log_path = "data_by_challenges/glacier_avatar/o1-preview-2024-09-12_success_subtask_avatar_run_log.json"
    fail_log_path = "data_by_challenges/glacier_avatar/o1-preview-2024-09-12_fail_avatar_run_log.json"
    
    # Load the logs
    success_log = load_json_file(success_log_path)
    fail_log = load_json_file(fail_log_path)
    
    if not success_log or not fail_log:
        print("Could not load sample logs.")
        return
    
    print("\n===== Avatar Challenge Log Analysis =====")
    
    # Initialize our detector
    detector = FailureDetector()
    
    # Analyze the logs
    print("\nAnalyzing success log...")
    success_failures = detector.analyze_agent_log(success_log)
    
    print("\nAnalyzing failure log...")
    fail_failures = detector.analyze_agent_log(fail_log)
    
    # Print the results
    print(f"\nSuccess log had {len(success_failures)} detected failures")
    print(f"Failure log had {len(fail_failures)} detected failures")
    
    # Analyze failure types
    print("\nFailure types in success log:")
    success_failure_types = Counter([f['failure_type'].name for f in success_failures])
    pprint(dict(success_failure_types))
    
    print("\nFailure types in failure log:")
    fail_failure_types = Counter([f['failure_type'].name for f in fail_failures])
    pprint(dict(fail_failure_types))
    
    # Detailed analysis of the failure log
    print("\nDetailed analysis of the first few failures in the failure log:")
    for i, failure in enumerate(fail_failures[:3]):
        print(f"\nFailure {i+1}:")
        print(f"Type: {failure['failure_type'].name}")
        print(f"Command: {failure.get('command', 'N/A')}")
        print(f"Output: {truncate_text(failure.get('output', 'N/A'), 100)}")
        print(f"Error: {truncate_text(failure.get('error', 'N/A'), 100)}")
    
    # Demonstrate how Agent-R would handle these failures
    print("\nDemonstrating how Agent-R would handle one of these failures:")
    if fail_failures:
        failure = fail_failures[0]
        agent_r = AgentR(BaseAgent())
        
        context = {
            'command': failure.get('command', ''),
            'output': failure.get('output', ''),
            'error': failure.get('error', ''),
            'command_history': [],
            'retries': 0
        }
        
        recovery_result = recover_from_failure(failure['failure_type'], agent_r, context)
        print("\nRecovery action:")
        pprint(recovery_result)


def analyze_agent_behaviors(log_data):
    """
    Analyze agent behaviors from log data.
    """
    # Extract iterations from the log data
    iterations = extract_iterations(log_data)
    
    # Initialize lists for our analysis
    commands = []
    outputs = []
    errors = []
    
    # Extract commands, outputs, and errors
    for iteration in iterations:
        if 'command' in iteration and 'command_str' in iteration['command']:
            commands.append(iteration['command']['command_str'])
        else:
            commands.append(None)
        
        outputs.append(iteration.get('stdout', ''))
        errors.append(iteration.get('stderr', ''))
    
    # Check for loops in commands
    loops = detect_loops(commands)
    
    # Check for invalid actions
    invalid_actions = []
    for i, error in enumerate(errors):
        if error and commands[i]:
            invalid_actions.append((i, commands[i], error))
    
    return {
        'command_count': len(commands),
        'unique_commands': len(set([c for c in commands if c])),
        'loops': loops,
        'invalid_actions': invalid_actions
    }


def extract_iterations(log_data):
    """Extract iterations from the log data."""
    iterations = []
    
    # Check if we're dealing with subtask completions
    if 'subtask_completions' in log_data:
        for subtask_key, subtask_data in log_data['subtask_completions'].items():
            if 'iterations' in subtask_data:
                for iteration in subtask_data['iterations']:
                    if 'execution_output' in iteration:
                        iterations.append(iteration['execution_output'])
    
    return iterations


def detect_loops(commands, threshold=3):
    """
    Detect loops in a sequence of commands.
    
    Args:
        commands: List of commands
        threshold: Number of consecutive repetitions to consider a loop
        
    Returns:
        List of detected loops (indices and commands)
    """
    loops = []
    
    # Check for exact repetitions
    for i in range(len(commands) - threshold + 1):
        if commands[i:i+threshold] and len(set(commands[i:i+threshold])) == 1:
            loops.append((i, commands[i], threshold))
    
    return loops


def compare_success_vs_failure(success_log, fail_log):
    """
    Compare successful and failed agent logs to identify differences.
    """
    # Analyze both logs
    success_analysis = analyze_agent_behaviors(success_log)
    fail_analysis = analyze_agent_behaviors(fail_log)
    
    # Create a comparison
    comparison = {
        'command_count': {
            'success': success_analysis['command_count'],
            'failure': fail_analysis['command_count']
        },
        'unique_commands': {
            'success': success_analysis['unique_commands'],
            'failure': fail_analysis['unique_commands']
        },
        'loops': {
            'success': len(success_analysis['loops']),
            'failure': len(fail_analysis['loops'])
        },
        'invalid_actions': {
            'success': len(success_analysis['invalid_actions']),
            'failure': len(fail_analysis['invalid_actions'])
        }
    }
    
    return comparison


def truncate_text(text, max_length=100):
    """Truncate text to a maximum length."""
    if not text:
        return "N/A"
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."


def visualize_comparison(comparison):
    """
    Visualize the comparison between successful and failed logs.
    """
    # Extract data for visualization
    metrics = list(comparison.keys())
    success_values = [comparison[metric]['success'] for metric in metrics]
    failure_values = [comparison[metric]['failure'] for metric in metrics]
    
    # Create a figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Set width of bars
    bar_width = 0.35
    
    # Set position of bar on X axis
    br1 = np.arange(len(metrics))
    br2 = [x + bar_width for x in br1]
    
    # Make the plot
    ax.bar(br1, success_values, width=bar_width, label='Success')
    ax.bar(br2, failure_values, width=bar_width, label='Failure')
    
    # Add labels and title
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Count')
    ax.set_title('Comparison of Successful vs Failed Agent Logs')
    ax.set_xticks([r + bar_width/2 for r in range(len(metrics))])
    ax.set_xticklabels(metrics)
    
    # Add legend
    ax.legend()
    
    # Save the figure
    plt.savefig('agent_comparison.png')
    plt.close()
    
    print("Visualization saved to 'agent_comparison.png'")


def load_json_file(file_path):
    """Load a JSON file."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return None


def agent_r_demonstration():
    """
    Demonstrate how Agent-R would be used in practice.
    """
    print("\n===== Agent-R Demonstration =====")
    
    # Create a base agent
    base_agent = BaseAgent()
    
    # Create an Agent-R enhanced agent
    agent_r = AgentR(base_agent)
    
    # Simulate a task with common failures
    task = {
        'name': 'avatar_challenge',
        'description': 'Obtain a flag in the CTF challenge'
    }
    
    # Simulate command execution with failures similar to those in the logs
    commands = [
        "cat chall.py",                       # Success
        "python3 chall.py",                   # Timeout
        "echo \"gctf\" | python3 chall.py",   # Invalid action
        "nc avatar 1337",                     # Success
        "echo '(*+*)' | nc avatar 1337",      # Success
        "echo '(*+*)' | nc avatar 1337",      # Loop - repeating command
        "echo '(*+*)' | nc avatar 1337",      # Loop - repeating command
        "echo '{}=={}' | nc avatar 1337",     # Success
        "python solve.py"                     # Success
    ]
    
    # Simulate failures
    failures = {
        1: (FailureType.TIMEOUT, "Command timed out after 120 seconds"),
        2: (FailureType.INVALID_ACTION, "NameError: name 'gctf' is not defined"),
        5: (FailureType.LOOP_DETECTED, ""),
        6: (FailureType.LOOP_DETECTED, "")
    }
    
    print("\nSimulating task execution with failure recovery:")
    
    # Execute each command
    for i, command in enumerate(commands):
        print(f"\nExecuting command: {command}")
        
        # Mock execution result
        result = {
            'command': command,
            'stdout': f'Output from {command}',
            'stderr': ''
        }
        
        # Simulate a failure if this command has one
        if i in failures:
            failure_type, error_msg = failures[i]
            result['stderr'] = error_msg
            
            # Create a context for recovery
            context = {
                'command': command,
                'output': result.get('stdout', ''),
                'error': error_msg,
                'command_history': agent_r.command_history,
                'retries': 0
            }
            
            # Handle the failure
            print(f"Detected failure: {failure_type.name}")
            recovery_result = recover_from_failure(failure_type, agent_r, context)
            print(f"Recovery action: {recovery_result['action']}")
            print(f"Recovery message: {recovery_result['message']}")
            
            # Apply the recovery
            if 'suggestion' in recovery_result:
                print(f"Suggestion: {recovery_result['suggestion']}")
        
        # Add to history
        agent_r.command_history.append(command)
    
    # Demonstrate generating a revision trajectory
    bad_trajectory = {
        'task': task,
        'commands': commands[:7],  # Up to and including the loop
        'result': {'success': False}
    }
    
    good_trajectory = {
        'task': task,
        'commands': [commands[0], commands[3], commands[7], commands[8]],  # Successful commands
        'result': {'success': True}
    }
    
    print("\nGenerating revision trajectory:")
    revision = agent_r.generate_revision_trajectory(bad_trajectory, good_trajectory)
    
    print("\nRevision trajectory:")
    print(f"Task: {revision['task']['name']}")
    print(f"Bad commands: {revision['bad_commands']}")
    print(f"Reflection: {revision['reflection']}")
    print(f"Good commands: {revision['good_commands']}")
    print(f"Transition point: {revision['transition_point']}")
    
    # Simulate training
    print("\nSimulating training with revision trajectories:")
    agent_r.revision_trajectories = [revision]  # Add the revision trajectory
    stats = agent_r.train_with_revision_trajectories(iterations=2)
    
    print("\nTraining stats:")
    pprint(stats)
    
    print("\n===== Demonstration Complete =====")


if __name__ == "__main__":
    # Analyze the avatar logs
    load_avatar_logs()
    
    # Demonstrate Agent-R
    agent_r_demonstration()
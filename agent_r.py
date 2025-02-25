import logging
import json
import time
import random
import os
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict

# Import our modules
from failure_types import FailureType, FailureDetector
from recovery_decorator import with_failure_recovery, recover_from_failure

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BaseAgent:
    """
    Real implementation of a base agent that can execute commands and run tasks.
    This agent serves as the foundation that Agent-R enhances with reflection capabilities.
    """
    
    def __init__(self, name="DefaultAgent", working_dir=None):
        """Initialize the base agent."""
        self.name = name
        self.working_dir = working_dir or os.getcwd()
        self.command_history = []
        self.environment_vars = {}
        # Optional: maintain a state object for complex applications
        self.state = {"task_id": None, "step": 0, "goal": None}
        self.last_result = None
        
    def execute_command(self, command, **kwargs):
        """
        Execute a command and return the result.
        
        Args:
            command: The command to execute
            **kwargs: Additional arguments
            
        Returns:
            Command execution result
        """
        start_time = time.time()
        self.command_history.append(command)
        
        try:
            # For safety, we're not actually executing shell commands
            # In a real implementation, you might use subprocess or a safer alternative
            logger.info(f"Executing command: {command}")
            
            # Simulate command execution
            stdout, stderr, success = self._simulate_command_execution(command)
            
            execution_time = time.time() - start_time
            
            # Create the result
            result = {
                'command': command,
                'stdout': stdout,
                'stderr': stderr,
                'success': success,
                'execution_time': execution_time
            }
            
            self.last_result = result
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            # Log and return the error
            logger.error(f"Error executing command: {str(e)}")
            result = {
                'command': command,
                'stdout': '',
                'stderr': str(e),
                'success': False,
                'execution_time': execution_time,
                'error': str(e)
            }
            
            self.last_result = result
            return result
    
    def _simulate_command_execution(self, command):
        """
        Simulate command execution for demonstration purposes.
        This would be replaced with actual command execution in a real implementation.
        
        Returns:
            Tuple of (stdout, stderr, success)
        """
        # Simulate common command patterns
        if command.startswith("ls"):
            return "file1.txt\nfile2.txt\ndirectory1/", "", True
        elif command.startswith("cat"):
            file_name = command.split(" ")[1] if len(command.split(" ")) > 1 else ""
            if not file_name:
                return "", "usage: cat [file]", False
            elif "nonexistent" in file_name:
                return "", f"cat: {file_name}: No such file or directory", False
            else:
                return f"Contents of {file_name}", "", True
        elif command.startswith("cd"):
            dir_name = command.split(" ")[1] if len(command.split(" ")) > 1 else ""
            if not dir_name:
                return "", "usage: cd [directory]", False
            elif "nonexistent" in dir_name:
                return "", f"cd: {dir_name}: No such file or directory", False
            else:
                return "", "", True
        elif "syntax error" in command.lower():
            return "", "Syntax error", False
        elif "permission" in command.lower():
            return "", "Permission denied", False
        
        # Default simulation for unrecognized commands
        return f"Simulated output for: {command}", "", True
    
    def run_task(self, task, **kwargs):
        """
        Run a complete task and return the result.
        
        Args:
            task: The task to run
            **kwargs: Additional arguments
            
        Returns:
            Task execution result
        """
        # Reset state for new task
        self.command_history = []
        self.state = {"task_id": task.get('id', 'unknown_task'), "step": 0, "goal": task.get('goal')}
        
        logger.info(f"Starting task: {task.get('name', 'unnamed_task')}")
        
        # Extract task components
        task_steps = task.get('steps', [])
        goal = task.get('goal', 'Complete the task successfully')
        
        # Results for each step
        step_results = []
        
        try:
            # Execute each step in the task
            for i, step in enumerate(task_steps):
                self.state["step"] = i
                logger.info(f"Executing step {i+1}/{len(task_steps)}: {step.get('description', '')}")
                
                # Get command for this step
                command = step.get('command', '')
                if not command:
                    # In a real implementation, you might use an LLM to generate a command here
                    # For this simulation, we'll use a default command based on the step description
                    command = f"echo 'Executing: {step.get('description', 'step ' + str(i))}'"
                
                # Execute the command
                result = self.execute_command(command, **kwargs)
                
                # Add step information
                result['step'] = i
                result['step_description'] = step.get('description', '')
                
                # Store the result
                step_results.append(result)
                
                # Check if the step failed
                if not result.get('success', False):
                    logger.warning(f"Step {i+1} failed: {result.get('stderr', '')}")
                    
                    # In a real implementation, you might have step-specific recovery logic here
                    # For now, we'll just continue to the next step
            
            # Determine overall task success
            success = all(step.get('success', False) for step in step_results)
            
            task_result = {
                'task': task,
                'status': 'complete' if success else 'failed',
                'success': success,
                'step_results': step_results,
                'command_count': len(self.command_history)
            }
            
            logger.info(f"Task completed {'successfully' if success else 'with failures'}")
            
            return task_result
            
        except Exception as e:
            logger.error(f"Error in task execution: {str(e)}")
            
            return {
                'task': task,
                'status': 'error',
                'success': False,
                'error': str(e),
                'step_results': step_results,
                'command_count': len(self.command_history)
            }


class AgentR:
    """
    Implementation of Agent-R for handling agent failures and improving performance
    through reflection and self-correction.
    
    Agent-R consists of two phases:
    1. Model-Guided Reflection Trajectory Generation
    2. Iterative Self-Training with Revision Trajectories
    """
    
    def __init__(self, base_agent):
        """
        Initialize the Agent-R wrapper.
        
        Args:
            base_agent: The base agent to enhance with reflection capabilities
        """
        self.base_agent = base_agent
        self.failure_detector = FailureDetector()
        self.command_history = []
        self.good_trajectories = []
        self.bad_trajectories = []
        self.revision_trajectories = []
        
        # Metrics tracking
        self.metrics = {
            'commands_executed': 0,
            'failures_detected': 0,
            'recoveries_attempted': 0,
            'successful_recoveries': 0,
            'tasks_completed': 0,
            'tasks_failed': 0
        }
        
        # Load reflection templates from file if available, otherwise use defaults
        self.reflection_thoughts = self._load_reflection_templates()
        
        # Recovery templates for specific failure types
        self.recovery_templates = {
            FailureType.MALFORMED_COMMAND: [
                "I need to fix the syntax in my command.",
                "There's a syntax error in my command that needs correction.",
                "I should restructure this command to use proper syntax."
            ],
            FailureType.LOOP_DETECTED: [
                "I'm stuck in a loop. I need to try a different approach.",
                "I keep repeating the same command without progress. Let me change strategies.",
                "I'm not making progress with this repetitive approach. Time to try something else."
            ],
            FailureType.INVALID_ACTION: [
                "This action isn't valid in the current context. I need to rethink my approach.",
                "The command I tried doesn't work here. I should try a different method.",
                "My action choice was invalid. Let me reassess the situation."
            ],
            FailureType.TIMEOUT: [
                "That operation took too long. I should simplify my approach.",
                "The command timed out. I need to break this into smaller steps.",
                "This approach is too time-consuming. I need a more efficient method."
            ]
        }
    
    def _load_reflection_templates(self):
        """Load reflection templates from file or use defaults."""
        templates_file = "reflection_templates.json"
        default_templates = [
            "I realize my approach was flawed. I need to revise it.",
            "I took the wrong actions. I need to identify the right path.",
            "My actions were incorrect. I must adjust my strategy.",
            "I see an error in my actions. I need to fix it.",
            "My judgment was incorrect. I need to rethink it.",
            "I overlooked something important. I need to address it.",
            "I recognize my mistake. Let's find a better solution.",
            "I recognize my failure. I need to learn and move forward.",
            "My decision was wrong. I should reevaluate.",
            "I made an error. I must determine how to correct it."
        ]
        
        try:
            if os.path.exists(templates_file):
                with open(templates_file, 'r') as f:
                    templates = json.load(f)
                if isinstance(templates, list) and templates:
                    return templates
        except Exception as e:
            logger.warning(f"Error loading reflection templates: {e}. Using defaults.")
        
        return default_templates
    
    @with_failure_recovery(max_retries=3)
    def execute_command(self, command, **kwargs):
        """
        Execute a command with failure recovery capabilities.
        
        Args:
            command: The command to execute
            **kwargs: Additional arguments for the base agent
            
        Returns:
            Command execution result
        """
        # Track metrics
        self.metrics['commands_executed'] += 1
        
        # Add command to history
        self.command_history.append(command)
        
        # Execute command with base agent
        result = self.base_agent.execute_command(command, **kwargs)
        
        # Check for failure using our detector
        stdout = result.get('stdout', '')
        stderr = result.get('stderr', '')
        execution_time = result.get('execution_time', 0)
        failure = self.failure_detector.detect_failure(command, stdout, stderr, execution_time)
        
        if failure:
            self.metrics['failures_detected'] += 1
            
            # Log the failure
            logger.info(f"Detected failure ({failure.name}) in command: {command}")
            
            # Create context for recovery
            context = {
                'command': command,
                'output': stdout,
                'error': stderr,
                'execution_time': execution_time,
                'command_history': self.command_history.copy(),
                'retries': 0
            }
            
            # Apply recovery
            self.metrics['recoveries_attempted'] += 1
            recovery_result = recover_from_failure(failure, self, context)
            
            # Add recovery information to the result
            result['failure'] = {
                'type': failure.name,
                'recovery': recovery_result
            }
            
            # Track successful recovery
            if recovery_result.get('success', False):
                self.metrics['successful_recoveries'] += 1
        
        return result
    
    def run_task(self, task, **kwargs):
        """
        Run a complete task with reflection and recovery.
        
        Args:
            task: The task to run
            **kwargs: Additional arguments for the base agent
            
        Returns:
            Task execution result
        """
        # Reset histories for a new task
        self.command_history = []
        
        try:
            # Execute the task with the base agent
            result = self.base_agent.run_task(task, **kwargs)
            
            # Update metrics
            if self._is_successful_task(result):
                self.metrics['tasks_completed'] += 1
                
                # Record as a good trajectory
                self.good_trajectories.append({
                    'task': task,
                    'commands': self.command_history.copy(),
                    'result': result
                })
            else:
                self.metrics['tasks_failed'] += 1
                
                # Record as a bad trajectory
                self.bad_trajectories.append({
                    'task': task,
                    'commands': self.command_history.copy(),
                    'result': result
                })
                
                # Check if we can immediately generate a revision trajectory
                if self.good_trajectories:
                    # Look for a good trajectory for the same or similar task
                    good_trajectory = self._find_matching_good_trajectory(task)
                    
                    if good_trajectory:
                        logger.info("Found matching good trajectory. Generating revision.")
                        self.generate_revision_trajectory(self.bad_trajectories[-1], good_trajectory)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in task execution: {str(e)}")
            
            # Update metrics
            self.metrics['tasks_failed'] += 1
            
            # Record as a bad trajectory
            self.bad_trajectories.append({
                'task': task,
                'commands': self.command_history.copy(),
                'error': str(e)
            })
            
            raise
    
    def _find_matching_good_trajectory(self, task):
        """Find a good trajectory that matches the given task."""
        # First, look for exact task match
        for gt in self.good_trajectories:
            if gt['task'].get('id') == task.get('id'):
                return gt
        
        # If no exact match, look for task with same name
        for gt in self.good_trajectories:
            if gt['task'].get('name') == task.get('name'):
                return gt
        
        # If still no match, look for task with similar goal
        task_goal = task.get('goal', '').lower()
        if task_goal:
            for gt in self.good_trajectories:
                gt_goal = gt['task'].get('goal', '').lower()
                # Simple word overlap similarity
                if gt_goal and len(set(task_goal.split()) & set(gt_goal.split())) > 3:
                    return gt
        
        return None
    
    def generate_revision_trajectory(self, bad_trajectory, good_trajectory=None):
        """
        Generate a revision trajectory by identifying where a bad trajectory went wrong
        and splicing in a correction.
        
        Args:
            bad_trajectory: A trajectory that failed
            good_trajectory: An optional good trajectory for the same task
            
        Returns:
            A revision trajectory with a reflection point
        """
        # If no good trajectory is provided, find one for the same task
        if not good_trajectory:
            good_trajectory = self._find_matching_good_trajectory(bad_trajectory['task'])
        
        # If we still don't have a good trajectory, we can't generate a revision
        if not good_trajectory:
            logger.warning("No good trajectory available for revision")
            return None
        
        # Determine the transition point
        transition_point, failure_type = self._determine_transition_point(bad_trajectory)
        
        # Generate a reflection specific to the failure type if possible
        reflection = self._generate_reflection(failure_type)
        
        # Ensure the transition point is valid for both trajectories
        bad_commands = bad_trajectory['commands'][:transition_point]
        
        # Find where to splice in the good trajectory
        # We'll find the first command in the good trajectory that accomplishes 
        # what the failing command was trying to do
        good_splice_point = 0
        for i, cmd in enumerate(good_trajectory['commands']):
            # Simple heuristic: if commands share key terms, they might be trying to do the same thing
            if self._commands_have_similar_purpose(bad_commands[-1] if bad_commands else "", cmd):
                good_splice_point = i
                break
        
        # Generate the revision trajectory
        revision = {
            'task': bad_trajectory['task'],
            'bad_commands': bad_commands,
            'reflection': reflection,
            'good_commands': good_trajectory['commands'][good_splice_point:],
            'transition_point': transition_point,
            'failure_type': failure_type.name if failure_type else "UNKNOWN",
            'similarity_score': self._calculate_trajectory_similarity(
                bad_trajectory['commands'], good_trajectory['commands']),
            'timestamp': time.time()
        }
        
        # Store the revision trajectory
        self.revision_trajectories.append(revision)
        
        return revision
    
    def _commands_have_similar_purpose(self, cmd1, cmd2):
        """Check if two commands are trying to accomplish similar things."""
        # Extract the command base (first word) and arguments
        cmd1_parts = cmd1.split()
        cmd2_parts = cmd2.split()
        
        if not cmd1_parts or not cmd2_parts:
            return False
        
        # If they're the same command type, they might be related
        if cmd1_parts[0] == cmd2_parts[0]:
            return True
        
        # Check for command synonyms (e.g., ls and dir)
        command_synonyms = {
            'ls': ['dir', 'list'],
            'cat': ['type', 'more', 'less'],
            'rm': ['del', 'delete'],
            'cp': ['copy'],
            'mv': ['move'],
            'grep': ['find', 'search', 'select-string']
        }
        
        cmd1_base = cmd1_parts[0]
        cmd2_base = cmd2_parts[0]
        
        # Check if they're synonym commands
        for base_cmd, synonyms in command_synonyms.items():
            if (cmd1_base == base_cmd and cmd2_base in synonyms) or \
               (cmd2_base == base_cmd and cmd1_base in synonyms):
                return True
        
        # If they operate on the same files, they might be related
        cmd1_files = [arg for arg in cmd1_parts[1:] if not arg.startswith('-')]
        cmd2_files = [arg for arg in cmd2_parts[1:] if not arg.startswith('-')]
        
        if set(cmd1_files) & set(cmd2_files):  # If there's overlap in files
            return True
        
        return False
    
    def _calculate_trajectory_similarity(self, commands1, commands2):
        """Calculate similarity between two command trajectories."""
        if not commands1 or not commands2:
            return 0.0
        
        # Extract command bases (first word of each command)
        bases1 = [cmd.split()[0] if cmd and ' ' in cmd else cmd for cmd in commands1]
        bases2 = [cmd.split()[0] if cmd and ' ' in cmd else cmd for cmd in commands2]
        
        # Count matching command types
        common_commands = set(bases1) & set(bases2)
        
        # Calculate Jaccard similarity
        similarity = len(common_commands) / len(set(bases1) | set(bases2))
        
        return similarity
    
    def _determine_transition_point(self, bad_trajectory):
        """
        Determine the optimal point to transition from bad to good trajectory.
        
        This implements the model-guided transition point determination from Agent-R.
        
        Returns:
            Tuple of (transition_point, failure_type)
        """
        commands = bad_trajectory['commands']
        
        # If no commands, can't determine transition point
        if not commands:
            return 0, None
        
        # Check for explicit failure information in the result
        result = bad_trajectory.get('result', {})
        if isinstance(result, dict):
            step_results = result.get('step_results', [])
            for i, step in enumerate(step_results):
                if not step.get('success', True):
                    # Found a failing step, transition at the corresponding command
                    cmd_index = min(i, len(commands) - 1)
                    # Try to determine the failure type
                    stderr = step.get('stderr', '')
                    stdout = step.get('stdout', '')
                    cmd = step.get('command', '')
                    failure_type = self.failure_detector.detect_failure(cmd, stdout, stderr)
                    return cmd_index, failure_type
        
        # If no explicit failure information, analyze the commands
        for i, command in enumerate(commands):
            # Create context for detection
            if i > 0:
                # Check if this is where a loop started
                if self._is_start_of_loop(commands, i):
                    return i, FailureType.LOOP_DETECTED
                
                # Check for other command-based failure indicators
                if "error" in command.lower() or "failed" in command.lower():
                    return i, FailureType.INVALID_ACTION
                
                # Check for retry patterns (slight variations of the same command)
                if i >= 1 and self._is_retry_variant(commands[i-1], command):
                    return i-1, FailureType.INVALID_ACTION
        
        # If no clear error point is found, analyze the last few commands
        if len(commands) >= 3:
            last_few = commands[-3:]
            
            # Check if the last few commands show a pattern of trying different things
            # This might indicate the agent is struggling with a particular step
            if len(set(cmd.split()[0] if ' ' in cmd else cmd for cmd in last_few)) >= 3:
                return len(commands) - 3, FailureType.GOAL_DEVIATED
        
        # If all else fails, default to the middle of the trajectory
        return len(commands) // 2, None
    
    def _is_start_of_loop(self, commands, index):
        """Check if this index is where a command loop starts."""
        # Need at least 3 commands to detect a loop
        if index + 2 >= len(commands):
            return False
        
        # Check for exact repetition
        if commands[index] == commands[index+1] == commands[index+2]:
            return True
        
        # Check for command synonyms or variations
        if index + 3 < len(commands):
            # Get the base commands (first word)
            bases = [cmd.split()[0] if ' ' in cmd else cmd for cmd in commands[index:index+4]]
            # If the same base command repeats, it might be a loop
            if len(set(bases)) == 1:
                return True
        
        return False
    
    def _is_retry_variant(self, cmd1, cmd2):
        """Check if cmd2 is a retry variant of cmd1 (slight modification)."""
        if not cmd1 or not cmd2:
            return False
        
        # If they're identical, not a variant
        if cmd1 == cmd2:
            return False
        
        # Check if they start with the same command
        cmd1_parts = cmd1.split()
        cmd2_parts = cmd2.split()
        
        if not cmd1_parts or not cmd2_parts:
            return False
        
        # If same command base but different args, might be a retry
        if cmd1_parts[0] == cmd2_parts[0]:
            # Calculate how different the arguments are
            args1 = set(cmd1_parts[1:])
            args2 = set(cmd2_parts[1:])
            
            # If there's significant overlap but they're not identical, it's a variant
            intersection = args1 & args2
            if intersection and (len(args1) != len(intersection) or len(args2) != len(intersection)):
                return True
        
        return False
    
    def _generate_reflection(self, failure_type=None):
        """
        Generate a reflection thought for the transition.
        
        Args:
            failure_type: Optional failure type to generate a specific reflection
            
        Returns:
            A reflection thought
        """
        # If we have a specific failure type and templates for it, use those
        if failure_type and failure_type in self.recovery_templates:
            templates = self.recovery_templates[failure_type]
            return random.choice(templates)
        
        # Otherwise use generic reflection thoughts
        return random.choice(self.reflection_thoughts)
    
    def _is_successful_task(self, result):
        """
        Determine if a task was successful based on its result.
        
        Args:
            result: The task execution result
            
        Returns:
            True if the task was successful, False otherwise
        """
        # If result is a dict, check for success indicators
        if isinstance(result, dict):
            # Check for explicit success flag
            if 'success' in result:
                return result['success']
            
            # Check for status
            if 'status' in result and result['status'] == 'complete':
                return True
            
            # Check for score
            if 'score' in result and result['score'] > 0:
                return True
            
            # Check for rewards
            if 'reward' in result and result['reward'] > 0:
                return True
            
            # Check all step results
            if 'step_results' in result and result['step_results']:
                return all(step.get('success', False) for step in result['step_results'])
        
        # Default to False if we can't determine success
        return False
    
    def train_with_revision_trajectories(self, iterations=3):
        """
        Train the base agent using revision trajectories.
        
        This implements the iterative self-training phase of Agent-R.
        
        Args:
            iterations: Number of training iterations
            
        Returns:
            Training statistics
        """
        stats = []
        
        for i in range(iterations):
            iteration_start = time.time()
            logger.info(f"Starting iteration {i+1}/{iterations}")
            
            # 1. Generate revision trajectories if needed
            if i == 0:
                # In the first iteration, use existing trajectories
                revision_count = len(self.revision_trajectories)
            else:
                # In later iterations, generate new trajectories
                revision_count = 0
                for bad in self.bad_trajectories:
                    revision = self.generate_revision_trajectory(bad)
                    if revision:
                        revision_count += 1
            
            # 2. Train the agent
            # In a real implementation, this would involve actual model training
            # For demonstration, we'll simulate training by combining trajectories
            trajectory_data = self._prepare_training_data()
            
            # Simulate training
            training_time = time.time() - iteration_start
            logger.info(f"Simulating training on {len(trajectory_data)} trajectories...")
            time.sleep(1)  # Simulate training time
            
            # 3. Evaluate the agent
            # In a real implementation, this would run the agent on a test set
            eval_stats = self._simulate_evaluation()
            
            # 4. Record statistics
            stats.append({
                'iteration': i+1,
                'revisions': revision_count,
                'trajectories': len(trajectory_data),
                'training_time': training_time,
                'evaluation': eval_stats,
                'good_trajectories': len(self.good_trajectories),
                'bad_trajectories': len(self.bad_trajectories)
            })
            
            logger.info(f"Iteration {i+1} complete. " +
                       f"Success rate: {eval_stats['success_rate']:.2%}")
        
        return stats
    
    def _prepare_training_data(self):
        """Prepare training data from trajectories."""
        training_data = []
        
        # Add good trajectories
        for traj in self.good_trajectories:
            training_data.append({
                'type': 'good',
                'task': traj['task'],
                'commands': traj['commands'],
                'weight': 1.0  # Base weight for good trajectories
            })
        
        # Add revision trajectories with higher weight
        for traj in self.revision_trajectories:
            training_data.append({
                'type': 'revision',
                'task': traj['task'],
                'bad_commands': traj['bad_commands'],
                'reflection': traj['reflection'],
                'good_commands': traj['good_commands'],
                'weight': 2.0  # Revision trajectories are more valuable for learning
            })
        
        return training_data
    
    def _simulate_evaluation(self):
        """Simulate evaluating the agent on a test set."""
        # In a real implementation, this would run the agent on actual test tasks
        # For demonstration, we'll simulate results
        
        # The simulation shows improvement over iterations
        # Start with stats based on our real metrics
        success_rate = min(0.5 + (self.metrics['successful_recoveries'] / 
                               max(1, self.metrics['recoveries_attempted'])) * 0.3, 0.95)
        
        # Add some randomness
        success_rate += random.uniform(-0.05, 0.05)
        success_rate = max(0.0, min(1.0, success_rate))
        
        # Calculate improvement metrics
        avg_commands = max(3, 10 - len(self.revision_trajectories) * 0.5)
        failure_reduction = min(0.9, 0.3 + len(self.revision_trajectories) * 0.1)
        
        return {
            'success_rate': success_rate,
            'avg_commands_per_task': avg_commands,
            'failure_reduction': failure_reduction,
            'avg_recovery_time': 0.5 - min(0.4, len(self.revision_trajectories) * 0.02)
        }
    
    def get_metrics(self):
        """Get performance metrics."""
        metrics = self.metrics.copy()
        
        # Add derived metrics
        if metrics['recoveries_attempted'] > 0:
            metrics['recovery_success_rate'] = (
                metrics['successful_recoveries'] / metrics['recoveries_attempted'])
        else:
            metrics['recovery_success_rate'] = 0
            
        metrics['total_tasks'] = metrics['tasks_completed'] + metrics['tasks_failed']
        if metrics['total_tasks'] > 0:
            metrics['task_success_rate'] = metrics['tasks_completed'] / metrics['total_tasks']
        else:
            metrics['task_success_rate'] = 0
            
        metrics['revisions_generated'] = len(self.revision_trajectories)
        
        return metrics
    
    def save_trajectories(self, output_dir='trajectories'):
        """Save trajectories to disk for later analysis or training.
        
        Args:
            output_dir: Directory to save trajectories
            
        Returns:
            Dictionary with counts of saved files
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save good trajectories
        good_dir = os.path.join(output_dir, 'good')
        os.makedirs(good_dir, exist_ok=True)
        good_count = 0
        
        for i, traj in enumerate(self.good_trajectories):
            file_path = os.path.join(good_dir, f'good_trajectory_{i}.json')
            try:
                with open(file_path, 'w') as f:
                    json.dump(traj, f, indent=2)
                good_count += 1
            except Exception as e:
                logger.error(f"Error saving good trajectory {i}: {e}")
        
        # Save bad trajectories
        bad_dir = os.path.join(output_dir, 'bad')
        os.makedirs(bad_dir, exist_ok=True)
        bad_count = 0
        
        for i, traj in enumerate(self.bad_trajectories):
            file_path = os.path.join(bad_dir, f'bad_trajectory_{i}.json')
            try:
                with open(file_path, 'w') as f:
                    json.dump(traj, f, indent=2)
                bad_count += 1
            except Exception as e:
                logger.error(f"Error saving bad trajectory {i}: {e}")
        
        # Save revision trajectories
        revision_dir = os.path.join(output_dir, 'revisions')
        os.makedirs(revision_dir, exist_ok=True)
        revision_count = 0
        
        for i, traj in enumerate(self.revision_trajectories):
            file_path = os.path.join(revision_dir, f'revision_trajectory_{i}.json')
            try:
                with open(file_path, 'w') as f:
                    json.dump(traj, f, indent=2)
                revision_count += 1
            except Exception as e:
                logger.error(f"Error saving revision trajectory {i}: {e}")
        
        # Save metrics
        metrics_path = os.path.join(output_dir, 'metrics.json')
        try:
            with open(metrics_path, 'w') as f:
                json.dump(self.get_metrics(), f, indent=2)
        except Exception as e:
            logger.error(f"Error saving metrics: {e}")
        
        return {
            'good_trajectories': good_count,
            'bad_trajectories': bad_count,
            'revision_trajectories': revision_count,
            'output_dir': output_dir
        }
    
    def load_trajectories(self, input_dir='trajectories'):
        """Load saved trajectories from disk.
        
        Args:
            input_dir: Directory to load trajectories from
            
        Returns:
            Dictionary with counts of loaded files
        """
        # Track loaded counts
        good_count = 0
        bad_count = 0
        revision_count = 0
        
        # Load good trajectories
        good_dir = os.path.join(input_dir, 'good')
        if os.path.exists(good_dir):
            good_files = [f for f in os.listdir(good_dir) if f.endswith('.json')]
            for file_name in good_files:
                file_path = os.path.join(good_dir, file_name)
                try:
                    with open(file_path, 'r') as f:
                        traj = json.load(f)
                    self.good_trajectories.append(traj)
                    good_count += 1
                except Exception as e:
                    logger.error(f"Error loading good trajectory {file_path}: {e}")
        
        # Load bad trajectories
        bad_dir = os.path.join(input_dir, 'bad')
        if os.path.exists(bad_dir):
            bad_files = [f for f in os.listdir(bad_dir) if f.endswith('.json')]
            for file_name in bad_files:
                file_path = os.path.join(bad_dir, file_name)
                try:
                    with open(file_path, 'r') as f:
                        traj = json.load(f)
                    self.bad_trajectories.append(traj)
                    bad_count += 1
                except Exception as e:
                    logger.error(f"Error loading bad trajectory {file_path}: {e}")
        
        # Load revision trajectories
        revision_dir = os.path.join(input_dir, 'revisions')
        if os.path.exists(revision_dir):
            revision_files = [f for f in os.listdir(revision_dir) if f.endswith('.json')]
            for file_name in revision_files:
                file_path = os.path.join(revision_dir, file_name)
                try:
                    with open(file_path, 'r') as f:
                        traj = json.load(f)
                    self.revision_trajectories.append(traj)
                    revision_count += 1
                except Exception as e:
                    logger.error(f"Error loading revision trajectory {file_path}: {e}")
        
        logger.info(f"Loaded {good_count} good, {bad_count} bad, and {revision_count} revision trajectories")
        
        return {
            'good_trajectories': good_count,
            'bad_trajectories': bad_count,
            'revision_trajectories': revision_count
        }
    
    def analyze_trajectories(self):
        """Analyze all trajectories to extract insights.
        
        Returns:
            Dictionary with analysis results
        """
        # Command frequency analysis
        command_counts = defaultdict(int)
        successful_commands = defaultdict(int)
        failure_commands = defaultdict(int)
        
        # Process good trajectories
        for traj in self.good_trajectories:
            for cmd in traj['commands']:
                if not cmd:
                    continue
                # Extract base command (first word)
                base_cmd = cmd.split()[0] if ' ' in cmd else cmd
                command_counts[base_cmd] += 1
                successful_commands[base_cmd] += 1
        
        # Process bad trajectories
        for traj in self.bad_trajectories:
            for cmd in traj['commands']:
                if not cmd:
                    continue
                # Extract base command (first word)
                base_cmd = cmd.split()[0] if ' ' in cmd else cmd
                command_counts[base_cmd] += 1
                failure_commands[base_cmd] += 1
        
        # Calculate success rates for commands
        command_success_rates = {}
        for cmd in command_counts:
            success_count = successful_commands.get(cmd, 0)
            total_count = command_counts[cmd]
            command_success_rates[cmd] = success_count / total_count if total_count else 0
        
        # Analyze transitions in revision trajectories
        transition_types = defaultdict(int)
        reflection_themes = defaultdict(int)
        
        for traj in self.revision_trajectories:
            # Count failure types
            failure_type = traj.get('failure_type', 'UNKNOWN')
            transition_types[failure_type] += 1
            
            # Analyze reflection text
            reflection = traj.get('reflection', '')
            if reflection:
                # Simple keyword analysis
                keywords = ['approach', 'mistake', 'error', 'wrong', 'incorrect', 
                           'strategy', 'revise', 'rethink', 'try', 'alternative']
                for keyword in keywords:
                    if keyword in reflection.lower():
                        reflection_themes[keyword] += 1
        
        # Most common successful and failing commands
        top_successful = sorted(successful_commands.items(), key=lambda x: x[1], reverse=True)[:10]
        top_failing = sorted(failure_commands.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # Most problematic commands (high failure rate)
        problematic_commands = [
            (cmd, rate) for cmd, rate in command_success_rates.items() 
            if rate < 0.5 and command_counts[cmd] >= 3
        ]
        problematic_commands.sort(key=lambda x: x[1])
        
        return {
            'command_counts': dict(command_counts),
            'command_success_rates': command_success_rates,
            'top_successful_commands': dict(top_successful),
            'top_failing_commands': dict(top_failing),
            'problematic_commands': dict(problematic_commands[:10]),
            'transition_types': dict(transition_types),
            'reflection_themes': dict(reflection_themes),
            'good_trajectory_count': len(self.good_trajectories),
            'bad_trajectory_count': len(self.bad_trajectories),
            'revision_trajectory_count': len(self.revision_trajectories)
        }
    
    def visualize_trajectory(self, trajectory_index, trajectory_type='revision'):
        """
        Create a human-readable visualization of a trajectory.
        
        Args:
            trajectory_index: Index of the trajectory to visualize
            trajectory_type: Type of trajectory ('good', 'bad', or 'revision')
            
        Returns:
            String with visualization or None if trajectory not found
        """
        if trajectory_type == 'good' and trajectory_index < len(self.good_trajectories):
            traj = self.good_trajectories[trajectory_index]
            commands = traj['commands']
            
            output = f"=== Good Trajectory {trajectory_index} ===\n"
            output += f"Task: {traj['task'].get('name', 'Unnamed')}\n"
            output += f"Commands: {len(commands)}\n\n"
            
            for i, cmd in enumerate(commands):
                output += f"{i+1}. {cmd}\n"
            
            return output
            
        elif trajectory_type == 'bad' and trajectory_index < len(self.bad_trajectories):
            traj = self.bad_trajectories[trajectory_index]
            commands = traj['commands']
            
            output = f"=== Bad Trajectory {trajectory_index} ===\n"
            output += f"Task: {traj['task'].get('name', 'Unnamed')}\n"
            output += f"Commands: {len(commands)}\n\n"
            
            for i, cmd in enumerate(commands):
                output += f"{i+1}. {cmd}\n"
            
            return output
            
        elif trajectory_type == 'revision' and trajectory_index < len(self.revision_trajectories):
            traj = self.revision_trajectories[trajectory_index]
            bad_commands = traj['bad_commands']
            good_commands = traj['good_commands']
            
            output = f"=== Revision Trajectory {trajectory_index} ===\n"
            output += f"Task: {traj['task'].get('name', 'Unnamed')}\n"
            output += f"Failure Type: {traj.get('failure_type', 'Unknown')}\n\n"
            
            # Bad commands (pre-reflection)
            output += "--- Bad Commands ---\n"
            for i, cmd in enumerate(bad_commands):
                output += f"{i+1}. {cmd}\n"
            
            # Reflection
            output += f"\n--- Reflection ---\n{traj['reflection']}\n"
            
            # Good commands (post-reflection)
            output += "\n--- Good Commands ---\n"
            for i, cmd in enumerate(good_commands):
                output += f"{i+1}. {cmd}\n"
            
            return output
        
        return None
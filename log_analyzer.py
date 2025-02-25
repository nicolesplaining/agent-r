import os
import json
import glob
from pathlib import Path
from typing import Dict, List, Optional
from collections import Counter
from failure_types import FailureDetector
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AgentLogAnalyzer:
    """
    Analyzes agent logs from data_by_challenges directory to identify failure patterns.
    """
    
    def __init__(self, data_dir="data_by_challenges"):
        self.data_dir = data_dir
        self.failure_detector = FailureDetector()
        # Statistics
        self.total_logs = 0
        self.success_logs = 0
        self.failure_logs = 0
        self.failure_types_count = Counter()
    
    def scan_logs(self) -> Dict:
        """
        Scan all logs in the data directory and analyze them.
        
        Returns:
            Dictionary with analysis results
        """
        log_files = self._find_all_log_files()
        logger.info(f"Found {len(log_files)} log files to analyze")
        
        # Reset statistics
        self.total_logs = 0
        self.success_logs = 0
        self.failure_logs = 0
        self.failure_types_count = Counter()
        
        # Process all log files
        for log_file in log_files:
            try:
                self.total_logs += 1
                
                # Determine if this is a success or failure log
                is_success = 'success' in log_file.lower()
                if is_success:
                    self.success_logs += 1
                else:
                    self.failure_logs += 1
                
                # Analyze the log file
                log_data = self._load_log_file(log_file)
                if not log_data:
                    logger.warning(f"Skipping empty or invalid log file: {log_file}")
                    continue
                    
                # Detect failures in the log
                logger.info(f"Analyzing log file: {log_file}")
                failures = self.failure_detector.analyze_agent_log(log_data)
                
                # Count failure types
                for failure in failures:
                    failure_type = failure['failure_type']
                    self.failure_types_count[failure_type.name] += 1
            
            except Exception as e:
                logger.error(f"Error processing log file {log_file}: {e}")
        
        # Prepare analysis results
        analysis = {
            'total_logs': self.total_logs,
            'success_logs': self.success_logs,
            'failure_logs': self.failure_logs,
            'failure_distribution': dict(self.failure_types_count),
            'success_rate': (self.success_logs / self.total_logs) if self.total_logs > 0 else 0
        }
        
        return analysis
    
    def analyze_log_file(self, log_file_path: str) -> Dict:
        """
        Analyze a specific log file in detail.
        
        Args:
            log_file_path: Path to the log file
            
        Returns:
            Detailed analysis of the log file
        """
        # Load the log data
        log_data = self._load_log_file(log_file_path)
        if not log_data:
            return {'error': 'Failed to load log file'}
        
        # Run the failure detector
        failures = self.failure_detector.analyze_agent_log(log_data)
        
        # Get basic task info
        task_info = self._extract_task_info(log_data)
        
        # Calculate success/failure
        is_success = 'success' in log_file_path.lower()
        
        return {
            'file_path': log_file_path,
            'task_info': task_info,
            'success': is_success,
            'failures': failures,
            'failure_count': len(failures),
            'failure_types': [f['failure_type'].name for f in failures]
        }
    
    def analyze_challenge_directory(self, challenge_dir: str) -> Dict:
        """
        Analyze all logs in a specific challenge directory.
        
        Args:
            challenge_dir: Path to the challenge directory
            
        Returns:
            Analysis of all logs in the challenge directory
        """
        # Find all log files in the directory
        log_files = glob.glob(os.path.join(self.data_dir, challenge_dir, "*.json"))
        
        analyses = []
        success_count = 0
        failure_count = 0
        
        for log_file in log_files:
            analysis = self.analyze_log_file(log_file)
            analyses.append(analysis)
    
            if analysis['success']:
                success_count += 1
            else:
                failure_count += 1
        
        return {
            'challenge_dir': challenge_dir,
            'total_logs': len(log_files),
            'success_count': success_count,
            'failure_count': failure_count,
            'success_rate': (success_count / len(log_files)) if log_files else 0,
            'analyses': analyses
        }
    
    def find_common_failure_patterns(self) -> List[Dict]:
        """
        Identify common failure patterns across logs.
        
        Returns:
            List of common failure patterns with frequency
        """
        # Find all failure logs
        log_files = self._find_all_log_files(success_only=False, failure_only=True)
        
        # Collect failure sequences
        failure_sequences = []
        
        for log_file in log_files:
            log_data = self._load_log_file(log_file)
            if not log_data:
                continue
                
            failures = self.failure_detector.analyze_agent_log(log_data)
            if failures:
                # Create a failure sequence (just the types for now)
                sequence = [f['failure_type'].name for f in failures]
                failure_sequences.append((log_file, sequence))
        
        # Count common patterns (for now just count individual failure types)
        failure_counts = Counter()
        for _, sequence in failure_sequences:
            for failure in sequence:
                failure_counts[failure] += 1
        
        # Find common sequences (patterns of 2 or more failures)
        common_patterns = []
        for i, (log_file, sequence) in enumerate(failure_sequences):
            if len(sequence) >= 2:
                pattern = ' -> '.join(sequence)
                # Check if we've seen this pattern before
                if not any(p['pattern'] == pattern for p in common_patterns):
                    # Count occurrences of this pattern
                    count = sum(1 for _, seq in failure_sequences if ' -> '.join(seq) == pattern)
                    if count > 1:  # Only include if it appears more than once
                        common_patterns.append({
                            'pattern': pattern,
                            'count': count,
                            'example_log': log_file
                        })
        
        return common_patterns
    
    def _find_all_log_files(self, success_only=False, failure_only=False) -> List[str]:
        """Find all JSON log files in the data directory."""
        # Get all JSON files
        log_files = glob.glob(os.path.join(self.data_dir, "**/*.json"), recursive=True)
        
        if success_only:
            log_files = [f for f in log_files if 'success' in f.lower()]
        elif failure_only:
            log_files = [f for f in log_files if 'fail' in f.lower()]
        
        return log_files
    
    def _load_log_file(self, file_path: str) -> Optional[Dict]:
        """Load a JSON log file."""
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Error loading log file {file_path}: {e}")
            return None
    
    def _extract_task_info(self, log_data: Dict) -> Dict:
        """Extract basic task information from the log data."""
        task_info = {}
        
        if 'task' in log_data:
            task = log_data['task']
            task_info['name'] = task.get('name', 'Unknown')
            task_info['path'] = task.get('path', 'Unknown')
            task_info['difficulty'] = task.get('difficulty', 'Unknown')
            
            # Extract competition info if available
            if 'competition' in task:
                task_info['competition'] = task['competition'].get('competition_name', 'Unknown')
        
        return task_info
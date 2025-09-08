"""
History management and persistence layer for the Bedrock Prompt Optimizer.

This module provides the HistoryManager class for saving and loading prompt iterations,
session data, and managing the local JSON-based storage system.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import uuid
import shutil

from models import PromptIteration, ExecutionResult, EvaluationResult, UserFeedback


class HistoryManager:
    """Manages persistence of prompt optimization sessions and iterations."""
    
    def __init__(self, storage_dir: str = "data"):
        """
        Initialize the HistoryManager with a storage directory.
        
        Args:
            storage_dir: Directory path for storing session data
        """
        self.storage_dir = Path(storage_dir)
        self.sessions_dir = self.storage_dir / "sessions"
        self.iterations_dir = self.storage_dir / "iterations"
        
        # Create directories if they don't exist
        self._ensure_directories()
    
    def _ensure_directories(self) -> None:
        """Create storage directories if they don't exist."""
        self.storage_dir.mkdir(exist_ok=True)
        self.sessions_dir.mkdir(exist_ok=True)
        self.iterations_dir.mkdir(exist_ok=True)
    
    def create_session(self, initial_prompt: str, context: Optional[str] = None) -> str:
        """
        Create a new optimization session.
        
        Args:
            initial_prompt: The initial prompt text
            context: Optional context information for the session
            
        Returns:
            str: Unique session identifier
        """
        session_id = str(uuid.uuid4())
        session_data = {
            "id": session_id,
            "created_at": datetime.now().isoformat(),
            "initial_prompt": initial_prompt,
            "context": context or "",
            "status": "active",
            "iteration_count": 0,
            "final_prompt": None,
            "finalized_at": None
        }
        
        session_file = self.sessions_dir / f"{session_id}.json"
        with open(session_file, 'w', encoding='utf-8') as f:
            json.dump(session_data, f, indent=2)
        
        return session_id
    
    def save_iteration(self, iteration: PromptIteration) -> bool:
        """
        Save a prompt iteration to storage.
        
        Args:
            iteration: The PromptIteration object to save
            
        Returns:
            bool: True if saved successfully, False otherwise
        """
        try:
            # Validate the iteration before saving
            if not iteration.validate():
                return False
            
            # Create session-specific directory
            session_dir = self.iterations_dir / iteration.session_id
            session_dir.mkdir(exist_ok=True)
            
            # Save iteration to file
            iteration_file = session_dir / f"iteration_{iteration.version:03d}.json"
            with open(iteration_file, 'w', encoding='utf-8') as f:
                f.write(iteration.to_json())
            
            # Update session metadata
            self._update_session_iteration_count(iteration.session_id, iteration.version)
            
            return True
        except Exception:
            return False
    
    def load_iteration(self, session_id: str, version: int) -> Optional[PromptIteration]:
        """
        Load a specific iteration from storage.
        
        Args:
            session_id: The session identifier
            version: The iteration version number
            
        Returns:
            PromptIteration or None if not found
        """
        try:
            session_dir = self.iterations_dir / session_id
            iteration_file = session_dir / f"iteration_{version:03d}.json"
            
            if not iteration_file.exists():
                return None
            
            with open(iteration_file, 'r', encoding='utf-8') as f:
                json_data = f.read()
            
            return PromptIteration.from_json(json_data)
        except Exception:
            return None
    
    def load_session_history(self, session_id: str) -> List[PromptIteration]:
        """
        Load all iterations for a session.
        
        Args:
            session_id: The session identifier
            
        Returns:
            List of PromptIteration objects, sorted by version
        """
        iterations = []
        session_dir = self.iterations_dir / session_id
        
        if not session_dir.exists():
            return iterations
        
        # Find all iteration files
        iteration_files = sorted(session_dir.glob("iteration_*.json"))
        
        for iteration_file in iteration_files:
            try:
                with open(iteration_file, 'r', encoding='utf-8') as f:
                    json_data = f.read()
                iteration = PromptIteration.from_json(json_data)
                iterations.append(iteration)
            except Exception:
                # Skip corrupted files
                continue
        
        # Sort by version number
        iterations.sort(key=lambda x: x.version)
        return iterations
    
    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get session metadata.
        
        Args:
            session_id: The session identifier
            
        Returns:
            Dictionary with session information or None if not found
        """
        try:
            session_file = self.sessions_dir / f"{session_id}.json"
            
            if not session_file.exists():
                return None
            
            with open(session_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return None
    
    def list_sessions(self) -> List[Dict[str, Any]]:
        """
        List all sessions with their metadata.
        
        Returns:
            List of session information dictionaries
        """
        sessions = []
        
        for session_file in self.sessions_dir.glob("*.json"):
            try:
                with open(session_file, 'r', encoding='utf-8') as f:
                    session_data = json.load(f)
                sessions.append(session_data)
            except Exception:
                # Skip corrupted files
                continue
        
        # Sort by creation date (newest first)
        sessions.sort(key=lambda x: x.get('created_at', ''), reverse=True)
        return sessions
    
    def finalize_session(self, session_id: str, final_prompt: str) -> bool:
        """
        Mark a session as finalized with the final optimized prompt.
        
        Args:
            session_id: The session identifier
            final_prompt: The final optimized prompt text
            
        Returns:
            bool: True if finalized successfully, False otherwise
        """
        try:
            session_info = self.get_session_info(session_id)
            if not session_info:
                return False
            
            session_info['status'] = 'finalized'
            session_info['final_prompt'] = final_prompt
            session_info['finalized_at'] = datetime.now().isoformat()
            
            session_file = self.sessions_dir / f"{session_id}.json"
            with open(session_file, 'w', encoding='utf-8') as f:
                json.dump(session_info, f, indent=2)
            
            return True
        except Exception:
            return False
    
    def delete_session(self, session_id: str) -> bool:
        """
        Delete a session and all its iterations.
        
        Args:
            session_id: The session identifier
            
        Returns:
            bool: True if deleted successfully, False otherwise
        """
        try:
            # Delete session metadata file
            session_file = self.sessions_dir / f"{session_id}.json"
            if session_file.exists():
                session_file.unlink()
            
            # Delete session iterations directory
            session_dir = self.iterations_dir / session_id
            if session_dir.exists():
                shutil.rmtree(session_dir)
            
            return True
        except Exception:
            return False
    
    def export_session(self, session_id: str, export_path: str) -> bool:
        """
        Export a complete session to a single JSON file.
        
        Args:
            session_id: The session identifier
            export_path: Path where to save the exported session
            
        Returns:
            bool: True if exported successfully, False otherwise
        """
        try:
            session_info = self.get_session_info(session_id)
            if not session_info:
                return False
            
            iterations = self.load_session_history(session_id)
            
            export_data = {
                "session_info": session_info,
                "iterations": [iteration.to_dict() for iteration in iterations]
            }
            
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2)
            
            return True
        except Exception:
            return False
    
    def get_latest_iteration(self, session_id: str) -> Optional[PromptIteration]:
        """
        Get the latest iteration for a session.
        
        Args:
            session_id: The session identifier
            
        Returns:
            PromptIteration or None if no iterations found
        """
        iterations = self.load_session_history(session_id)
        return iterations[-1] if iterations else None
    
    def _update_session_iteration_count(self, session_id: str, version: int) -> None:
        """
        Update the iteration count in session metadata.
        
        Args:
            session_id: The session identifier
            version: The latest iteration version
        """
        try:
            session_info = self.get_session_info(session_id)
            if session_info:
                session_info['iteration_count'] = max(session_info.get('iteration_count', 0), version)
                
                session_file = self.sessions_dir / f"{session_id}.json"
                with open(session_file, 'w', encoding='utf-8') as f:
                    json.dump(session_info, f, indent=2)
        except Exception:
            pass  # Non-critical operation
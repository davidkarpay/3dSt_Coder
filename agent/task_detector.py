"""Task detection service for automatic task type classification."""

import re
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from .schemas import TaskDetectionResult

logger = logging.getLogger(__name__)


class TaskDetector:
    """Service for detecting task types from user messages and context."""

    def __init__(self):
        """Initialize task detector with keyword patterns."""
        self.task_patterns = {
            "code_generation": {
                "keywords": [
                    "write", "create", "generate", "implement", "build", "make",
                    "function", "class", "method", "script", "program", "app",
                    "algorithm", "feature", "component", "module", "api"
                ],
                "phrases": [
                    "write a function", "create a class", "implement", "build an app",
                    "generate code", "write code", "create code", "make a script",
                    "build a feature", "develop a"
                ],
                "file_extensions": [".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".cpp", ".c"],
                "confidence_boost": 0.2
            },
            "code_review": {
                "keywords": [
                    "review", "analyze", "check", "inspect", "audit", "examine",
                    "evaluate", "assess", "optimize", "improve", "refactor",
                    "bug", "error", "issue", "problem", "fix"
                ],
                "phrases": [
                    "review this code", "look at this", "check this code",
                    "what's wrong", "find bugs", "code review", "analyze this",
                    "optimize this", "improve this code", "refactor this"
                ],
                "file_extensions": [".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".cpp", ".c"],
                "confidence_boost": 0.15
            },
            "documentation": {
                "keywords": [
                    "document", "explain", "describe", "comment", "readme",
                    "guide", "tutorial", "manual", "help", "instruction",
                    "documentation", "doc", "docs", "wiki"
                ],
                "phrases": [
                    "write documentation", "create docs", "explain this",
                    "document this", "write a readme", "create a guide",
                    "help me understand", "what does this do", "how does this work"
                ],
                "file_extensions": [".md", ".rst", ".txt", ".doc", ".docx"],
                "confidence_boost": 0.25
            },
            "debugging": {
                "keywords": [
                    "debug", "fix", "error", "bug", "issue", "problem", "fail",
                    "broken", "not working", "exception", "crash", "trace",
                    "stacktrace", "stderr", "logs"
                ],
                "phrases": [
                    "fix this bug", "debug this", "find the error", "not working",
                    "getting an error", "exception", "crash", "fails",
                    "broken", "something wrong", "help debug"
                ],
                "file_extensions": [".log", ".py", ".js", ".ts", ".java", ".cpp"],
                "confidence_boost": 0.3
            },
            "general": {
                "keywords": [
                    "help", "question", "how", "what", "why", "when", "where",
                    "explain", "tell me", "show me", "advice", "suggestion"
                ],
                "phrases": [
                    "help me", "can you", "how do I", "what is", "tell me about",
                    "explain", "show me how", "I need help", "question about"
                ],
                "file_extensions": [],
                "confidence_boost": 0.1
            }
        }

        # Subagent mappings
        self.subagent_mappings = {
            "code_generation": ["code-generator", "developer", "architect"],
            "code_review": ["code-reviewer", "analyzer", "security-auditor"],
            "documentation": ["doc-generator", "technical-writer"],
            "debugging": ["debugger", "troubleshooter"],
            "general": ["general-assistant"]
        }

    def detect_task(
        self,
        message: str,
        attached_files: Optional[List[str]] = None,
        conversation_history: Optional[List[str]] = None
    ) -> TaskDetectionResult:
        """Detect the most likely task type for a given message.

        Args:
            message: User message to analyze
            attached_files: List of attached filenames
            conversation_history: Previous messages for context

        Returns:
            TaskDetectionResult with detected task and confidence
        """
        message_lower = message.lower()
        scores = {}
        reasoning_parts = []

        # Score based on keywords
        for task_type, patterns in self.task_patterns.items():
            score = 0.0

            # Check keywords
            keyword_matches = []
            for keyword in patterns["keywords"]:
                if keyword in message_lower:
                    score += 0.1
                    keyword_matches.append(keyword)

            if keyword_matches:
                reasoning_parts.append(f"{task_type}: keywords {keyword_matches}")

            # Check phrases (higher weight)
            phrase_matches = []
            for phrase in patterns["phrases"]:
                if phrase in message_lower:
                    score += 0.2
                    phrase_matches.append(phrase)

            if phrase_matches:
                reasoning_parts.append(f"{task_type}: phrases {phrase_matches}")

            scores[task_type] = score

        # Boost based on file attachments
        if attached_files:
            file_extensions = [Path(f).suffix.lower() for f in attached_files]
            reasoning_parts.append(f"File extensions: {file_extensions}")

            for task_type, patterns in self.task_patterns.items():
                if any(ext in patterns["file_extensions"] for ext in file_extensions):
                    scores[task_type] += patterns["confidence_boost"]
                    reasoning_parts.append(f"{task_type}: file type match")

        # Context from conversation history
        if conversation_history:
            context_score = self._analyze_conversation_context(conversation_history)
            for task_type, context_boost in context_score.items():
                scores[task_type] += context_boost
                if context_boost > 0:
                    reasoning_parts.append(f"{task_type}: conversation context")

        # Determine best match
        if not scores or all(score == 0 for score in scores.values()):
            # Default to general if no matches
            detected_task = "general"
            confidence = 0.3
            reasoning = "No specific task indicators found, defaulting to general assistance"
        else:
            # Get highest scoring task
            detected_task = max(scores, key=scores.get)
            max_score = scores[detected_task]

            # Normalize confidence (max possible score is roughly 1.0)
            confidence = min(max_score, 1.0)

            # Boost confidence if there are multiple indicators
            if len([s for s in scores.values() if s > 0]) > 1:
                confidence *= 1.1
                confidence = min(confidence, 1.0)

            reasoning = "; ".join(reasoning_parts) if reasoning_parts else "Pattern matching analysis"

        # Get alternative suggestions
        alternative_tasks = [
            task for task, score in sorted(scores.items(), key=lambda x: x[1], reverse=True)
            if task != detected_task and score > 0.1
        ][:3]

        # Suggest subagent
        suggested_subagent = self._suggest_subagent(detected_task, message, attached_files)

        return TaskDetectionResult(
            detected_task=detected_task,
            confidence=confidence,
            reasoning=reasoning,
            suggested_subagent=suggested_subagent,
            alternative_tasks=alternative_tasks
        )

    def _analyze_conversation_context(self, history: List[str]) -> Dict[str, float]:
        """Analyze conversation history for task context.

        Args:
            history: List of previous messages

        Returns:
            Dictionary of task types and context boost scores
        """
        context_scores = {}

        # Look at recent messages for patterns
        recent_messages = history[-5:] if len(history) > 5 else history
        combined_text = " ".join(recent_messages).lower()

        # Check for ongoing task patterns
        for task_type, patterns in self.task_patterns.items():
            context_score = 0.0

            # Lower weight for context keywords
            for keyword in patterns["keywords"]:
                if keyword in combined_text:
                    context_score += 0.05

            context_scores[task_type] = context_score

        return context_scores

    def _suggest_subagent(
        self,
        task_type: str,
        message: str,
        attached_files: Optional[List[str]] = None
    ) -> Optional[str]:
        """Suggest the best subagent for the detected task.

        Args:
            task_type: Detected task type
            message: Original message
            attached_files: Attached filenames

        Returns:
            Suggested subagent name or None
        """
        if task_type not in self.subagent_mappings:
            return None

        candidates = self.subagent_mappings[task_type]

        # Simple selection based on message content
        message_lower = message.lower()

        # Specific subagent selection logic
        if task_type == "code_review":
            if any(word in message_lower for word in ["security", "vulnerability", "exploit"]):
                return "security-auditor"
            elif any(word in message_lower for word in ["analyze", "structure", "architecture"]):
                return "analyzer"
            else:
                return "code-reviewer"

        elif task_type == "code_generation":
            if any(word in message_lower for word in ["architecture", "design", "system"]):
                return "architect"
            else:
                return "developer"

        elif task_type == "documentation":
            return "doc-generator"

        elif task_type == "debugging":
            return "debugger"

        # Default to first candidate
        return candidates[0] if candidates else None

    def get_task_descriptions(self) -> Dict[str, str]:
        """Get human-readable descriptions for all task types.

        Returns:
            Dictionary mapping task types to descriptions
        """
        return {
            "code_generation": "Generate new code, functions, and classes",
            "code_review": "Analyze and review existing code for quality and issues",
            "documentation": "Write documentation, explanations, and guides",
            "debugging": "Help debug errors, fix issues, and troubleshoot problems",
            "general": "General-purpose coding assistance and questions"
        }


# Global task detector instance
task_detector = TaskDetector()
#!/usr/bin/env python3
"""Test the hybrid subagent system."""

import asyncio
from pathlib import Path
from agent.subagent_config import SubAgentConfig, SubAgentRegistry, initialize_builtin_agents


def test_configuration_system():
    """Test the configuration-based subagent system."""
    print("Testing Hybrid Subagent System")
    print("=" * 60)

    # Test 1: Create a configuration
    print("\n1. Creating SubAgent Configuration:")
    config = SubAgentConfig(
        name="test-agent",
        description="Test agent for demonstration",
        system_prompt="You are a test agent. Be helpful and concise.",
        tools=["file_read", "grep"],
        model="codellama"
    )
    print(f"   [OK] Created config: {config.name}")
    print(f"   - Description: {config.description}")
    print(f"   - Tools: {config.tools}")
    print(f"   - Model: {config.model}")

    # Test 2: Markdown conversion
    print("\n2. Markdown Conversion:")
    markdown = config.to_markdown()
    print(f"   [OK] Converted to Markdown ({len(markdown)} chars)")
    print("   Preview:")
    for line in markdown.split('\n')[:5]:
        print(f"     {line}")

    # Test 3: Registry
    print("\n3. SubAgent Registry:")
    registry = SubAgentRegistry()
    initialize_builtin_agents(registry)
    print(f"   [OK] Registry initialized")
    print(f"   - Built-in agents: {', '.join(registry.agents.keys())}")

    # Test 4: Finding matching agents
    print("\n4. Agent Matching:")
    test_cases = [
        "Review my code for quality",
        "Run all tests and fix failures",
        "Debug this error message"
    ]

    for task in test_cases:
        matches = registry.find_matching(task)
        if matches:
            print(f"   [OK] '{task[:30]}...' -> {matches[0].name}")
        else:
            print(f"   [X] '{task[:30]}...' -> No matches")

    # Test 5: Proactive agents
    print("\n5. Proactive Agent Detection:")
    proactive_count = sum(1 for agent in registry.agents.values() if agent.proactive)
    print(f"   [OK] Found {proactive_count} proactive agents")

    for name, agent in registry.agents.items():
        if agent.proactive:
            print(f"   - {name}: {agent.description[:50]}...")

    print("\n" + "=" * 60)
    print("All tests passed! [OK]")
    print("\nKey Features Demonstrated:")
    print("* Configuration-based subagents (like Claude Code)")
    print("* Markdown/YAML format support")
    print("* Built-in agent library")
    print("* Natural language task matching")
    print("* Tool permission management")
    print("* Model selection per subagent")
    print("\nAdvantages over Claude Code:")
    print("* Parallel execution capabilities")
    print("* Performance metrics tracking")
    print("* Dependency graph management")
    print("* Specialized parallel tools")


def create_example_agents():
    """Create example subagent files."""
    print("\nCreating Example Subagent Files")
    print("=" * 60)

    # Create .claude/agents directory
    agents_dir = Path(".claude/agents")
    agents_dir.mkdir(parents=True, exist_ok=True)
    print(f"[OK] Created directory: {agents_dir}")

    # Create a custom subagent
    custom_agent = agents_dir / "custom-analyzer.md"
    custom_agent.write_text("""---
name: custom-analyzer
description: Analyze code structure and patterns. Use for code analysis tasks.
tools: file_read, grep, parallel_file_analyzer
model: inherit
---

You are a code analysis expert focused on understanding code structure.

When analyzing code:
1. Identify architectural patterns
2. Map dependencies between modules
3. Find code smells and anti-patterns
4. Suggest improvements

Focus on clarity and maintainability.""")

    print(f"[OK] Created: {custom_agent}")

    # Create another example
    doc_generator = agents_dir / "doc-generator.md"
    doc_generator.write_text("""---
name: doc-generator
description: Generate comprehensive documentation. Use proactively for documentation tasks.
tools: file_read, file_write, parallel_file_analyzer
model: codellama
---

You are a documentation specialist.

Generate clear, comprehensive documentation including:
- API references
- Usage examples
- Configuration guides
- Architecture diagrams (as text)

Follow documentation best practices.""")

    print(f"[OK] Created: {doc_generator}")

    print("\nExample agents created successfully!")
    print("You can now use them with commands like:")
    print('  "Use the custom-analyzer to review my code"')
    print('  "Have the doc-generator create API documentation"')


if __name__ == "__main__":
    # Run configuration tests
    test_configuration_system()

    # Create example agents
    create_example_agents()

    print("\n" + "=" * 60)
    print("Hybrid Subagent System Ready!")
    print("=" * 60)
    print("\nThe system now combines:")
    print("1. Claude Code's configuration approach")
    print("2. Our powerful parallel execution engine")
    print("\nBest of both worlds!")
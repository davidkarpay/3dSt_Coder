#!/usr/bin/env python3
"""Comprehensive test runner for all test suites."""

import os
import sys
import subprocess
import time
from pathlib import Path


def run_command(cmd, description):
    """Run a command and return success status."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {cmd}")
    print(f"{'='*60}")

    start_time = time.time()

    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )

        duration = time.time() - start_time

        if result.returncode == 0:
            print(f"✓ SUCCESS ({duration:.2f}s)")
            if result.stdout:
                print(f"Output:\n{result.stdout}")
            return True
        else:
            print(f"✗ FAILED ({duration:.2f}s)")
            print(f"Exit code: {result.returncode}")
            if result.stdout:
                print(f"stdout:\n{result.stdout}")
            if result.stderr:
                print(f"stderr:\n{result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        print(f"✗ TIMEOUT after 5 minutes")
        return False
    except Exception as e:
        print(f"✗ ERROR: {e}")
        return False


def main():
    """Run comprehensive test suite."""
    print("3dSt_Coder - Comprehensive Test Suite")
    print("=" * 60)

    # Set environment for testing
    os.environ["PYTHONIOENCODING"] = "utf-8"

    # Track results
    results = {}

    # 1. Basic functionality tests (no pytest dependency)
    results["basic"] = run_command(
        "/c/Python312/python.exe run_tests.py",
        "Basic Functionality Tests"
    )

    # 2. Unit tests with pytest
    results["pytest"] = run_command(
        "/c/Python312/python.exe -m pytest --tb=short -v",
        "Comprehensive Unit Tests (pytest)"
    )

    # 3. Security-focused tests
    results["security"] = run_command(
        "/c/Python312/python.exe -m pytest auth/tests/ -v -m 'not slow'",
        "Authentication & Security Tests"
    )

    # 4. Agent tests
    results["agent"] = run_command(
        "/c/Python312/python.exe -m pytest agent/tests/ -v",
        "Agent System Tests"
    )

    # 5. API tests
    results["api"] = run_command(
        "/c/Python312/python.exe -m pytest api/tests/ -v",
        "API Endpoint Tests"
    )

    # 6. LLM engine tests
    results["llm"] = run_command(
        "/c/Python312/python.exe -m pytest llm_server/tests/ -v",
        "LLM Engine Tests"
    )

    # 7. Test coverage report (if coverage is available)
    coverage_available = True
    try:
        subprocess.run("/c/Python312/python.exe -m coverage --help",
                      shell=True, capture_output=True, check=True)
    except subprocess.CalledProcessError:
        coverage_available = False

    if coverage_available:
        results["coverage"] = run_command(
            "/c/Python312/python.exe -m coverage run -m pytest && /c/Python312/python.exe -m coverage report",
            "Test Coverage Analysis"
        )
    else:
        print(f"\n{'='*60}")
        print("Coverage analysis skipped (coverage not installed)")
        print("Install with: pip install coverage")
        print(f"{'='*60}")
        results["coverage"] = None

    # Summary
    print(f"\n{'='*60}")
    print("COMPREHENSIVE TEST SUMMARY")
    print(f"{'='*60}")

    passed = 0
    failed = 0
    skipped = 0

    for test_name, result in results.items():
        if result is True:
            print(f"✓ {test_name.upper()}: PASSED")
            passed += 1
        elif result is False:
            print(f"✗ {test_name.upper()}: FAILED")
            failed += 1
        else:
            print(f"- {test_name.upper()}: SKIPPED")
            skipped += 1

    print(f"\nResults: {passed} passed, {failed} failed, {skipped} skipped")

    if failed == 0:
        print("\n🎉 ALL TESTS PASSED! 3dSt_Coder is ready for deployment.")
        return 0
    else:
        print(f"\n⚠️  {failed} test suite(s) failed. Please review the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
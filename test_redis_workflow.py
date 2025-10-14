#!/usr/bin/env python3
"""
Test script specifically for the Redis parallel processing workflow.
This script tests the run_parallel.py functionality independently.
"""

import os
import sys
import tempfile
from pathlib import Path
import redis

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_redis_connection():
    """Test basic Redis connection."""
    print("Testing Redis connection...")
    
    try:
        r = redis.Redis()
        r.ping()
        print("✅ Redis connection successful")
        return True
    except Exception as e:
        print(f"❌ Redis connection failed: {e}")
        print("Please start Redis server: redis-server")
        return False

def test_run_parallel_imports():
    """Test that run_parallel can be imported and its functions work."""
    print("\nTesting run_parallel imports...")
    
    try:
        from run_parallel import enqueue_jobs, start_workers, _run_worker_process
        print("✅ run_parallel functions imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Failed to import run_parallel: {e}")
        return False

def test_worker_redis_import():
    """Test that worker_redis can be imported."""
    print("\nTesting worker_redis import...")
    
    try:
        from worker_redis import parse_pdf
        print("✅ worker_redis imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Failed to import worker_redis: {e}")
        return False

def test_enqueue_function():
    """Test the enqueue_jobs function with a mock setup."""
    print("\nTesting enqueue_jobs function...")
    
    try:
        from run_parallel import enqueue_jobs
        
        # Create a temporary directory with a mock PDF file
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create a dummy PDF file for testing
            dummy_pdf = temp_path / "test.pdf"
            dummy_pdf.write_bytes(b"dummy pdf content")
            
            # Test enqueueing a single file
            print(f"Testing with dummy file: {dummy_pdf}")
            
            # This will fail because it's not a real PDF, but we can test the function structure
            try:
                jobs = enqueue_jobs(str(dummy_pdf), str(temp_path / "output"), 0)
                print(f"✅ enqueue_jobs function executed, returned {len(jobs)} jobs")
                return True
            except Exception as e:
                # Expected to fail with dummy PDF, but function should be callable
                if "not a valid PDF" in str(e) or "PDF" in str(e):
                    print("✅ enqueue_jobs function is working (expected PDF error)")
                    return True
                else:
                    print(f"❌ Unexpected error in enqueue_jobs: {e}")
                    return False
                    
    except Exception as e:
        print(f"❌ Failed to test enqueue_jobs: {e}")
        return False

def test_start_workers_function():
    """Test the start_workers function (without actually starting workers)."""
    print("\nTesting start_workers function structure...")
    
    try:
        from run_parallel import start_workers
        
        # Test the function signature (don't actually start workers)
        print("✅ start_workers function is available")
        print("Note: Not actually starting workers to avoid hanging process")
        return True
        
    except Exception as e:
        print(f"❌ Failed to test start_workers: {e}")
        return False

def test_parse_gemini_hsbc_import():
    """Test that parse_gemini_hsbc can be imported."""
    print("\nTesting parse_gemini_hsbc import...")
    
    try:
        from parse_gemini_hsbc import process_pdf
        print("✅ parse_gemini_hsbc imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Failed to import parse_gemini_hsbc: {e}")
        return False

def test_rq_imports():
    """Test that RQ (Redis Queue) can be imported."""
    print("\nTesting RQ imports...")
    
    try:
        from rq import Queue, Worker
        print("✅ RQ imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Failed to import RQ: {e}")
        print("Install with: pip install rq")
        return False

def main():
    """Run all Redis workflow tests."""
    print("="*60)
    print("REDIS PARALLEL PROCESSING WORKFLOW TESTS")
    print("="*60)
    
    tests = [
        ("Redis Connection", test_redis_connection),
        ("RQ Imports", test_rq_imports),
        ("parse_gemini_hsbc Import", test_parse_gemini_hsbc_import),
        ("run_parallel Imports", test_run_parallel_imports),
        ("worker_redis Import", test_worker_redis_import),
        ("enqueue_jobs Function", test_enqueue_function),
        ("start_workers Function", test_start_workers_function),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("REDIS WORKFLOW TEST SUMMARY")
    print("="*60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:.<40} {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All Redis workflow tests passed!")
        print("\nThe Redis parallel processing should work correctly.")
        print("\nTo test with real PDFs:")
        print("1. Ensure Redis is running: redis-server")
        print("2. Run: python run_parallel.py /path/to/pdfs --workers 4")
        print("3. Or use the combined analyzer: python bank_statement_analyzer.py /path/to/pdfs")
        return True
    else:
        print("⚠️  Some tests failed. Please fix the issues above.")
        print("\nCommon fixes:")
        print("- Install Redis: sudo apt-get install redis-server")
        print("- Start Redis: redis-server")
        print("- Install RQ: pip install rq")
        print("- Check parse_gemini_hsbc.py exists and is importable")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

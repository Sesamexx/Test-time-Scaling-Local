# =============================================================================
# test_server.py
# Created:  [Lc-v0.0.1] 2026-02-03
# Updated:  [Lc-v0.0.1] 2026-02-03 Basic Tests
# =============================================================================

"""
Test scripts for remote LLM server.
Run these to verify server is working correctly.
"""

import asyncio
import time
import traceback
from client import get_client, Config

# Disable all proxies
import os
os.environ["HTTP_PROXY"] = ""
os.environ["HTTPS_PROXY"] = ""
os.environ["http_proxy"] = ""
os.environ["https_proxy"] = ""
os.environ["ALL_PROXY"] = ""
os.environ["all_proxy"] = ""

# =============================================================================
# Test Functions
# =============================================================================

async def test_health():
    """Test 1: Check server health."""
    print("=== Test: Health Check ===")
    client = get_client()
    
    print(f"Connecting to: {client.base_url}")
    
    try:
        health = await client.health()
        print(f"Status: {health['status']}")
        print(f"Ollama: {health['ollama_status']}")
        print(f"Models: {health['available_models']}")
        print(f"Pending: {health['pending_requests']}")
        print("✓ Health check passed\n")
        return True
    except Exception as e:
        print(f"✗ Health check failed: {type(e).__name__}: {e}")
        # Print more details for connection errors
        if "ConnectError" in str(type(e)) or "timeout" in str(e).lower():
            print(f"   Server is not reachable at {client.base_url}")
        print()
        return False

async def test_list_models():
    """Test 2: List available models."""
    print("=== Test: List Models ===")
    client = get_client()
    
    try:
        models = await client.list_models()
        print(f"Available models: {models}")
        print("✓ List models passed\n")
        return True
    except Exception as e:
        print(f"✗ List models failed: {type(e).__name__}: {e}\n")
        return False

async def test_single_chat():
    """Test 3: Single chat request."""
    print("=== Test: Single Chat ===")
    client = get_client()
    
    try:
        start = time.time()
        response = await client.chat(
            model="gelab-zero-4b-preview",
            messages=[{"role": "user", "content": "Say 'Hello, test successful!' in exactly those words."}],
            max_tokens=50
        )
        elapsed = time.time() - start
        
        print(f"Response: {response}")
        print(f"Time: {elapsed:.2f}s")
        print("✓ Single chat passed\n")
        return True
    except Exception as e:
        print(f"✗ Single chat failed: {e}\n")
        return False

async def test_concurrent_chat():
    """Test 4: Multiple concurrent requests."""
    print("=== Test: Concurrent Requests ===")
    client = get_client()
    
    prompts = [
        "Count from 1 to 5.",
        "What is 2+2?",
        "Say 'test'.",
    ]
    
    try:
        start = time.time()
        
        # Create concurrent tasks
        tasks = [
            client.chat(
                model="qwen3-vl",
                messages=[{"role": "user", "content": p}],
                max_tokens=100
            )
            for p in prompts
        ]
        
        # Run all at once
        responses = await asyncio.gather(*tasks)
        elapsed = time.time() - start
        
        for i, (prompt, response) in enumerate(zip(prompts, responses)):
            print(f"[{i+1}] Prompt: {prompt}")
            print(f"    Response: {response[:100]}...")
        
        print(f"\nTotal time for {len(prompts)} requests: {elapsed:.2f}s")
        print("✓ Concurrent chat passed\n")
        return True
    except Exception as e:
        print(f"✗ Concurrent chat failed: {e}\n")
        return False

# =============================================================================
# Main
# =============================================================================

async def run_all_tests():
    """Run all tests in sequence."""
    print("\n" + "="*50)
    print("Running Server Tests")
    print("="*50 + "\n")
    
    results = []
    results.append(("Health", await test_health()))
    results.append(("Models", await test_list_models()))
    results.append(("Single Chat", await test_single_chat()))
    results.append(("Concurrent", await test_concurrent_chat()))
    
    print("="*50)
    print("Test Summary")
    print("="*50)
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name}: {status}")
    
    all_passed = all(r[1] for r in results)
    print(f"\nOverall: {'All tests passed!' if all_passed else 'Some tests failed.'}")
    return all_passed

if __name__ == "__main__":
    asyncio.run(run_all_tests())

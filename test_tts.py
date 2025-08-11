#!/usr/bin/env python3
"""
Simple test script for Orpheus TTS server
"""
import requests
import json
import time

def test_tts_endpoint():
    url = "http://localhost:7007/tts"
    
    payload = {
        "prompt": "Hello, this is a test of the Orpheus TTS system.",
        "voice": "tara",
        "sample_rate": 24000
    }
    
    print(f"Testing TTS endpoint: {url}")
    print(f"Payload: {json.dumps(payload, indent=2)}")
    
    try:
        start_time = time.time()
        response = requests.post(url, json=payload, timeout=120)  # 2 minute timeout
        end_time = time.time()
        
        print(f"Response status: {response.status_code}")
        print(f"Response time: {end_time - start_time:.2f} seconds")
        print(f"Response headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            # Save the audio file
            with open("test_output.wav", "wb") as f:
                f.write(response.content)
            print(f"Audio saved to test_output.wav ({len(response.content)} bytes)")
            return True
        else:
            print(f"Error response: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print("Request timed out after 120 seconds")
        return False
    except Exception as e:
        print(f"Request failed: {e}")
        return False

def test_health_endpoint():
    url = "http://localhost:7007/health"
    
    try:
        response = requests.get(url, timeout=10)
        print(f"Health check: {response.status_code} - {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Health check failed: {e}")
        return False

if __name__ == "__main__":
    print("=== Orpheus TTS Server Test ===")
    
    # Test health first
    if test_health_endpoint():
        print("✓ Health check passed")
        
        # Test TTS
        if test_tts_endpoint():
            print("✓ TTS test passed")
        else:
            print("✗ TTS test failed")
    else:
        print("✗ Health check failed - server may not be running")

"""
Simple test script for CogCanvas Web Backend API.

Run this after starting the server to verify all endpoints work correctly.

Usage:
    python test_api.py
"""

import requests
import json
import time

BASE_URL = "http://localhost:3701"
SESSION_ID = "test-session"


def test_health():
    """Test health endpoint."""
    print("Testing /health endpoint...")
    response = requests.get(f"{BASE_URL}/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    print("✓ Health check passed")


def test_root():
    """Test root endpoint."""
    print("\nTesting / endpoint...")
    response = requests.get(BASE_URL)
    assert response.status_code == 200
    data = response.json()
    assert "name" in data
    assert data["name"] == "CogCanvas API"
    print("✓ Root endpoint passed")


def test_simple_chat():
    """Test simple chat endpoint."""
    print("\nTesting /api/chat/simple endpoint...")
    payload = {
        "message": "Let's use PostgreSQL for our database",
        "session_id": SESSION_ID
    }
    response = requests.post(f"{BASE_URL}/api/chat/simple", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "role" in data
    assert "content" in data
    assert data["role"] == "assistant"
    print(f"✓ Chat response: {data['content'][:50]}...")

    if data.get("extracted_objects"):
        print(f"✓ Extracted {len(data['extracted_objects'])} objects")
        for obj in data['extracted_objects']:
            print(f"  - {obj['type']}: {obj['content'][:50]}...")


def test_canvas_stats():
    """Test canvas stats endpoint."""
    print("\nTesting /api/canvas/stats endpoint...")
    response = requests.get(f"{BASE_URL}/api/canvas/stats", params={"session_id": SESSION_ID})
    assert response.status_code == 200
    data = response.json()
    assert "object_count" in data
    assert "turn_count" in data
    assert "type_counts" in data
    print(f"✓ Canvas stats: {data['object_count']} objects, {data['turn_count']} turns")
    if data['type_counts']:
        print(f"  Type breakdown: {data['type_counts']}")


def test_canvas_objects():
    """Test getting all canvas objects."""
    print("\nTesting /api/canvas endpoint...")
    response = requests.get(f"{BASE_URL}/api/canvas", params={"session_id": SESSION_ID})
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    print(f"✓ Retrieved {len(data)} objects")

    for obj in data[:3]:  # Show first 3 objects
        print(f"  - [{obj['type']}] {obj['content'][:50]}... (confidence: {obj['confidence']:.2f})")


def test_canvas_graph():
    """Test graph endpoint."""
    print("\nTesting /api/canvas/graph endpoint...")
    response = requests.get(f"{BASE_URL}/api/canvas/graph", params={"session_id": SESSION_ID})
    assert response.status_code == 200
    data = response.json()
    assert "nodes" in data
    assert "links" in data
    print(f"✓ Graph data: {len(data['nodes'])} nodes, {len(data['links'])} links")


def test_retrieve():
    """Test retrieve endpoint."""
    print("\nTesting /api/canvas/retrieve endpoint...")
    payload = {
        "query": "database",
        "top_k": 3,
        "method": "semantic",
        "session_id": SESSION_ID
    }
    response = requests.post(f"{BASE_URL}/api/canvas/retrieve", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "objects" in data
    assert "scores" in data
    assert "query" in data
    print(f"✓ Retrieved {len(data['objects'])} objects for query: '{data['query']}'")

    for obj, score in zip(data['objects'], data['scores']):
        print(f"  - {obj['content'][:50]}... (score: {score:.3f})")


def test_chat_stream():
    """Test streaming chat endpoint."""
    print("\nTesting /api/chat (SSE streaming) endpoint...")
    payload = {
        "message": "We need to implement authentication",
        "session_id": SESSION_ID
    }

    try:
        response = requests.post(
            f"{BASE_URL}/api/chat",
            json=payload,
            stream=True,
            headers={"Accept": "text/event-stream"}
        )

        assert response.status_code == 200

        token_count = 0
        extraction_count = 0

        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith('data: '):
                    data = json.loads(line[6:])

                    if data['type'] == 'token':
                        token_count += 1
                    elif data['type'] == 'extraction':
                        extraction_count = data.get('count', 0)
                    elif data['type'] == 'done':
                        print(f"✓ Received {token_count} tokens")
                        if extraction_count > 0:
                            print(f"✓ Extracted {extraction_count} new objects")
                        break

        print("✓ Streaming chat completed successfully")

    except Exception as e:
        print(f"✗ Streaming test failed: {e}")


def test_clear_canvas():
    """Test clearing canvas."""
    print("\nTesting /api/canvas/clear endpoint...")
    payload = {"session_id": SESSION_ID}
    response = requests.post(f"{BASE_URL}/api/canvas/clear", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    print(f"✓ {data['message']}")

    # Verify canvas is empty
    response = requests.get(f"{BASE_URL}/api/canvas/stats", params={"session_id": SESSION_ID})
    stats = response.json()
    assert stats['object_count'] == 0
    print("✓ Canvas cleared successfully")


def main():
    """Run all tests."""
    print("=" * 60)
    print("CogCanvas Web Backend API Test Suite")
    print("=" * 60)

    try:
        test_health()
        test_root()
        test_simple_chat()
        test_canvas_stats()
        test_canvas_objects()
        test_canvas_graph()
        test_retrieve()
        test_chat_stream()
        test_clear_canvas()

        print("\n" + "=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)

    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        return 1
    except requests.exceptions.ConnectionError:
        print("\n✗ Cannot connect to server. Make sure it's running on port 3701")
        print("  Start server with: uvicorn main:app --reload --port 3701")
        return 1
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())

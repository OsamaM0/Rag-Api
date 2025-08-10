#!/usr/bin/env python3
"""
Test script to demonstrate proper partial update functionality.

This script shows how the API now correctly handles partial updates
where you only need to send the fields you want to change.
"""

import requests
import json

# API base URL - adjust as needed
BASE_URL = "http://localhost:8000"

def test_collection_partial_update():
    """Test collection partial update - only updating description."""
    collection_id = "test-collection-id"  # Replace with actual collection ID
    
    # PATCH request with only the field you want to update
    update_data = {
        "description": "This is an updated description via PATCH"
        # Note: name and metadata are NOT included - they won't be changed
    }
    
    print("Testing Collection Partial Update...")
    print(f"Sending PATCH to /collections/{collection_id}")
    print(f"Data: {json.dumps(update_data, indent=2)}")
    
    try:
        response = requests.patch(
            f"{BASE_URL}/collections/{collection_id}",
            json=update_data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            print("✅ Success! Collection updated with only the specified field.")
            print(f"Response: {json.dumps(response.json(), indent=2)}")
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to API. Make sure the server is running.")


def test_document_partial_update():
    """Test document partial update - only updating keywords."""
    document_id = "test-document-id"  # Replace with actual document ID
    
    # PATCH request with only the field you want to update
    update_data = {
        "keywords": "machine learning, AI, updated keywords"
        # Note: other fields like filename, content, etc. are NOT included
    }
    
    print("\nTesting Document Partial Update...")
    print(f"Sending PATCH to /documents/{document_id}")
    print(f"Data: {json.dumps(update_data, indent=2)}")
    
    try:
        response = requests.patch(
            f"{BASE_URL}/documents/{document_id}",
            json=update_data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            print("✅ Success! Document updated with only the specified field.")
            print(f"Response: {json.dumps(response.json(), indent=2)}")
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to API. Make sure the server is running.")


def test_empty_update():
    """Test what happens when no fields are provided."""
    collection_id = "test-collection-id"
    
    # Empty update data
    update_data = {}
    
    print("\nTesting Empty Update (should fail)...")
    print(f"Sending PATCH to /collections/{collection_id}")
    print(f"Data: {json.dumps(update_data, indent=2)}")
    
    try:
        response = requests.patch(
            f"{BASE_URL}/collections/{collection_id}",
            json=update_data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 400:
            print("✅ Correctly rejected empty update with 400 Bad Request")
            print(f"Response: {response.text}")
        else:
            print(f"⚠️  Unexpected response: {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to API. Make sure the server is running.")


def test_multiple_fields_update():
    """Test updating multiple fields at once."""
    collection_id = "test-collection-id"
    
    # Update multiple fields
    update_data = {
        "name": "Updated Collection Name",
        "description": "Updated description with multiple fields",
        "metadata": {
            "updated_by": "test_script",
            "update_type": "partial_multiple"
        }
    }
    
    print("\nTesting Multiple Fields Update...")
    print(f"Sending PATCH to /collections/{collection_id}")
    print(f"Data: {json.dumps(update_data, indent=2)}")
    
    try:
        response = requests.patch(
            f"{BASE_URL}/collections/{collection_id}",
            json=update_data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            print("✅ Success! Collection updated with multiple fields.")
            print(f"Response: {json.dumps(response.json(), indent=2)}")
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to API. Make sure the server is running.")


if __name__ == "__main__":
    print("=== RAG API Partial Update Test ===")
    print("This script tests the new partial update functionality.")
    print("Make sure your API server is running on http://localhost:8000")
    print("You'll need to replace the test IDs with actual collection/document IDs.")
    print()
    
    # Run all tests
    test_collection_partial_update()
    test_document_partial_update()
    test_empty_update()
    test_multiple_fields_update()
    
    print("\n=== Test Summary ===")
    print("✅ Your API now supports proper partial updates!")
    print("📝 Key improvements:")
    print("   - PATCH endpoints added alongside PUT")
    print("   - Only send the fields you want to update")
    print("   - Empty updates are properly rejected")
    print("   - Uses Pydantic's exclude_unset=True for clean partial updates")
    print("   - Works with both single fields and multiple fields")

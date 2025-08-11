#!/usr/bin/env python3
"""
Quick test to verify UUID conversion is working properly.
"""

import asyncio
import sys
import os

# Add the app directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from app.services.database import create_collection, PSQLDatabase


async def test_uuid_conversion():
    """Test that UUID conversion works correctly."""
    
    print("🔧 Testing UUID conversion in create_collection...")
    
    try:
        # Create a collection
        result = await create_collection(
            name="Test UUID Conversion",
            description="Testing that UUIDs are returned as strings",
            custom_id="test-uuid-conversion-123"
        )
        
        print(f"✅ Collection created: {result}")
        print(f"📝 UUID field type: {type(result['uuid'])}")
        print(f"📝 UUID value: {result['uuid']}")
        
        # Verify the UUID is a string
        if isinstance(result['uuid'], str):
            print("✅ UUID is correctly returned as a string!")
        else:
            print(f"❌ ERROR: UUID is not a string, it's {type(result['uuid'])}")
        
        return result['uuid']
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        # Clean up database connections
        await PSQLDatabase.close_pool()


if __name__ == "__main__":
    asyncio.run(test_uuid_conversion())

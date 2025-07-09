#!/usr/bin/env python3
"""
Test script to demonstrate database download functionality from aggregator service.

This script shows how the miner_manager can download the database from the aggregator.
"""

import requests
import json
import time
from pathlib import Path
import hashlib

def test_database_info(aggregator_url: str):
    """Test the database info endpoint"""
    print("📊 Testing database info endpoint...")
    
    try:
        response = requests.get(f"{aggregator_url}/database-info")
        response.raise_for_status()
        
        info = response.json()
        print(f"✅ Database info retrieved successfully:")
        print(f"   Database type: {info.get('database_type', 'unknown')}")
        print(f"   Database path: {info.get('database_path', 'unknown')}")
        print(f"   Download available: {info.get('download_available', False)}")
        
        if 'file_info' in info:
            file_info = info['file_info']
            print(f"   File size: {file_info.get('size_human', 'unknown')}")
            print(f"   Modified: {file_info.get('modified_time', 'unknown')}")
        
        if 'table_info' in info:
            table_info = info['table_info']
            print(f"   Miner records: {table_info.get('miner_metadata_count', 0)}")
            print(f"   History records: {table_info.get('performance_history_count', 0)}")
        
        return info
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Failed to get database info: {e}")
        return None

def test_database_download(aggregator_url: str, output_path: str = "downloaded_database.db"):
    """Test downloading the database file"""
    print(f"📥 Testing database download to: {output_path}")
    
    try:
        # Download the database
        response = requests.get(f"{aggregator_url}/download-database")
        response.raise_for_status()
        
        # Save to file
        with open(output_path, 'wb') as f:
            f.write(response.content)
        
        # Calculate file hash
        file_hash = hashlib.md5(response.content).hexdigest()
        file_size = len(response.content)
        
        print(f"✅ Database downloaded successfully:")
        print(f"   File: {output_path}")
        print(f"   Size: {file_size:,} bytes")
        print(f"   MD5: {file_hash}")
        print(f"   Headers: {dict(response.headers)}")
        
        return output_path
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Failed to download database: {e}")
        return None

def test_database_backup(aggregator_url: str):
    """Test creating a database backup"""
    print("📁 Testing database backup creation...")
    
    try:
        response = requests.post(f"{aggregator_url}/backup-database")
        response.raise_for_status()
        
        backup_info = response.json()
        print(f"✅ Database backup created successfully:")
        print(f"   Backup file: {backup_info['backup_info']['backup_filename']}")
        print(f"   Backup size: {backup_info['backup_info']['backup_size']:,} bytes")
        print(f"   Created at: {backup_info['backup_info']['created_at']}")
        
        return backup_info
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Failed to create database backup: {e}")
        return None

def test_aggregator_health(aggregator_url: str):
    """Test the aggregator health endpoint"""
    print("🏥 Testing aggregator health...")
    
    try:
        response = requests.get(f"{aggregator_url}/health")
        response.raise_for_status()
        
        health = response.json()
        print(f"✅ Aggregator is healthy:")
        print(f"   Status: {health.get('status', 'unknown')}")
        print(f"   Service: {health.get('service', 'unknown')}")
        print(f"   Version: {health.get('version', 'unknown')}")
        
        return health
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Aggregator health check failed: {e}")
        return None

def simulate_miner_manager_download(aggregator_url: str):
    """Simulate how miner_manager would download the database"""
    print("\n🔄 Simulating miner_manager database download process...")
    
    # Step 1: Check if database is available
    print("1. Checking database availability...")
    db_info = test_database_info(aggregator_url)
    if not db_info or not db_info.get('download_available', False):
        print("❌ Database not available for download")
        return False
    
    # Step 2: Download the database
    print("2. Downloading database...")
    db_path = test_database_download(aggregator_url, "video_subnet_validator.db")
    if not db_path:
        print("❌ Failed to download database")
        return False
    
    # Step 3: Verify the download
    print("3. Verifying downloaded database...")
    if Path(db_path).exists():
        file_size = Path(db_path).stat().st_size
        print(f"✅ Database file verified: {db_path} ({file_size:,} bytes)")
        return True
    else:
        print("❌ Downloaded database file not found")
        return False

def main():
    """Main test function"""
    print("🧪 Testing Aggregator Database Download Functionality")
    print("=" * 60)
    
    # Configuration
    aggregator_url = "http://localhost:20000"  # Change this to your aggregator URL
    
    print(f"Testing aggregator at: {aggregator_url}")
    print()
    
    # Test 1: Health check
    health = test_aggregator_health(aggregator_url)
    if not health:
        print("❌ Aggregator is not running or not accessible")
        print("💡 Make sure to start the aggregator service first:")
        print("   python aggregator.py")
        return
    
    print()
    
    # Test 2: Database info
    db_info = test_database_info(aggregator_url)
    print()
    
    # Test 3: Database download
    if db_info and db_info.get('download_available', False):
        db_path = test_database_download(aggregator_url)
        print()
        
        # Test 4: Database backup
        backup_info = test_database_backup(aggregator_url)
        print()
        
        # Test 5: Simulate miner_manager workflow
        success = simulate_miner_manager_download(aggregator_url)
        print()
        
        if success:
            print("🎉 All tests passed! The database download functionality is working correctly.")
            print("\nTo use with miner_manager, set:")
            print(f"   export DATABASE_URL=\"{aggregator_url}/download-database\"")
        else:
            print("❌ Some tests failed. Please check the aggregator service.")
    else:
        print("⚠️  Database download not available (may be using remote database)")
    
    print("\n" + "=" * 60)
    print("✨ Test completed!")

if __name__ == "__main__":
    main() 
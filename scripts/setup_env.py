#!/usr/bin/env python3
"""
Helper script to setup .env file for ENT Challenge project.
"""

import os
import sys
from pathlib import Path


def create_env_file():
    """Create .env file from template."""
    project_root = Path(__file__).parent
    env_example_path = project_root / ".env.example"
    env_path = project_root / ".env"
    
    # Check if .env.example exists
    if not env_example_path.exists():
        print(f"❌ File not found: {env_example_path}")
        return False
    
    # Check if .env already exists
    if env_path.exists():
        response = input(f"📁 File .env already exists. Overwrite? (y/N): ").strip().lower()
        if response != 'y':
            print("⏭️ Skipping .env file creation")
            return True
    
    # Copy content
    try:
        with open(env_example_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        with open(env_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"✅ Created .env file from {env_example_path}")
        return True
        
    except Exception as e:
        print(f"❌ Error creating .env file: {e}")
        return False


def setup_api_key():
    """Guide to setup API key."""
    print("\n🔑 SETTING UP API KEY")
    print("=" * 50)
    
    print("1. Visit Google AI Studio: https://makersuite.google.com/app/apikey")
    print("2. Create new API key")
    print("3. Copy API key")
    
    api_key = input("\n🔐 Enter Google API key (or Enter to skip): ").strip()
    
    if not api_key:
        print("⏭️ Skipping API key setup. You can edit .env file later.")
        return True
    
    # Update .env file
    env_path = Path(__file__).parent / ".env"
    if not env_path.exists():
        print("❌ File .env does not exist. Run script again to create .env file first.")
        return False
    
    try:
        # Read current content
        with open(env_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Update GOOGLE_API_KEY line
        updated = False
        for i, line in enumerate(lines):
            if line.startswith('GOOGLE_API_KEY='):
                lines[i] = f'GOOGLE_API_KEY={api_key}\n'
                updated = True
                break
        
        # If not found, add it
        if not updated:
            lines.append(f'\nGOOGLE_API_KEY={api_key}\n')
        
        # Write back
        with open(env_path, 'w', encoding='utf-8') as f:
            f.writelines(lines)
        
        print("✅ Updated API key in .env file")
        return True
        
    except Exception as e:
        print(f"❌ Error updating API key: {e}")
        return False


def verify_setup():
    """Verify setup."""
    print("\n🔍 VERIFYING SETUP")
    print("=" * 50)
    
    env_path = Path(__file__).parent / ".env"
    
    # Check .env file
    if env_path.exists():
        print("✅ File .env exists")
        
        # Check API key
        try:
            with open(env_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if 'GOOGLE_API_KEY=' in content and 'your_' not in content:
                print("✅ GOOGLE_API_KEY is set")
            else:
                print("⚠️ GOOGLE_API_KEY not set or still placeholder")
                
        except Exception as e:
            print(f"❌ Error reading .env file: {e}")
    else:
        print("❌ File .env does not exist")
    
    # Check dependencies
    try:
        import dotenv
        print("✅ python-dotenv is installed")
    except ImportError:
        print("❌ python-dotenv not installed")
        print("   Run: pip install python-dotenv")
    
    try:
        import google.generativeai
        print("✅ google-generativeai is installed")
    except ImportError:
        print("❌ google-generativeai not installed")
        print("   Run: pip install google-generativeai")


def test_api_connection():
    """Test API connection."""
    print("\n🔗 TESTING API CONNECTION")
    print("=" * 50)
    
    try:
        # Load .env
        from dotenv import load_dotenv
        load_dotenv()
        
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key or 'your_' in api_key:
            print("❌ API key not set correctly")
            return False
        
        # Test connection
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        
        # Try to list models
        models = list(genai.list_models())
        print(f"✅ Connection successful! Found {len(models)} models")
        
        # Find Gemini models
        gemini_models = [m for m in models if 'gemini' in m.name.lower()]
        if gemini_models:
            print(f"✅ Found {len(gemini_models)} Gemini models:")
            for model in gemini_models[:3]:  # Show first 3
                print(f"   - {model.name}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        return False
    except Exception as e:
        print(f"❌ API connection error: {e}")
        return False


def main():
    """Main function."""
    print("🚀 ENT CHALLENGE - SETUP ENVIRONMENT")
    print("=" * 60)
    
    # Step 1: Create .env file
    print("\n📁 STEP 1: CREATE .ENV FILE")
    if not create_env_file():
        print("❌ Cannot create .env file")
        return 1
    
    # Step 2: Setup API key
    print("\n🔑 STEP 2: SETUP API KEY")
    setup_api_key()
    
    # Step 3: Verify setup
    verify_setup()
    
    # Step 4: Test API (optional)
    response = input("\n🔗 Test API connection? (y/N): ").strip().lower()
    if response == 'y':
        test_api_connection()
    
    print("\n🎉 SETUP COMPLETED!")
    print("=" * 60)
    print("Now you can run:")
    print("  python eval_with_dataset.py --use_gemini_reranking ...")
    print("  python eval_with_reranking_comparison.py ...")
    print("\nSee RERANKING_README.md for more details.")


if __name__ == "__main__":
    sys.exit(main())

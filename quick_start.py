#!/usr/bin/env python3
"""
PriceWise Quick Start Script
Automatically detects the best deployment method and guides users through setup
"""

import os
import sys
import platform
import subprocess
import json
from pathlib import Path

def detect_environment():
    """Detect the current environment and available tools"""
    env_info = {
        'os': platform.system(),
        'python_version': sys.version_info,
        'docker_available': False,
        'python_available': True,
        'recommended_method': 'script'
    }
    
    # Check Docker availability
    try:
        result = subprocess.run(['docker', '--version'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            env_info['docker_available'] = True
            env_info['recommended_method'] = 'docker'
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    
    # Check Python version
    if env_info['python_version'] < (3, 8):
        env_info['python_available'] = False
        env_info['recommended_method'] = 'docker'
    
    return env_info

def create_secrets_file():
    """Create the secrets.toml file if it doesn't exist"""
    secrets_dir = Path('.streamlit')
    secrets_file = secrets_dir / 'secrets.toml'
    
    if not secrets_dir.exists():
        secrets_dir.mkdir(parents=True)
        print("✅ Created .streamlit directory")
    
    if not secrets_file.exists():
        print("\n🔑 OpenAI API Key Setup Required")
        print("=" * 40)
        api_key = input("Enter your OpenAI API key (or press Enter to skip): ").strip()
        
        if api_key:
            secrets_content = f"""[openai]
api_key = "{api_key}"
"""
            with open(secrets_file, 'w') as f:
                f.write(secrets_content)
            print("✅ Created .streamlit/secrets.toml with your API key")
        else:
            print("⚠️  No API key provided. AI features will be limited.")
            print("   You can add it later in .streamlit/secrets.toml")
    else:
        print("✅ .streamlit/secrets.toml already exists")

def run_docker_setup():
    """Set up and run with Docker"""
    print("\n🐳 Setting up PriceWise with Docker...")
    
    try:
        # Build the Docker image
        print("🔨 Building Docker image...")
        subprocess.run(['docker', 'build', '-t', 'pricewise', '.'], check=True)
        
        # Run with Docker Compose if available
        if Path('docker-compose.yml').exists():
            print("🚀 Starting with Docker Compose...")
            subprocess.run(['docker-compose', 'up', '--build'], check=True)
        else:
            print("🚀 Starting with Docker...")
            subprocess.run([
                'docker', 'run', '-p', '8501:8501',
                '-v', f'{os.getcwd()}/data:/app/data',
                '-v', f'{os.getcwd()}/.streamlit:/app/.streamlit',
                'pricewise'
            ], check=True)
            
    except subprocess.CalledProcessError as e:
        print(f"❌ Docker setup failed: {e}")
        return False
    except KeyboardInterrupt:
        print("\n⏹️  Stopped by user")
        return False
    
    return True

def run_script_setup():
    """Set up and run with Python scripts"""
    print("\n🐍 Setting up PriceWise with Python...")
    
    # Determine the appropriate script
    if platform.system() == 'Windows':
        script = 'run_windows.bat'
        print(f"🚀 Running {script}...")
        subprocess.run([script], shell=True)
    elif platform.system() == 'Darwin':  # macOS
        script = 'run_macos.command'
        print(f"🚀 Running {script}...")
        subprocess.run(['bash', script])
    else:  # Linux
        script = 'run_unix.sh'
        print(f"🚀 Running {script}...")
        subprocess.run(['bash', script])

def main():
    """Main quick start process"""
    print("🚀 PriceWise Quick Start")
    print("=" * 30)
    
    # Detect environment
    env = detect_environment()
    print(f"🖥️  OS: {env['os']}")
    print(f"🐍 Python: {env['python_version'].major}.{env['python_version'].minor}")
    print(f"🐳 Docker: {'✅ Available' if env['docker_available'] else '❌ Not available'}")
    
    # Create secrets file
    create_secrets_file()
    
    # Choose deployment method
    if env['docker_available'] and env['recommended_method'] == 'docker':
        print(f"\n🎯 Recommended method: Docker (most portable)")
        choice = input("Use Docker? (Y/n): ").strip().lower()
        if choice in ['', 'y', 'yes']:
            if run_docker_setup():
                return
            else:
                print("🔄 Falling back to Python script method...")
    
    # Fall back to script method
    print(f"\n🎯 Using method: Python Scripts")
    print("This will install dependencies and run the application")
    
    if not env['python_available']:
        print("❌ Python 3.8+ is required for this method")
        print("Please install Python 3.8+ and try again")
        return
    
    try:
        run_script_setup()
    except KeyboardInterrupt:
        print("\n⏹️  Stopped by user")
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        print("\n🔧 Manual setup options:")
        print("1. Run the appropriate script for your OS")
        print("2. Use Docker: docker-compose up --build")
        print("3. Manual installation: pip install -r requirements.txt")

if __name__ == "__main__":
    main()

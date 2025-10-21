#!/usr/bin/env python3
"""
Startup script for the RAG API server.

Provides easy commands for starting, testing, and managing the API.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


def check_environment():
    """Check if required environment variables are set."""
    required_vars = ["OPENAI_API_KEY"]
    optional_vars = ["TAVILY_API_KEY", "COHERE_API_KEY", "QDRANT_URL", "QDRANT_API_KEY"]
    
    print("🔍 Checking environment variables...")
    
    missing_required = []
    for var in required_vars:
        if not os.getenv(var):
            missing_required.append(var)
    
    if missing_required:
        print(f"❌ Missing required environment variables: {', '.join(missing_required)}")
        print("   Please set these in your .env file or environment")
        return False
    
    print("✅ Required environment variables found")
    
    available_optional = []
    for var in optional_vars:
        if os.getenv(var):
            available_optional.append(var)
    
    if available_optional:
        print(f"ℹ️  Optional features available: {', '.join(available_optional)}")
    else:
        print("ℹ️  No optional features configured")
    
    return True


def install_dependencies():
    """Install required dependencies using uv."""
    print("📦 Installing dependencies with uv...")
    try:
        subprocess.run(["uv", "sync"], check=True)
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        print("💡 Make sure uv is installed: pip install uv")
        return False


def start_server(host="0.0.0.0", port=8000, reload=True):
    """Start the FastAPI server."""
    print(f"🚀 Starting RAG API server on {host}:{port}")
    
    cmd = [
        "uv", "run", "uvicorn",
        "src.api.main:app",
        "--host", host,
        "--port", str(port)
    ]
    
    if reload:
        cmd.append("--reload")
    
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to start server: {e}")


def test_streaming():
    """Test streaming endpoints."""
    print("🌊 Testing streaming endpoints...")
    try:
        subprocess.run(["uv", "run", "streaming_client.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Streaming test failed: {e}")


def test_llm_streaming():
    """Test LLM streaming."""
    print("🤖 Testing LLM streaming...")
    try:
        subprocess.run(["uv", "run", "streaming_client.py", "llm"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ LLM streaming test failed: {e}")


def interactive_streaming():
    """Start interactive streaming client."""
    print("🎯 Starting interactive streaming client...")
    try:
        subprocess.run(["uv", "run", "streaming_client.py", "interactive"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Interactive streaming client failed: {e}")


def test_api():
    """Test the API endpoints."""
    print("🧪 Testing API endpoints...")
    try:
        subprocess.run(["uv", "run", "client.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ API test failed: {e}")


def interactive_client():
    """Start interactive client."""
    print("🎯 Starting interactive client...")
    try:
        subprocess.run(["uv", "run", "client.py", "interactive"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Interactive client failed: {e}")


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(description="RAG API Server Management")
    parser.add_argument("command", choices=["start", "test", "interactive", "install", "check", "stream", "llm-stream", "stream-interactive"], 
                       help="Command to execute")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to (default: 8000)")
    parser.add_argument("--no-reload", action="store_true", help="Disable auto-reload")
    
    args = parser.parse_args()
    
    # Change to script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    if args.command == "check":
        if check_environment():
            print("✅ Environment check passed")
            sys.exit(0)
        else:
            print("❌ Environment check failed")
            sys.exit(1)
    
    elif args.command == "install":
        if install_dependencies():
            print("✅ Installation completed")
            sys.exit(0)
        else:
            print("❌ Installation failed")
            sys.exit(1)
    
    elif args.command == "start":
        if not check_environment():
            print("❌ Environment check failed, cannot start server")
            sys.exit(1)
        
        start_server(args.host, args.port, not args.no_reload)
    
    elif args.command == "test":
        test_api()
    
    elif args.command == "interactive":
        interactive_client()
    
    elif args.command == "stream":
        test_streaming()
    
    elif args.command == "llm-stream":
        test_llm_streaming()
    
    elif args.command == "stream-interactive":
        interactive_streaming()


if __name__ == "__main__":
    main()

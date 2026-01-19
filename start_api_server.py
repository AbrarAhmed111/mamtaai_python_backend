#!/usr/bin/env python3
"""
Start the template API server from project root
"""
import sys
import os

# Ensure the project root is in the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

if __name__ == "__main__":
    import uvicorn
    
    print("""
╔═══════════════════════════════════════════════════════════╗
║              Backend Template API v1.0.0                  ║
║               Starting from Project Root                  ║
╚═══════════════════════════════════════════════════════════╝

🚀 Starting API server...

Access Points:
  🌐 Main Page             → http://localhost:8000/
  📚 API Documentation     → http://localhost:8000/docs
  💚 Health Check          → http://localhost:8000/health

Press CTRL+C to stop the server
═══════════════════════════════════════════════════════════
    """)
    
    # Get port from environment or use default
    port = int(os.getenv("PORT", 8000))
    
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info"
    )


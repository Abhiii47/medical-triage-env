import os
import sys

# Ensure this directory is in the path
sys.path.append(os.path.dirname(__file__))

from server.app import app

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "7860"))
    uvicorn.run("app:app", host="0.0.0.0", port=port)

import os
import sys
sys.path.append(os.path.dirname(__file__))

from server.app import app

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "7860"))
    uvicorn.run("app:app", host="0.0.0.0", port=port)

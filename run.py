"""
YOO PROJECT — Quick Launcher
Starts the FastAPI Backend and Flagship Web Application.
"""

import sys
import os
import uvicorn

if __name__ == "__main__":
    print("=" * 65)
    print("  YOO PROJECT — AI-Powered Business Intelligence & Analytics")
    print("  Automating Power BI Workflows")
    print("=" * 65)
    print("  Web Application: http://127.0.0.1:8000")
    print("  API Docs:        http://127.0.0.1:8000/docs")
    print("=" * 65)
    print("Starting server...")

    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=False)

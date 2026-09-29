from fastapi import FastAPI

app = FastAPI(title = "Aegis AI Platform")

@app.get("/health")
def health():
    return {"status": "healthy"}
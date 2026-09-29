from fastapi import FastAPI

app = FastAPI(title="Aegis AI Platform")


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/health/ready")
def readiness():
    return {"status": "ready"}


@app.get("/health/live")
def liveness():
    return {"status": "live"}

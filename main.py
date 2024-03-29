from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def root():
    return{"Welcome to LP5":"fastapi for sepsis prediction"}
    

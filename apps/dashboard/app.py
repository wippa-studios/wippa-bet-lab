from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

app = FastAPI(title="Wippa Bet Lab Dashboard")
app.mount("/static", StaticFiles(directory="apps/dashboard/static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def index():
    return open("apps/dashboard/templates/index.html").read()

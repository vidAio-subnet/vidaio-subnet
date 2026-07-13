"""
Lightweight URL accessibility check proxy.

Deploy this as a cloud function (AWS Lambda, GCP Cloud Function, proxy etc.)
on a different IP than the validator. The scoring server calls this
endpoint to verify that a miner's S3 URL is publicly downloadable,
not IP-restricted to just the validator.

Usage:
  GET /?url=<encoded_url>

Response:
  {"accessible": true/false, "status_code": 200}

Set the deployed URL as URL_ACCESSIBILITY_CHECK_ENDPOINT env var
on your scoring server.
"""

import aiohttp
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse

app = FastAPI()


@app.get("/")
async def check_url(url: str = Query(..., description="URL to check accessibility")):
    try:
        timeout = aiohttp.ClientTimeout(total=10)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.head(url) as resp:
                accessible = resp.status in (200, 206)
                return JSONResponse({"accessible": accessible, "status_code": resp.status})
    except aiohttp.ClientError as e:
        return JSONResponse({"accessible": False, "status_code": -1, "error": str(e)})
    except Exception as e:
        return JSONResponse({"accessible": False, "status_code": -1, "error": str(e)})

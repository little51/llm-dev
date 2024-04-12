import asyncio
import json
import time
import os
import aiohttp_cors
import requests
import argparse
from aiohttp import web
from generate import generate_text


async def generate(request):
    params = await request.json()
    context = params["context"]
    if len(context.strip()) == 0:
        return web.Response(
            content_type="application/json",
            text=json.dumps(
                {"result": "[]", "time": 0}
            ),
        )
    maxlength = params["maxlength"]
    samples = params["samples"]
    if samples == 0:
        samples = 1
    start = time.perf_counter()
    result = generate_text(context, maxlength, samples)
    end = time.perf_counter()
    return web.Response(
        content_type="application/json",
        text=json.dumps(
            {"result": result, "time": end-start}
        ),
    )

app = web.Application()
cors = aiohttp_cors.setup(app)
app.router.add_post("/generate", generate)

for route in list(app.router.routes()):
    cors.add(route, {
        "*": aiohttp_cors.ResourceOptions(
            allow_credentials=True,
            expose_headers="*",
            allow_headers="*",
            allow_methods="*"
        )
    })

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', default=5005,
                        type=int, required=False)
    args = parser.parse_args()
    print("Start web server")
    web.run_app(
        app, access_log=None, host="0.0.0.0",
        port=args.port,
        ssl_context=None
    )

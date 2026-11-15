"""
MiniCPM-V-4.5 Hand Tracking Server (GGUF + llama.cpp version)
Optimized for CPU inference with INT4 quantization using persistent llama-server

IMPORTANT: Configure your model paths below before running!
"""

import re
import io
import time
import base64
import subprocess
import os
from typing import Optional, List
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from pydantic import BaseModel
import uvicorn
import requests


# ============ CONFIGURATION - UPDATE THESE PATHS ============
# TODO: Update these paths to point to your local GGUF models
MODEL_PATH = "/path/to/your/ggml-model-Q4_K_M.gguf"
MMPROJ_PATH = "/path/to/your/mmproj-model-f16.gguf"
LLAMA_SERVER_PATH = "/path/to/your/llama-server"
# ============================================================


# Response model
class DetectionResponse(BaseModel):
    success: bool
    bbox: Optional[List[int]] = None
    error: Optional[str] = None
    inference_time: float
    prompt: str


class LlamaServerProcess:
    """
    Persistent llama.cpp HTTP server wrapper.

    Why: spawning llama-mtmd-cli per request reloads the GGUF every time (very slow).
    With llama-server, the model stays loaded and requests become fast.
    """

    def __init__(
        self,
        model_path: str,
        mmproj_path: str,
        llama_server_path: str,
        host: str = "127.0.0.1",
        port: int = 8085,
        alias: str = "minicpm-v4_5-gguf",
    ):
        self.model_path = model_path
        self.mmproj_path = mmproj_path
        self.llama_server_path = llama_server_path
        self.host = host
        self.port = port
        self.alias = alias

        self._proc: Optional[subprocess.Popen] = None
        self._ready = False

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def start(self) -> None:
        if self._proc and self._proc.poll() is None:
            return

        # Verify files exist
        for path, name in [
            (self.model_path, "Language model"),
            (self.mmproj_path, "Vision model"),
            (self.llama_server_path, "llama-server"),
        ]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"{name} not found at: {path}")

        # Server params (override via env vars)
        ctx_size = os.getenv("LLAMA_CTX_SIZE", "4096")  # Increased from 2048
        threads = os.getenv("LLAMA_THREADS")  # if unset, llama.cpp decides (-1)
        n_parallel = os.getenv("LLAMA_SERVER_PARALLEL", "1")
        no_mmap = os.getenv("LLAMA_NO_MMAP", "0") in {"1", "true", "on", "yes"}

        cmd = [
            self.llama_server_path,
            "-m",
            self.model_path,
            "-mm",
            self.mmproj_path,
            "--host",
            self.host,
            "--port",
            str(self.port),
            "-c",
            str(ctx_size),
            "-np",
            str(n_parallel),
            "-a",
            self.alias,
            "--no-webui",
            "--reasoning-budget", "0",
        ]
        if threads:
            cmd.extend(["-t", threads])
        if no_mmap:
            cmd.append("--no-mmap")

        # Keep logs from blocking the process: default to DEVNULL.
        # Set LLAMA_SERVER_LOG=1 to inherit stdout/stderr for debugging.
        if os.getenv("LLAMA_SERVER_LOG", "0") in {"1", "true", "on", "yes"}:
            stdout = None
            stderr = None
        else:
            stdout = subprocess.DEVNULL
            stderr = subprocess.DEVNULL

        self._proc = subprocess.Popen(cmd, stdout=stdout, stderr=stderr)
        self._ready = False

    def stop(self) -> None:
        self._ready = False
        if not self._proc:
            return
        if self._proc.poll() is None:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=5)
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass
        self._proc = None

    def wait_ready(self, timeout_s: float = 180.0) -> None:
        """
        Wait until llama-server is ready (GET /health -> 200).
        """
        self.start()
        t0 = time.time()
        last_err = None
        while time.time() - t0 < timeout_s:
            if self._proc and self._proc.poll() is not None:
                raise RuntimeError("llama-server exited unexpectedly during startup")
            try:
                r = requests.get(f"{self.base_url}/health", timeout=2)
                if r.status_code == 200:
                    self._ready = True
                    return
            except Exception as e:
                last_err = e
            time.sleep(0.5)
        raise TimeoutError(f"llama-server not ready after {timeout_s:.0f}s (last error: {last_err})")

    def is_ready(self) -> bool:
        if self._ready:
            return True
        try:
            r = requests.get(f"{self.base_url}/health", timeout=1)
            self._ready = r.status_code == 200
        except Exception:
            self._ready = False
        return self._ready

    def chat_completion(self, question: str, image_jpeg_bytes: bytes) -> str:
        """
        Call llama-server multimodal chat completion endpoint.
        Returns assistant message content (best-effort across response shapes).
        """
        if not self.is_ready():
            self.wait_ready()

        # Generation knobs
        n_predict = int(os.getenv("LLAMA_N_PREDICT", "1024"))
        temperature = float(os.getenv("LLAMA_TEMPERATURE", "0.7"))
        top_p = float(os.getenv("LLAMA_TOP_P", "0.8"))
        top_k = int(os.getenv("LLAMA_TOP_K", "100"))
        repeat_penalty = float(os.getenv("LLAMA_REPEAT_PENALTY", "1.05"))

        b64 = base64.b64encode(image_jpeg_bytes).decode("ascii")
        data_url = f"data:image/jpeg;base64,{b64}"

        payload = {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "n_predict": n_predict,
            "repeat_penalty": repeat_penalty,
            "stream": False,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                },
            ],
        }

        r = requests.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            timeout=180,
        )
        if r.status_code != 200:
            raise RuntimeError(f"llama-server error {r.status_code}: {r.text[:300]}")
        j = r.json()

        # Debug: Print the full response structure
        print(f"[DEBUG] Full llama-server JSON response: {j}")

        # Typical shape: {"choices":[{"message":{"role":"assistant","content":"..."}}]}
        choice0 = (j.get("choices") or [{}])[0]
        msg = choice0.get("message") or {}
        content = msg.get("content")

        if isinstance(content, str):
            return content

        # Some servers/clients may represent content as a list of parts.
        if isinstance(content, list):
            parts: list[str] = []
            for part in content:
                if isinstance(part, dict):
                    t = part.get("text")
                    if isinstance(t, str):
                        parts.append(t)
            return "".join(parts).strip()

        # Fallback: sometimes content can be missing; return empty string and let caller log JSON.
        return ""


class TrackingServerGGUF:
    def __init__(
        self, 
        model_path: str,
        mmproj_path: str,
        llama_server_path: str,
    ):
        """
        Initialize the tracking server with GGUF model.
        """
        print("=" * 60)
        print("MiniCPM-V-4.5 Hand Tracking Server (GGUF INT4)")
        print("=" * 60)
        
        self.model_path = model_path
        self.mmproj_path = mmproj_path
        self.llama_server_path = llama_server_path

        llama_host = os.getenv("LLAMA_SERVER_HOST", "127.0.0.1")
        llama_port = int(os.getenv("LLAMA_SERVER_PORT", "8085"))
        llama_alias = os.getenv("LLAMA_SERVER_ALIAS", "minicpm-v4_5-gguf")
        self.llama = LlamaServerProcess(
            model_path=self.model_path,
            mmproj_path=self.mmproj_path,
            llama_server_path=self.llama_server_path,
            host=llama_host,
            port=llama_port,
            alias=llama_alias,
        )
        
        # Verify files exist
        for path, name in [
            (model_path, "Language model"),
            (mmproj_path, "Vision model"),
            (llama_server_path, "llama-server"),
        ]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"{name} not found at: {path}")
            print(f"[OK] {name}: {path}")
        
        # Statistics
        self.total_requests = 0
        self.total_inference_time = 0.0
        self.successful_detections = 0
        
        print("\n" + "=" * 60)
        print("Server ready! Using persistent llama-server")
        print("=" * 60 + "\n")
    
    def _extract_bbox_from_response(self, response: str) -> Optional[List[int]]:
        """Extract bounding box from model response."""
        # Try <box> tags first
        match = re.search(r"<box>(.*?)</box>", response, flags=re.DOTALL)
        if match:
            bbox_str = match.group(1)
            nums = re.findall(r"[-+]?\d*\.?\d+", bbox_str)
            bbox = [int(round(float(x))) for x in nums]
            if len(bbox) == 4:
                return bbox
        
        # Fallback: extract any 4 consecutive numbers from response
        nums = re.findall(r"\d+", response)
        if len(nums) >= 4:
            bbox = [int(nums[i]) for i in range(4)]
            return bbox
        
        return None
    
    def detect(self, image: Image.Image, prompt: str) -> DetectionResponse:
        """
        Run object detection using persistent llama.cpp server.
        """
        start_time = time.time()
        
        try:
            # Prepare grounding question
            question = f'Please provide the bounding box coordinate of the region this sentence describes: <ref>{prompt}</ref>. Return format: <box>x1 y1 x2 y2</box>'

            # Encode image to JPEG bytes (in-memory)
            buf = io.BytesIO()
            image.save(buf, format="JPEG", quality=90)
            image_bytes = buf.getvalue()

            # Call persistent llama-server
            response = self.llama.chat_completion(question=question, image_jpeg_bytes=image_bytes).strip()
            inference_time = time.time() - start_time
            if response:
                print(f"[DEBUG] llama-server response ({len(response)} chars): {response[:300]}")
            else:
                # If we get an empty response, provide actionable diagnostics.
                print("[ERROR] Empty response from llama-server (no assistant content)")
                return DetectionResponse(
                    success=False,
                    error="Empty response from model",
                    inference_time=inference_time,
                    prompt=prompt,
                )
            
            # Extract bbox
            bbox = self._extract_bbox_from_response(response)
            
            # Update statistics
            self.total_requests += 1
            self.total_inference_time += inference_time
            
            if bbox:
                self.successful_detections += 1
                print(f"[DETECTION] Success - Prompt: '{prompt}' | BBox: {bbox} | Time: {inference_time:.2f}s")
                return DetectionResponse(
                    success=True,
                    bbox=bbox,
                    inference_time=inference_time,
                    prompt=prompt
                )
            else:
                print(f"[DETECTION] Not found - Prompt: '{prompt}' | Response: {response[:200]}... | Time: {inference_time:.2f}s")
                return DetectionResponse(
                    success=False,
                    error=f"Object not found in image. Model response: {response[:200]}",
                    inference_time=inference_time,
                    prompt=prompt
                )
        
        except Exception as e:
            print(f"[ERROR] {str(e)}")
            return DetectionResponse(
                success=False,
                error=str(e),
                inference_time=time.time() - start_time,
                prompt=prompt
            )
    
    def get_stats(self) -> dict:
        avg_time = (self.total_inference_time / self.total_requests) if self.total_requests > 0 else 0
        success_rate = (self.successful_detections / self.total_requests * 100) if self.total_requests > 0 else 0
        
        return {
            "total_requests": self.total_requests,
            "successful_detections": self.successful_detections,
            "success_rate": f"{success_rate:.1f}%",
            "avg_inference_time": f"{avg_time:.2f}s",
            "device": "CPU (INT4 GGUF)"
        }


# ============ FastAPI App ============

print("Initializing GGUF tracking server...")
tracking_server = TrackingServerGGUF(
    model_path=MODEL_PATH,
    mmproj_path=MMPROJ_PATH,
    llama_server_path=LLAMA_SERVER_PATH,
)

app = FastAPI(
    title="MiniCPM-V-4.5 Hand Tracking Server (GGUF)",
    description="Fast inference with persistent llama.cpp server (GGUF)",
    version="2.0.0"
)

@app.on_event("startup")
async def _startup():
    # Start llama-server once and keep it loaded
    try:
        print("[STARTUP] Starting persistent llama-server...")
        tracking_server.llama.wait_ready(timeout_s=float(os.getenv("LLAMA_SERVER_STARTUP_TIMEOUT", "180")))
        print(f"[STARTUP] llama-server is ready at {tracking_server.llama.base_url}")
    except Exception as e:
        # Don't crash FastAPI startup; /health will reflect not-ready and /detect will error.
        print(f"[STARTUP ERROR] Failed to start llama-server: {e}")


@app.on_event("shutdown")
async def _shutdown():
    print("[SHUTDOWN] Stopping llama-server...")
    tracking_server.llama.stop()


@app.get("/")
async def root():
    return {
        "status": "online",
        "message": "MiniCPM-V-4.5 Hand Tracking Server (GGUF)",
        "quantization": "INT4 (Q4_K_M)",
        "llama_server": tracking_server.llama.base_url,
    }


@app.get("/health")
async def health():
    llama_ready = tracking_server.llama.is_ready()
    return {
        "status": "healthy" if llama_ready else "starting",
        "model_loaded": llama_ready,
        "quantization": "INT4",
        "llama_server": tracking_server.llama.base_url,
    }


@app.get("/stats")
async def stats():
    return tracking_server.get_stats()


@app.post("/detect", response_model=DetectionResponse)
async def detect(
    image: UploadFile = File(...),
    prompt: str = Form(...)
):
    try:
        image_bytes = await image.read()
        pil_image = Image.open(io.BytesIO(image_bytes))
        
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        result = tracking_server.detect(pil_image, prompt)
        return result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def main():
    import argparse
    parser = argparse.ArgumentParser(description="MiniCPM-V-4.5 Hand Tracking Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"Starting server on {args.host}:{args.port}")
    print(f"Access the API at: http://localhost:{args.port}")
    print(f"API docs available at: http://localhost:{args.port}/docs")
    print(f"{'='*60}\n")
    
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

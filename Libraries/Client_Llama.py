# Libraries/Client_Llama.py

import json
import time
import requests
from typing import Optional, Dict, Any, List

# ==============================

class LocalLlamaClient:
    def __init__(self, 
                 host: str = "http://localhost:8080", 
                 timeout: int = 270,
                 retry: int = 3, 
                 wait_timeout: int = 10):
        
        self.host = host.rstrip("/")
        self.timeout = timeout 
        self.health_timeout = 5
        self.retry = retry

        # chọn endpoint mặc định, có thể cập nhật sau health-check
        self._completion_endpoint = "/completion"    # llama.cpp server
        self._alt_endpoints = ["/v1/completions", "/v1/chat/completions"]

        self.wait_for_server_ready(wait_timeout)
        
    def wait_for_server_ready(self, wait_timeout: int):
        """
        Hỏi thăm (poll) endpoint /health cho đến khi server "ready"
        hoặc hết thời gian chờ (wait_timeout).
        Chấp nhận {"status":"ok"} hoặc {"ready":true} hoặc HTTP 200 với text "ok".
        """
        start_time = time.time()
        url = f"{self.host}/health"
        
        print(f"⏳ Đang kiểm tra Llama server tại {self.host}...")
        
        while True:
            elapsed = time.time() - start_time
            if elapsed > wait_timeout:
                print(f"❌ Server không sẵn sàng sau {wait_timeout} giây.")
                raise TimeoutError(f"Server tại {self.host} không sẵn sàng sau {wait_timeout}s.")

            try:
                res = requests.get(url, timeout=self.health_timeout)
                ok = False
                try:
                    data = res.json()
                    status = str(data.get("status", "")).lower()
                    ready = bool(data.get("ready", False))
                    if status == "ok" or ready:
                        ok = True
                except json.JSONDecodeError:
                    if res.status_code == 200 and "ok" in res.text.lower():
                        ok = True

                if ok:
                    print(f"✅ Server đã sẵn sàng.")
                    break
                else:
                    print(f"⏳ Server chưa sẵn sàng (HTTP {res.status_code}). Thử lại sau 5s...")

            except requests.exceptions.ConnectionError:
                print(f"⏳ Đang chờ kết nối đến server tại {self.host}... Thử lại sau 5s...")
            
            except requests.exceptions.RequestException as e:
                print(f"⚠️ Lỗi health check: {e}. Thử lại sau 5s...")

            time.sleep(5)

    def _post(self, endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Gửi yêu cầu POST và xử lý retry + fallback endpoint."""
        tried = [endpoint] + [ep for ep in self._alt_endpoints if ep != endpoint]

        for ep in tried:
            url = f"{self.host}{ep}"
            for attempt in range(self.retry):
                try:
                    res = requests.post(url, json=payload, timeout=self.timeout) 

                    if res.status_code == 200:
                        try:
                            return res.json()
                        except json.JSONDecodeError:
                            return {"error": f"INVALID_JSON_RESPONSE at {ep}"}

                    print(f"[LLAMA API WARNING] HTTP {res.status_code} at {ep}: {res.text}")
                    # Nếu 404, thử endpoint khác ngay
                    if res.status_code == 404:
                        break
                    time.sleep(1)

                except requests.exceptions.RequestException as e:
                    print(f"[LLAMA API ERROR] {str(e)} (endpoint {ep})")
                    time.sleep(1)

        return {"error": "LLAMA_REQUEST_FAILED"}

    def reset(self):
        """Reset model session / KV cache trên llama.cpp (nếu hỗ trợ)."""
        url = f"{self.host}/reset"
        try:
            res = requests.post(url, timeout=10)
            if res.status_code == 200:
                print("🧹 Phiên đã reset (KV cache cleared)")
            else:
                print(f"⚠️ Reset lỗi: HTTP {res.status_code} → {res.text}")
        except Exception as e:
            print(f"❌ Không reset được Llama server: {e}")

    def __call__(self,
                 prompt: str,
                 max_tokens: int = 512,
                 temperature: float = 0.7,
                 top_p: float = 0.9,
                 stop: Optional[List[str]] = None,
                 grammar: Optional[str] = None,
                 json_mode: bool = False) -> Dict[str, Any]:
        
        payload = {
            "prompt": prompt,
            "stream": False,
            "n_predict": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }

        if stop:
            payload["stop"] = stop
        
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        elif grammar:
            payload["grammar"] = grammar
            
        data = self._post(self._completion_endpoint, payload)

        if "error" in data:
            return {"choices": [{"text": f"[LLAMA_ERROR] {data['error']}"}]}
        
        # Chuẩn hóa theo schema llama.cpp server
        content = data.get("content", "")
        if not content and isinstance(data.get("choices"), list):
            # openai-like
            try:
                content = data["choices"][0]["message"]["content"]
            except Exception:
                try:
                    content = data["choices"][0].get("text","")
                except Exception:
                    content = ""

        return {
            "choices": [
                {"text": content}
            ]
        }

"""
Health checks for all system components.
Pure Python, no LLM. Returns structured status for each component.
"""
import asyncio
import os
import shutil
import time
import psutil
import structlog
import httpx
from redis.asyncio import Redis

from config.settings import get_settings

logger = structlog.get_logger()


class HealthChecks:
    def __init__(self):
        self.settings = get_settings()

    # === Server Resources ===

    async def check_cpu(self) -> dict:
        cpu_percent = psutil.cpu_percent(interval=1)
        return {
            "name": "CPU",
            "status": "critical" if cpu_percent > 90 else "warning" if cpu_percent > 75 else "ok",
            "value": f"{cpu_percent}%",
        }

    async def check_memory(self) -> dict:
        mem = psutil.virtual_memory()
        return {
            "name": "RAM",
            "status": "critical" if mem.percent > 90 else "warning" if mem.percent > 80 else "ok",
            "value": f"{mem.percent}% ({mem.used // (1024**3)}/{mem.total // (1024**3)} GB)",
        }

    async def check_disk(self) -> dict:
        disk = shutil.disk_usage("/")
        used_pct = (disk.used / disk.total) * 100
        free_gb = disk.free / (1024**3)
        return {
            "name": "Disk",
            "status": "critical" if free_gb < 2 else "warning" if free_gb < 5 else "ok",
            "value": f"{used_pct:.1f}% used, {free_gb:.1f} GB free",
        }

    # === Services ===

    async def check_gateway(self) -> dict:
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(f"http://127.0.0.1:{self.settings.gateway_port}/health")
                if resp.status_code == 200:
                    return {"name": "Gateway API", "status": "ok", "value": "HTTP 200"}
                return {"name": "Gateway API", "status": "critical", "value": f"HTTP {resp.status_code}"}
        except Exception as e:
            return {"name": "Gateway API", "status": "critical", "value": str(e)[:80]}

    async def check_redis(self) -> dict:
        try:
            r = Redis.from_url(self.settings.redis_url, decode_responses=True)
            pong = await r.ping()
            await r.close()
            return {"name": "Redis", "status": "ok" if pong else "critical", "value": "PONG" if pong else "no response"}
        except Exception as e:
            return {"name": "Redis", "status": "critical", "value": str(e)[:80]}

    async def check_postgres(self) -> dict:
        try:
            import asyncpg
            conn = await asyncpg.connect(
                host="127.0.0.1", port=5432,
                user="salesbot", password="salesbot_secure_2026",
                database="salesbot", timeout=5,
            )
            result = await conn.fetchval("SELECT 1")
            await conn.close()
            return {"name": "PostgreSQL", "status": "ok" if result == 1 else "critical", "value": "connected"}
        except Exception as e:
            return {"name": "PostgreSQL", "status": "critical", "value": str(e)[:80]}

    async def check_docker_containers(self) -> list[dict]:
        """Check Docker containers via Docker Engine API (unix socket)."""
        results = []
        expected = {"salesbot-postgres", "salesbot-redis", "salesbot-webui"}
        found = set()
        try:
            transport = httpx.AsyncHTTPTransport(uds="/var/run/docker.sock")
            async with httpx.AsyncClient(transport=transport, timeout=5) as client:
                resp = await client.get("http://docker/containers/json?all=true")
                containers = resp.json()
                for c in containers:
                    name = c.get("Names", [""])[0].lstrip("/")
                    if name not in expected:
                        continue
                    found.add(name)
                    state = c.get("State", "unknown")
                    status_text = c.get("Status", "")
                    is_healthy = "healthy" in status_text.lower()
                    if state == "running" and is_healthy:
                        s = "ok"
                    elif state == "running":
                        s = "warning"
                    else:
                        s = "critical"
                    results.append({"name": f"Docker:{name}", "status": s, "value": status_text})
            for missing in expected - found:
                results.append({"name": f"Docker:{missing}", "status": "critical", "value": "NOT RUNNING"})
        except Exception as e:
            results.append({"name": "Docker", "status": "warning", "value": f"Cannot check: {str(e)[:60]}"})
        return results

    async def check_systemd_services(self) -> list[dict]:
        """Check systemd services via checking if process is alive."""
        results = []
        services = {
            "salesbot-gateway": 8000,
            "salesbot-telegram": None,
            "salesbot-repair": None,
        }
        for svc, port in services.items():
            if port:
                # Check by port
                try:
                    async with httpx.AsyncClient(timeout=3) as client:
                        resp = await client.get(f"http://127.0.0.1:{port}/health")
                        results.append({
                            "name": f"Service:{svc}",
                            "status": "ok" if resp.status_code == 200 else "critical",
                            "value": f"port {port} responding",
                        })
                except Exception:
                    results.append({"name": f"Service:{svc}", "status": "critical", "value": f"port {port} not responding"})
            else:
                # Check by process name
                found = False
                search_term = {
                    "salesbot-telegram": "interfaces.telegram.bot",
                    "salesbot-repair": "agents.repair.agent",
                }
                term = search_term.get(svc, svc)
                for proc in psutil.process_iter(['cmdline']):
                    try:
                        cmdline = " ".join(proc.info.get('cmdline') or [])
                        if term in cmdline:
                            found = True
                            break
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                results.append({
                    "name": f"Service:{svc}",
                    "status": "ok" if found else "critical",
                    "value": "running" if found else "process not found",
                })
        return results

    async def check_dispatcher_agents(self) -> list[dict]:
        """Check agent registry via Gateway."""
        results = []
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(f"http://127.0.0.1:{self.settings.gateway_port}/api/dispatcher/status")
                data = resp.json()
                for name, info in data.get("registry", {}).get("agents", {}).items():
                    results.append({
                        "name": f"Agent:{name}",
                        "status": "ok" if info["status"] == "online" else "info",
                        "value": f"{info['status']} (errors: {info['error_count']})",
                    })
        except Exception as e:
            results.append({"name": "Dispatcher", "status": "critical", "value": str(e)[:80]})
        return results

    # === Full Check ===

    async def run_all(self) -> list[dict]:
        results = []
        server_checks = await asyncio.gather(
            self.check_cpu(), self.check_memory(), self.check_disk(),
            return_exceptions=True,
        )
        for r in server_checks:
            results.append(r if isinstance(r, dict) else {"name": "check", "status": "critical", "value": str(r)[:80]})

        service_checks = await asyncio.gather(
            self.check_gateway(), self.check_redis(), self.check_postgres(),
            return_exceptions=True,
        )
        for r in service_checks:
            results.append(r if isinstance(r, dict) else {"name": "check", "status": "critical", "value": str(r)[:80]})

        docker_results = await self.check_docker_containers()
        results.extend(docker_results)

        svc_results = await self.check_systemd_services()
        results.extend(svc_results)

        agent_results = await self.check_dispatcher_agents()
        results.extend(agent_results)

        return results

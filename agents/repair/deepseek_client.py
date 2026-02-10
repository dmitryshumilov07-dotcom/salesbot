"""
DeepSeek Reasoner Client - API client for DeepSeek AI models.

Used for code review and analysis before creating repair tasks.
Provides intelligent analysis of code issues and suggested fixes.
"""
import asyncio
import time
from typing import Optional
import structlog
import httpx

from config.settings import get_settings

logger = structlog.get_logger()

DEEPSEEK_API_BASE = "https://api.deepseek.com/v1"


class DeepSeekClient:
    """Client for DeepSeek API for code analysis and review."""
    
    def __init__(self):
        self.settings = get_settings()
        self.api_key = getattr(self.settings, "deepseek_api_key", "")
        self._request_count: int = 0
        self._last_request_time: Optional[float] = None
        self._total_tokens_used: int = 0
        self._error_count: int = 0
        
        if not self.api_key:
            logger.warning("deepseek_api_key_not_configured",
                          message="DeepSeek API key is not set. Code review will be skipped.")
    
    async def analyze_code(
        self,
        code: str,
        context: str = "",
        language: str = "python",
    ) -> dict:
        """
        Analyze code for issues and improvements.
        
        Args:
            code: Source code to analyze
            context: Additional context about the code
            language: Programming language
            
        Returns:
            Analysis result with issues and suggestions
        """
        if not self.api_key:
            return {
                "success": False,
                "error": "DeepSeek API key not configured",
                "issues": [],
            }
        
        prompt = self._build_analysis_prompt(code, context, language)
        
        try:
            response = await self._chat(
                messages=[
                    {"role": "system", "content": "You are a senior code reviewer. Analyze the code for bugs, security issues, and improvements."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
            )
            
            if response.get("success"):
                return {
                    "success": True,
                    "analysis": response.get("content", ""),
                    "tokens_used": response.get("tokens_used", 0),
                }
            else:
                return {
                    "success": False,
                    "error": response.get("error", "Unknown error"),
                    "issues": [],
                }
        except Exception as e:
            self._error_count += 1
            logger.error("deepseek_analyze_error", error=str(e))
            return {
                "success": False,
                "error": str(e),
                "issues": [],
            }
    
    async def review_fix(
        self,
        problem_description: str,
        proposed_fix: str,
        affected_files: list[str] | None = None,
    ) -> dict:
        """
        Review a proposed fix before applying.
        
        Args:
            problem_description: Description of the problem
            proposed_fix: The fix to be applied
            affected_files: List of files that will be modified
            
        Returns:
            Review result with verdict (approve/reject) and reasoning
        """
        if not self.api_key:
            # If API key not configured, auto-approve to allow repairs to proceed
            return {
                "success": True,
                "verdict": "approve",
                "reason": "DeepSeek review skipped - API key not configured",
                "risk_level": "unknown",
            }
        
        prompt = f"""Review this proposed code fix:

PROBLEM:
{problem_description}

PROPOSED FIX:
{proposed_fix}

AFFECTED FILES:
{', '.join(affected_files or ['unknown'])}

Analyze:
1. Will this fix solve the problem?
2. Could it introduce new bugs or security issues?
3. Is the fix minimal and targeted?

Respond with:
- VERDICT: approve or reject
- RISK_LEVEL: low, medium, high, critical
- REASON: Brief explanation
"""
        
        try:
            response = await self._chat(
                messages=[
                    {"role": "system", "content": "You are a code review expert. Evaluate fixes for correctness and safety."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
            )
            
            if response.get("success"):
                content = response.get("content", "")
                return self._parse_review_response(content)
            else:
                return {
                    "success": False,
                    "verdict": "reject",
                    "reason": f"Review failed: {response.get('error')}",
                    "risk_level": "unknown",
                }
        except Exception as e:
            self._error_count += 1
            logger.error("deepseek_review_error", error=str(e))
            return {
                "success": False,
                "verdict": "reject",
                "reason": f"Review error: {e}",
                "risk_level": "unknown",
            }
    
    async def _chat(
        self,
        messages: list[dict],
        temperature: float = 0.3,
        max_tokens: int = 2000,
    ) -> dict:
        """Send chat request to DeepSeek API."""
        self._request_count += 1
        self._last_request_time = time.time()
        
        try:
            async with httpx.AsyncClient(timeout=60) as client:
                response = await client.post(
                    f"{DEEPSEEK_API_BASE}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": "deepseek-reasoner",
                        "messages": messages,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                    },
                )
                
                if response.status_code == 200:
                    data = response.json()
                    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                    tokens = data.get("usage", {}).get("total_tokens", 0)
                    self._total_tokens_used += tokens
                    
                    logger.info("deepseek_chat_success", tokens=tokens)
                    return {
                        "success": True,
                        "content": content,
                        "tokens_used": tokens,
                    }
                else:
                    error_msg = f"HTTP {response.status_code}: {response.text[:200]}"
                    logger.error("deepseek_chat_error", error=error_msg)
                    return {
                        "success": False,
                        "error": error_msg,
                    }
        except Exception as e:
            logger.error("deepseek_chat_exception", error=str(e))
            return {
                "success": False,
                "error": str(e),
            }
    
    def _build_analysis_prompt(self, code: str, context: str, language: str) -> str:
        """Build prompt for code analysis."""
        prompt_parts = [
            f"Analyze this {language} code for issues:",
            "",
            "```" + language,
            code,
            "```",
        ]
        
        if context:
            prompt_parts.extend(["", "Context:", context])
        
        prompt_parts.extend([
            "",
            "Look for:",
            "1. Bugs and logic errors",
            "2. Security vulnerabilities",
            "3. Performance issues",
            "4. Code quality problems",
            "",
            "Format each issue as:",
            "- LINE: <line number>",
            "- SEVERITY: critical/high/medium/low",
            "- ISSUE: <description>",
            "- FIX: <suggested fix>",
        ])
        
        return "\n".join(prompt_parts)
    
    def _parse_review_response(self, content: str) -> dict:
        """Parse DeepSeek review response."""
        content_lower = content.lower()
        
        # Extract verdict
        verdict = "approve"
        if "reject" in content_lower or "verdict: reject" in content_lower:
            verdict = "reject"
        
        # Extract risk level
        risk_level = "medium"
        for level in ["critical", "high", "medium", "low"]:
            if f"risk_level: {level}" in content_lower or f"risk level: {level}" in content_lower:
                risk_level = level
                break
        
        # Extract reason (first line after REASON: or just use content)
        reason = content[:200]
        if "reason:" in content_lower:
            reason_start = content_lower.find("reason:") + 7
            reason_end = content.find("\n", reason_start)
            if reason_end == -1:
                reason_end = len(content)
            reason = content[reason_start:reason_end].strip()
        
        return {
            "success": True,
            "verdict": verdict,
            "reason": reason,
            "risk_level": risk_level,
        }
    
    def get_stats(self) -> dict:
        """
        Return usage statistics.
        
        Returns:
            Dictionary with request count, last request time, tokens used, errors
        """
        # FIX: Complete implementation of get_stats method
        # This method was truncated in the original code, causing SyntaxError
        return {
            "total_requests": self._request_count,
            "last_request_time": self._last_request_time,
            "total_tokens_used": self._total_tokens_used,
            "error_count": self._error_count,
            "api_configured": bool(self.api_key),
        }
    
    def reset_stats(self) -> None:
        """Reset usage statistics."""
        self._request_count = 0
        self._last_request_time = None
        self._total_tokens_used = 0
        self._error_count = 0


# Singleton instance
_client: Optional[DeepSeekClient] = None


def get_deepseek_client() -> DeepSeekClient:
    """Get or create DeepSeekClient singleton."""
    global _client
    if _client is None:
        _client = DeepSeekClient()
    return _client

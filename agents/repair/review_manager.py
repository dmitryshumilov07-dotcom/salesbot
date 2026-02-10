"""
Review Manager - Orchestrates code analysis and repair creation.

Manages the workflow:
1. Run code analysis to find issues
2. Check daily limits
3. Build problem descriptions
4. Get DeepSeek review
5. Launch Cursor agent for fixes
6. Track PR creation

Follows Single Responsibility Principle with helper methods.
"""
import asyncio
import time
from datetime import datetime, date
from typing import Optional
from dataclasses import dataclass, field
import structlog

from config.settings import get_settings
from agents.repair.code_scanner import CodeScanner, Finding, FindingSeverity, get_code_scanner
from agents.repair.deepseek_client import DeepSeekClient, get_deepseek_client
from agents.repair.cursor_engine import CursorEngine, get_cursor_engine

logger = structlog.get_logger()

# Daily limits to prevent runaway repairs
MAX_DAILY_PRS = 5
MAX_DAILY_REVIEWS = 20


@dataclass
class AnalysisResult:
    """Result of code analysis run."""
    status: str  # completed, failed, no_issues
    findings_count: int = 0
    critical_count: int = 0
    high_count: int = 0
    trigger: str = ""
    duration: float = 0.0
    error: Optional[str] = None
    
    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "findings_count": self.findings_count,
            "critical_count": self.critical_count,
            "high_count": self.high_count,
            "trigger": self.trigger,
            "duration": self.duration,
            "error": self.error,
        }


class ReviewManager:
    """Manages code review and repair creation workflow."""
    
    def __init__(self):
        self.settings = get_settings()
        self.scanner = get_code_scanner()
        self.deepseek = get_deepseek_client()
        self.cursor = get_cursor_engine()
        
        # Daily counters (reset at midnight)
        self._daily_pr_count: int = 0
        self._daily_review_count: int = 0
        self._last_reset_date: Optional[date] = None
        
        # Analysis history
        self._analysis_history: list[AnalysisResult] = []
        self._last_analysis_time: Optional[float] = None
    
    def _reset_daily_counters_if_needed(self) -> None:
        """Reset daily counters if it's a new day."""
        today = date.today()
        if self._last_reset_date != today:
            self._daily_pr_count = 0
            self._daily_review_count = 0
            self._last_reset_date = today
            logger.info("review_manager_daily_reset")
    
    async def run_analysis(self, trigger: str = "manual") -> AnalysisResult:
        """
        Run code analysis and return results.
        
        Args:
            trigger: What triggered the analysis (manual, scheduled, monitoring)
            
        Returns:
            AnalysisResult with findings summary
        """
        start_time = time.time()
        logger.info("review_manager_analysis_start", trigger=trigger)
        
        try:
            findings = await self.scanner.scan_directory()
            duration = time.time() - start_time
            
            if not findings:
                result = AnalysisResult(
                    status="no_issues",
                    trigger=trigger,
                    duration=duration,
                )
            else:
                critical = len([f for f in findings if f.severity == FindingSeverity.CRITICAL])
                high = len([f for f in findings if f.severity == FindingSeverity.HIGH])
                
                result = AnalysisResult(
                    status="completed",
                    findings_count=len(findings),
                    critical_count=critical,
                    high_count=high,
                    trigger=trigger,
                    duration=duration,
                )
            
            self._analysis_history.append(result)
            if len(self._analysis_history) > 100:
                self._analysis_history = self._analysis_history[-100:]
            
            self._last_analysis_time = time.time()
            
            logger.info("review_manager_analysis_complete",
                       status=result.status,
                       findings=result.findings_count,
                       critical=result.critical_count,
                       high=result.high_count)
            
            return result
            
        except Exception as e:
            duration = time.time() - start_time
            logger.error("review_manager_analysis_error", error=str(e))
            return AnalysisResult(
                status="failed",
                trigger=trigger,
                duration=duration,
                error=str(e),
            )
    
    # FIX: Refactored create_fix_task into smaller helper methods
    # Original function was 94 lines and violated single responsibility principle
    
    def _check_daily_limit(self) -> dict:
        """
        Check if daily PR limit has been reached.
        
        Returns:
            dict with success=True if under limit, or error message if exceeded
        """
        self._reset_daily_counters_if_needed()
        
        if self._daily_pr_count >= MAX_DAILY_PRS:
            logger.warning("review_manager_daily_limit_reached",
                          count=self._daily_pr_count,
                          limit=MAX_DAILY_PRS)
            return {
                "success": False,
                "error": f"Daily PR limit reached ({MAX_DAILY_PRS}). Try again tomorrow.",
            }
        
        return {"success": True}
    
    def _build_problem_description(
        self,
        findings: list[dict],
        admin_comment: str = "",
    ) -> tuple[str, list[str]]:
        """
        Build problem description from findings.
        
        Args:
            findings: List of finding dictionaries
            admin_comment: Optional admin comment
            
        Returns:
            Tuple of (problem description, affected files)
        """
        affected_files = list(set(f.get("file_path", "") for f in findings if f.get("file_path")))
        
        description_parts = ["Code analysis found the following issues:", ""]
        
        for i, finding in enumerate(findings[:10], 1):  # Limit to 10 issues
            severity = finding.get("severity", "unknown").upper()
            category = finding.get("category", "unknown")
            title = finding.get("title", "Issue")
            desc = finding.get("description", "")
            file_path = finding.get("file_path", "unknown")
            line = finding.get("line_number", 0)
            
            description_parts.append(f"[{severity}] [{category}] {title}")
            description_parts.append(f"  File: {file_path}:{line}")
            description_parts.append(f"  Problem: {desc}")
            if finding.get("suggested_fix"):
                description_parts.append(f"  Fix: {finding['suggested_fix']}")
            description_parts.append("")
        
        if len(findings) > 10:
            description_parts.append(f"... and {len(findings) - 10} more issues")
        
        if admin_comment:
            description_parts.extend(["", f"Admin notes: {admin_comment}"])
        
        return "\n".join(description_parts), affected_files
    
    async def _get_deepseek_review(self, problem_description: str) -> dict:
        """
        Get DeepSeek review of the proposed fix.
        
        Args:
            problem_description: Description of the problem to fix
            
        Returns:
            Review result with verdict
        """
        self._daily_review_count += 1
        
        if self._daily_review_count > MAX_DAILY_REVIEWS:
            logger.warning("review_manager_review_limit_reached")
            return {
                "success": True,
                "verdict": "approve",
                "reason": "Review limit reached, auto-approving",
                "risk_level": "medium",
            }
        
        return await self.deepseek.review_fix(
            problem_description=problem_description,
            proposed_fix="Cursor Background Agent will analyze and fix",
            affected_files=None,
        )
    
    def _handle_rejected_review(self, review: dict) -> dict:
        """
        Handle a rejected review.
        
        Args:
            review: DeepSeek review result
            
        Returns:
            Error response
        """
        reason = review.get("reason", "Unknown")
        logger.info("review_manager_fix_rejected", reason=reason)
        return {
            "success": False,
            "error": f"Fix rejected by DeepSeek review: {reason}",
            "review": review,
        }
    
    async def _launch_cursor_agent(
        self,
        problem_description: str,
        affected_files: list[str],
        review: dict,
    ) -> dict:
        """
        Launch Cursor Background Agent to fix issues.
        
        Args:
            problem_description: Description of the problem
            affected_files: List of affected file paths
            review: DeepSeek review result
            
        Returns:
            Launch result
        """
        context = f"DeepSeek Review: {review.get('reason', 'No review')} (Risk: {review.get('risk_level', 'unknown')})"
        
        result = await self.cursor.launch_repair(
            problem_description=problem_description,
            affected_files=affected_files,
            context=context,
        )
        
        return result
    
    def _increment_pr_count(self) -> None:
        """Increment the daily PR counter."""
        self._daily_pr_count += 1
        logger.info("review_manager_pr_created",
                   daily_count=self._daily_pr_count,
                   limit=MAX_DAILY_PRS)
    
    async def create_fix_task(
        self,
        findings: list[dict],
        admin_comment: str = "",
    ) -> dict:
        """
        Create a fix task from findings.
        
        Orchestrates the full flow:
        1. Check daily limits
        2. Build problem description
        3. Get DeepSeek review
        4. Launch Cursor agent
        5. Update counters
        
        Args:
            findings: List of finding dictionaries from code scanner
            admin_comment: Optional admin comment/context
            
        Returns:
            Result dict with success status and details
        """
        # Step 1: Check daily limit
        limit_check = self._check_daily_limit()
        if not limit_check["success"]:
            return limit_check
        
        # Step 2: Build problem description
        problem_desc, affected_files = self._build_problem_description(findings, admin_comment)
        
        logger.info("review_manager_creating_fix",
                   findings_count=len(findings),
                   affected_files=len(affected_files))
        
        # Step 3: Get DeepSeek review
        review = await self._get_deepseek_review(problem_desc)
        
        if review.get("verdict") == "reject":
            return self._handle_rejected_review(review)
        
        # Step 4: Launch Cursor agent
        result = await self._launch_cursor_agent(problem_desc, affected_files, review)
        
        # Step 5: Update counters on success
        if result.get("success"):
            self._increment_pr_count()
        
        return result
    
    def get_status(self) -> dict:
        """Get current status of review manager."""
        self._reset_daily_counters_if_needed()
        
        return {
            "daily_pr_count": self._daily_pr_count,
            "daily_pr_limit": MAX_DAILY_PRS,
            "daily_review_count": self._daily_review_count,
            "daily_review_limit": MAX_DAILY_REVIEWS,
            "last_analysis_time": self._last_analysis_time,
            "analysis_history_count": len(self._analysis_history),
            "scanner_summary": self.scanner.get_summary(),
            "deepseek_stats": self.deepseek.get_stats(),
        }
    
    def get_recent_analyses(self, limit: int = 10) -> list[dict]:
        """Get recent analysis results."""
        recent = self._analysis_history[-limit:]
        return [a.to_dict() for a in reversed(recent)]


# Singleton instance
_manager: Optional[ReviewManager] = None


def get_review_manager() -> ReviewManager:
    """Get or create ReviewManager singleton."""
    global _manager
    if _manager is None:
        _manager = ReviewManager()
    return _manager

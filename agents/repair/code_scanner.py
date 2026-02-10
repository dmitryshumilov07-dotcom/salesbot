"""
Code Scanner - Static analysis module for SalesBot codebase.

Scans Python files for common issues:
- Syntax errors
- Mutable default arguments
- Security issues
- Performance problems
- Code quality issues

Uses AST analysis for accurate detection.
"""
import ast
import asyncio
import re
import os
from pathlib import Path
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional
import structlog

logger = structlog.get_logger()


class FindingCategory(str, Enum):
    """Category of code finding."""
    BUG = "bug"
    SECURITY = "security"
    PERFORMANCE = "performance"
    QUALITY = "quality"
    TODO = "todo"


class FindingSeverity(str, Enum):
    """Severity level of finding."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class Finding:
    """Represents a single code finding."""
    file_path: str
    line_number: int
    category: FindingCategory
    severity: FindingSeverity
    title: str
    description: str
    suggested_fix: Optional[str] = None
    
    def to_dict(self) -> dict:
        return {
            "file_path": self.file_path,
            "line_number": self.line_number,
            "category": self.category.value,
            "severity": self.severity.value,
            "title": self.title,
            "description": self.description,
            "suggested_fix": self.suggested_fix,
        }


class CodeScanner:
    """Static code analyzer for Python files."""
    
    def __init__(self, root_path: str = "/opt/salesbot"):
        self.root_path = Path(root_path)
        self._findings: list[Finding] = []
        self._scanned_files: int = 0
        self._last_scan_time: Optional[float] = None
        
    def _add(
        self,
        file_path: str,
        line_number: int,
        category: FindingCategory,
        severity: FindingSeverity,
        title: str,
        description: str,
        suggested_fix: Optional[str] = None,
    ) -> None:
        """Add a finding to the results."""
        finding = Finding(
            file_path=file_path,
            line_number=line_number,
            category=category,
            severity=severity,
            title=title,
            description=description,
            suggested_fix=suggested_fix,
        )
        self._findings.append(finding)
        logger.debug("code_scanner_finding",
                    file=file_path,
                    line=line_number,
                    category=category.value,
                    title=title)
    
    async def scan_directory(
        self,
        target_dir: Optional[str] = None,
        exclude_patterns: Optional[list[str]] = None,
    ) -> list[Finding]:
        """
        Scan a directory for code issues.
        
        Args:
            target_dir: Directory to scan (defaults to root_path)
            exclude_patterns: Glob patterns to exclude
            
        Returns:
            List of Finding objects
        """
        import time
        start_time = time.time()
        self._findings = []
        self._scanned_files = 0
        
        scan_path = Path(target_dir) if target_dir else self.root_path
        exclude = exclude_patterns or ["__pycache__", ".git", "*.pyc", "venv", ".venv"]
        
        logger.info("code_scanner_starting", path=str(scan_path))
        
        # Find all Python files
        python_files = []
        for file_path in scan_path.rglob("*.py"):
            # Check exclusions
            skip = False
            for pattern in exclude:
                if pattern in str(file_path):
                    skip = True
                    break
            if not skip:
                python_files.append(file_path)
        
        # Process files with throttling to avoid blocking
        for i, file_path in enumerate(python_files):
            try:
                rel_path = str(file_path.relative_to(scan_path))
                await self._scan_file(file_path, rel_path)
                self._scanned_files += 1
                
                # FIX: Use asyncio.sleep instead of time.sleep to avoid blocking event loop
                # Yield control every 10 files to keep the event loop responsive
                if i > 0 and i % 10 == 0:
                    await asyncio.sleep(0.01)
            except Exception as e:
                logger.error("code_scanner_file_error",
                            file=str(file_path),
                            error=str(e))
        
        self._last_scan_time = time.time() - start_time
        
        logger.info("code_scanner_complete",
                   files_scanned=self._scanned_files,
                   findings=len(self._findings),
                   duration=f"{self._last_scan_time:.2f}s")
        
        return self._findings
    
    async def _scan_file(self, file_path: Path, rel_path: str) -> None:
        """Scan a single Python file."""
        try:
            content = file_path.read_text(encoding="utf-8")
        except Exception as e:
            self._add(rel_path, 1, FindingCategory.BUG, FindingSeverity.MEDIUM,
                     "Cannot read file", f"Failed to read file: {e}")
            return
        
        # Check for syntax errors first
        try:
            tree = ast.parse(content)
        except SyntaxError as e:
            self._add(rel_path, e.lineno or 1, FindingCategory.BUG, FindingSeverity.CRITICAL,
                     "Syntax error", f"Syntax error: {e.msg}")
            return
        
        # Run various checks
        self._check_ast_patterns(tree, rel_path)
        self._check_regex_patterns(content, rel_path)
    
    def _check_ast_patterns(self, tree: ast.AST, rel_path: str) -> None:
        """Check AST for common issues."""
        for node in ast.walk(tree):
            # FIX: Corrected mutable default argument detection
            # Check function definitions for mutable default arguments
            if isinstance(node, ast.FunctionDef):
                for default in node.args.defaults:
                    # Check for literal mutable defaults: [], {}
                    if isinstance(default, ast.List):
                        self._add(rel_path, node.lineno, FindingCategory.BUG,
                                 FindingSeverity.MEDIUM,
                                 "Mutable default argument",
                                 f"Function {node.name} has mutable default argument []. Use None instead.",
                                 f"Change `def {node.name}(..., arg=[])` to `def {node.name}(..., arg=None)` and initialize in function body.")
                    elif isinstance(default, ast.Dict):
                        self._add(rel_path, node.lineno, FindingCategory.BUG,
                                 FindingSeverity.MEDIUM,
                                 "Mutable default argument",
                                 f"Function {node.name} has mutable default argument {{}}. Use None instead.",
                                 f"Change `def {node.name}(..., arg={{}})` to `def {node.name}(..., arg=None)` and initialize in function body.")
                    elif isinstance(default, ast.Set):
                        self._add(rel_path, node.lineno, FindingCategory.BUG,
                                 FindingSeverity.MEDIUM,
                                 "Mutable default argument",
                                 f"Function {node.name} has mutable default argument set(). Use None instead.",
                                 f"Change to `def {node.name}(..., arg=None)` and initialize in function body.")
                    # FIX: Also check for list(), dict(), set() calls as defaults
                    elif isinstance(default, ast.Call) and isinstance(default.func, ast.Name):
                        if default.func.id in ('list', 'dict', 'set'):
                            self._add(rel_path, node.lineno, FindingCategory.BUG,
                                     FindingSeverity.MEDIUM,
                                     "Mutable default argument",
                                     f"Function {node.name} has mutable default argument {default.func.id}(). Use None instead.",
                                     f"Change `def {node.name}(..., arg={default.func.id}())` to `def {node.name}(..., arg=None)` and initialize in function body.")
            
            # Check for bare except
            if isinstance(node, ast.ExceptHandler):
                if node.type is None:
                    self._add(rel_path, node.lineno, FindingCategory.QUALITY,
                             FindingSeverity.MEDIUM,
                             "Bare except clause",
                             "Bare except catches all exceptions including SystemExit and KeyboardInterrupt.",
                             "Use `except Exception:` instead of bare `except:`")
            
            # Check for blocking calls in async functions
            if isinstance(node, ast.AsyncFunctionDef):
                for child in ast.walk(node):
                    if isinstance(child, ast.Call):
                        if isinstance(child.func, ast.Attribute):
                            if child.func.attr == "sleep" and isinstance(child.func.value, ast.Name):
                                if child.func.value.id == "time":
                                    self._add(rel_path, child.lineno, FindingCategory.PERFORMANCE,
                                             FindingSeverity.HIGH,
                                             "time.sleep() in async context",
                                             "time.sleep() blocks the event loop. Use await asyncio.sleep() instead.",
                                             "Replace `time.sleep(n)` with `await asyncio.sleep(n)`")
    
    def _check_regex_patterns(self, content: str, rel_path: str) -> None:
        """Check content using regex patterns."""
        lines = content.split("\n")
        
        patterns = [
            # Hardcoded secrets
            (r'password\s*=\s*["\'][^"\']{5,}["\']', FindingCategory.SECURITY, FindingSeverity.CRITICAL,
             "Hardcoded password", "Password appears to be hardcoded. Use environment variables."),
            (r'api_key\s*=\s*["\'][A-Za-z0-9_-]{20,}["\']', FindingCategory.SECURITY, FindingSeverity.CRITICAL,
             "Hardcoded API key", "API key appears to be hardcoded. Use environment variables."),
            # TODO comments
            (r'#\s*TODO\s*:', FindingCategory.TODO, FindingSeverity.INFO,
             "TODO comment", "Found TODO comment that may need addressing."),
            (r'#\s*FIXME\s*:', FindingCategory.TODO, FindingSeverity.MEDIUM,
             "FIXME comment", "Found FIXME comment indicating a known issue."),
            (r'#\s*HACK\s*:', FindingCategory.QUALITY, FindingSeverity.MEDIUM,
             "HACK comment", "Found HACK comment indicating potentially problematic code."),
            # Debug statements
            (r'\bprint\s*\(', FindingCategory.QUALITY, FindingSeverity.LOW,
             "Print statement", "Consider using logging instead of print statements."),
            # SQL injection risks
            (r'execute\s*\(\s*f["\']', FindingCategory.SECURITY, FindingSeverity.HIGH,
             "Potential SQL injection", "F-string in SQL execute() may be vulnerable to SQL injection. Use parameterized queries."),
        ]
        
        for line_num, line in enumerate(lines, 1):
            for pattern, category, severity, title, description in patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    self._add(rel_path, line_num, category, severity, title, description)
    
    def get_summary(self) -> dict:
        """
        Get summary of last scan.
        
        Returns:
            Dictionary with total findings, counts by category and severity
        """
        # FIX: Complete implementation of get_summary method
        # This method was truncated in the original code
        if not self._findings:
            return {
                "total": 0,
                "by_category": {},
                "by_severity": {},
                "files_scanned": self._scanned_files,
                "scan_duration": self._last_scan_time,
            }
        
        summary = {
            "total": len(self._findings),
            "by_category": {},
            "by_severity": {},
            "files_scanned": self._scanned_files,
            "scan_duration": self._last_scan_time,
        }
        
        # Count findings by category
        for finding in self._findings:
            category_key = finding.category.value if isinstance(finding.category, FindingCategory) else finding.category
            severity_key = finding.severity.value if isinstance(finding.severity, FindingSeverity) else finding.severity
            
            summary["by_category"][category_key] = summary["by_category"].get(category_key, 0) + 1
            summary["by_severity"][severity_key] = summary["by_severity"].get(severity_key, 0) + 1
        
        return summary
    
    def get_findings(self, 
                    category: Optional[FindingCategory] = None,
                    severity: Optional[FindingSeverity] = None,
                    min_severity: Optional[FindingSeverity] = None) -> list[Finding]:
        """
        Get findings with optional filtering.
        
        Args:
            category: Filter by category
            severity: Filter by exact severity
            min_severity: Filter by minimum severity (critical > high > medium > low > info)
            
        Returns:
            Filtered list of findings
        """
        severity_order = {
            FindingSeverity.CRITICAL: 5,
            FindingSeverity.HIGH: 4,
            FindingSeverity.MEDIUM: 3,
            FindingSeverity.LOW: 2,
            FindingSeverity.INFO: 1,
        }
        
        results = self._findings
        
        if category:
            results = [f for f in results if f.category == category]
        
        if severity:
            results = [f for f in results if f.severity == severity]
        elif min_severity:
            min_level = severity_order.get(min_severity, 0)
            results = [f for f in results if severity_order.get(f.severity, 0) >= min_level]
        
        return results
    
    def clear(self) -> None:
        """Clear all findings."""
        self._findings = []
        self._scanned_files = 0
        self._last_scan_time = None


# Singleton instance
_scanner: Optional[CodeScanner] = None


def get_code_scanner(root_path: str = "/opt/salesbot") -> CodeScanner:
    """Get or create CodeScanner singleton."""
    global _scanner
    if _scanner is None:
        _scanner = CodeScanner(root_path)
    return _scanner

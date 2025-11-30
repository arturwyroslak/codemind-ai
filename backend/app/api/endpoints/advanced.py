from fastapi import APIRouter, HTTPException, Depends, Header, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import aiohttp
import json
import logging
from datetime import datetime

router = APIRouter(prefix="/api/v1/advanced", tags=["advanced"])
logger = logging.getLogger(__name__)

class PRReviewRequest(BaseModel):
    repository: str = Field(..., description="Repository in format owner/repo")
    pr_number: int = Field(..., description="Pull request number")
    action_type: str = Field(default="pr_review", description="Action type")
    context: Optional[Dict[str, Any]] = Field(default=None, description="Additional context")

class PRReviewResponse(BaseModel):
    status: str
    pr_analysis: Dict[str, Any]
    audit_id: str
    recommendations: List[str]
    execution_time: float

class AnalyzeRequest(BaseModel):
    code: str = Field(..., description="Code to analyze")
    language: str = Field(..., description="Programming language")
    analysis_type: str = Field(default="full", description="Type: security, performance, architecture, full")
    context: Optional[Dict[str, Any]] = Field(default=None, description="Additional context")

class AnalyzeResponse(BaseModel):
    status: str
    overall_score: float
    analysis_type: str
    findings: List[Dict[str, Any]]
    execution_time: float
    audit_id: str

@router.post("/pr-review", response_model=PRReviewResponse)
async def pr_review(
    request: PRReviewRequest,
    background_tasks: BackgroundTasks,
    authorization: str = Header(None, alias="Authorization")
):
    """
    Perform AI-powered PR review with swarm agents.
    
    This endpoint:
    1. Fetches PR details from GitHub API
    2. Analyzes code changes using swarm orchestrator
    3. Posts inline comments to the PR
    4. Returns analysis summary
    """
    
    # Authentication
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization header")
    
    token = authorization.split(" ")[1]
    start_time = datetime.now()
    
    try:
        # Import required modules (lazy import to avoid circular dependencies)
        from app.advanced_features import (
            SwarmOrchestrator, 
            PluginManager, 
            PrivacyLayer,
            GitHubActionsPlugin
        )
        
        privacy_layer = PrivacyLayer()
        plugin_manager = PluginManager()
        swarm_orchestrator = SwarmOrchestrator()
        
        logger.info(f"Starting PR review for {request.repository}#{request.pr_number}")
        
        # Fetch PR data from GitHub
        github_token = request.context.get("github_token") if request.context else None
        
        async with aiohttp.ClientSession() as session:
            # Get PR details
            pr_url = f"https://api.github.com/repos/{request.repository}/pulls/{request.pr_number}"
            headers = {"Accept": "application/vnd.github.v3+json"}
            if github_token:
                headers["Authorization"] = f"token {github_token}"
            
            async with session.get(pr_url, headers=headers) as resp:
                if resp.status != 200:
                    raise HTTPException(
                        status_code=400, 
                        detail=f"Failed to fetch PR: {resp.status}"
                    )
                pr_data = await resp.json()
            
            # Get PR files
            files_url = f"{pr_url}/files"
            async with session.get(files_url, headers=headers) as resp:
                if resp.status != 200:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Failed to fetch PR files: {resp.status}"
                    )
                files = await resp.json()
        
        # Prepare analysis task
        analysis_task = {
            "type": "pr_review",
            "files": files[:20],  # Limit to first 20 files for performance
            "pr_data": {
                "title": pr_data.get("title"),
                "description": pr_data.get("body"),
                "base_branch": pr_data.get("base", {}).get("ref"),
                "head_branch": pr_data.get("head", {}).get("ref"),
            },
            "context": request.context or {},
            "repository": request.repository,
            "pr_number": request.pr_number
        }
        
        # Apply privacy layer (mask secrets)
        masked_task = privacy_layer.mask_secrets(json.dumps(analysis_task))
        analysis_task = json.loads(masked_task)
        
        # Create audit log
        audit_id = privacy_layer.audit_operation(
            operation="pr_review",
            user_id=token[:8],  # Use token prefix as user ID
            metadata={
                "repository": request.repository,
                "pr_number": request.pr_number,
                "files_count": len(files)
            }
        )
        
        # Execute swarm analysis
        logger.info(f"Executing swarm analysis for audit_id: {audit_id}")
        swarm_result = await swarm_orchestrator.execute_swarm(analysis_task)
        
        # Register and execute GitHub Actions plugin
        plugin_manager.register_plugin("github_actions", GitHubActionsPlugin())
        
        # Post comments to PR in background (if GitHub token provided)
        if github_token and swarm_result.get("comments"):
            background_tasks.add_task(
                post_pr_comments,
                request.repository,
                request.pr_number,
                swarm_result["comments"],
                github_token
            )
        
        # Generate recommendations
        recommendations = []
        for finding in swarm_result.get("findings", [])[:10]:  # Top 10 findings
            if finding.get("severity") in ["high", "critical"]:
                rec = (
                    f"{finding.get('severity', 'unknown').upper()}: "
                    f"{finding.get('type', 'Issue')} in {finding.get('file', 'unknown')} "
                    f"line {finding.get('line', '?')} - {finding.get('message', 'No description')}"
                )
                recommendations.append(rec)
        
        # Calculate execution time
        execution_time = (datetime.now() - start_time).total_seconds()
        
        # Prepare response
        analysis = {
            "overall_score": swarm_result.get("overall_score", 0.0),
            "files_analyzed": len(files),
            "issues_found": len(swarm_result.get("findings", [])),
            "comments_posted": len(swarm_result.get("comments", [])),
            "execution_time": swarm_result.get("execution_time", execution_time),
            "agents_used": swarm_result.get("agents_used", []),
            "security_score": swarm_result.get("security_score", 0.0),
            "performance_score": swarm_result.get("performance_score", 0.0),
            "architecture_score": swarm_result.get("architecture_score", 0.0)
        }
        
        logger.info(f"PR review completed for audit_id: {audit_id}")
        
        return PRReviewResponse(
            status="success",
            pr_analysis=analysis,
            audit_id=audit_id,
            recommendations=recommendations,
            execution_time=execution_time
        )
        
    except Exception as e:
        logger.error(f"PR review failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"PR review failed: {str(e)}")


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze_code(
    request: AnalyzeRequest,
    authorization: str = Header(None, alias="Authorization")
):
    """
    Analyze code snippet with AI agents.
    
    Used by IDE extensions (VS Code, JetBrains) for inline diagnostics.
    """
    
    # Authentication
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization header")
    
    token = authorization.split(" ")[1]
    start_time = datetime.now()
    
    try:
        from app.advanced_features import SwarmOrchestrator, PrivacyLayer
        
        privacy_layer = PrivacyLayer()
        swarm_orchestrator = SwarmOrchestrator()
        
        logger.info(f"Analyzing code snippet: language={request.language}, type={request.analysis_type}")
        
        # Apply privacy masking
        masked_code = privacy_layer.mask_secrets(request.code)
        
        # Create audit log
        audit_id = privacy_layer.audit_operation(
            operation="analyze_code",
            user_id=token[:8],
            metadata={
                "language": request.language,
                "analysis_type": request.analysis_type,
                "code_length": len(request.code)
            }
        )
        
        # Prepare analysis task
        analysis_task = {
            "type": "code_analysis",
            "code": masked_code,
            "language": request.language,
            "analysis_type": request.analysis_type,
            "context": request.context or {}
        }
        
        # Execute swarm analysis
        result = await swarm_orchestrator.execute_swarm(analysis_task)
        
        # Calculate execution time
        execution_time = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"Code analysis completed: audit_id={audit_id}")
        
        return AnalyzeResponse(
            status="success",
            overall_score=result.get("overall_score", 0.0),
            analysis_type=request.analysis_type,
            findings=result.get("findings", []),
            execution_time=execution_time,
            audit_id=audit_id
        )
        
    except Exception as e:
        logger.error(f"Code analysis failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


async def post_pr_comments(
    repository: str,
    pr_number: int,
    comments: List[Dict[str, Any]],
    github_token: str
):
    """
    Background task to post comments to GitHub PR.
    """
    try:
        async with aiohttp.ClientSession() as session:
            comments_url = f"https://api.github.com/repos/{repository}/pulls/{pr_number}/comments"
            headers = {
                "Authorization": f"token {github_token}",
                "Accept": "application/vnd.github.v3+json"
            }
            
            for comment in comments[:15]:  # Limit to 15 comments to avoid spam
                payload = {
                    "body": comment.get("body"),
                    "commit_id": comment.get("commit_id"),
                    "path": comment.get("path"),
                    "line": comment.get("line"),
                    "side": comment.get("side", "RIGHT")
                }
                
                async with session.post(comments_url, headers=headers, json=payload) as resp:
                    if resp.status == 201:
                        logger.info(f"Posted comment to PR {pr_number}: {comment.get('path')}:{comment.get('line')}")
                    else:
                        logger.warning(f"Failed to post comment: {resp.status}")
                        
    except Exception as e:
        logger.error(f"Failed to post PR comments: {str(e)}")
"""Agent orchestrator for coordinating multiple agents"""

import asyncio
from typing import List, Dict, Any
import logging
import time

from app.agents.security_agent import SecurityAgent
from app.agents.performance_agent import PerformanceAgent
from app.agents.architecture_agent import ArchitectureAgent
from app.agents.documentation_agent import DocumentationAgent

logger = logging.getLogger(__name__)


class AgentOrchestrator:
    """Orchestrates multiple specialized agents for code analysis"""
    
    def __init__(self):
        """Initialize agent orchestrator"""
        self.agents = {
            "security": SecurityAgent(),
            "performance": PerformanceAgent(),
            "architecture": ArchitectureAgent(),
            "documentation": DocumentationAgent()
        }
    
    async def analyze_code(
        self,
        code: str,
        language: str,
        agents: List[str],
        context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Run analysis with specified agents
        
        Args:
            code: Source code to analyze
            language: Programming language
            agents: List of agent names to use
            context: Additional context for analysis
            
        Returns:
            List of agent results
        """
        logger.info(f"Starting orchestrated analysis with {len(agents)} agents")
        
        # Create tasks for parallel execution
        tasks = []
        for agent_name in agents:
            if agent_name in self.agents:
                agent = self.agents[agent_name]
                task = self._run_agent(agent, agent_name, code, language, context)
                tasks.append(task)
            else:
                logger.warning(f"Unknown agent: {agent_name}")
        
        # Execute agents in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and format results
        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Agent {agents[i]} failed: {str(result)}")
            else:
                valid_results.append(result)
        
        logger.info(f"Analysis completed with {len(valid_results)} successful agents")
        return valid_results
    
    async def _run_agent(
        self,
        agent: Any,
        agent_name: str,
        code: str,
        language: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run a single agent and measure execution time"""
        start_time = time.time()
        
        try:
            findings = await agent.analyze(code, language, context)
            execution_time = time.time() - start_time
            
            return {
                "agent": agent_name,
                "status": "success",
                "findings": findings,
                "summary": agent.generate_summary(findings),
                "execution_time": execution_time
            }
        except Exception as e:
            logger.error(f"Agent {agent_name} failed: {str(e)}")
            raise
    
    def calculate_overall_score(self, results: List[Dict[str, Any]]) -> float:
        """
        Calculate overall code quality score from agent results
        
        Returns:
            Score from 0 to 100
        """
        if not results:
            return 0.0
        
        total_score = 0.0
        weights = {
            "security": 0.35,
            "performance": 0.25,
            "architecture": 0.25,
            "documentation": 0.15
        }
        
        for result in results:
            agent_name = result["agent"]
            weight = weights.get(agent_name, 0.25)
            
            # Calculate agent score based on findings severity
            findings = result["findings"]
            agent_score = self._calculate_agent_score(findings)
            total_score += agent_score * weight
        
        return round(total_score, 2)
    
    def _calculate_agent_score(self, findings: List[Dict[str, Any]]) -> float:
        """Calculate score for individual agent based on findings"""
        if not findings:
            return 100.0
        
        severity_weights = {
            "critical": -30,
            "high": -15,
            "medium": -5,
            "low": -2,
            "info": 0
        }
        
        score = 100.0
        for finding in findings:
            severity = finding.get("severity", "medium").lower()
            score += severity_weights.get(severity, -5)
        
        return max(0.0, min(100.0, score))
    
    def generate_recommendations(self, results: List[Dict[str, Any]]) -> List[str]:
        """
        Generate prioritized recommendations from all agent results
        
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        # Collect critical and high severity findings
        for result in results:
            for finding in result.get("findings", []):
                severity = finding.get("severity", "medium").lower()
                if severity in ["critical", "high"]:
                    recommendation = finding.get("recommendation", "")
                    if recommendation and recommendation not in recommendations:
                        recommendations.append(recommendation)
        
        # Limit to top 10 recommendations
        return recommendations[:10]
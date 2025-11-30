#!/usr/bin/env python3
"""Swarm Agent Orchestrator with Ray integration"""

import asyncio
import logging
import ray
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from langgraph import Graph, Node
from langchain_openai import ChatOpenAI
from app.core.config import settings

logger = logging.getLogger(__name__)

@dataclass
class AgentTask:
    """Task definition for agent execution"""
    code: str
    language: str
    context: Dict[str, Any]
    priority: int = 1
    timeout: int = 300  # 5 minutes

@dataclass
class AgentResult:
    """Result from agent execution"""
    agent_id: str
    status: str
    findings: List[Dict[str, Any]]
    execution_time: float
    confidence: float

class RaySwarmOrchestrator:
    """Advanced swarm orchestrator using Ray for distributed execution"""
    
    def __init__(self):
        # Initialize Ray cluster
        if not ray.is_initialized():
            ray.init(
                address=None,  # Local mode for development
                num_cpus=4,
                num_gpus=0,
                dashboard_host='0.0.0.0',
                dashboard_port=8265
            )
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model=settings.OPENAI_MODEL,
            api_key=settings.OPENAI_API_KEY,
            temperature=0.1
        )
        
        # Agent registry
        self.agents = self._initialize_agents()
        
        # Metrics
        self.execution_metrics = {}
        
    def _initialize_agents(self) -> Dict[str, Any]:
        """Initialize all available agents"""
        agents = {
            'security': SecuritySwarmAgent(),
            'performance': PerformanceSwarmAgent(),
            'architecture': ArchitectureSwarmAgent(),
            'documentation': DocumentationSwarmAgent(),
            'refactoring': RefactoringSwarmAgent(),
            'testing': TestingSwarmAgent(),
        }
        
        logger.info(f"Initialized {len(agents)} swarm agents")
        return agents
    
    def select_optimal_agents(self, task: AgentTask) -> List[str]:
        """
        Dynamically select optimal agents based on task characteristics
        
        Args:
            task: AgentTask with code and context
            
        Returns:
            List of recommended agent IDs
        """
        agent_recommendations = []
        
        # Base analysis for all code
        agent_recommendations.extend(['security', 'performance', 'documentation'])
        
        # Language-specific agents
        if task.language.lower() in ['python', 'javascript', 'typescript']:
            agent_recommendations.append('architecture')
            
        # Context-specific agents
        if task.context.get('project_type') == 'microservice':
            agent_recommendations.append('refactoring')
            
        if task.context.get('needs_testing', False):
            agent_recommendations.append('testing')
        
        # Remove duplicates and limit to 4 agents max
        unique_agents = list(dict.fromkeys(agent_recommendations))[:4]
        
        logger.info(f"Selected agents for task: {unique_agents}")
        return unique_agents
    
    async def execute_swarm_analysis(self, task: AgentTask) -> Dict[str, Any]:
        """
        Execute swarm analysis with distributed processing
        
        Args:
            task: AgentTask to process
            
        Returns:
            Complete analysis results with coordination
        """
        start_time = asyncio.get_event_loop().time()
        
        # Select optimal agents
        selected_agents = self.select_optimal_agents(task)
        
        # Create distributed tasks
        ray_tasks = [
            self._create_ray_task(agent_id, task)
            for agent_id in selected_agents
        ]
        
        # Execute in parallel using Ray
        logger.info(f"Executing {len(ray_tasks)} agents in parallel")
        results = await asyncio.gather(*ray_tasks, return_exceptions=True)
        
        # Coordinate results
        coordinated_results = await self._coordinate_swarm_results(results)
        
        # Calculate metrics
        total_time = asyncio.get_event_loop().time() - start_time
        overall_score = self._calculate_swarm_score(coordinated_results)
        
        return {
            'status': 'completed',
            'execution_time': total_time,
            'agents_executed': len([r for r in results if not isinstance(r, Exception)]),
            'overall_score': overall_score,
            'coordinated_results': coordinated_results,
            'recommendations': self._generate_swarm_recommendations(coordinated_results)
        }
    
    def _create_ray_task(self, agent_id: str, task: AgentTask) -> asyncio.Future:
        """Create Ray task for specific agent"""
        agent = self.agents[agent_id]
        
        async def execute_agent_task():
            try:
                start_time = time.time()
                result = await agent.analyze(task.code, task.language, task.context)
                execution_time = time.time() - start_time
                
                return AgentResult(
                    agent_id=agent_id,
                    status='success',
                    findings=result.get('findings', []),
                    execution_time=execution_time,
                    confidence=result.get('confidence', 0.8)
                )
            except Exception as e:
                logger.error(f"Agent {agent_id} failed: {str(e)}")
                return AgentResult(
                    agent_id=agent_id,
                    status='error',
                    findings=[],
                    execution_time=0,
                    confidence=0.0
                )
        
        # Use Ray for distributed execution
        @ray.remote(num_cpus=1)
        def ray_wrapper():
            return asyncio.run(execute_agent_task())
        
        # Submit to Ray and return future
        ray_future = ray.get(ray_wrapper.remote())
        return asyncio.Future()  # Simplified for demo
    
    async def _coordinate_swarm_results(self, results: List[Any]) -> List[Dict[str, Any]]:
        """Coordinate and merge results from multiple agents"""
        coordinated = []
        
        for result in results:
            if isinstance(result, AgentResult) and result.status == 'success':
                # Enhance findings with cross-agent context
                enhanced_findings = await self._enhance_with_context(result.findings, result.agent_id)
                
                coordinated.append({
                    'agent': result.agent_id,
                    'status': result.status,
                    'findings': enhanced_findings,
                    'execution_time': result.execution_time,
                    'confidence': result.confidence,
                    'summary': self._generate_agent_summary(enhanced_findings)
                })
        
        # Cross-reference findings between agents
        coordinated = await self._cross_reference_findings(coordinated)
        
        return coordinated
    
    async def _enhance_with_context(self, findings: List[Dict], agent_id: str) -> List[Dict]:
        """Enhance individual findings with swarm context"""
        enhanced = []
        
        for finding in findings:
            # Add cross-agent context if available
            context_enhancement = await self._get_context_enhancement(finding, agent_id)
            
            enhanced.append({
                **finding,
                'swarm_context': context_enhancement,
                'cross_referenced': False,
                'priority_score': self._calculate_priority(finding)
            })
        
        return enhanced
    
    def _calculate_swarm_score(self, coordinated_results: List[Dict]) -> float:
        """Calculate overall swarm quality score"""
        if not coordinated_results:
            return 0.0
        
        total_score = 0.0
        weights = {
            'security': 0.30,
            'performance': 0.25,
            'architecture': 0.25,
            'documentation': 0.10,
            'refactoring': 0.05,
            'testing': 0.05
        }
        
        for result in coordinated_results:
            agent_weight = weights.get(result['agent'], 0.15)
            agent_score = self._calculate_agent_score(result['findings'])
            total_score += agent_score * agent_weight
        
        # Apply swarm coordination bonus
        coordination_bonus = min(0.1, len(coordinated_results) * 0.02)
        final_score = total_score + coordination_bonus * 100
        
        return round(max(0, min(100, final_score)), 2)
    
    def _calculate_agent_score(self, findings: List[Dict]) -> float:
        """Calculate score for individual agent"""
        if not findings:
            return 100.0
        
        severity_weights = {
            'critical': -25,
            'high': -15,
            'medium': -8,
            'low': -3,
            'info': 0
        }
        
        score = 100.0
        for finding in findings:
            severity = finding.get('severity', 'medium').lower()
            score += severity_weights.get(severity, -5)
            
            # Adjust for priority and confidence
            priority_adjustment = finding.get('priority_score', 1.0) * 2
            confidence_adjustment = finding.get('confidence', 0.8) * 10
            score += priority_adjustment - confidence_adjustment
        
        return max(0.0, min(100.0, score))
    
    def _generate_swarm_recommendations(self, results: List[Dict]) -> List[str]:
        """Generate coordinated recommendations from swarm analysis"""
        recommendations = []
        priority_recommendations = []
        
        for result in results:
            for finding in result['findings']:
                if finding.get('priority_score', 1.0) > 1.5:
                    rec = finding.get('recommendation', '')
                    if rec and rec not in priority_recommendations:
                        priority_recommendations.append(rec)
                elif finding.get('severity') in ['critical', 'high']:
                    rec = finding.get('recommendation', '')
                    if rec and rec not in recommendations:
                        recommendations.append(rec)
        
        # Sort by priority and limit
        all_recs = priority_recommendations + recommendations[:7]
        return all_recs
    
    async def _cross_reference_findings(self, results: List[Dict]) -> List[Dict]:
        """Cross-reference findings between different agents"""
        cross_referenced = results.copy()
        
        for i, result_i in enumerate(cross_referenced):
            for finding in result_i['findings']:
                # Check if other agents found related issues
                related_findings = self._find_related_findings(finding, cross_referenced, i)
                
                if related_findings:
                    finding['cross_referenced'] = True
                    finding['related_agents'] = related_findings
                    finding['impact_multiplier'] = len(related_findings) * 0.2
        
        return cross_referenced
    
    def _find_related_findings(self, finding: Dict, results: List[Dict], exclude_index: int) -> List[str]:
        """Find related findings from other agents"""
        related = []
        
        for i, result in enumerate(results):
            if i == exclude_index:
                continue
                
            for other_finding in result['findings']:
                if self._are_related(finding, other_finding):
                    related.append(result['agent'])
        
        return related
    
    def _are_related(self, finding1: Dict, finding2: Dict) -> bool:
        """Determine if two findings are related"""
        # Simple semantic similarity check
        keywords1 = set(finding1.get('description', '').lower().split())
        keywords2 = set(finding2.get('description', '').lower().split())
        
        common = keywords1.intersection(keywords2)
        return len(common) > 1 or finding1.get('category') == finding2.get('category')
    
    def _calculate_priority(self, finding: Dict) -> float:
        """Calculate priority score for finding"""
        base_priority = 1.0
        
        # Severity multiplier
        severity_map = {
            'critical': 3.0,
            'high': 2.0,
            'medium': 1.5,
            'low': 1.0,
            'info': 0.5
        }
        
        severity = finding.get('severity', 'medium').lower()
        base_priority *= severity_map.get(severity, 1.0)
        
        # Cross-reference bonus
        if finding.get('cross_referenced', False):
            base_priority *= 1.3
        
        # Business impact
        if finding.get('impact', 'code') == 'business':
            base_priority *= 1.5
        
        return round(base_priority, 2)
    
    async def _get_context_enhancement(self, finding: Dict, agent_id: str) -> Dict:
        """Get additional context for finding from other agents"""
        # This would query other agents for related context
        # Simplified for demo
        return {
            'related_concepts': [],
            'business_impact': 'medium',
            'fix_complexity': 'low',
            'confidence': 0.85
        }
    
    def _generate_agent_summary(self, findings: List[Dict]) -> str:
        """Generate summary for agent findings"""
        if not findings:
            return "No issues detected."
        
        critical = sum(1 for f in findings if f.get('severity') == 'critical')
        high = sum(1 for f in findings if f.get('severity') == 'high')
        total = len(findings)
        
        return f"Detected {total} issues ({critical} critical, {high} high priority)"


# Example specialized swarm agents
class SecuritySwarmAgent:
    """Advanced security agent with swarm capabilities"""
    
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4", temperature=0.1)
        self.static_analyzers = ['bandit', 'semgrep', 'snyk']
    
    async def analyze(self, code: str, language: str, context: Dict) -> Dict:
        """Analyze code for security vulnerabilities using multiple techniques"""
        
        # Run static analysis
        static_results = await self._run_static_analysis(code, language)
        
        # Run dynamic LLM analysis
        llm_results = await self._run_llm_analysis(code, language, context)
        
        # Combine and prioritize
        combined = self._combine_security_findings(static_results, llm_results)
        
        return {
            'findings': combined,
            'confidence': self._calculate_security_confidence(combined),
            'coverage': {
                'static': len(static_results),
                'dynamic': len(llm_results),
                'total': len(combined)
            }
        }
    
    async def _run_static_analysis(self, code: str, language: str) -> List[Dict]:
        """Run multiple static analysis tools"""
        # This would integrate with actual SAST tools
        # Mock implementation for demo
        return [
            {
                'severity': 'high',
                'description': 'Potential SQL injection in query construction',
                'line': 42,
                'tool': 'semgrep',
                'confidence': 0.95,
                'recommendation': 'Use parameterized queries',
                'cwe': 'CWE-89'
            },
            {
                'severity': 'medium',
                'description': 'Hardcoded API key detected',
                'line': 127,
                'tool': 'bandit',
                'confidence': 0.98,
                'recommendation': 'Use environment variables',
                'cwe': 'CWE-798'
            }
        ]
    
    async def _run_llm_analysis(self, code: str, language: str, context: Dict) -> List[Dict]:
        """Run LLM-based security analysis"""
        # Advanced prompt engineering for security
        prompt = f"""
        Analyze this {language} code for advanced security vulnerabilities:
        
        CODE:
        {code}
        
        CONTEXT:
        {context}
        
        Focus on:
        1. Business logic flaws
        2. Authentication bypass scenarios
        3. Privilege escalation paths
        4. Supply chain attacks
        5. Cryptographic misconfigurations
        
        Return detailed findings with:
        - Attack vectors
        - Impact assessment
        - Exploit complexity
        - Remediation strategies
        """
        
        response = await self.llm.ainvoke(prompt)
        
        # Parse LLM response (simplified)
        return [
            {
                'severity': 'critical',
                'description': 'Weak session management could allow account takeover',
                'line': None,
                'tool': 'llm-analysis',
                'confidence': 0.92,
                'recommendation': 'Implement secure session handling with CSRF protection',
                'impact': 'High business impact',
                'attack_vector': 'Authentication bypass'
            }
        ]
    
    def _combine_security_findings(self, static: List[Dict], llm: List[Dict]) -> List[Dict]:
        """Combine static and dynamic analysis results"""
        all_findings = static + llm
        
        # Deduplicate and prioritize
        unique_findings = []
        seen_descriptions = set()
        
        for finding in sorted(all_findings, key=lambda x: x.get('confidence', 0), reverse=True):
            desc_hash = hash(finding['description'][:100])
            if desc_hash not in seen_descriptions:
                seen_descriptions.add(desc_hash)
                
                # Enhance with cross-verification
                finding['verification_sources'] = [
                    src['tool'] for src in all_findings 
                    if src['description'][:50] == finding['description'][:50]
                ]
                
                unique_findings.append(finding)
        
        return unique_findings[:20]  # Limit results
    
    def _calculate_security_confidence(self, findings: List[Dict]) -> float:
        """Calculate overall security confidence score"""
        if not findings:
            return 1.0
        
        avg_confidence = sum(f.get('confidence', 0.5) for f in findings) / len(findings)
        verification_bonus = min(0.2, len(set(f.get('tool', '') for f in findings)) * 0.05)
        
        return round(avg_confidence + verification_bonus, 2)


# Similar implementations for other agents...
class PerformanceSwarmAgent:
    """Performance analysis with profiling and optimization suggestions"""
    def __init__(self):
        pass
    
    async def analyze(self, code: str, language: str, context: Dict) -> Dict:
        # Performance profiling, complexity analysis, optimization
        pass


class ArchitectureSwarmAgent:
    """Architecture and design pattern analysis"""
    def __init__(self):
        pass
    
    async def analyze(self, code: str, language: str, context: Dict) -> Dict:
        # SOLID principles, design patterns, scalability analysis
        pass


# Usage example
async def main():
    orchestrator = RaySwarmOrchestrator()
    
    task = AgentTask(
        code="""def vulnerable_function(user_input):
    query = f"SELECT * FROM users WHERE name = '{user_input}'"
    return execute(query)""",
        language='python',
        context={'project_type': 'web_app', 'framework': 'django'}
    )
    
    result = await orchestrator.execute_swarm_analysis(task)
    
    print(f"Swarm analysis completed in {result['execution_time']:.2f}s")
    print(f"Overall score: {result['overall_score']}/100")
    print(f"Recommendations: {len(result['recommendations'])}")

if __name__ == "__main__":
    asyncio.run(main())
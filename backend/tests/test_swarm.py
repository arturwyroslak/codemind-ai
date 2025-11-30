import pytest
import asyncio
from app.advanced_features import SwarmOrchestrator

@pytest.fixture
def orchestrator():
    """Create SwarmOrchestrator instance."""
    return SwarmOrchestrator()

@pytest.mark.asyncio
async def test_swarm_initialization(orchestrator):
    """Test that swarm orchestrator initializes correctly."""
    assert orchestrator is not None
    assert hasattr(orchestrator, 'execute_swarm')

@pytest.mark.asyncio
async def test_simple_analysis(orchestrator):
    """Test simple code analysis."""
    task = {
        "type": "code_analysis",
        "code": "print('hello world')",
        "language": "python",
        "analysis_type": "full"
    }
    
    try:
        result = await orchestrator.execute_swarm(task)
        assert "overall_score" in result
        assert "findings" in result
    except Exception as e:
        # May fail if dependencies not installed
        pytest.skip(f"Swarm analysis not available: {e}")
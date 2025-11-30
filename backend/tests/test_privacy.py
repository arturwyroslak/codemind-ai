import pytest
from app.advanced_features import PrivacyLayer

@pytest.fixture
def privacy_layer():
    return PrivacyLayer()

def test_mask_api_key(privacy_layer):
    """Test that API keys are masked."""
    code = "api_key = 'sk-1234567890abcdef'"
    masked = privacy_layer.mask_secrets(code)
    assert "sk-1234567890abcdef" not in masked
    assert "[REDACTED]" in masked

def test_mask_password(privacy_layer):
    """Test that passwords are masked."""
    code = 'password = "SuperSecret123"'
    masked = privacy_layer.mask_secrets(code)
    assert "SuperSecret123" not in masked
    assert "[REDACTED]" in masked

def test_no_false_positives(privacy_layer):
    """Test that normal code is not masked."""
    code = "def hello(): return 'world'"
    masked = privacy_layer.mask_secrets(code)
    assert code == masked

def test_audit_log(privacy_layer):
    """Test audit logging."""
    audit_id = privacy_layer.audit_operation(
        operation="test",
        user_id="test_user",
        metadata={"test": True}
    )
    assert audit_id.startswith("audit_")
    assert len(audit_id) > 10
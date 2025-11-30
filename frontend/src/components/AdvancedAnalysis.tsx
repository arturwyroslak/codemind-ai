import React, { useState } from 'react';
import axios from 'axios';
import {
  Box,
  Button,
  Card,
  CardContent,
  Typography,
  Select,
  MenuItem,
  TextField,
  LinearProgress,
  Chip,
  Alert,
  Grid,
  Paper
} from '@mui/material';
import {
  Security,
  Speed,
  Architecture,
  BugReport
} from '@mui/icons-material';

interface Finding {
  severity: string;
  type: string;
  message: string;
  file: string;
  line: number;
  fix?: string;
}

interface AnalysisResult {
  status: string;
  overall_score: number;
  analysis_type: string;
  findings: Finding[];
  execution_time: number;
  audit_id: string;
}

const AdvancedAnalysis: React.FC = () => {
  const [code, setCode] = useState('');
  const [language, setLanguage] = useState('python');
  const [analysisType, setAnalysisType] = useState('full');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const apiUrl = process.env.REACT_APP_API_URL || 'http://localhost:8000';
  const apiToken = localStorage.getItem('codemind_token') || 'test-token-dev-only';

  const handleAnalyze = async () => {
    if (!code.trim()) {
      setError('Please enter some code to analyze');
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await axios.post(
        `${apiUrl}/api/v1/advanced/analyze`,
        {
          code,
          language,
          analysis_type: analysisType,
          context: {
            source: 'web_ui'
          }
        },
        {
          headers: {
            'Authorization': `Bearer ${apiToken}`,
            'Content-Type': 'application/json'
          }
        }
      );

      setResult(response.data);
    } catch (err: any) {
      setError(
        err.response?.data?.detail || err.message || 'Analysis failed'
      );
    } finally {
      setLoading(false);
    }
  };

  const getSeverityColor = (severity: string) => {
    switch (severity.toLowerCase()) {
      case 'critical': return 'error';
      case 'high': return 'warning';
      case 'medium': return 'info';
      case 'low': return 'success';
      default: return 'default';
    }
  };

  const getScoreColor = (score: number) => {
    if (score >= 80) return 'success';
    if (score >= 60) return 'warning';
    return 'error';
  };

  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h4" gutterBottom>
        🤖 Advanced Code Analysis
      </Typography>

      <Grid container spacing={3}>
        {/* Input Section */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Code Input
              </Typography>

              <Box sx={{ mb: 2 }}>
                <Select
                  fullWidth
                  value={language}
                  onChange={(e) => setLanguage(e.target.value)}
                  sx={{ mb: 2 }}
                >
                  <MenuItem value="python">Python</MenuItem>
                  <MenuItem value="javascript">JavaScript</MenuItem>
                  <MenuItem value="typescript">TypeScript</MenuItem>
                  <MenuItem value="java">Java</MenuItem>
                  <MenuItem value="cpp">C++</MenuItem>
                  <MenuItem value="go">Go</MenuItem>
                  <MenuItem value="rust">Rust</MenuItem>
                </Select>

                <Select
                  fullWidth
                  value={analysisType}
                  onChange={(e) => setAnalysisType(e.target.value)}
                  sx={{ mb: 2 }}
                >
                  <MenuItem value="full">
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      <BugReport /> Full Analysis
                    </Box>
                  </MenuItem>
                  <MenuItem value="security">
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      <Security /> Security Only
                    </Box>
                  </MenuItem>
                  <MenuItem value="performance">
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      <Speed /> Performance Only
                    </Box>
                  </MenuItem>
                  <MenuItem value="architecture">
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      <Architecture /> Architecture Only
                    </Box>
                  </MenuItem>
                </Select>

                <TextField
                  fullWidth
                  multiline
                  rows={15}
                  placeholder="Paste your code here..."
                  value={code}
                  onChange={(e) => setCode(e.target.value)}
                  sx={{
                    fontFamily: 'monospace',
                    '& textarea': {
                      fontFamily: 'monospace',
                      fontSize: '0.9rem'
                    }
                  }}
                />
              </Box>

              <Button
                fullWidth
                variant="contained"
                onClick={handleAnalyze}
                disabled={loading || !code.trim()}
                size="large"
              >
                {loading ? 'Analyzing...' : '🔍 Analyze Code'}
              </Button>
            </CardContent>
          </Card>
        </Grid>

        {/* Results Section */}
        <Grid item xs={12} md={6}>
          {loading && (
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Analyzing with AI Swarm...
                </Typography>
                <LinearProgress />
                <Typography variant="body2" color="text.secondary" sx={{ mt: 2 }}>
                  Running security, performance, and architecture agents...
                </Typography>
              </CardContent>
            </Card>
          )}

          {error && (
            <Alert severity="error" sx={{ mb: 2 }}>
              {error}
            </Alert>
          )}

          {result && (
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Analysis Results
                </Typography>

                {/* Score */}
                <Paper sx={{ p: 2, mb: 2, bgcolor: `${getScoreColor(result.overall_score)}.light` }}>
                  <Typography variant="h3" align="center">
                    {result.overall_score.toFixed(1)}/100
                  </Typography>
                  <Typography variant="body2" align="center" color="text.secondary">
                    Overall Quality Score
                  </Typography>
                </Paper>

                {/* Metadata */}
                <Box sx={{ mb: 2 }}>
                  <Chip
                    label={`Type: ${result.analysis_type}`}
                    size="small"
                    sx={{ mr: 1, mb: 1 }}
                  />
                  <Chip
                    label={`Time: ${result.execution_time.toFixed(2)}s`}
                    size="small"
                    sx={{ mr: 1, mb: 1 }}
                  />
                  <Chip
                    label={`Issues: ${result.findings.length}`}
                    size="small"
                    color={result.findings.length === 0 ? 'success' : 'warning'}
                    sx={{ mb: 1 }}
                  />
                </Box>

                {/* Findings */}
                <Typography variant="h6" gutterBottom sx={{ mt: 2 }}>
                  Findings ({result.findings.length})
                </Typography>

                {result.findings.length === 0 ? (
                  <Alert severity="success">
                    ✨ No issues found! Your code looks great!
                  </Alert>
                ) : (
                  <Box>
                    {result.findings.map((finding, index) => (
                      <Card key={index} sx={{ mb: 2, borderLeft: 4, borderColor: `${getSeverityColor(finding.severity)}.main` }}>
                        <CardContent>
                          <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                            <Chip
                              label={finding.severity}
                              color={getSeverityColor(finding.severity) as any}
                              size="small"
                            />
                            <Typography variant="caption" color="text.secondary">
                              {finding.file}:{finding.line}
                            </Typography>
                          </Box>

                          <Typography variant="subtitle2" gutterBottom>
                            {finding.type}
                          </Typography>

                          <Typography variant="body2" color="text.secondary">
                            {finding.message}
                          </Typography>

                          {finding.fix && (
                            <Alert severity="info" sx={{ mt: 1 }}>
                              <Typography variant="caption">
                                💡 Suggested Fix:
                              </Typography>
                              <Typography
                                variant="body2"
                                sx={{ fontFamily: 'monospace', mt: 0.5 }}
                              >
                                {finding.fix}
                              </Typography>
                            </Alert>
                          )}
                        </CardContent>
                      </Card>
                    ))}
                  </Box>
                )}

                {/* Audit ID */}
                <Typography variant="caption" color="text.secondary" sx={{ mt: 2, display: 'block' }}>
                  Audit ID: {result.audit_id}
                </Typography>
              </CardContent>
            </Card>
          )}
        </Grid>
      </Grid>
    </Box>
  );
};

export default AdvancedAnalysis;
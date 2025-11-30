import React, { useState } from 'react';
import axios from 'axios';
import {
  Box,
  Button,
  Card,
  CardContent,
  Typography,
  TextField,
  Alert,
  CircularProgress,
  List,
  ListItem,
  ListItemText,
  Divider,
  Chip
} from '@mui/material';
import { GitHub, CheckCircle, Error as ErrorIcon } from '@mui/icons-material';

interface PRAnalysis {
  overall_score: number;
  files_analyzed: number;
  issues_found: number;
  comments_posted: number;
  execution_time: number;
}

interface PRReviewResult {
  status: string;
  pr_analysis: PRAnalysis;
  audit_id: string;
  recommendations: string[];
  execution_time: number;
}

const PRReview: React.FC = () => {
  const [repository, setRepository] = useState('');
  const [prNumber, setPrNumber] = useState('');
  const [githubToken, setGithubToken] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<PRReviewResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const apiUrl = process.env.REACT_APP_API_URL || 'http://localhost:8000';
  const apiToken = localStorage.getItem('codemind_token') || 'test-token-dev-only';

  const handleReview = async () => {
    if (!repository || !prNumber) {
      setError('Please provide repository and PR number');
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await axios.post(
        `${apiUrl}/api/v1/advanced/pr-review`,
        {
          repository,
          pr_number: parseInt(prNumber),
          action_type: 'pr_review',
          context: {
            github_token: githubToken || undefined,
            language: 'auto',
            project_type: 'auto'
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
        err.response?.data?.detail || err.message || 'PR review failed'
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <Box sx={{ p: 3, maxWidth: 1200, mx: 'auto' }}>
      <Typography variant="h4" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
        <GitHub /> AI-Powered PR Review
      </Typography>

      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Configure PR Review
          </Typography>

          <TextField
            fullWidth
            label="Repository (owner/repo)"
            placeholder="e.g., microsoft/vscode"
            value={repository}
            onChange={(e) => setRepository(e.target.value)}
            sx={{ mb: 2 }}
          />

          <TextField
            fullWidth
            label="PR Number"
            type="number"
            placeholder="e.g., 123"
            value={prNumber}
            onChange={(e) => setPrNumber(e.target.value)}
            sx={{ mb: 2 }}
          />

          <TextField
            fullWidth
            label="GitHub Token (Optional)"
            type="password"
            placeholder="ghp_..."
            value={githubToken}
            onChange={(e) => setGithubToken(e.target.value)}
            helperText="Required to post comments directly to PR"
            sx={{ mb: 2 }}
          />

          <Button
            fullWidth
            variant="contained"
            onClick={handleReview}
            disabled={loading || !repository || !prNumber}
            size="large"
            startIcon={loading ? <CircularProgress size={20} /> : <GitHub />}
          >
            {loading ? 'Analyzing PR...' : 'Start AI Review'}
          </Button>
        </CardContent>
      </Card>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}

      {result && (
        <Card>
          <CardContent>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
              <CheckCircle color="success" />
              <Typography variant="h6">
                Review Complete!
              </Typography>
            </Box>

            {/* Stats */}
            <Box sx={{ display: 'flex', gap: 2, mb: 3, flexWrap: 'wrap' }}>
              <Chip
                label={`Score: ${result.pr_analysis.overall_score.toFixed(1)}/100`}
                color={result.pr_analysis.overall_score >= 80 ? 'success' : 'warning'}
              />
              <Chip label={`Files: ${result.pr_analysis.files_analyzed}`} />
              <Chip
                label={`Issues: ${result.pr_analysis.issues_found}`}
                color={result.pr_analysis.issues_found === 0 ? 'success' : 'error'}
              />
              <Chip label={`Comments: ${result.pr_analysis.comments_posted}`} />
              <Chip label={`Time: ${result.pr_analysis.execution_time.toFixed(2)}s`} />
            </Box>

            <Divider sx={{ my: 2 }} />

            {/* Recommendations */}
            <Typography variant="h6" gutterBottom>
              Key Recommendations
            </Typography>

            {result.recommendations.length === 0 ? (
              <Alert severity="success">
                ✨ No critical issues found! This PR looks good to merge.
              </Alert>
            ) : (
              <List>
                {result.recommendations.map((rec, index) => (
                  <React.Fragment key={index}>
                    <ListItem alignItems="flex-start">
                      <ListItemText
                        primary={
                          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                            <ErrorIcon color="error" fontSize="small" />
                            <Typography variant="body1">
                              {rec}
                            </Typography>
                          </Box>
                        }
                      />
                    </ListItem>
                    {index < result.recommendations.length - 1 && <Divider />}
                  </React.Fragment>
                ))}
              </List>
            )}

            {/* Audit Info */}
            <Typography variant="caption" color="text.secondary" sx={{ mt: 2, display: 'block' }}>
              Audit ID: {result.audit_id}
            </Typography>
          </CardContent>
        </Card>
      )}
    </Box>
  );
};

export default PRReview;
import React from 'react';
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import {
  AppBar,
  Toolbar,
  Typography,
  Container,
  Box,
  Button,
  ThemeProvider,
  createTheme,
  CssBaseline
} from '@mui/material';
import { Code, GitHub, Analytics } from '@mui/icons-material';

import AdvancedAnalysis from './components/AdvancedAnalysis';
import PRReview from './components/PRReview';

const theme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#2d74c7',
    },
    secondary: {
      main: '#f50057',
    },
  },
});

function App() {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Router>
        <Box sx={{ flexGrow: 1 }}>
          <AppBar position="static">
            <Toolbar>
              <Code sx={{ mr: 2 }} />
              <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
                CodeMind AI
              </Typography>
              <Button color="inherit" component={Link} to="/">
                <Analytics sx={{ mr: 1 }} /> Analysis
              </Button>
              <Button color="inherit" component={Link} to="/pr-review">
                <GitHub sx={{ mr: 1 }} /> PR Review
              </Button>
            </Toolbar>
          </AppBar>

          <Container maxWidth="xl" sx={{ mt: 4, mb: 4 }}>
            <Routes>
              <Route path="/" element={<AdvancedAnalysis />} />
              <Route path="/pr-review" element={<PRReview />} />
            </Routes>
          </Container>

          <Box
            component="footer"
            sx={{
              py: 3,
              px: 2,
              mt: 'auto',
              backgroundColor: (theme) =>
                theme.palette.mode === 'light'
                  ? theme.palette.grey[200]
                  : theme.palette.grey[900],
              textAlign: 'center'
            }}
          >
            <Typography variant="body2" color="text.secondary">
              Made with ❤️ and 🤖 by CodeMind AI Team
            </Typography>
          </Box>
        </Box>
      </Router>
    </ThemeProvider>
  );
}

export default App;
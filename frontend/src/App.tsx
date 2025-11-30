import React, { useState } from 'react'
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import { Toaster } from 'react-hot-toast'
import Navbar from './components/Navbar'
import Sidebar from './components/Sidebar'
import CodeAnalysis from './pages/CodeAnalysis'
import CodeGeneration from './pages/CodeGeneration'
import RAGQuery from './pages/RAGQuery'
import Dashboard from './pages/Dashboard'
import Settings from './pages/Settings'

function App() {
  const [sidebarOpen, setSidebarOpen] = useState(true)

  return (
    <Router>
      <div className="flex h-screen bg-slate-900 text-white">
        <Toaster position="top-right" />
        
        <Sidebar isOpen={sidebarOpen} />
        
        <div className="flex-1 flex flex-col overflow-hidden">
          <Navbar onMenuClick={() => setSidebarOpen(!sidebarOpen)} />
          
          <main className="flex-1 overflow-auto p-6">
            <Routes>
              <Route path="/" element={<Dashboard />} />
              <Route path="/analyze" element={<CodeAnalysis />} />
              <Route path="/generate" element={<CodeGeneration />} />
              <Route path="/query" element={<RAGQuery />} />
              <Route path="/settings" element={<Settings />} />
            </Routes>
          </main>
        </div>
      </div>
    </Router>
  )
}

export default App
import React, { useState } from 'react'
import Editor from '@monaco-editor/react'
import { Play, Download, AlertCircle, CheckCircle, Info } from 'lucide-react'
import toast from 'react-hot-toast'
import axios from 'axios'

const CodeAnalysis: React.FC = () => {
  const [code, setCode] = useState('# Paste your code here\ndef example_function():\n    pass')
  const [language, setLanguage] = useState('python')
  const [analyzing, setAnalyzing] = useState(false)
  const [results, setResults] = useState<any>(null)
  const [selectedAgents, setSelectedAgents] = useState(['security', 'performance', 'architecture'])
  
  const agents = [
    { id: 'security', name: 'Security', color: 'red' },
    { id: 'performance', name: 'Performance', color: 'yellow' },
    { id: 'architecture', name: 'Architecture', color: 'blue' },
    { id: 'documentation', name: 'Documentation', color: 'green' },
  ]
  
  const handleAnalyze = async () => {
    if (!code.trim()) {
      toast.error('Please enter some code to analyze')
      return
    }
    
    setAnalyzing(true)
    
    try {
      const response = await axios.post('/api/v1/analyze/', {
        code,
        language,
        agents: selectedAgents,
        context: {}
      })
      
      setResults(response.data)
      toast.success('Analysis completed successfully!')
    } catch (error: any) {
      toast.error(error.response?.data?.detail || 'Analysis failed')
    } finally {
      setAnalyzing(false)
    }
  }
  
  const toggleAgent = (agentId: string) => {
    if (selectedAgents.includes(agentId)) {
      setSelectedAgents(selectedAgents.filter(id => id !== agentId))
    } else {
      setSelectedAgents([...selectedAgents, agentId])
    }
  }
  
  const getSeverityIcon = (severity: string) => {
    switch (severity?.toLowerCase()) {
      case 'critical':
      case 'high':
        return <AlertCircle className="w-5 h-5 text-red-500" />
      case 'medium':
        return <AlertCircle className="w-5 h-5 text-yellow-500" />
      case 'low':
        return <Info className="w-5 h-5 text-blue-500" />
      default:
        return <CheckCircle className="w-5 h-5 text-green-500" />
    }
  }
  
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold mb-2">Code Analysis</h1>
        <p className="text-slate-400">Analyze your code with AI-powered multi-agent system</p>
      </div>
      
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Code Editor */}
        <div className="lg:col-span-2 space-y-4">
          <div className="bg-slate-800 rounded-lg p-4 border border-slate-700">
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-4">
                <select
                  value={language}
                  onChange={(e) => setLanguage(e.target.value)}
                  className="bg-slate-900 border border-slate-700 rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                  <option value="python">Python</option>
                  <option value="javascript">JavaScript</option>
                  <option value="typescript">TypeScript</option>
                  <option value="java">Java</option>
                  <option value="csharp">C#</option>
                  <option value="go">Go</option>
                </select>
              </div>
              
              <div className="flex gap-2">
                <button
                  onClick={handleAnalyze}
                  disabled={analyzing || selectedAgents.length === 0}
                  className="flex items-center gap-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-slate-700 disabled:cursor-not-allowed rounded-lg font-medium transition-colors"
                >
                  <Play className="w-4 h-4" />
                  {analyzing ? 'Analyzing...' : 'Analyze Code'}
                </button>
              </div>
            </div>
            
            <div className="border border-slate-700 rounded-lg overflow-hidden">
              <Editor
                height="500px"
                language={language}
                value={code}
                onChange={(value) => setCode(value || '')}
                theme="vs-dark"
                options={{
                  minimap: { enabled: false },
                  fontSize: 14,
                  lineNumbers: 'on',
                  scrollBeyondLastLine: false,
                  automaticLayout: true,
                }}
              />
            </div>
          </div>
        </div>
        
        {/* Agent Selection & Settings */}
        <div className="space-y-4">
          <div className="bg-slate-800 rounded-lg p-4 border border-slate-700">
            <h3 className="font-bold mb-4">Select Agents</h3>
            <div className="space-y-2">
              {agents.map((agent) => (
                <label
                  key={agent.id}
                  className="flex items-center gap-3 p-3 bg-slate-900 rounded-lg cursor-pointer hover:bg-slate-700 transition-colors"
                >
                  <input
                    type="checkbox"
                    checked={selectedAgents.includes(agent.id)}
                    onChange={() => toggleAgent(agent.id)}
                    className="w-4 h-4"
                  />
                  <div className={`w-3 h-3 rounded-full bg-${agent.color}-500`}></div>
                  <span className="font-medium">{agent.name}</span>
                </label>
              ))}
            </div>
          </div>
          
          {results && (
            <div className="bg-slate-800 rounded-lg p-4 border border-slate-700">
              <h3 className="font-bold mb-4">Overall Score</h3>
              <div className="text-center">
                <div className="text-5xl font-bold mb-2">
                  {results.overall_score}
                  <span className="text-2xl text-slate-400">/100</span>
                </div>
                <div className="w-full bg-slate-700 rounded-full h-3 mb-4">
                  <div
                    className={`h-3 rounded-full ${
                      results.overall_score >= 80 ? 'bg-green-500' :
                      results.overall_score >= 60 ? 'bg-yellow-500' : 'bg-red-500'
                    }`}
                    style={{ width: `${results.overall_score}%` }}
                  ></div>
                </div>
                <p className="text-sm text-slate-400">
                  Analysis completed in {results.total_execution_time?.toFixed(2)}s
                </p>
              </div>
            </div>
          )}
        </div>
      </div>
      
      {/* Results */}
      {results && (
        <div className="space-y-4">
          {results.results?.map((result: any, index: number) => (
            <div key={index} className="bg-slate-800 rounded-lg p-6 border border-slate-700">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-xl font-bold capitalize">{result.agent} Agent</h3>
                <span className="text-sm text-slate-400">
                  {result.execution_time?.toFixed(2)}s
                </span>
              </div>
              
              <p className="text-slate-300 mb-4">{result.summary}</p>
              
              {result.findings && result.findings.length > 0 && (
                <div className="space-y-3">
                  <h4 className="font-semibold text-sm text-slate-400">Findings:</h4>
                  {result.findings.map((finding: any, fIndex: number) => (
                    <div key={fIndex} className="flex gap-3 p-4 bg-slate-900 rounded-lg">
                      {getSeverityIcon(finding.severity)}
                      <div className="flex-1">
                        <div className="flex items-center gap-2 mb-1">
                          <span className="font-semibold">{finding.description}</span>
                          {finding.line && (
                            <span className="text-xs bg-slate-700 px-2 py-1 rounded">
                              Line {finding.line}
                            </span>
                          )}
                        </div>
                        {finding.recommendation && (
                          <p className="text-sm text-slate-400">
                            💡 {finding.recommendation}
                          </p>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          ))}
          
          {results.recommendations && results.recommendations.length > 0 && (
            <div className="bg-blue-900/20 border border-blue-500/30 rounded-lg p-6">
              <h3 className="text-xl font-bold mb-4">🎯 Recommendations</h3>
              <ul className="space-y-2">
                {results.recommendations.map((rec: string, index: number) => (
                  <li key={index} className="flex items-start gap-2">
                    <span className="text-blue-400">•</span>
                    <span>{rec}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

export default CodeAnalysis
import React from 'react'
import { TrendingUp, AlertTriangle, CheckCircle, Code, Clock, Zap } from 'lucide-react'
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts'

const Dashboard: React.FC = () => {
  const stats = [
    { label: 'Total Analyses', value: '127', icon: Code, color: 'blue', trend: '+12%' },
    { label: 'Code Generated', value: '43', icon: Zap, color: 'green', trend: '+8%' },
    { label: 'Issues Found', value: '89', icon: AlertTriangle, color: 'red', trend: '-5%' },
    { label: 'Issues Resolved', value: '71', icon: CheckCircle, color: 'purple', trend: '+15%' },
  ]
  
  const activityData = [
    { date: 'Mon', analyses: 12, generated: 5 },
    { date: 'Tue', analyses: 19, generated: 8 },
    { date: 'Wed', analyses: 15, generated: 6 },
    { date: 'Thu', analyses: 22, generated: 10 },
    { date: 'Fri', analyses: 28, generated: 12 },
    { date: 'Sat', analyses: 18, generated: 7 },
    { date: 'Sun', analyses: 13, generated: 4 },
  ]
  
  const severityData = [
    { name: 'Critical', value: 8, color: '#ef4444' },
    { name: 'High', value: 23, color: '#f97316' },
    { name: 'Medium', value: 35, color: '#eab308' },
    { name: 'Low', value: 23, color: '#22c55e' },
  ]
  
  const recentActivity = [
    { time: '2 minutes ago', action: 'Code analysis completed', file: 'auth.py', status: 'success' },
    { time: '15 minutes ago', action: 'Security scan found 3 issues', file: 'api.ts', status: 'warning' },
    { time: '1 hour ago', action: 'Code generated successfully', file: 'utils.js', status: 'success' },
    { time: '2 hours ago', action: 'Performance analysis completed', file: 'database.py', status: 'success' },
  ]
  
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold mb-2">Dashboard</h1>
        <p className="text-slate-400">Overview of your code analysis and generation activities</p>
      </div>
      
      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {stats.map((stat) => {
          const Icon = stat.icon
          return (
            <div key={stat.label} className="bg-slate-800 rounded-lg p-6 border border-slate-700">
              <div className="flex items-center justify-between mb-4">
                <Icon className={`w-8 h-8 text-${stat.color}-500`} />
                <span className={`text-sm font-semibold text-${stat.color}-500`}>{stat.trend}</span>
              </div>
              <h3 className="text-3xl font-bold mb-1">{stat.value}</h3>
              <p className="text-slate-400 text-sm">{stat.label}</p>
            </div>
          )
        })}
      </div>
      
      {/* Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Activity Chart */}
        <div className="bg-slate-800 rounded-lg p-6 border border-slate-700">
          <h2 className="text-xl font-bold mb-4">Weekly Activity</h2>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={activityData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis dataKey="date" stroke="#94a3b8" />
              <YAxis stroke="#94a3b8" />
              <Tooltip contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155' }} />
              <Line type="monotone" dataKey="analyses" stroke="#3b82f6" strokeWidth={2} />
              <Line type="monotone" dataKey="generated" stroke="#10b981" strokeWidth={2} />
            </LineChart>
          </ResponsiveContainer>
        </div>
        
        {/* Severity Distribution */}
        <div className="bg-slate-800 rounded-lg p-6 border border-slate-700">
          <h2 className="text-xl font-bold mb-4">Issue Severity Distribution</h2>
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie
                data={severityData}
                cx="50%"
                cy="50%"
                labelLine={false}
                label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                outerRadius={100}
                fill="#8884d8"
                dataKey="value"
              >
                {severityData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </Pie>
              <Tooltip contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155' }} />
            </PieChart>
          </ResponsiveContainer>
        </div>
      </div>
      
      {/* Recent Activity */}
      <div className="bg-slate-800 rounded-lg p-6 border border-slate-700">
        <h2 className="text-xl font-bold mb-4">Recent Activity</h2>
        <div className="space-y-4">
          {recentActivity.map((activity, index) => (
            <div key={index} className="flex items-center justify-between p-4 bg-slate-900 rounded-lg">
              <div className="flex items-center gap-4">
                <div className={`w-2 h-2 rounded-full ${
                  activity.status === 'success' ? 'bg-green-500' : 'bg-yellow-500'
                }`}></div>
                <div>
                  <p className="font-medium">{activity.action}</p>
                  <p className="text-sm text-slate-400">{activity.file}</p>
                </div>
              </div>
              <div className="flex items-center gap-2 text-slate-400 text-sm">
                <Clock className="w-4 h-4" />
                {activity.time}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

export default Dashboard
import React from 'react'
import { Link, useLocation } from 'react-router-dom'
import { LayoutDashboard, Code, Sparkles, Search, Settings, FileCode } from 'lucide-react'

interface SidebarProps {
  isOpen: boolean
}

const Sidebar: React.FC<SidebarProps> = ({ isOpen }) => {
  const location = useLocation()
  
  const menuItems = [
    { path: '/', icon: LayoutDashboard, label: 'Dashboard' },
    { path: '/analyze', icon: Code, label: 'Code Analysis' },
    { path: '/generate', icon: Sparkles, label: 'Generate Code' },
    { path: '/query', icon: Search, label: 'RAG Query' },
    { path: '/settings', icon: Settings, label: 'Settings' },
  ]
  
  if (!isOpen) return null
  
  return (
    <aside className="w-64 bg-slate-800 border-r border-slate-700 p-4">
      <nav className="space-y-2">
        {menuItems.map((item) => {
          const Icon = item.icon
          const isActive = location.pathname === item.path
          
          return (
            <Link
              key={item.path}
              to={item.path}
              className={`flex items-center gap-3 px-4 py-3 rounded-lg transition-colors ${
                isActive
                  ? 'bg-blue-600 text-white'
                  : 'text-slate-300 hover:bg-slate-700'
              }`}
            >
              <Icon className="w-5 h-5" />
              <span className="font-medium">{item.label}</span>
            </Link>
          )
        })}
      </nav>
      
      <div className="mt-8 p-4 bg-slate-900 rounded-lg">
        <h3 className="text-sm font-semibold mb-2 text-slate-400">Quick Stats</h3>
        <div className="space-y-2 text-sm">
          <div className="flex justify-between">
            <span className="text-slate-400">Analyses</span>
            <span className="font-semibold text-blue-400">127</span>
          </div>
          <div className="flex justify-between">
            <span className="text-slate-400">Code Generated</span>
            <span className="font-semibold text-green-400">43</span>
          </div>
          <div className="flex justify-between">
            <span className="text-slate-400">Issues Found</span>
            <span className="font-semibold text-red-400">89</span>
          </div>
        </div>
      </div>
    </aside>
  )
}

export default Sidebar
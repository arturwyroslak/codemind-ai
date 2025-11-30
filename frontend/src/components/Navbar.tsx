import React from 'react'
import { Menu, Brain, Bell, User } from 'lucide-react'

interface NavbarProps {
  onMenuClick: () => void
}

const Navbar: React.FC<NavbarProps> = ({ onMenuClick }) => {
  return (
    <nav className="bg-slate-800 border-b border-slate-700 px-6 py-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <button
            onClick={onMenuClick}
            className="p-2 hover:bg-slate-700 rounded-lg transition-colors"
          >
            <Menu className="w-5 h-5" />
          </button>
          
          <div className="flex items-center gap-2">
            <Brain className="w-8 h-8 text-blue-500" />
            <div>
              <h1 className="text-xl font-bold">CodeMind AI</h1>
              <p className="text-xs text-slate-400">Intelligent Code Analysis Platform</p>
            </div>
          </div>
        </div>
        
        <div className="flex items-center gap-4">
          <button className="p-2 hover:bg-slate-700 rounded-lg transition-colors relative">
            <Bell className="w-5 h-5" />
            <span className="absolute top-1 right-1 w-2 h-2 bg-red-500 rounded-full"></span>
          </button>
          
          <button className="flex items-center gap-2 px-3 py-2 hover:bg-slate-700 rounded-lg transition-colors">
            <User className="w-5 h-5" />
            <span className="text-sm">Profile</span>
          </button>
        </div>
      </div>
    </nav>
  )
}

export default Navbar
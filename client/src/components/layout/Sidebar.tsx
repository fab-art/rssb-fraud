import React from 'react';
import {
  LayoutDashboard,
  FileText,
  ShieldAlert,
  Settings,
  LogOut,
  Database
} from 'lucide-react';
import { cn } from '@/lib/utils';

const Sidebar = () => {
  const menuItems = [
    { icon: LayoutDashboard, label: 'Dashboard', href: '/' },
    { icon: Database, label: 'Cases', href: '/cases' },
    { icon: ShieldAlert, label: 'Fraud Detection', href: '#' },
    { icon: FileText, label: 'Reports', href: '/reports' },
    { icon: Settings, label: 'Settings', href: '#' },
  ];

  return (
    <div className="flex flex-col w-64 bg-slate-900 text-slate-100 h-screen border-r border-slate-800 shrink-0">
      <div className="p-6">
        <h1 className="text-2xl font-bold flex items-center gap-2">
          <span className="text-emerald-400">💊</span> PharmaScan
        </h1>
        <p className="text-xs text-slate-500 mt-1 uppercase tracking-widest font-mono">
          Voucher Intelligence
        </p>
      </div>

      <nav className="flex-1 px-4 py-4 space-y-1">
        {menuItems.map((item) => (
          <a
            key={item.label}
            href={item.href}
            className={cn(
              "flex items-center gap-3 px-3 py-2 rounded-lg text-sm font-medium transition-colors",
              "hover:bg-slate-800 hover:text-emerald-400 text-slate-400"
            )}
          >
            <item.icon className="w-4 h-4" />
            {item.label}
          </a>
        ))}
      </nav>

      <div className="p-4 border-t border-slate-800">
        <button className="flex items-center gap-3 px-3 py-2 w-full rounded-lg text-sm font-medium text-slate-400 hover:bg-slate-800 hover:text-red-400 transition-colors">
          <LogOut className="w-4 h-4" />
          Logout
        </button>
      </div>
    </div>
  );
};

export default Sidebar;

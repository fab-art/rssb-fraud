import {
  Activity,
  AlertTriangle,
  CheckCircle2,
  Clock,
  FileStack,
  TrendingUp
} from "lucide-react";

export default function Dashboard() {
  const stats = [
    { label: "Active Cases", value: "12", icon: FileStack, color: "text-blue-400" },
    { label: "Pending Vouchers", value: "1,248", icon: Clock, color: "text-amber-400" },
    { label: "High Risk Flags", value: "84", icon: AlertTriangle, color: "text-red-400" },
    { label: "Verified Today", value: "312", icon: CheckCircle2, color: "text-emerald-400" },
  ];

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-3xl font-bold font-syne tracking-tight text-white">Dashboard</h1>
        <p className="text-slate-500 mt-2 font-mono text-sm uppercase tracking-wider">
          System Overview & Metrics
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {stats.map((stat) => (
          <div key={stat.label} className="bg-slate-900/50 border border-slate-800 p-6 rounded-2xl hover:border-slate-700 transition-all group">
            <div className="flex justify-between items-start">
              <div>
                <p className="text-xs font-bold text-slate-500 uppercase tracking-widest mb-1 group-hover:text-slate-400 transition-colors">
                  {stat.label}
                </p>
                <h3 className="text-3xl font-bold text-slate-100">{stat.value}</h3>
              </div>
              <div className={`p-2 bg-slate-800 rounded-xl ${stat.color}`}>
                <stat.icon className="w-5 h-5" />
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

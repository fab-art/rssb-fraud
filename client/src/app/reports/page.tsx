'use client';

import React, { useState, useEffect } from 'react';
import { FileSpreadsheet, Download, Calendar, CheckCircle2, Clock } from 'lucide-react';
import axios from 'axios';

const API_BASE = 'http://localhost:8000/api';

export default function ReportsPage() {
  const [cases, setCases] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => { fetchCases(); }, []);

  const fetchCases = async () => {
    try {
      const res = await axios.get(`${API_BASE}/cases`);
      setCases(res.data);
    } catch (err) { console.error(err); } finally { setLoading(false); }
  };

  const downloadReport = async (caseId: string, caseName: string) => {
    try {
      const res = await axios.get(`${API_BASE}/cases/${caseId}/report`, { responseType: 'blob' });
      const url = window.URL.createObjectURL(new Blob([res.data]));
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', `report_${caseName}.xlsx`);
      document.body.appendChild(link);
      link.click();
      link.remove();
    } catch (err) { console.error(err); }
  };

  if (loading) return <div className="p-8 text-white">Loading...</div>;

  return (
    <div className="p-8 space-y-8 text-white max-w-6xl mx-auto">
      <h1 className="text-3xl font-bold font-syne">Report Archive</h1>
      <div className="space-y-4">
        {cases.map((c) => (
          <div key={c.id} className="bg-slate-900 p-6 rounded-2xl flex justify-between items-center border border-slate-800">
            <div>
               <h3 className="font-bold text-lg">{c.name}</h3>
               <p className="text-sm text-slate-500">{c.status}</p>
            </div>
            <button onClick={() => downloadReport(c.id, c.name)} className="bg-emerald-500 text-black px-4 py-2 rounded font-bold">Download Excel</button>
          </div>
        ))}
      </div>
    </div>
  );
}

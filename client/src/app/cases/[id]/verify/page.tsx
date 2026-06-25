'use client';

import React, { useState, useEffect } from 'react';
import { useParams } from 'next/navigation';
import { CheckCircle2, AlertCircle, Search, ArrowLeft, ShieldCheck, ShieldAlert, Loader2, Calendar, User, CreditCard, Building2, Database, Check, Flag } from 'lucide-react';
import axios from 'axios';
import { cn } from '@/lib/utils';

const API_BASE = 'http://localhost:8000/api';

export default function VerificationPage() {
  const { id } = useParams();
  const [caseData, setCaseData] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [selectedRecord, setSelectedRecord] = useState<any>(null);
  const [searchTerm, setSearch] = useState('');
  const [deductionAmount, setDeductionAmount] = useState('0');
  const [deductionReason, setDeductionReason] = useState('');

  useEffect(() => { fetchCase(); }, [id]);
  useEffect(() => { if (selectedRecord) { setDeductionAmount(selectedRecord.deductionAmount?.toString() || '0'); setDeductionReason(selectedRecord.deductionReason || ''); } }, [selectedRecord]);

  const fetchCase = async () => {
    try {
      const res = await axios.get(`${API_BASE}/cases/${id}`);
      setCaseData(res.data);
      if (res.data.pharmacyRecords?.length > 0 && !selectedRecord) setSelectedRecord(res.data.pharmacyRecords[0]);
    } catch (err) { console.error(err); } finally { setLoading(false); }
  };

  const handleUpdateStatus = async (status: string) => {
    if (!selectedRecord) return;
    try {
      await axios.post(`${API_BASE}/cases/${id}/records/${selectedRecord.voucher_id}`, { status, deductionAmount: parseFloat(deductionAmount), deductionReason });
      await fetchCase();
    } catch (err) { console.error(err); }
  };

  if (loading) return <div className="text-white p-8">Loading...</div>;

  const records = caseData?.pharmacyRecords || [];

  return (
    <div className="h-screen flex flex-col gap-4 text-white">
      <div className="flex justify-between items-center px-4 py-2 bg-slate-900 border-b border-slate-800">
        <h1 className="font-bold">{caseData?.name}</h1>
        <a href="/reports" className="bg-emerald-500 text-black px-3 py-1 rounded text-sm">Reports</a>
      </div>
      <div className="flex flex-1 overflow-hidden">
        <div className="w-80 bg-slate-900 border-r border-slate-800 overflow-y-auto">
          {records.map((r: any) => (
            <button key={r.voucher_id} onClick={() => setSelectedRecord(r)} className={cn("w-full text-left p-4 border-b border-slate-800 hover:bg-slate-800", selectedRecord?.voucher_id === r.voucher_id && "bg-slate-800")}>
              <div className="text-xs text-slate-500">{r.voucher_id}</div>
              <div className="font-bold truncate">{r.patient_name}</div>
              <div className="text-xs">{r.status}</div>
            </button>
          ))}
        </div>
        <div className="flex-1 p-8 overflow-y-auto bg-slate-950">
          {selectedRecord ? (
            <div className="space-y-8 max-w-4xl mx-auto">
              <div className="bg-slate-900 p-6 rounded-xl border border-slate-800 flex justify-between items-center">
                 <div>
                    <div className="text-xl font-bold">{selectedRecord.patient_name}</div>
                    <div className="text-slate-400">Match: {selectedRecord.match?.status} ({(selectedRecord.match?.confidence*100).toFixed(0)}%)</div>
                 </div>
                 <div className="flex gap-2">
                    <button onClick={() => handleUpdateStatus('FLAGGED')} className="bg-red-500 px-4 py-2 rounded">Flag</button>
                    <button onClick={() => handleUpdateStatus('REVIEWED')} className="bg-emerald-500 px-4 py-2 rounded">Approve</button>
                 </div>
              </div>
              <div className="grid grid-cols-2 gap-4">
                 <div className="bg-slate-900 p-4 rounded border border-slate-800">
                    <h3 className="font-bold mb-2">Claim Details</h3>
                    <p>Date: {selectedRecord.visit_date}</p>
                    <p>Insurance: {selectedRecord.insurance_copay}</p>
                 </div>
                 <div className="bg-slate-900 p-4 rounded border border-slate-800">
                    <h4 className="font-bold mb-2">Manual Adjustment</h4>
                    <input type="number" className="w-full bg-black p-2 border border-slate-800 rounded" value={deductionAmount} onChange={(e) => setDeductionAmount(e.target.value)} />
                    <textarea className="w-full bg-black p-2 mt-2 border border-slate-800 rounded" value={deductionReason} onChange={(e) => setDeductionReason(e.target.value)} />
                 </div>
              </div>
            </div>
          ) : <div>Select a record</div>}
        </div>
      </div>
    </div>
  );
}

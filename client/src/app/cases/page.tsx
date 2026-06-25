'use client';

import React, { useState, useEffect } from 'react';
import { Plus, Upload, FileText, CheckCircle2, ArrowRight, Loader2, Database } from 'lucide-react';
import axios from 'axios';

const API_BASE = 'http://localhost:8000/api';

export default function CasesPage() {
  const [isCreating, setIsCreating] = useState(false);
  const [caseName, setCaseName] = useState('');
  const [currentCase, setcurrentCase] = useState<any>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState<string>('');

  const handleCreateCase = async () => {
    if (!caseName) return;
    setIsCreating(true);
    try {
      const formData = new FormData();
      formData.append('name', caseName);
      const res = await axios.post(`${API_BASE}/cases`, formData);
      setcurrentCase(res.data);
      setUploadStatus('Case created.');
    } catch (err) { console.error(err); } finally { setIsCreating(false); }
  };

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>, type: 'pharmacy' | 'facility') => {
    if (!e.target.files || !currentCase) return;
    const file = e.target.files[0];
    setIsUploading(true);
    setUploadStatus(`Uploading ${type}...`);
    const formData = new FormData();
    formData.append('file', file);
    try {
      const endpoint = type === 'pharmacy' ? 'upload-pharmacy' : 'upload-facility';
      await axios.post(`${API_BASE}/cases/${currentCase.id}/${endpoint}`, formData);
      const updated = await axios.get(`${API_BASE}/cases/${currentCase.id}`);
      setcurrentCase(updated.data);
      setUploadStatus(`${type} uploaded.`);
    } catch (err) { setUploadStatus('Upload failed.'); } finally { setIsUploading(false); }
  };

  const handleProcess = async () => {
    if (!currentCase) return;
    setIsUploading(true);
    setUploadStatus('Processing...');
    try {
      await axios.post(`${API_BASE}/cases/${currentCase.id}/process`);
      const updated = await axios.get(`${API_BASE}/cases/${currentCase.id}`);
      setcurrentCase(updated.data);
      setUploadStatus('Done!');
    } catch (err) { setUploadStatus('Failed.'); } finally { setIsUploading(false); }
  };

  return (
    <div className="space-y-8 max-w-5xl mx-auto text-white">
      <h1 className="text-3xl font-bold font-syne">Case Management</h1>
      {!currentCase ? (
        <div className="bg-slate-900 border border-slate-800 p-8 rounded-2xl text-center space-y-6">
          <input type="text" placeholder="Case Name" className="bg-slate-950 border border-slate-800 p-2 rounded" value={caseName} onChange={(e) => setCaseName(e.target.value)} />
          <button onClick={handleCreateCase} className="bg-emerald-500 text-black px-4 py-2 rounded">Create</button>
        </div>
      ) : (
        <div className="space-y-6">
          <div className="bg-slate-900 p-6 rounded-2xl border border-slate-800">
            <h3 className="text-lg font-bold">{currentCase.name} ({currentCase.status})</h3>
            <div className="grid grid-cols-2 gap-4 mt-4">
              <div className="p-4 border border-slate-800 rounded-xl">
                <p>Pharmacy Records: {currentCase.pharmacyRecords?.length || 0}</p>
                <input type="file" data-testid="pharmacy-upload" onChange={(e) => handleFileUpload(e, 'pharmacy')} />
              </div>
              <div className="p-4 border border-slate-800 rounded-xl">
                <p>Hospital Records: {currentCase.facilityRecords?.length || 0}</p>
                <input type="file" data-testid="facility-upload" onChange={(e) => handleFileUpload(e, 'facility')} />
              </div>
            </div>
            {uploadStatus && <p className="mt-4 font-mono text-emerald-400">{uploadStatus}</p>}
            <button onClick={handleProcess} className="mt-4 bg-emerald-500 text-black px-4 py-2 rounded">Process</button>
            {currentCase.status === 'REVIEWING' && (
               <a href={`/cases/${currentCase.id}/verify`} className="block mt-4 text-emerald-400 font-bold underline">Go to Verification Inbox</a>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

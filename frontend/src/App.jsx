import React, { useState, useEffect } from 'react';
import { FileText, Shuffle, ArrowRight } from 'lucide-react';

import FileUpload from './components/FileUpload';
import ProcessingStatus from './components/ProcessingStatus';
import ResultsView from './components/ResultsView';
import { uploadPDF, getJobStatus, getProcessingLogs } from './utils/api';

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [isUploading, setIsUploading] = useState(false);
  const [jobResult, setJobResult] = useState(null);
  const [logs, setLogs] = useState([]);
  const [currentStep, setCurrentStep] = useState('upload'); // upload, processing, completed

  // Poll for job status updates
  useEffect(() => {
    if (!jobResult || jobResult.status === 'completed' || jobResult.status === 'failed') {
      return;
    }

    const pollInterval = setInterval(async () => {
      try {
        const updatedResult = await getJobStatus(jobResult.job_id);
        setJobResult(updatedResult);
        
        // Get latest logs
        const logsResponse = await getProcessingLogs(jobResult.job_id);
        setLogs(logsResponse.logs || []);

        if (updatedResult.status === 'completed' || updatedResult.status === 'failed') {
          setCurrentStep('completed');
          clearInterval(pollInterval);
        }
      } catch (error) {
        console.error('Error polling job status:', error);
      }
    }, 2000);

    return () => clearInterval(pollInterval);
  }, [jobResult]);

  const handleFileSelect = (file) => {
    setSelectedFile(file);
  };

  const handleClearFile = () => {
    setSelectedFile(null);
  };

  const handleUpload = async () => {
    if (!selectedFile) return;

    setIsUploading(true);
    try {
      const response = await uploadPDF(selectedFile);
      
      // Initialize job result
      setJobResult({
        job_id: response.job_id,
        status: 'pending',
        original_filename: selectedFile.name,
        pages_reordered: [],
        document_analysis: { total_pages: 0 },
        processing_logs: []
      });
      
      setCurrentStep('processing');
    } catch (error) {
      console.error('Upload failed:', error);
      alert('Upload failed. Please try again.');
    } finally {
      setIsUploading(false);
    }
  };

  const handleStartNew = () => {
    setSelectedFile(null);
    setJobResult(null);
    setLogs([]);
    setCurrentStep('upload');
  };

  const handleRefresh = async () => {
    if (!jobResult) return;
    
    try {
      const updatedResult = await getJobStatus(jobResult.job_id);
      setJobResult(updatedResult);
      
      const logsResponse = await getProcessingLogs(jobResult.job_id);
      setLogs(logsResponse.logs || []);
    } catch (error) {
      console.error('Refresh failed:', error);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b">
        <div className="max-w-4xl mx-auto px-6 py-4">
          <div className="flex items-center space-x-3">
            <div className="flex items-center space-x-2">
              <FileText className="h-8 w-8 text-primary-600" />
              <Shuffle className="h-6 w-6 text-gray-400" />
              <ArrowRight className="h-5 w-5 text-gray-400" />
              <FileText className="h-8 w-8 text-green-600" />
            </div>
            <div>
              <h1 className="text-2xl font-bold text-gray-900">
                PDF Reconstruction Tool
              </h1>
              <p className="text-sm text-gray-600">
                Intelligently reorder shuffled PDF pages using AI
              </p>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-4xl mx-auto px-6 py-8">
        {currentStep === 'upload' && (
          <div className="space-y-8">
            {/* Upload Section */}
            <div className="text-center">
              <h2 className="text-xl font-semibold text-gray-900 mb-4">
                Upload Your Shuffled PDF
              </h2>
              <p className="text-gray-600 mb-8 max-w-2xl mx-auto">
                Upload a PDF with randomly shuffled pages, and our AI will analyze the content 
                to restore the document to its logical sequence. Perfect for mortgage documents, 
                contracts, and other multi-page files.
              </p>
            </div>

            <FileUpload
              onFileSelect={handleFileSelect}
              selectedFile={selectedFile}
              onClearFile={handleClearFile}
              isUploading={isUploading}
            />

            {selectedFile && (
              <div className="text-center">
                <button
                  onClick={handleUpload}
                  disabled={isUploading}
                  className="btn-primary text-lg px-8 py-3 disabled:opacity-50"
                >
                  {isUploading ? 'Uploading...' : 'Start Processing'}
                </button>
              </div>
            )}

            {/* Features */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mt-12">
              <div className="text-center p-6">
                <div className="bg-blue-100 rounded-full w-12 h-12 flex items-center justify-center mx-auto mb-4">
                  <FileText className="h-6 w-6 text-blue-600" />
                </div>
                <h3 className="font-medium text-gray-900 mb-2">Smart Analysis</h3>
                <p className="text-sm text-gray-600">
                  Uses OCR, semantic analysis, and AI reasoning to understand document structure
                </p>
              </div>
              
              <div className="text-center p-6">
                <div className="bg-green-100 rounded-full w-12 h-12 flex items-center justify-center mx-auto mb-4">
                  <Shuffle className="h-6 w-6 text-green-600" />
                </div>
                <h3 className="font-medium text-gray-900 mb-2">Multi-Strategy</h3>
                <p className="text-sm text-gray-600">
                  Combines page numbers, dates, content flow, and AI insights for optimal ordering
                </p>
              </div>
              
              <div className="text-center p-6">
                <div className="bg-purple-100 rounded-full w-12 h-12 flex items-center justify-center mx-auto mb-4">
                  <ArrowRight className="h-6 w-6 text-purple-600" />
                </div>
                <h3 className="font-medium text-gray-900 mb-2">Industry Focus</h3>
                <p className="text-sm text-gray-600">
                  Optimized for mortgage documents, lending files, and business contracts
                </p>
              </div>
            </div>
          </div>
        )}

        {(currentStep === 'processing' || currentStep === 'completed') && (
          <div className="space-y-8">
            <ProcessingStatus jobResult={jobResult} logs={logs} />
            
            {currentStep === 'completed' && jobResult?.status === 'completed' && (
              <ResultsView
                jobResult={jobResult}
                onStartNew={handleStartNew}
                onRefresh={handleRefresh}
              />
            )}

            {jobResult?.status === 'failed' && (
              <div className="text-center">
                <div className="card bg-red-50 border-red-200">
                  <h3 className="text-lg font-medium text-red-800 mb-2">
                    Processing Failed
                  </h3>
                  <p className="text-red-700 mb-4">
                    We encountered an error while processing your PDF. This might be due to:
                  </p>
                  <ul className="text-sm text-red-600 text-left max-w-md mx-auto space-y-1">
                    <li>• Corrupted or unreadable PDF file</li>
                    <li>• Extremely poor image quality for OCR</li>
                    <li>• Unsupported PDF format or encryption</li>
                    <li>• Server processing error</li>
                  </ul>
                  <button
                    onClick={handleStartNew}
                    className="btn-primary mt-6"
                  >
                    Try Another File
                  </button>
                </div>
              </div>
            )}
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="bg-white border-t mt-12">
        <div className="max-w-4xl mx-auto px-6 py-4">
          <div className="text-center text-sm text-gray-500">
            <p>Built for Value AI Labs • Developed by Prem</p>
            <p className="mt-1">
              Supports Gemini and Azure AI services for intelligent document analysis
            </p>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default App;
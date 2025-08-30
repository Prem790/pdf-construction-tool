import React from 'react';
import { Download, RefreshCw, Trash2, ExternalLink } from 'lucide-react';
import { downloadResult, deleteJob } from '../utils/api';

const ResultsView = ({ jobResult, onStartNew, onRefresh }) => {
  const handleDownload = async () => {
    try {
      await downloadResult(jobResult.job_id, `reordered_${jobResult.original_filename}`);
    } catch (error) {
      console.error('Download failed:', error);
      alert('Download failed. Please try again.');
    }
  };

  const handleDelete = async () => {
    if (window.confirm('Are you sure you want to delete this job?')) {
      try {
        await deleteJob(jobResult.job_id);
        onStartNew();
      } catch (error) {
        console.error('Delete failed:', error);
        alert('Delete failed. Please try again.');
      }
    }
  };

  const getConfidenceColor = (confidence) => {
    if (confidence > 0.8) return 'text-green-600 bg-green-50';
    if (confidence > 0.6) return 'text-yellow-600 bg-yellow-50';
    return 'text-red-600 bg-red-50';
  };

  const getOverallConfidence = () => {
    if (jobResult.pages_reordered?.length > 0) {
      const avgConfidence = jobResult.pages_reordered.reduce(
        (sum, page) => sum + page.confidence_score, 0
      ) / jobResult.pages_reordered.length;
      return avgConfidence;
    }
    return 0;
  };

  if (jobResult.status !== 'completed') {
    return null;
  }

  const overallConfidence = getOverallConfidence();

  return (
    <div className="space-y-6">
      {/* Success Header */}
      <div className="card bg-gradient-to-r from-green-50 to-emerald-50 border-green-200">
        <div className="text-center">
          <div className="mx-auto flex items-center justify-center h-12 w-12 rounded-full bg-green-100 mb-4">
            <Download className="h-6 w-6 text-green-600" />
          </div>
          <h2 className="text-xl font-semibold text-gray-900 mb-2">
            PDF Successfully Reconstructed!
          </h2>
          <p className="text-gray-600 mb-4">
            Your shuffled PDF has been analyzed and reordered with{' '}
            <span className={`px-2 py-1 rounded-full text-sm font-medium ${getConfidenceColor(overallConfidence)}`}>
              {(overallConfidence * 100).toFixed(0)}% confidence
            </span>
          </p>
        </div>
      </div>

      {/* Action Buttons */}
      <div className="flex flex-wrap gap-3 justify-center">
        <button
          onClick={handleDownload}
          className="btn-primary flex items-center space-x-2"
        >
          <Download className="h-4 w-4" />
          <span>Download Reordered PDF</span>
        </button>
        
        <button
          onClick={onRefresh}
          className="btn-secondary flex items-center space-x-2"
        >
          <RefreshCw className="h-4 w-4" />
          <span>Refresh Status</span>
        </button>
        
        <button
          onClick={onStartNew}
          className="btn-secondary flex items-center space-x-2"
        >
          <ExternalLink className="h-4 w-4" />
          <span>Process Another PDF</span>
        </button>
        
        <button
          onClick={handleDelete}
          className="text-red-600 hover:text-red-700 hover:bg-red-50 font-medium py-2 px-4 rounded-lg transition-colors duration-200 flex items-center space-x-2"
        >
          <Trash2 className="h-4 w-4" />
          <span>Delete Job</span>
        </button>
      </div>

      {/* Summary Stats */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="card text-center">
          <div className="text-2xl font-bold text-blue-600 mb-1">
            {jobResult.document_analysis?.total_pages || 0}
          </div>
          <div className="text-sm text-gray-500">Pages Processed</div>
        </div>
        
        <div className="card text-center">
          <div className="text-2xl font-bold text-green-600 mb-1">
            {jobResult.total_processing_time?.toFixed(1) || 0}s
          </div>
          <div className="text-sm text-gray-500">Processing Time</div>
        </div>
        
        <div className="card text-center">
          <div className={`text-2xl font-bold mb-1 ${
            overallConfidence > 0.8 ? 'text-green-600' :
            overallConfidence > 0.6 ? 'text-yellow-600' : 'text-red-600'
          }`}>
            {(overallConfidence * 100).toFixed(0)}%
          </div>
          <div className="text-sm text-gray-500">Confidence Score</div>
        </div>
      </div>

      {/* Document Information */}
      <div className="card">
        <h3 className="font-medium text-gray-900 mb-4">Document Information</h3>
        <div className="space-y-3">
          <div className="flex justify-between">
            <span className="text-gray-500">Original Filename:</span>
            <span className="font-medium">{jobResult.original_filename}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-500">Document Type:</span>
            <span className="font-medium capitalize">
              {jobResult.document_analysis?.document_type?.replace('_', ' ') || 'Unknown'}
            </span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-500">Job ID:</span>
            <span className="font-mono text-sm text-gray-600">{jobResult.job_id}</span>
          </div>
        </div>
      </div>

      {/* Issues Found */}
      {(jobResult.document_analysis?.missing_pages?.length > 0 || 
        jobResult.document_analysis?.duplicate_pages?.length > 0) && (
        <div className="card bg-yellow-50 border-yellow-200">
          <h3 className="font-medium text-yellow-800 mb-4">Issues Detected</h3>
          <div className="space-y-2 text-sm">
            {jobResult.document_analysis.missing_pages?.length > 0 && (
              <div className="flex items-start space-x-2">
                <span className="text-yellow-700 font-medium">Missing Pages:</span>
                <span className="text-yellow-800">
                  Pages {jobResult.document_analysis.missing_pages.join(', ')} appear to be missing from the sequence
                </span>
              </div>
            )}
            {jobResult.document_analysis.duplicate_pages?.length > 0 && (
              <div className="flex items-start space-x-2">
                <span className="text-yellow-700 font-medium">Duplicate Pages:</span>
                <span className="text-yellow-800">
                  Pages {jobResult.document_analysis.duplicate_pages.join(', ')} appear to be duplicated
                </span>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default ResultsView;
import React from 'react';
import { Loader2, CheckCircle, XCircle, Clock, FileText } from 'lucide-react';

const ProcessingStatus = ({ jobResult, logs = [] }) => {
  const getStatusIcon = (status) => {
    switch (status) {
      case 'pending':
        return <Clock className="h-5 w-5 text-yellow-500" />;
      case 'processing':
        return <Loader2 className="h-5 w-5 text-blue-500 animate-spin" />;
      case 'completed':
        return <CheckCircle className="h-5 w-5 text-green-500" />;
      case 'failed':
        return <XCircle className="h-5 w-5 text-red-500" />;
      default:
        return <Clock className="h-5 w-5 text-gray-500" />;
    }
  };

  const getStatusText = (status) => {
    switch (status) {
      case 'pending':
        return 'Queued for processing...';
      case 'processing':
        return 'Processing your PDF...';
      case 'completed':
        return 'Processing completed successfully!';
      case 'failed':
        return 'Processing failed';
      default:
        return 'Unknown status';
    }
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'pending':
        return 'text-yellow-700 bg-yellow-50 border-yellow-200';
      case 'processing':
        return 'text-blue-700 bg-blue-50 border-blue-200';
      case 'completed':
        return 'text-green-700 bg-green-50 border-green-200';
      case 'failed':
        return 'text-red-700 bg-red-50 border-red-200';
      default:
        return 'text-gray-700 bg-gray-50 border-gray-200';
    }
  };

  const getLogIcon = (level) => {
    switch (level.toLowerCase()) {
      case 'error':
        return <XCircle className="h-4 w-4 text-red-500" />;
      case 'warning':
        return <Clock className="h-4 w-4 text-yellow-500" />;
      default:
        return <CheckCircle className="h-4 w-4 text-blue-500" />;
    }
  };

  if (!jobResult) {
    return null;
  }

  return (
    <div className="space-y-6">
      {/* Status Header */}
      <div className={`card border-l-4 ${getStatusColor(jobResult.status)}`}>
        <div className="flex items-center space-x-3">
          {getStatusIcon(jobResult.status)}
          <div className="flex-1">
            <h3 className="font-medium text-gray-900">
              {getStatusText(jobResult.status)}
            </h3>
            <p className="text-sm text-gray-600 mt-1">
              Processing: {jobResult.original_filename}
            </p>
            {jobResult.total_processing_time && (
              <p className="text-xs text-gray-500 mt-1">
                Completed in {jobResult.total_processing_time.toFixed(2)} seconds
              </p>
            )}
          </div>
        </div>
      </div>

      {/* Document Analysis */}
      {jobResult.document_analysis && jobResult.status === 'completed' && (
        <div className="card">
          <h4 className="font-medium text-gray-900 mb-4 flex items-center">
            <FileText className="h-5 w-5 mr-2" />
            Document Analysis
          </h4>
          <div className="grid grid-cols-2 gap-4 text-sm">
            <div>
              <span className="text-gray-500">Total Pages:</span>
              <span className="ml-2 font-medium">{jobResult.document_analysis.total_pages}</span>
            </div>
            <div>
              <span className="text-gray-500">Document Type:</span>
              <span className="ml-2 font-medium capitalize">
                {jobResult.document_analysis.document_type?.replace('_', ' ') || 'Unknown'}
              </span>
            </div>
            {jobResult.document_analysis.missing_pages?.length > 0 && (
              <div className="col-span-2">
                <span className="text-red-600">Missing Pages:</span>
                <span className="ml-2 font-medium text-red-700">
                  {jobResult.document_analysis.missing_pages.join(', ')}
                </span>
              </div>
            )}
            {jobResult.document_analysis.duplicate_pages?.length > 0 && (
              <div className="col-span-2">
                <span className="text-yellow-600">Duplicate Pages:</span>
                <span className="ml-2 font-medium text-yellow-700">
                  {jobResult.document_analysis.duplicate_pages.join(', ')}
                </span>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Processing Logs */}
      {logs.length > 0 && (
        <div className="card">
          <h4 className="font-medium text-gray-900 mb-4">Processing Logs</h4>
          <div className="space-y-2 max-h-64 overflow-y-auto">
            {logs.map((log, index) => (
              <div
                key={index}
                className="flex items-start space-x-3 p-3 bg-gray-50 rounded-lg"
              >
                {getLogIcon(log.level)}
                <div className="flex-1 min-w-0">
                  <p className="text-sm text-gray-900">{log.message}</p>
                  <p className="text-xs text-gray-500 mt-1">
                    {new Date(log.timestamp).toLocaleTimeString()}
                  </p>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Page Reordering Details */}
      {jobResult.pages_reordered?.length > 0 && (
        <div className="card">
          <h4 className="font-medium text-gray-900 mb-4">Page Reordering</h4>
          <div className="space-y-2 max-h-48 overflow-y-auto">
            {jobResult.pages_reordered.map((page, index) => (
              <div
                key={index}
                className="flex items-center justify-between p-2 bg-gray-50 rounded"
              >
                <span className="text-sm">
                  Page {page.page_number} → Position {page.new_position + 1}
                </span>
                <div className="flex items-center space-x-2">
                  <span className="text-xs text-gray-500">
                    Confidence: {(page.confidence_score * 100).toFixed(0)}%
                  </span>
                  <div
                    className={`w-2 h-2 rounded-full ${
                      page.confidence_score > 0.8
                        ? 'bg-green-500'
                        : page.confidence_score > 0.6
                        ? 'bg-yellow-500'
                        : 'bg-red-500'
                    }`}
                  />
                </div>
              </div>
            ))}
          </div>
          {jobResult.pages_reordered[0]?.reasoning && (
            <div className="mt-4 p-3 bg-blue-50 rounded-lg">
              <p className="text-sm text-blue-800">
                <strong>Reasoning:</strong> {jobResult.pages_reordered[0].reasoning}
              </p>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default ProcessingStatus;
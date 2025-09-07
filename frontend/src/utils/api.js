import axios from 'axios';

const API_BASE = 'https://pdf-construction-tool.onrender.com';
// const API_BASE = 'http://localhost:8000/api';

const api = axios.create({
  baseURL: API_BASE,
  timeout: 600000,
});

export const uploadPDF = async (file) => {
  const formData = new FormData();
  formData.append('file', file);
  
  const response = await api.post('/upload', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });
  
  return response.data;
};

export const getJobStatus = async (jobId) => {
  const response = await api.get(`/status/${jobId}`);
  return response.data;
};

export const downloadResult = async (jobId, filename) => {
  try {
    const response = await api.get(`/download/${jobId}`, {
      responseType: 'blob',
    });
    
    // Create blob with explicit PDF type
    const blob = new Blob([response.data], { type: 'application/pdf' });
    
    // Create download link
    const url = window.URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = filename || 'reordered.pdf';
    
    // Important: Add to body before clicking
    document.body.appendChild(link);
    link.click();
    
    // Cleanup after a short delay
    setTimeout(() => {
      document.body.removeChild(link);
      window.URL.revokeObjectURL(url);
    }, 100);
    
  } catch (error) {
    console.error('Download error details:', error);
    throw error;
  }
};

export const getProcessingLogs = async (jobId) => {
  const response = await api.get(`/logs/${jobId}`);
  return response.data;
};

export const deleteJob = async (jobId) => {
  const response = await api.delete(`/jobs/${jobId}`);
  return response.data;
};

export default api;




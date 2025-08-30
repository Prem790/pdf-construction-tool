import axios from 'axios';

const API_BASE = 'http://localhost:8000/api';

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
  const response = await api.get(`/download/${jobId}`, {
    responseType: 'blob',
  });
  
  // Create download link
  const url = window.URL.createObjectURL(new Blob([response.data]));
  const link = document.createElement('a');
  link.href = url;
  link.setAttribute('download', filename || 'reordered.pdf');
  document.body.appendChild(link);
  link.click();
  link.remove();
  window.URL.revokeObjectURL(url);
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
import { useState, useEffect } from 'react';
import axios from 'axios';

const API_BASE_URL = "http://localhost:5001"; 

export default function useVideoProcessing() {
  const [state, setState] = useState({
    loading: false,
    results: [],
    message: "",
    file: null,
    ballVideos: [],
    frameData: [],
    videoName: null,
    status: null,
    bounceResults: [],
    progress: 0,
    stage: ""
  });

  // Load any existing processing state from localStorage
  useEffect(() => {
    const savedState = localStorage.getItem('videoProcessingState');
    if (savedState) {
      const parsedState = JSON.parse(savedState);
      setState(prev => ({ ...prev, ...parsedState }));
      
      // If we were in the middle of processing, resume polling
      if (parsedState.loading && parsedState.videoName) {
        startPolling(parsedState.videoName);
      }
    }
  }, []);

  // Save state to localStorage whenever it changes
  useEffect(() => {
    if (state.videoName) {
      localStorage.setItem('videoProcessingState', JSON.stringify(state));
    }
  }, [state]);

  const checkStatus = async (videoName) => {
    try {
      const response = await axios.get(
        `http://localhost:5001/status/${videoName}`
      );
      return response.data;
    } catch (error) {
      console.error("Error checking status:", error);
      return { status: "error", error: "Failed to check status" };
    }
  };

  const handleUpload = async (file) => {
    if (!file) {
      setState(prev => ({ 
        ...prev, 
        message: "Please select a video file before uploading." 
      }));
      return null;
    }

    const formData = new FormData();
    formData.append("file", file);

    setState(prev => ({
      ...prev,
      loading: true,
      message: "",
      results: [],
      frameData: [],
      ballVideos: [],
      file,
      videoName: file.name.split('.')[0],
      bounceResults: [],
      progress: 0,
      status: "uploading"
    }));

    try {
      const response = await axios.post(
        `${API_BASE_URL}/upload`,
        formData,
        { headers: { "Content-Type": "multipart/form-data" }, timeout: 3000 } 
      );
      
      setState(prev => ({
        ...prev,
        filePath: response.data.file_path,
        videoName: response.data.video_name,
        status: "processing"
      }));

      return response.data.video_name;
    } catch (error) {
      console.error("Error uploading file:", error);
      setState(prev => ({
        ...prev,
        message: `Upload failed: ${error.message}. Is the backend server running?`,
        loading: false,
        status: "error"
      }));
      return null;
    }
  };

  const startPolling = (videoName) => {
    const intervalId = setInterval(async () => {
      const status = await checkStatus(videoName);
      
      if (status.status === "completed") {
        clearInterval(intervalId);
        setState(prev => ({
          ...prev,
          loading: false,
          results: status.results || [],
          frameData: status.frame_data || [],
          ballVideos: status.ball_videos || [],
          progress: 100,
          bounceResults: status.bounce_results || [],
          status: "completed",
          stage: ""
        }));
        localStorage.removeItem('videoProcessingState');
      } else if (status.status === "error") {
        clearInterval(intervalId);
        setState(prev => ({
          ...prev,
          loading: false,
          message: status.error || "Error processing file",
          status: "error",
          stage: ""
        }));
      } else {
        setState(prev => ({
          ...prev,
          progress: status.progress || 0,
          stage: status.stage || "",
          status: status.status || "processing"
        }));
      }
    }, 3000);

    return () => clearInterval(intervalId);
  };

  const resetState = () => {
    setState({
      loading: false,
      results: [],
      message: "",
      file: null,
      ballVideos: [],
      frameData: [],
      videoName: null,
      status: null,
      progress: 0,
      stage: ""
    });
    localStorage.removeItem('videoProcessingState');
  };

  return { 
    ...state, 
    handleUpload, 
    startPolling, 
    checkStatus,
    resetState
  };
}
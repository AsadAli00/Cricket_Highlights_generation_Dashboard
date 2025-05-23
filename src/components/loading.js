import React from 'react';
import '../CSS/loading.css';

const Loading = () => {
  return (
    <div className="loading-container">
      <div className="loading-spinner"></div>
      <p>Processing video...</p>
    </div>
  );
};

export default Loading;
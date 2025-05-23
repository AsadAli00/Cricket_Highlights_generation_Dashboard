
import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Bar } from 'react-chartjs-2';
import Loading from './loading';
import '../CSS/dashboard.css';
import { Chart } from 'chart.js/auto';

const Dashboard = () => {
  const [file, setFile] = useState(null);
  const [message, setMessage] = useState('');
  const [videoName, setVideoName] = useState('');
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [videoSrc, setVideoSrc] = useState(null);
  const [bounceData, setBounceData] = useState([]);
  const [highlightsVideo, setHighlightsVideo] = useState(null);
  const [hoveredFrame, setHoveredFrame] = useState(null);


  const videoRef = useRef(null);

  const handleFileChange = (event) => {
    setFile(event.target.files[0]);
  };

  const handleUpload = async () => {
    const formData = new FormData();
    formData.append('file', file);

    setLoading(true);
    setMessage('');
    setResults([]);

    try {
      const response = await axios.post('http://localhost:5000/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      setMessage(response.data.message);
      setResults(response.data.results);
      setVideoSrc(response.data.video_name);
      setVideoName(response.data.video_name);
      setBounceData(response.data.bounce_results);
      setHighlightsVideo(response.data.highlights_video);
    } catch (error) {
      setMessage('Error uploading file');
    } finally {
      setLoading(false);
    }
  };

  const handleMouseMove = (event) => {
    const rect = event.currentTarget.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const frameIndex = Math.floor((x / rect.width) * results.length);
    setHoveredFrame(results[frameIndex]);
  };

  const handleMouseLeave = () => {
    setHoveredFrame(null);
  };

  const data = {
    labels: results.map(result => result.class_label),
    datasets: [
      {
        label: 'Predicted Class',
        data: results.map(result => result.predicted_class),
        backgroundColor: 'rgba(75, 192, 192, 0.2)',
        borderColor: 'rgba(75, 192, 192, 1)',
        borderWidth: 1,
      },
    ],
  };


  const options = {
    scales: {
      y: {
        beginAtZero: true,
        type: 'linear',
      },
    },
    onHover: (event, elements) => {
      if (elements.length) {
        const index = elements[0].index;
        setHoveredFrame(results[index]);
      }
    },
  };


  useEffect(() => {
    // Clean up the canvas element when the component unmounts
    return () => {
      const canvas = document.getElementById('myChart');
      if (canvas) {
        canvas.remove();
      }
    };
  }, []);


  return (
    <div className="dashboard container mt-5">
      <h1 className="text-center mb-4 text-info font-weight-bold">Cricket Shot Classification Dashboard</h1>
      <p className="text-center mb-4 text-info font-weight-bold font-italic text-secondary">By Asad Ali</p>
      <div className="card">
        <div className="card-body">
          <div className="input-group mb-3">
            <input type="file" className="form-control" onChange={handleFileChange} />
            <button className="btn btn-secondary" onClick={handleUpload}>Upload</button>
          </div>
          {loading ? (
            <Loading />
          ) : (
            <>
                {message && <p className="text-center">{message}</p>}
              {results.length > 0 && (
                <div>
                  <h2 className="text-center mb-4">Classification Results</h2>
                  <div
                    className="highlights-bar"
                    onMouseMove={handleMouseMove}
                    onMouseLeave={handleMouseLeave}
                  >
                    <Bar data={data} options={options} />
                  </div>
                  {hoveredFrame && (
                    <div className="frame-preview">
                      <img src={`http://localhost:5000/video/${hoveredFrame.video}`} alt="Frame preview" />
                      <p>Frame: {hoveredFrame.class_label}</p>
                    </div>
                  )}
                  <ul className="list-group mt-4">
                    {results.map((result, index) => (
                      <li key={index} className="list-group-item">
                        <p>Video: {result.video}</p>
                        <p>Predicted Class: {result.predicted_class}</p>
                        <p>Frame Range: {result.class_label}</p>
                      </li>
                    ))}
                  </ul>
                  {highlightsVideo && (
                    <div className="video-container mt-4">
                      <h3>Complete Highlights Video</h3>
                      <video ref={videoRef} width="640" height="360" controls>
                        <source src={`http://localhost:5000/video/${highlightsVideo}`} type="video/mp4" />
                      </video>
                    </div>
                  )}
                </div>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
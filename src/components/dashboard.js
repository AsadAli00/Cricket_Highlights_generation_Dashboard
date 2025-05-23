import React, { useState, useEffect, useRef } from "react";
import axios from "axios";
import { Bar, Doughnut } from "react-chartjs-2";
import ReactPlayer from "react-player";
import Slider from "react-slick";
import "slick-carousel/slick/slick.css";
import "slick-carousel/slick/slick-theme.css";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
  ArcElement,
  LineElement,
  PointElement,
} from "chart.js";
import useVideoProcessing from "./useVideoProcessing.js";
import Loading from "./loading.js";
import "../CSS/dashboard.css";

// const highlights = [
//   { time: 10, type: "boundary", label: "4 runs" },
//   { time: 75, type: "wicket", label: "Wicket" },
//   { time: 125, type: "boundary", label: "6 runs" },
//   { time: 190, type: "replay", label: "Replay" },
// ];

// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
  ArcElement,
  LineElement,
  PointElement
);

const Dashboard = ({ isSidebarOpen }) => {
  const {
    loading,
    results,
    message,
    file,
    ballVideos,
    frameData,
    videoName,
    status,
    bounceResults,
    progress,
    stage,
    handleUpload,
    startPolling,
    resetState,
  } = useVideoProcessing();
  const videoRef = useRef(null);

  const settings = {
    dots: true,
    infinite: false,
    speed: 500,
    slidesToShow: 3, // Number of slides to show at once
    slidesToScroll: 3, // Number of slides to scroll
    responsive: [
      {
        breakpoint: 1024,
        settings: {
          slidesToShow: 3,
          slidesToScroll: 3,
        },
      },
      {
        breakpoint: 600,
        settings: {
          slidesToShow: 2,
          slidesToScroll: 2,
        },
      },
    ],
  };

  // const [loading, setLoading] = useState(false);
  // const [results, setResults] = useState([]);
  // const [message, setMessage] = useState("");
  // const [file, setFile] = useState(null);
  // const [ballVideos, setBallVideos] = useState([]);
  // const [hoveredFrame, setHoveredFrame] = useState(null);
  // const [frameData, setFrameData] = useState([]);
  // const [filePath, setFilePath] = useState(null);
  // const [status, setStatus] = useState(null);
  // const videoRef = useRef(null);

  const [hoveredFrame, setHoveredFrame] = useState(null);
  const [selectedFile, setSelectedFile] = useState(null);
  const [playing, setPlaying] = useState(false);

  const handleFileChange = (event) => {
    setSelectedFile(event.target.files[0]);
  };

  const handleUploadClick = async () => {
    if (!selectedFile) {
      alert("Please select a file first");
      return;
    }
    const name = await handleUpload(selectedFile);
    if (name) {
      startPolling(name);
    }
  };

  const handleReset = () => {
    resetState();
    window.location.reload();
    setSelectedFile(null);
    setHoveredFrame(null);
  };

  //   const formData = new FormData();
  //   formData.append("file", selectedFile);

  //   try {
  //     const response = await axios.post(
  //       "http://localhost:5001/upload",
  //       formData,
  //       {
  //         headers: {
  //           "Content-Type": "multipart/form-data",
  //         },
  //       }
  //     );
  //     setFilePath(response.data.file_path);
  //     checkStatus();
  //     console.log("processing")
  //   } catch (error) {
  //     console.error("Error uploading file:", error);
  //     setMessage("Error uploading file. Please try again.");
  //     setLoading(false);
  //   }
  // };

  // const checkStatus = async () => {
  //   if (!filePath) return;
  //   try {
  //     const response = await axios.get(
  //       `http://localhost:5001/status/${encodeURIComponent(filePath)}`
  //     );
  //     const data = response.data;
  //     console.log("status processing")
  //     setStatus(data.status);
  //     if (data.status === "completed") {
  //       setResults(data.results);
  //       setFrameData(data.frame_data);
  //       // setBallVideos(data.ball_videos);
  //       console.log("Done")
  //       setLoading(false);
  //     } else if (data.status === "error") {
  //       setMessage("Error processing file.");
  //       setLoading(false);
  //     }
  //   } catch (error) {
  //     console.error("Error checking status:", error);
  //      setMessage("Error checking status. Please try again.")
  //     setLoading(false);
  //   }
  // };

  const classCounts = results.reduce((acc, curr) => {
    const label = curr.class_label;
    acc[label] = (acc[label] || 0) + 1;
    return acc;
  }, {});

  const barData = {
    labels: results.map((r) => r.class_label),
    datasets: [
      {
        label: "Predicted Class",
        data: results.map((r) => r.predicted_class),
        backgroundColor: "rgba(54, 162, 235, 0.5)",
        borderColor: "rgba(54, 162, 235, 1)",
        borderWidth: 1,
      },
    ],
  };

  const barOptions = {
    indexAxis: "y",
    responsive: true,
    plugins: {
      legend: {
        position: "top",
      },
      title: {
        display: true,
        text: "Shot Classification Results",
      },
    },
  };

  const doughnutData = {
  labels: Object.keys(classCounts),
  datasets: [
    {
      label: "Class Distribution",
      data: Object.values(classCounts),
      backgroundColor: [
        "rgba(255, 99, 132, 0.5)",
        "rgba(54, 162, 235, 0.5)",
        "rgba(255, 206, 86, 0.5)",
        "rgba(75, 192, 192, 0.5)",
        "rgba(153, 102, 255, 0.5)",
      ],
      borderColor: [
        "rgba(255, 99, 132, 1)",
        "rgba(54, 162, 235, 1)",
        "rgba(255, 206, 86, 1)",
        "rgba(75, 192, 192, 1)",
        "rgba(153, 102, 255, 1)",
      ],
      borderWidth: 1,
    },
  ],
};

  return (
    <div
      className={`dashboard ${
        isSidebarOpen ? "sidebar-open" : "sidebar-closed"
      }`}
    >
      <div className="title-bar justify-content-center">
        <div className="text-center">
          <h1>Intelligent Shot Detection And Cricket Highlights</h1>
          <p className="owner-name text-center">
            {" "}
            <i> Presented By: </i>
            <b>Asad Ali</b>
          </p>
          <p className="description text-center">
            Batsman statistics and performance analytics for cricket matches Using AI.
          </p>
        </div>
      </div>

      {/* Dashboard Content */}

      <div className="container-fluid  dashboard-main bg-light p-5 mb-4 rounded shadow-sm">
        <div className="title-content justify-content-center">
          <h5 className="title-content justify-content-center">Upload Video</h5>
          {loading && (
            <div className="progress mt-3 mb-3">
              <div
                className="progress-bar progress-bar-striped progress-bar-animated"
                role="progressbar"
                style={{ width: `${progress}%` }}
                aria-valuenow={progress}
                aria-valuemin="0"
                aria-valuemax="100"
              >
                {stage} ({progress}%)
              </div>
            </div>
          )}
        </div>
        <div className="justify-content-center align-items-center">
          <div className="input-group mb-3">
            <input
              type="file"
              className="form-control"
              onChange={handleFileChange}
              disabled={loading}
              accept="video/*"
            />
            <button
              className="btn btn-secondary"
              onClick={handleUploadClick}
              disabled={loading || !selectedFile}
            >
              {loading ? "Processing..." : "Upload"}
            </button>
            {!loading && (results.length > 0 || message) && (
              <button className="btn btn-secondary ms-2" onClick={handleReset}>
                Reset
              </button>
            )}
          </div>

          {message && (
            <div
              className={`alert ${
                status === "error" ? "alert-danger" : "alert-info"
              }`}
            >
              {message}
            </div>
          )}
          {loading && (
            <div className="text-center">
              <Loading />
              <p className="text-muted mt-2">
                Processing {videoName} - {stage}...
              </p>
            </div>
          )}
          {!loading && results.length > 0 && (
            <div>
              <h4 className="title-content justify-content-center">
                Classification Results
              </h4>
              <div className="row mt-4 justify-content-between">
                <div className="col-md-6 mb-4">
                  <div className="card h-60">
                    <div className="card-body">
                      <h5 className="card-title">Shot Classification</h5>
                      <Bar data={barData} options={barOptions} />
                    </div>
                  </div>
                </div>
                <div className="col-md-5 mb-4">
                  <div className="card h-60">
                    <div className="card-body">
                      <h5 className="card-title">Shot Distribution</h5>
                      <Doughnut data={doughnutData} />
                    </div>
                  </div>
                </div>
              </div>
              {/* {frameData.length > 0 && (
                <div className="timeline-fixed-bar mt-4">
                  <h4 className="mb-2">Timeline</h4>
                  <div className="frame-strip d-flex flex-nowrap overflow-auto p-2 bg-dark rounded">
                    {frameData.map((frame, index) => (
                      <div
                        key={index}
                        className="frame-thumbnail text-center mx-1"
                        onMouseEnter={() => setHoveredFrame(frame)}
                        onMouseLeave={() => setHoveredFrame(null)}
                      >
                        <img
                          src={`/annotated_frames/${frame.frame}`}
                          alt={frame.frame}
                          className="img-thumbnail"
                          style={{
                            height: "50px",
                            width: "auto",
                            objectFit: "cover",
                          }}
                        />
                        <p
                          className="text-light"
                          style={{ fontSize: "0.7rem" }}
                        >
                          {frame.frame}
                        </p>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {hoveredFrame && (
                <div className="frame-preview-container text-center mt-3">
                  <h6>Preview</h6>
                  <img
                    src={`/annotated_frames/${hoveredFrame.frame}`}
                    alt="Preview"
                    className="img-fluid rounded shadow"
                    style={{ maxHeight: "250px", border: "2px solid #ccc" }}
                  />
                </div>
              )} */}
              {/* {frameData.map((frame,index)=>{
                console.log(frame.frame)
              })} */}
              {/* {frameData.length > 0 && (
                <div className="row mt-4">
                  <div className="col-16">
                    <h4 className="title-content justify-content-center">
                      Processed Frames for each ball
                    </h4>
                    <div className="video-bar justify-content-between">
                      {frameData.map((frame, index) => (
                        <div
                          key={index}
                          className="frame-thumbnail m-2"
                          onMouseEnter={() => setHoveredFrame(frame)}
                          onMouseLeave={() => setHoveredFrame(null)}
                        >
                          <img
                            src={`/annotated_frames/${frame.frame}`}
                            alt={frame.frame}
                            className="img-thumbnail"
                            style={{ height: "50px", width: "50px" }}
                          />
                          <p>
                            <small>{frame.frame}</small>
                          </p>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              )}
              {hoveredFrame && (
                <div className="row mt-3">
                  <div className="col-12 text-center">
                    <div className="frame-preview">
                      <img
                        src={`/annotated_frames/${hoveredFrame.frame}`}
                        alt="Preview"
                        className="img-fluid rounded"
                        style={{ maxHeight: "400px" }}
                      />
                    </div>
                  </div>
                </div>
              )} */}
              <div className="container">
                {frameData.length > 0 && (
                  <div className="row mt-4">
                    <div className="col-12">
                      <h4 className="title-content text-center">
                        Processed Frames for each ball
                      </h4>
                      <div
                        id="frameCarousel"
                        className="carousel slide"
                        data-ride="carousel"
                      >
                        <div className="carousel-inner">
                          {frameData.map((frame, index) => (
                            <div
                              key={index}
                              className={`carousel-item ${
                                index === 0 ? "active" : ""
                              }`}
                            >
                              <img
                                src={`/${videoName}/annotated_frames/${frame.frame}`}
                                className="d-block"
                                alt={frame.frame}
                                style={{
                                  height: "400px",
                                  objectFit: "fit",
                                }}
                              />
                              <div className="carousel-caption d-none d-md-block">
                                <p>{frame.frame}</p>
                              </div>
                            </div>
                          ))}
                        </div>
                        <a
                          className="carousel-control-prev"
                          href="#frameCarousel"
                          role="button"
                          data-slide="prev"
                        >
                          <span
                            className="carousel-control-prev-icon"
                            aria-hidden="true"
                          ></span>
                          <span className="sr-only">Previous</span>
                        </a>
                        <a
                          className="carousel-control-next"
                          href="#frameCarousel"
                          role="button"
                          data-slide="next"
                        >
                          <span
                            className="carousel-control-next-icon"
                            aria-hidden="true"
                          ></span>
                          <span className="sr-only">Next</span>
                        </a>
                      </div>
                    </div>
                  </div>
                )}
                {hoveredFrame && (
                  <div className="row mt-3">
                    <div className="col-12 text-center">
                      <div className="frame-preview">
                        <img
                          src={`/${videoName}/annotated_frames/${hoveredFrame.frame}`}
                          alt="Preview"
                          className="img-fluid rounded"
                          style={{ maxHeight: "400px" }}
                        />
                      </div>
                    </div>
                  </div>
                )}
              </div>
              <div />
              <div />
            </div>
          )}
          {ballVideos && ballVideos.length > 0 && (
            <div className="video-analysis-section">
              <h3 className="section-title">Batsman Shot Analysis</h3>
              <div className="video-grid">
                {ballVideos.map((ball, index) => {
                  const videoUrl = `/${videoName}/${ball.ball_video}`;
                  return (
                    <div key={index} className="video-card">
                      <div className="player-wrapper">
                        <ReactPlayer
                          ref={videoRef}
                          url={videoUrl}
                          width="100%"
                          height="100%"
                          controls={true}
                          playing={playing}
                          light={""}
                          playIcon={
                            <div className="play-button">
                              <i className="fas fa-play"></i>
                            </div>
                          }
                          onError={(e) => console.error("Video error:", e)}
                          config={{
                            file: {
                              attributes: {
                                controlsList: "nodownload", // Disable download option
                              },
                            },
                          }}
                        />
                      </div>
                      <div className="video-info">
                        <span className="ball-number">Ball {index + 1}</span>
                        <span className="shot-type">
                          {results[index]?.class_label || "Cricket Shot"}
                        </span>
                        <div className="video-stats">
                          <span>
                            <i className="fas fa-clock"></i>{" "}
                            {ball.duration || "00:15"}
                          </span>
                          {/* <span>
                            <i className="fas fa-tachometer-alt"></i>{" "}
                            {ball.speed || "120"} km/h
                          </span> */}
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}
          {/* {console.log(bounceResults)} */}
          {status == "completed" && (
            <div className="results-container">
              {status === "completed" && (
                <div className="container">
                  <h2 className="section-title text-center my-4">
                    Ball Bounce Classification Results
                  </h2>
                  <div className="row">
                    {Object.keys(bounceResults).map((key, index) => (
                      <div
                        className="col-sm-6 col-md-4 col-lg-3 mb-4"
                        key={index}
                      >
                        <div className="card h-100">
                          <div className="card-img-container">
                            <img
                              src={`/${videoName}/ball_${key}_hit.jpg`}
                              className="card-img-top"
                              alt={`Ball ${index + 1}`}
                            />
                          </div>
                          <div className="card-body">
                            <h5 className="card-title">Ball {index + 1}</h5>
                            <p className="card-text">
                              <strong>Classification:</strong>{" "}
                              {bounceResults[key]}
                            </p>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
          {/* <div className="container">
            <h2 className="text-center my-4">
              Ball Bounce Classification Results
            </h2>
            <div className="row">
              {bounceResults.map((ball, index) => (
                <div key={index} className="col-md-4 mb-4">
                  <div className="card">
                    <img
                      src={`/annotated_frames/${ball.frame}`}
                      className="card-img-top"
                      alt={`Ball ${index + 1}`}
                      style={{ height: "200px", objectFit: "cover" }}
                    />
                    <div className="card-body">
                      <h5 className="card-title">Ball {index + 1}</h5>
                      <p className="card-text">
                        <strong>Classification:</strong> {ball.classification}
                      </p>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div> */}
          {status == "completed" && (
            <div>
              <h3 className="section-title">Complete Highlights Video</h3>
              <div className="video-card">
                <div className="player-wrapper">
                  <ReactPlayer
                    ref={videoRef}
                    url={`/${videoName}/${videoName}_highlights.mp4`}
                    width="100%"
                    height="100%"
                    controls={true}
                    playing={false} // Set to false if you don't want it to autoplay
                    light={""}
                    playIcon={
                      <div className="play-button">
                        <i className="fas fa-play"></i>
                      </div>
                    }
                    onError={(e) => console.error("Video error:", e)}
                    config={{
                      file: {
                        attributes: {
                          controlsList: "nodownload", // Disable download option
                        },
                      },
                    }}
                  />
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default Dashboard;

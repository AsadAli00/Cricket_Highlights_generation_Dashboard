// src/components/Layout.js
import React, { useState } from "react";
import { Link } from "react-router-dom";
import "bootstrap/dist/css/bootstrap.min.css";
import "../CSS/Layout.css";
import image from "../Images/main_background.jpg";
import Dashboard from "./dashboard";

const Layout = () => {
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);

  const toggleSidebar = () => {
    setIsSidebarOpen(!isSidebarOpen);
  };

  return (
    <div className="layout">
      {/* Sidebar */}
      <div className={`sidebar ${isSidebarOpen ? "open" : "closed"}`}>
        <div className="sidebar-header">
          <img src={image} alt="Logo" className="logo" />
          <h2>Cricket Dashboard</h2>
        </div>
        <div className="sidebar-menu">
          <Link to="/" className="dashboard-link">
            Dashboard
          </Link>
        </div>
        <div className="sidebar-footer">
          <button className="btn btn-danger" onClick={toggleSidebar}>
            Close Sidebar
          </button>
        </div>
      </div>

      {/* Main Content Area */}
      <div
        className={`main-content ${
          isSidebarOpen ? "sidebar-open" : "sidebar-closed"
        }`}
      >
        {/* Menu button OUTSIDE title bar */}
        {!isSidebarOpen && (
          <button className="menu-button" onClick={toggleSidebar}>
            &#9776;
          </button>
        )}
        {/* Dashboard inside */}
        <MemoizedDashboard />
      </div>
    </div>
  );
};

const MemoizedDashboard = React.memo(Dashboard);

export default Layout;

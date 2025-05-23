import React from "react";
import Dashboard from "./components/dashboard";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Layout from "./components/layout";
// import Dashboard from './components/dashboard';

const App = () => {
  return (
    <Router>
      <Layout>
        <Routes>
          <Route path="/dashboard" element={<MemoizedDashboard />} />
          <Route path="/" element={<MemoizedDashboard />} /> {/* Default route */}
        </Routes>
      </Layout>
    </Router>
    // <>
    // <Dashboard />
    // </>
  );
};

// Memoize the Dashboard component
const MemoizedDashboard = React.memo(Dashboard);

export default App;

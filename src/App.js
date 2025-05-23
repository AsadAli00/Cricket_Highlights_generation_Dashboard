<<<<<<< HEAD
import React from "react";
import Dashboard from "./components/dashboard";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Layout from "./components/layout";
// import Dashboard from './components/dashboard';
=======
import logo from './logo.svg';
import './App.css';
>>>>>>> parent of 742c67b (dashboard testing done all working)

const App = () => {
  return (
<<<<<<< HEAD
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
=======
    <div className="App">
      <header className="App-header">
        <img src={logo} className="App-logo" alt="logo" />
        <p>
          Edit <code>src/App.js</code> and save to reload.
        </p>
        <a
          className="App-link"
          href="https://reactjs.org"
          target="_blank"
          rel="noopener noreferrer"
        >
          Learn React
        </a>
      </header>
    </div>
>>>>>>> parent of 742c67b (dashboard testing done all working)
  );
};

<<<<<<< HEAD
// Memoize the Dashboard component
const MemoizedDashboard = React.memo(Dashboard);

=======
>>>>>>> parent of 742c67b (dashboard testing done all working)
export default App;

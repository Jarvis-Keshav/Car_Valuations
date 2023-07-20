import React from 'react';
import 'bootstrap/dist/css/bootstrap.css';
import { NavLink } from 'react-router-dom';
import "./Navbar.css";

const Navbar = () => {
  return (
    <>

    <nav className="navbar navbar-expand-lg bg-dark">
    <div className="container-fluid">
        <img className = "tvs-logo" src={ require('../../src/tvs_logo (1).png') } />
        <button className="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarSupportedContent" aria-controls="navbarSupportedContent" aria-expanded="false" aria-label="Toggle navigation">
        <span className="navbar-toggler-icon"></span>
        </button>
        <div className="collapse navbar-collapse" id="navbarSupportedContent">
        <ul class="navbar-nav me-auto mb-2 mb-lg-0" id = "move-right">
            <li className="nav-item">
            <NavLink className="nav-link" to="/home">Home</NavLink>
            </li>
            <li className="nav-item">
            <NavLink className="nav-link" to="/registeration">Registeration</NavLink>
            </li>
            <li className="nav-item">
            <NavLink className="nav-link" to="/login">Login</NavLink>
            </li>
            <li className="nav-item">
            <NavLink className="nav-link" to="/logout">Logout</NavLink>
            </li>
        </ul>
        </div>
    </div>
    </nav>

    </>
  )
}

export default Navbar
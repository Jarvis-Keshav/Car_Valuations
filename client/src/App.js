import React, { createContext } from 'react';
import { Route } from 'react-router-dom';
import Navbar from './components/Navbar';
import Home from './components/Home';
import Login from './components/Login';
import Registeration from './components/Registeration';
import Logout from './components/Logout';

const App = () => {

  return (
    <>
      <Navbar />

      <Route exact path="/home">
        <Home />
      </Route>

      <Route path="/login">
        <Login />
      </Route>
      
      <Route path="/registeration">
        <Registeration />
      </Route>

      <Route path = "/logout">
        <Logout />
      </Route>
    </>
  )
}

export default App
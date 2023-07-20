import React, {useState} from 'react';
import { useHistory } from 'react-router-dom';
import "./Login.css";

const Login = () => {

  const history = useHistory();

  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');

  const loginUser = async (e) => {
    e.preventDefault();

    const res = await fetch("http://localhost:5000/login", {
      method : "post",
      headers : {
        "Content-Type" : "application/json"
      },
      body : JSON.stringify({
        username, password
      })
    });

    const data = res.json();
    if(res.status === 422 || !data){
      window.alert("Invalid Credentials");
    }else{
      window.alert("Login successful");
      history.push("/");
    }

  }

  return (
    <div>
      <div className = "outer-form-div-login">
        <form>
          <div>
            <input type = "text" name = "username" id = "UserName-login" autoComplete = "off"
            value = {username} onChange = {(e) => setUsername(e.target.value)} placeholder='UserName'/>
          </div>
          <div>
            <input type = "password" name = "password" id = "Password-login" autoComplete = "off"
            value = {password} onChange = {(e) => setPassword(e.target.value)} placeholder='Password'/>
          </div>
          <div className='submit-login'>
            <input type = "submit" name = "login" id = "Login" value = "Login" onClick={loginUser}/> 
          </div>
        </form>
      </div>
    </div>
  )
}

export default Login
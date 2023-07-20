import React, {useState} from 'react'
import { NavLink , useHistory} from 'react-router-dom'
import "./Registeration.css"

const Registeration = () => {

  const history = useHistory();
  const [name, setName] = useState("");
  const [username, setUsername] = useState("");
  const [phoneno, setPhoneno] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");

  const PostData = async (e) => {
    e.preventDefault();
    const {res} = await fetch("http://localhost:5000/register", {
      method : "post",
      headers : {
        "Content-Type" : "application/json"
      },
      body : JSON.stringify({
        name : name,
        username : username,
        phoneno : phoneno,
        email : email,
        password : password
      })
    });
    
    const data = await res.json();

    if(data.status === 500 || !data){
      window.alert("Invalid Registeration");
    }else{
      window.alert("Registeration successful");
      history.push("/login");
    }

  }

  return (
    <div>
      <div className = "outer-form-div-register">
        <form>
          <div>
            <input type = "text" name = "name" id = "Name-register" autoComplete = "on" placeholder='Name'
            onChange={(e) => setName(e.target.value)}/>
          </div>
          <div>
            <input type = "text" name = "username" id = "UserName-register" autoComplete = "on" placeholder='UserName'
            onChange={(e) => setUsername(e.target.value)}/>
          </div>
          <div>
            <input type = "number" name = "phoneno" id = "PhoneNo-register" autoComplete = "on" placeholder='PhoneNo'
            onChange={(e) => setPhoneno(e.target.value)}/>
          </div>
          <div>
            <input type = "email" name = "email" id = "Email-register" autoComplete = "on" placeholder='Email'
            onChange={(e) => setEmail(e.target.value)}/>
          </div>
          <div>
            <input type = "password" name = "password" id = "Password-register" autoComplete = "on" placeholder='Password'
            onChange={(e) => setPassword(e.target.value)}/>
          </div>
          <div className='evaluate-button'>
            <input type = "submit" name = "register" id = "register-register" 
            value = "Register" onClick={PostData}/>
            <br/>
          </div>
        </form>
      </div>
    </div>
  )
}

export default Registeration
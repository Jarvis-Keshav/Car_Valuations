import React, { useEffect } from 'react'
import { useHistory } from 'react-router-dom';

const Logout = () => {

  const history = useHistory();
  useEffect(() => {
    fetch("/logout", {
      method : "GET",
      headers : {
        Accept : "application/json",
        "Content-Type" : "application/json"
      },
      credentials : "include"
    }).then((res) => {
      if(!res.status === 200){
        const error = new Error(res.error);
        throw error;
      }else history.push("/login", {replace : true});
    }).catch((err) => {
      console.log(err);
    })
  })

  return (
    <div>
        <h1>Logout page</h1>
    </div>
  )
}

export default Logout
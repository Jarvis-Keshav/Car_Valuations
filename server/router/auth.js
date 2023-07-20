require("../db/Mongoose_Connection");
const express = require("express");
const router = express.Router();
const User = require("../model/UserSchema");
const bcrypt = require("bcryptjs");
const jwt = require("jsonwebtoken");
const authenticate = require("../middleware/Authenticate");
const { application } = require("express");

router.post("/register", async (req, res) => {
    console.log()
    const {name, username, phoneno, email, password} = req.body;
    if(!name || !username || !email || !password || !phoneno){
        return res.status(422).json({error : "Please Fill the fields properly"});
    }
    User.findOne({Email : email}).then((userExist) => {
        if(userExist){
            return res.status(422).json({error : "Email Already Exists"});
        }
        const user = new User({
            Name : name,
            Username : username,
            PhoneNo : phoneno,
            Email : email,
            Password : password
        });

        user.save().then(() => {
            res.status(201).json({message : "User Registerd Successfully"});
        }).catch((err) => {
            console.log(err);
            res.status(500).json({err : "Failed to Register"});
        })
    }).catch((err) => {
        console.log(err);
    })
});

router.post("/login", async (req, res) => {
    try{
        const {username, password} = req.body;
        if(!username || !password){
            return res.status(422).json({err : "Fill all the fields"});
        }
        const userLogin = await User.findOne({Username : username});
        if(!userLogin) return res.status(422).json({err : "Invalid Credentials"});
        const isMatch = await bcrypt.compare(password, userLogin.Password);
        const token = await userLogin.generateAuthToken();
        res.cookie("jwtoken", token, {
            expires : new Date(Date.now() + 25892000000),
            httpOnly : true
        });

        if(!isMatch){   
            return res.status(422).json({err : "Invalid credentials"});
        }else{
            return res.status(200).json({msg : "User Signed in successfully"});
        }
    }catch(err){
        console.log(err);
    }
})

router.get("/logout", (req, res) => { 
    res.clearCookie('jwtoken', {path : "/"});
    res.status(200).send("User Logout"); 
});

module.exports = router;
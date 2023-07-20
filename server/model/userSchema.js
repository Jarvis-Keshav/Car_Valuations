const mongoose = require("mongoose");
const bcrypt = require("bcryptjs");
const jwt = require("jsonwebtoken");
const dotenv = require("dotenv");

dotenv.config({path : './config.env'});

const userSchema = new mongoose.Schema({
    Name : {
        type : String,
        required : true
    },
    Username : {
        type : String,
        required : true,
        unique : true
    },
    PhoneNo : {
        type : Number,
        required : true
    },
    Email : {
        type : String,
        required : true,
        unique : true
    },
    Password : {
        type : String,
        required : true
    },
    tokens : [
        {
            token : {
                type : String,
                required : true
            }
        }
    ]
})

userSchema.pre('save', async function(next){
    if(this.isModified('Password')){
        this.Password = await bcrypt.hash(this.Password, 12);
    }
    next();
});

userSchema.methods.generateAuthToken = async function (){
    try{
        let token = jwt.sign({_id : this.id}, process.env.SECRET_KEY);
        this.tokens = this.tokens.concat({token : token});
        await this.save();
        return token;
    }catch(err){
        console.log(err);
    }
}

const User = mongoose.model('USERS', userSchema);
module.exports = User;

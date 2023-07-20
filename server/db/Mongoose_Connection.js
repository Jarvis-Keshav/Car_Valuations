const mongoose = require("mongoose");

mongoose.connect("mongodb+srv://harshgupta:codemongodb@cluster0.9zjkhpb.mongodb.net/userverification?retryWrites=true&w=majority").then(() => {
    console.log(`connection is successful`);
}).catch((err) => {
    console.log(`connection is failed`);
});
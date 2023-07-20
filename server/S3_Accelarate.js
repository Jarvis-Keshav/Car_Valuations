const dotenv = require("dotenv");
const aws = require("aws-sdk");
const { model } = require("mongoose");

dotenv.config();

const region = "us-east-1"
const bucketName = "audio-already-upload"
const accessKeyId = process.env.AWS_ACCESS_KEY_ID_NEW
const secretAccessKey = process.env.AWS_SECRET_ACCESS_KEY_NEW

const s3 = new aws.S3({
    region, 
    accessKeyId,
    secretAccessKey,
    signatureVersion: 'v4'
})

module.exports = async function generateUploadURLAccelarate(brand, model){
    const imageName = brand + "_" + model + "_" + "accelarate";
    const params = ({
        Bucket : bucketName,
        Key : imageName,
        Expires : 10000
    }) 
    const uploadURL = await s3.getSignedUrlPromise('putObject',params)
    return uploadURL
};
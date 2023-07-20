const dotenv = require("dotenv");
const aws = require("aws-sdk");

dotenv.config();

const region = "us-east-1"
const bucketName = "image-already-upload"
const accessKeyId = process.env.AWS_ACCESS_KEY_ID
const secretAccessKey = process.env.AWS_SECRET_ACCESS_KEY

const s3right = new aws.S3({
    region, 
    accessKeyId,
    secretAccessKey,
    signatureVersion: 'v4'
})

module.exports = async function generateUploadURLRight(brand, model){
    const imageName = brand + "_" + model + "_" + "right";
    const params = ({
        Bucket : bucketName,
        Key : imageName,
        Expires : 10000
    }) 
    const uploadURL = await s3right.getSignedUrlPromise('putObject',params)
    return uploadURL
};
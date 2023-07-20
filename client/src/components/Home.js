import React, { useState, useEffect } from "react";
import "./Home.css";
import ClockLoader from 'react-spinners/ClockLoader';

const Home = () => {
  const [brand, setBrand] = useState("");
  const [model, setModel] = useState("");
  const [oldprice, setOldprice] = useState("");
  const [oldyears, setOldyear] = useState("");
  const [ownership, setOwnership] = useState("");
  const [location, setLocation] = useState("");
  const [kms, setKms] = useState("");
  const [front, setFront] = useState("");
  const [back, setBack] = useState("");
  const [left, setLeft] = useState("");
  const [right, setRight] = useState("");
  const [accelarate, setAccelarate] = useState("");
  const [deaccelarate, setDeaccelarate] = useState("");
  const [finalprice, setFinalprice] = useState("");
  const [loading1, setLoading1] = useState(0);
  const [loading2, setLoading2] = useState(0);
  const [loading3, setLoading3] = useState(0);
  const [loading4, setLoading4] = useState(0);
  const [loading5, setLoading5] = useState(0);
  const [loading6, setLoading6] = useState(0);

  const uploadFront = async (e) => {
    e.preventDefault();
    const ImageInput1 = document.querySelector("#inpFile1")
    const file1 = ImageInput1.files[0]

    const inpFile = document.getElementById("inpFile1");
    const previewContainer = document.getElementById("imagePreview1")
    const previewImage = previewContainer.querySelector(".image-preview__image1"); 
    const previewDefaultText = previewContainer.querySelector(".image-preview__default-text1");

    const reader = new FileReader();
    previewDefaultText.style.display = "none";
    previewImage.style.display = "block";

    reader.addEventListener("load",function(){
      previewImage.setAttribute("src", this.result);
    });

    reader.readAsDataURL(file1);

    setLoading1(1);

    const {urlfront} = await fetch('http://localhost:5000/s3front', {
      method : "post",
      headers : {
        "Content-Type" : "application/json"
      },
      body : JSON.stringify({
        brand : brand,
        model : model
      })
    }).then(res => res.json());

    console.log(urlfront);
    
    await fetch(urlfront, {
      method: "PUT",
      headers: {
        "Content-Type" : "multipart/form-data",
      },
      body: file1
    })
    setLoading1(2);

  }

  const uploadBack = async (e) => {
    e.preventDefault();
    const ImageInput2 = document.querySelector("#inpFile2");
    const file2 = ImageInput2.files[0];
    const inpFile = document.getElementById("inpFile2");
    const previewContainer = document.getElementById("imagePreview2")
    const previewImage = previewContainer.querySelector(".image-preview__image2"); 
    const previewDefaultText = previewContainer.querySelector(".image-preview__default-text2");

    const reader = new FileReader();
    previewDefaultText.style.display = "none";
    previewImage.style.display = "block";

    reader.addEventListener("load",function(){
      previewImage.setAttribute("src", this.result);
    });

    reader.readAsDataURL(file2);
    setLoading2(1);
    const {urlback} = await fetch('http://localhost:5000/s3back', {
      method : "post",
      headers : {
        "Content-Type" : "application/json"
      },
      body : JSON.stringify({
        brand : brand,
        model : model
      })
    }).then(res => res.json())
    
    await fetch(urlback, {
      method: "PUT",
      headers: {
        "Content-Type" : "multipart/form-data"
      },
      body: file2
    })
    setLoading2(2);
  }
  
  const uploadLeft = async (e) => {
    e.preventDefault();
    const ImageInput3 = document.querySelector("#inpFile3");
    const file3 = ImageInput3.files[0];
    const inpFile = document.getElementById("inpFile3");
    const previewContainer = document.getElementById("imagePreview3")
    const previewImage = previewContainer.querySelector(".image-preview__image3"); 
    const previewDefaultText = previewContainer.querySelector(".image-preview__default-text3");

    const reader = new FileReader();
    previewDefaultText.style.display = "none";
    previewImage.style.display = "block";

    reader.addEventListener("load",function(){
      previewImage.setAttribute("src", this.result);
    });

    reader.readAsDataURL(file3);
    setLoading3(1);
    const {urlleft} = await fetch('http://localhost:5000/s3left', {
      method : "post",
      headers : {
        "Content-Type" : "application/json"
      },
      body : JSON.stringify({
        brand : brand,
        model : model
      })
    }).then(res => res.json())

    await fetch(urlleft, {
      method: "PUT",
      headers: {
        "Content-Type" : "multipart/form-data"
      },
      body: file3
    })
    setLoading3(2);
  }
  
  const uploadRight = async (e) => {
    e.preventDefault();
    const ImageInput4 = document.querySelector("#inpFile4");
    const file4 = ImageInput4.files[0];
    const inpFile = document.getElementById("inpFile4");
    const previewContainer = document.getElementById("imagePreview4")
    const previewImage = previewContainer.querySelector(".image-preview__image4"); 
    const previewDefaultText = previewContainer.querySelector(".image-preview__default-text4");

    const reader = new FileReader();
    previewDefaultText.style.display = "none";
    previewImage.style.display = "block";

    reader.addEventListener("load",function(){
      previewImage.setAttribute("src", this.result);
    });

    reader.readAsDataURL(file4);
    setLoading4(1);
    const {urlright} = await fetch('http://localhost:5000/s3right', {
      method : "post",
      headers : {
        "Content-Type" : "application/json"
      },
      body : JSON.stringify({
        brand : brand,
        model : model
      })
    }).then(res => res.json())

    console.log(urlright);
    
    await fetch(urlright, {
      method: "PUT",
      headers: {
        "Content-Type" : "multipart/form-data"
      },
      body: file4
    })
    setLoading4(2);
  }
  
  const evaluate = async (e) => {
    e.preventDefault();
    const {result} = await fetch('http://localhost:5000/evaluate', {
      method : "post",
      headers : {
        "Content-Type" : "application/json"
      },
      body : JSON.stringify({
        brand : brand,
        model : model
      })
    }).then(res => res.json())
  }

  const uploadAccelarate = async (e) => {
    e.preventDefault();
    const AudioInput5 = document.querySelector("#inpFile5")
    const file5 = AudioInput5.files[0]

    const {urlaccelarate} = await fetch('http://localhost:5000/s3accelarate', {
      method : "post",
      headers : {
        "Content-Type" : "application/json"
      },
      body : JSON.stringify({
        brand : brand,
        model : model
      })
    }).then(res => res.json())

    // const {urlaccelarate} = await fetch('http://localhost:5000/s3accelarate').then(res => res.json());

    // console.log(urlaccelarate);
    
    await fetch(urlaccelarate, {
      method: "PUT",
      headers: {
        "Content-Type" : "multipart/form-data",
      },
      body: file5
    })
    
    alert("Accelarating Audio Uploaded");
  }

  const uploadDeaccelarate = async (e) => {
    e.preventDefault();
    const AudioInput6 = document.querySelector("#inpFile6")
    const file6 = AudioInput6.files[0]

    // const {urldeaccelarate} = await fetch('http://localhost:5000/s3deaccelarate').then(res => res.json());

    const {urldeaccelarate} = await fetch('http://localhost:5000/s3deaccelarate', {
      method : "post",
      headers : {
        "Content-Type" : "application/json"
      },
      body : JSON.stringify({
        brand : brand,
        model : model
      })
    }).then(res => res.json())

    console.log(urldeaccelarate);
    
    await fetch(urldeaccelarate, {
      method: "PUT",
      headers: {
        "Content-Type" : "multipart/form-data",
      },
      body: file6
    })
    
    alert("Deaccelarating Audio Uploaded");
  }

  const audioevaluate = async (e) => {
    e.preventDefault();
    const {result} = await fetch('http://localhost:5000/evaluateaudio', {
      method : "post",
      headers : {
        "Content-Type" : "application/json"
      },
      body : JSON.stringify({
        brand : brand,
        model : model
      })
    }).then(res => res.json())
  }

  function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  const audiofinalscore = async (e) => {
    e.preventDefault();
    console.log("Step 1");
    setLoading6(1);
    const res = fetch("http://localhost:5000/writefinalscore", {
      method : "post",
      headers : {
        "Content-Type" : "application/json"
      },
      body : JSON.stringify({
        brand : brand,
        model : model,
        oldprice : oldprice,
        yearsold : oldyears,
        ownership : ownership,
        location : location,
        kmsdriven : kms
      })
    });

    const size = await fetch('http://localhost:5000/getsize');
    const sizee = await size.json();
    const x = sizee['Size'];

    if(x > 1000){
      console.log("greater than 1000");
      const data1 = await fetch('http://localhost:5000/mldisplay');
      const target1 = await data1.json();
      setFinalprice(target1['Finalscoreml']);
    }else{
      console.log("less than 1000");
      const result = await fetch('http://localhost:5000/finalscore');
      const y = await result.json();
      await sleep(5000);
      const data = await fetch('http://localhost:5000/display');
      const target = await data.json();
      setFinalprice(target['Finalscore']);
    }
    setLoading6(2);
  }

  return (
    <div>
      <div className="home__container">
      <div id="carouselExampleInterval" class="carousel slide" data-bs-ride="carousel">
        <div class="carousel-inner">
          <div class="carousel-item active" data-bs-interval="2000">
            <img src={ require('../../src/banner3.jpg') } class="d-block w-100" alt="..." />
          </div>
          <div class="carousel-item" data-bs-interval="2000">
            <img src={ require('../../src/honda_cb350.jpg') } class="d-block w-100" alt="..." />
          </div>
        </div>
        <button class="carousel-control-prev" type="button" data-bs-target="#carouselExampleInterval" data-bs-slide="prev">
          <span class="carousel-control-prev-icon" aria-hidden="true"></span>
          <span class="visually-hidden">Previous</span>
        </button>
        <button class="carousel-control-next" type="button" data-bs-target="#carouselExampleInterval" data-bs-slide="next">
          <span class="carousel-control-next-icon" aria-hidden="true"></span>
          <span class="visually-hidden">Next</span>
        </button>
      </div>
      </div>
      <br></br>
      <div className="outer-form-div-home">
        <form id="form-outer" className="bg-container">
          <div className="input-text">
            <h2>Enter Vehicle Information</h2>
            <br></br>
            <br></br>

            <div className="outer-row-1">
              <div className="change">
                <div id="change-select-div">
                  <b>Select Brand</b>
                </div>
                <select
                  className="custom-select"
                  id="dropdown"
                  onChange={(e) => setBrand(e.target.value)}
                >
                  <option disabled selected value></option>
                  <option value="Honda">Honda</option>
                  <option value="Tvs">Tvs</option>
                </select>
              </div>

              <div>
                <div id="change-select-div">
                  <b>Select Model</b>
                </div>
                <select
                  className="custom-select"
                  id="dropdown"
                  onChange={(e) => setModel(e.target.value)}
                >
                  <option disabled selected value></option>
                  <option value="CB_350">CB_350</option>
                  <option value="Apache">Apache</option>
                </select>
              </div>

            </div>

            <div className="outer-row-2">
              <div>
                <div id="change-select-div">
                  <b>Old Price</b>
                </div>
                <input
                  type="number"
                  name="oldprice"
                  id="dropdown"
                  onChange={(e) => setOldprice(e.target.value)}
                  autoComplete="on"
                />
              </div>
              <div>
                <div id="change-select-div">
                  <b>Years Old</b>
                </div>
                <select
                  className="custom-select"
                  id="dropdown"
                  onChange={(e) => setOldyear(e.target.value)}
                >
                  <option disabled selected value></option>
                  <option value="2010">2010</option>
                  <option value="2011">2011</option>
                  <option value="2012">2012</option>
                  <option value="2013">2013</option>
                  <option value="2014">2014</option>
                  <option value="2015">2015</option>
                  <option value="2016">2016</option>
                  <option value="2017">2017</option>
                  <option value="2018">2018</option>
                  <option value="2019">2019</option>
                  <option value="2020">2020</option>
                  <option value="2021">2021</option>
                </select>
              </div>

            </div>

            <div>
              <div className="outer-row-3">
                <div>
                  <div id="change-select-div">
                    <b>OwnerShip</b>
                  </div>
                  <input
                    type="number"
                    name="ownership"
                    id="dropdown"
                    onChange={(e) => setOwnership(e.target.value)}
                    autoComplete="on"
                  />
                </div>

                <div>
                  <div id="change-select-div">
                    <b>Location</b>
                  </div>
                  <select
                    className="custom-select"
                    id="dropdown"
                    onChange={(e) => setLocation(e.target.value)}
                  >
                    <option disabled selected value></option>
                    <option value="Delhi">Delhi</option>
                    <option value="Gujrat">Gujrat</option>
                  </select>
                  </div>
                </div>
              </div>
              <div className="outer-row-4">
              <div className="change">
                <div id="change-select-div">
                      <b>Kms Driven</b>
                    </div>
                    <input
                      type="number"
                      name="kmsdriven"
                      id="dropdown"
                      onChange={(e) => setKms(e.target.value)}
                      autoComplete="on"
                    />
              </div>

            </div>
          </div>
          <br></br>
          <br></br>
          <hr></hr>
          <br></br>

          <div className="input-text-1">
            <br></br>
            <h2>Upload Vehicle Images</h2>
            <br></br>
            <br></br>
            <div className="outer-row-5">
              <div className="divv-up">
                <div className="image-preview1" id = "imagePreview1">
                  <img src = "" alt = "Image Preview1" className="image-preview__image1"/>
                  <span className="image-preview__default-text1">Image Preview</span>
                </div>
                <input
                  type="file"
                  id="inpFile1"
                  className="inner-css"
                  onChange={(e) => setFront(e.target.value)}
                  accept="image/*"
                />
                <div id="change-select-div">
                  {
                    loading1 === 0 ? <button type = "submit" id = "btnUpload1" onClick={uploadFront}>Upload Front Image</button>
                    : loading1 === 1 ? <ClockLoader className="loader" size={30} color={'#FFFFFF'} loading1={loading1 === 1}/>
                    : <button type = "submit" id = "btnUpload1" onClick={uploadFront}>Upload Success <b>&#x2713;</b> </button>
                  }
                </div>
              </div>
              <hr></hr>
              <div className="divv-down">
                <div className="image-preview2" id = "imagePreview2">
                  <img src = "" alt = "Image Preview2" className="image-preview__image2"/>
                  <span className="image-preview__default-text2">Image Preview</span>
                </div>
                <input
                  type="file"
                  id="inpFile2"
                  className="inner-css"
                  onChange={(e) => setBack(e.target.value)}
                  accept="image/*"
                />
                <div id="change-select-div">
                  {
                    loading2 === 0 ? <button type = "submit" id = "btnUpload1" onClick={uploadBack}>Upload Back Image</button>
                    : loading2 === 1 ? <ClockLoader className="loader" size={30} color={'#FFFFFF'} loading2={loading2 === 1}/>
                    : <button type = "submit" id = "btnUpload1" onClick={uploadBack}>Upload Success <b>&#x2713;</b> </button>
                  }
                </div>
              </div>
            </div>
            <div className="outer-row-6">
              <hr></hr>
              <div className="divv-left">
                <div className="image-preview3" id = "imagePreview3">
                  <img src = "" alt = "Image Preview3" className="image-preview__image3"/>
                  <span className="image-preview__default-text3">Image Preview</span>
                </div>
                <input
                  type="file"
                  id="inpFile3"
                  className="inner-css"
                  onChange={(e) => setLeft(e.target.value)}
                  accept="image/*"
                />
                <div id="change-select-div">
                  {
                    loading3 === 0 ? <button type = "submit" id = "btnUpload1" onClick={uploadLeft}>Upload Left Image</button>
                    : loading3 === 1 ? <ClockLoader className="loader" size={30} color={'#FFFFFF'} loading3={loading3 === 1}/>
                    : <button type = "submit" id = "btnUpload1" onClick={uploadLeft}>Upload Success <b>&#x2713;</b> </button>
                  }
                </div>
              </div>
              <hr></hr>
              <div className="divv-right">
                <div className="image-preview4" id = "imagePreview4">
                  <img src = "" alt = "Image Preview4" className="image-preview__image4"/>
                  <span className="image-preview__default-text4">Image Preview</span>
                </div>
                <input
                  type="file"
                  id="inpFile4"
                  className="inner-css"
                  onChange={(e) => setRight(e.target.value)}
                  accept="image/*"
                />
                <div id="change-select-div">
                  {
                    loading4 === 0 ? <button type = "submit" id = "btnUpload1" onClick={uploadRight}>Upload Right Image</button>
                    : loading4 === 1 ? <ClockLoader className="loader" size={30} color={'#FFFFFF'} loading4={loading4 === 1}/>
                    : <button type = "submit" id = "btnUpload1" onClick={uploadRight}>Upload Success <b>&#x2713;</b> </button>
                  }
                  
                </div>
              </div>
            </div>
            {
              loading5 === 0 ? <button type = "submit" className = "btn btn-light" id = "right-image" onClick={evaluate}>Upload Images</button>
              : loading5 === 1 ? <ClockLoader className="loader" size={30} color={'#FFFFFF'} loading5={loading5 === 1}/>
              : <button type = "submit" className = "btn btn-light" id = "right-image" onClick={evaluate}>Upload Success <b>&#x2713;</b> </button>
            }
          </div>

          <br></br>
          <hr></hr>
          <br></br>

          <div className="input-text-1">
            <br></br>
            <h2>Upload Vehicle Audio</h2>
            <br></br>
            <br></br>
            <div className="outer-row-5">
              <div className="divv-up">
              <input
                  type="file"
                  id="inpFile5"
                  className="inner-css"
                  onChange={(e) => setAccelarate(e.target.value)}
                  accept="audio/*"
                />
                <div id="change-select-div-accelarate">
                  <button type = "submit" id = "btnUpload5" onClick={uploadAccelarate}>Upload Accelarating Audio</button>
                </div>
              </div>
            </div>
            <div className="outer-row-6">
              <div className="divv-right">
              <input
                  type="file"
                  id="inpFile6"
                  className="inner-css"
                  onChange={(e) => setDeaccelarate(e.target.value)}
                  accept="audio/*"
                />
                <div id= "change-select-div-accelarate">
                  <button type = "submit" id = "btnUpload6" onClick={uploadDeaccelarate}>Upload Deaccelarating Audio</button>
                </div>
              </div>
            </div>
            <button
              type="submit"
              className="btn btn-dark"
              onClick={audioevaluate}
              id="audio-button"
            >
              Upload Audio
            </button>
          </div>
          {
            loading6 === 0 ? <button type = "submit" className = "btn btn-dark" id = "final-score-button" onClick={audiofinalscore}>Calculate Final Score</button>
            : loading6 === 1 ? <ClockLoader className="loader-final" size={30} color={'#000000'} loading6={loading6 === 1}/>
            : <button type = "submit" className = "btn btn-dark" id = "final-score-button" onClick={audiofinalscore}>Re-Evaluate <b>&#x2713;</b> </button>
          }
          {
            loading6 == 2 ? <div className="score"> <div className="final-score-center">The Final Price is {finalprice}</div></div> : <div></div>
          }
        </form>
      </div>
    </div>
  );
};

export default Home;

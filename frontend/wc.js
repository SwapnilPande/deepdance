const vid = document.getElementById('video');
const teacher = document.getElementById('teacher'); 
navigator.mediaDevices.getUserMedia({video: true, audio:true}) // request cam
.then(stream => {
  vid.srcObject = stream; // don't use createObjectURL(MediaStream)
  return vid.play(); // returns a Promise
})
.then(()=>{ // enable the button
  const btn = document.querySelector('button');
  btn.disabled = false;
  btn.onclick = startRecording;

});

function startRecording(){
  // switch button's behavior
  const teacher = document.getElementById('teacher'); 

  const btn = this;
  btn.textContent = 'stop recording';
  btn.onclick = stopRecording;


  teacher.play() 
  teacher.onended = stopRecording;
  const chunks = []; // here we will save all video data
  const rec = new MediaRecorder(vid.srcObject);
  // this event contains our data
  rec.ondataavailable = e => chunks.push(e.data);
  // when done, concatenate our chunks in a single Blob
  rec.onstop = e => score(new Blob(chunks), teacher.src);
  rec.start();
  function stopRecording(){
    rec.stop();
    // switch button's behavior
    btn.textContent = 'Start Dancing!';
    btn.onclick = startRecording;
  }
}
async function score(student, teacher){
  // uses the <a download> to download a Blob
  // let a = document.createElement('a'); 
  // a.href = URL.createObjectURL(student);
  // a.download = 'recorded.webm';
  // document.body.appendChild(a);
  // a.click();

 var myHeaders = new Headers();
//myHeaders.append("Content-Type", "application/x-www-form-urlencoded");
let master = await fetch(teacher).then(r => r.blob());
var formdata = new FormData();
formdata.append("teacher", master, 'teacher.webm');
formdata.append("student", student,  'student.webm' );
//localStorage.myfile = blob;

var requestOptions = {
  method: 'POST',
  headers: myHeaders,
  body: formdata,
  redirect: 'follow'
};

fetch("http://127.0.0.1:5000/test", requestOptions)
  .then(response => response.text())
  .then(result => console.log(result))
  .catch(error => console.log('error', error));
}



(function localFileVideoPlayer() {
  'use strict'
  var URL = window.URL || window.webkitURL
  var displayMessage = function (message, isError) {
    var element = document.querySelector('#message')
    element.innerHTML = message
    element.className = isError ? 'error' : 'info'
  }
  var playSelectedFile = function (event) {
    var file = this.files[0]
    var type = file.type
    var videoNode = document.getElementById('teacher')

    var canPlay = videoNode.canPlayType(type)
    if (canPlay === '') canPlay = 'no'
    var message = 'Can play type "' + type + '": ' + canPlay
    var isError = canPlay === 'no'
    displayMessage(message, isError)

    if (isError) {
      return
    }

    var fileURL = URL.createObjectURL(file)
    videoNode.src = fileURL
    videoNode.pause() 
  }
  var inputNode = document.querySelector('input')
  inputNode.addEventListener('change', playSelectedFile, false)
})()
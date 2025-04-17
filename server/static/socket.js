// var socket = io.connect('http://127.0.0.1:5000');

// socket.on('connect', function() {
//     socket.send('User has connected!');
// });


// function startTask() {
//     let param1 = document.getElementById("param1").value;
//     let param2 = document.getElementById("param2").value;

//     socket.emit("start_task", { param1: param1, param2: param2 }); // Send parameters to backend
// }

// socket.on("progress", function(data) {
//     let progressBar = document.getElementById("progress-bar");
//     if (progressBar) {
//         progressBar.style.width = data.progress + "%";
//         progressBar.innerText = data.progress + "%";
//     }
// });

// socket.on("task_complete", function(data) {
//     alert(data.message);  // Show alert when task completes
// });
const socket = io();
let socketid = undefined
socket.connect("https://localhost:5000");


socket.on("connect", function () {
        console.log("Connected!");
        socketid = socket.id;
        console.log("ID: " + socketid);
})
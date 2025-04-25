
/* script.js */
document.getElementById("captureFace").addEventListener("click", function() {
    window.location.href = "capture";
});

document.getElementById("initiateRecognition").addEventListener("click", function() {
    window.location.href = "recognize";  // Navigate to the recognition page
});

document.getElementById("faceLibrary").addEventListener("click", function() {
    window.location.href = "face-library";
});
let video = null;
let canvas = null;
let ctx = null;

window.onload = async () => {
    video = document.getElementById("video");
    canvas = document.getElementById("canvas");
    ctx = canvas.getContext("2d");

    let stream = await navigator.mediaDevices.getUserMedia({ video: true });
    video.srcObject = stream;
};

function captureFrame() {
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    let imgData = canvas.toDataURL("image/jpeg");

    fetch("/live_predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image: imgData })
    })
    .then(res => res.json())
    .then(data => {
        document.getElementById("result").innerHTML =
            `<h3>Prediction: ${data.label} (${data.confidence}%)</h3>`;
    });
}

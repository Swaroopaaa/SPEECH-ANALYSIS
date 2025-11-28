// Optional: basic skeleton for recording if you want to extend later
// This file is currently not wired to the UI.
let mediaRecorder;
let audioChunks = [];

async function initRecorder() {
  if (!navigator.mediaDevices) return;
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  mediaRecorder = new MediaRecorder(stream);
  mediaRecorder.ondataavailable = e => audioChunks.push(e.data);
  mediaRecorder.onstop = exportAudio;
}

function startRecording() {
  audioChunks = [];
  mediaRecorder.start();
}

function stopRecording() {
  mediaRecorder.stop();
}

function exportAudio() {
  const blob = new Blob(audioChunks);
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'recording.wav';
  a.click();
}

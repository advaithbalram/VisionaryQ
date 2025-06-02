// Handle video upload
document.getElementById('videoUpload').addEventListener('change', function(event) {
  const file = event.target.files[0];
  if (!file) return;

  const videoStatus = document.getElementById('videoStatus');
  videoStatus.textContent = 'Processing the video uploaded...';

  const formData = new FormData();
  formData.append('file', file);

  fetch('/upload', {
    method: 'POST',
    body: formData,
  })
  .then(response => response.json())
  .then(data => {
    if (data.message) {
      document.getElementById('videoStatus').textContent = `Uploaded and Processed the video: ${file.name}`;
    } else {
      document.getElementById('videoStatus').textContent = 'Upload failed!';
    }
  })
  .catch(error => {
    console.error('Error uploading video:', error);
    document.getElementById('videoStatus').textContent = 'Upload error!';
  });
});


// Handle chat messages
document.getElementById('sendButton').addEventListener('click', sendMessage);
document.getElementById('userInput').addEventListener('keypress', function(event) {
  if (event.key === 'Enter') {
    sendMessage();
  }
});

function sendMessage() {
  const userInput = document.getElementById('userInput').value.trim();
  if (userInput === '') return;

  displayMessage(userInput, 'user');
  document.getElementById('userInput').value = ''; // Clear input
  fetch('/ask', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query: userInput })
  })
  .then(response => response.json())
  .then(data => {
    if (data.answer) {
      displayMessage(data.answer, 'bot');
    } else {
      displayMessage('Error: No response from server', 'bot');
    }
  })
  .catch(error => {
    displayMessage('Error: Failed to communicate with server', 'bot');
    console.error('Error:', error);
  });
}

function displayMessage(message, sender) {
  const chatWindow = document.getElementById('chatWindow');
  const messageElement = document.createElement('div');
  messageElement.classList.add('chat-message', sender === 'user' ? 'user-message' : 'bot-message');
  messageElement.textContent = sender === 'bot' ? `VisionaryQ: ${message}` : message;
  chatWindow.appendChild(messageElement);
  chatWindow.scrollTop = chatWindow.scrollHeight; // Scroll to latest message
}

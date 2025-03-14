// Load TensorFlow.js dynamically (only if needed for other functionality)
const tfScript = document.createElement("script");
tfScript.src = "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs";
tfScript.onload = () => console.log("TensorFlow.js loaded!");
document.head.appendChild(tfScript);

// File input handling
const imageInput = document.getElementById("imageInput");
const dragDropLabel = document.querySelector(".drag-drop-label span");
const resultDiv = document.getElementById("prediction-result");

imageInput.addEventListener("change", function () {
    const fileName = this.files[0]?.name || "No file chosen";
    dragDropLabel.textContent = fileName;
    handleImageUpload(this.files[0]);
});

// Process the uploaded image for prediction
async function handleImageUpload(file) {
    if (!file) {
        console.log("No file selected.");
        return;
    }

    // Display preview
    const imgPreview = document.createElement("img");
    imgPreview.src = URL.createObjectURL(file);
    imgPreview.width = 200;
    
    // Clear previous results and add image preview
    resultDiv.innerHTML = "";
    resultDiv.appendChild(imgPreview);

    // Create a div to show the prediction result
    const predictionText = document.createElement("p");
    predictionText.id = "prediction-result";
    predictionText.innerText = "Processing image...";
    resultDiv.appendChild(predictionText);

    // Send image to Flask API for prediction
    await sendToFlaskAPI(file, predictionText);
}

// Send image to Flask API for prediction
async function sendToFlaskAPI(file, predictionText) {
    let formData = new FormData();
    formData.append("file", file);

    try {
        let response = await fetch("http://127.0.0.1:5000/predict", {
            method: "POST",
            body: formData
        });

        let result = await response.json();

        if (response.ok) {
            predictionText.innerText = `Prediction: ${result.prediction} (Confidence: ${(result.confidence * 100).toFixed(2)}%)`;
        } else {
            predictionText.innerText = `Error: ${result.error}`;
        }
    } catch (error) {
        console.error("Error:", error);
        predictionText.innerText = "Something went wrong with the server. Please try again.";
    }
}

// Chatbot elements
const chatbot = document.getElementById("chatbot");
const sendMessageButton = document.getElementById("send-message");
const userInput = document.getElementById("user-input");
const chatlog = document.getElementById("chatlog");
const closeChatButton = document.getElementById("close-chat");
const openChatButton = document.getElementById("open-chat");

// Show the chatbot when clicking the open button
openChatButton.addEventListener("click", function () {
    chatbot.style.display = "block";
    openChatButton.style.display = "none";
});

// Close the chatbot
closeChatButton.addEventListener("click", function () {
    chatbot.style.display = "none";
    openChatButton.style.display = "block";
});

// Handle user message submission when clicking "Send" button
sendMessageButton.addEventListener("click", sendMessage);

// Handle user message submission when pressing "Enter"
userInput.addEventListener("keypress", function (event) {
    if (event.key === "Enter") {
        event.preventDefault();
        sendMessage();
    }
});

// Function to send message
function sendMessage() {
    const message = userInput.value.trim();

    if (message !== "") {
        appendMessage("You", message);
        userInput.value = ""; // Clear input field

        // Simulate chatbot response
        setTimeout(function () {
            const botResponse = getChatbotResponse(message);
            appendMessage("Bot", botResponse);
        }, 1000);
    }
}

// Append message to chatlog
function appendMessage(sender, message) {
    const messageDiv = document.createElement("div");
    messageDiv.classList.add("chat-message");
    messageDiv.innerHTML = `<strong>${sender}:</strong> ${message}`;
    chatlog.appendChild(messageDiv);
    chatlog.scrollTop = chatlog.scrollHeight;
}

// Simple function to generate bot response (expandable)
function getChatbotResponse(userMessage) {
    const normalizedMessage = userMessage.toLowerCase();
    const responses = {
        "hi": "Hello! How can I assist you?",
        "hello": "Hi there! How can I help?",
        "can alzheimer's be cured?": "Currently, there is no cure for Alzheimer's disease. However, treatments can help manage symptoms and improve quality of life.",
        "can alzheimer be cured": "Currently, there is no cure for Alzheimer's disease, but research is ongoing.",
        "alzheimer's disease": "Alzheimer's is a progressive neurological disorder affecting memory, thinking, and behavior.",
    };

    return responses[normalizedMessage] || "I'm sorry, I don't understand that. Can you rephrase?";
}

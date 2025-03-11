// Load TensorFlow.js dynamically and ensure it's ready before loading the model
const tfScript = document.createElement("script");
tfScript.src = "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs";
tfScript.onload = async function () {
    console.log("TensorFlow.js loaded!");
    await loadModel(); // Load model only after TensorFlow is ready
};
document.head.appendChild(tfScript);

let model;

// Load the trained CNN model
async function loadModel() {
    try {
        model = await tf.loadLayersModel("http://127.0.0.1:5500/model/Model_CNN.json");
        console.log("Model loaded successfully!");
    } catch (error) {
        console.error("Error loading the model:", error);
    }
}

// File input handling
const imageInput = document.getElementById("imageInput");
const dragDropLabel = document.querySelector(".drag-drop-label span");

imageInput.addEventListener("change", function () {
    const fileName = this.files[0]?.name || "No file chosen";
    dragDropLabel.textContent = fileName;
    handleImageUpload(this.files[0]);
});

// Process the uploaded image for prediction
async function handleImageUpload(file) {
    if (!file || !model) {
        console.log("No file selected or model not loaded.");
        return;
    }

    const reader = new FileReader();
    reader.onload = async function (event) {
        const img = new Image();
        img.src = event.target.result;
        img.onload = async function () {
            await predictImage(img);
        };
    };
    reader.readAsDataURL(file);
}

// Process image and make prediction
async function predictImage(imageElement) {
    if (!model) {
        console.log("Model not loaded yet!");
        return;
    }

    // Convert image to tensor
    const tensor = tf.browser.fromPixels(imageElement)
        .resizeBilinear([224, 224]) // Ensure it matches model input size
        .toFloat()
        .expandDims(); // Add batch dimension

    // Make prediction
    const prediction = await model.predict(tensor).data();
    console.log("Prediction:", prediction);

    // Convert prediction to class label
    const result = processPrediction(prediction);
    appendMessage("Bot", `Prediction: ${result}`);
}

// Convert prediction to a meaningful label
function processPrediction(prediction) {
    const classes = ["Class 1", "Class 2", "Class 3"]; // Change these based on your model
    const maxIndex = prediction.indexOf(Math.max(...prediction));
    return classes[maxIndex];
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
        event.preventDefault(); // Prevents new line in input field
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

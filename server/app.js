const express = require("express");
const fs = require("fs");
const cors = require("cors");
const path = require("path");

const app = express();
const PORT = 3000;

fetch("http://127.0.0.1:5000/analyze", {
  method: "POST",
  headers: {
    "Content-Type": "application/json"
  },
  body: JSON.stringify({
    avgKeystrokeInterval: 0.4,
    mouseVelocity: 1.5,
    clickFrequency: 3,
    scrollPattern: 0.8,
    navigationFlow: 1.1,
    sessionDuration: 5.0
  })
})
.then(response => response.json())
.then(data => {
  console.log("Received from Flask:", data);
  alert("Risk Score: " + data.risk_score + "\nConfidence: " + data.confidence + "%");
  // You can also display this in the dashboard instead of alert
})
.catch(err => {
  console.error("Error from /analyze:", err);
});


// Enable CORS and JSON parsing
app.use(cors());
app.use(express.json());

app.post("/login", (req, res) => {
  const { username, password } = req.body;

  // Temporary dummy check (replace with DB/auth later)
  if (username === "admin" && password === "1234") {
    res.json({ success: true });
  } else {
    res.json({ success: false, message: "Invalid credentials" });
  }
});
app.get("/dashboard", (req, res) => {
  res.sendFile(path.join(__dirname, "../public/dashboard.html"));
});

app.get("/", (req, res) => {
  res.sendFile(path.join(__dirname, "../public/login.html"));
});



// ✅ Serve frontend static files - corrected path
app.use(express.static(path.join(__dirname, "../public")));

app.get("/", (req, res) => {
  res.sendFile(path.join(__dirname, "../public/autoencoder.html"));
});

// Log user behavior
app.post("/log", (req, res) => {
  const log = req.body;
  const threat = parseFloat(log.velocity) > 0.25 || log.touch_pressure > 0.9;
  fs.appendFileSync("server/activity_log.json", JSON.stringify(log) + "\n");
  res.json({ redirect_to_honeypot: threat });
});

app.listen(PORT, () => {
  console.log(`🌐 Website running at: http://localhost:${PORT}`);
});
const sessionId = sessionStorage.getItem("session_id") || Date.now();
sessionStorage.setItem("session_id", sessionId);

let lastTimestamp = performance.now();

document.addEventListener("mousemove", logEvent);
document.addEventListener("click", logEvent);
document.addEventListener("touchstart", logTouch);

function logEvent(e) {
  const now = performance.now();
  const log = {
    session_id: sessionId,
    action_type: e.type,
    timestamp: Date.now(),
    pos_x: e.clientX,
    pos_y: e.clientY,
    inter_touch_time: (now - lastTimestamp).toFixed(2),
    velocity: (Math.random() * 0.2).toFixed(3)
  };
  lastTimestamp = now;
  send(log);
}

function logTouch(e) {
  const t = e.touches[0];
  const log = {
    session_id: sessionId,
    action_type: "touch",
    timestamp: Date.now(),
    pos_x: t.clientX,
    pos_y: t.clientY,
    touch_pressure: t.force || 0,
    touch_size: t.radiusX || 0,
    velocity: (Math.random() * 0.2).toFixed(3)
  };
  send(log);
}

function send(data) {
  fetch("http://localhost:3000/log", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  }).then(res => res.json()).then(response => {
    if (response.redirect_to_honeypot) {
      window.location.href = "honeypot.html";
    }
  });
}

document.getElementById("login-form").addEventListener("submit", function (e) {
  e.preventDefault();
  const user = document.getElementById("username").value;
  const pass = document.getElementById("password").value;

  if (user === "admin" && pass === "admin123") {
    sessionStorage.setItem("session_id", Date.now());
    window.location.href = "/dashboard.html";
  } else {
    alert("Invalid credentials");
  }
});

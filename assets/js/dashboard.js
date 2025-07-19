const tabs = document.querySelectorAll(".tab");
const content = document.getElementById("tab-content");

// Remove and add active tab style
tabs.forEach(tab => {
  tab.addEventListener("click", () => {
    document.querySelector(".tab.active").classList.remove("active");
    tab.classList.add("active");
    loadTab(tab.dataset.tab);
  });
});

// Load content for each tab
function loadTab(tab) {
  switch (tab) {
    case "home":
      content.innerHTML = `
        <h2>Welcome to Canara Bank</h2>
        <p>This is your secure online banking dashboard. Select any tab above to manage your services.</p>
      `;
      break;

    case "accounts":
      content.innerHTML = `
        <h2>Your Accounts</h2>
        <ul>
          <li>💳 Savings Account: ₹45,000</li>
          <li>🏢 Current Account: ₹1,25,000</li>
          <li>🪙 Fixed Deposit: ₹5,00,000 (Matures in Oct 2025)</li>
        </ul>
        <button>📄 Download Statement</button>
        <button>✏️ Update Details</button>
        <button>➕ Open New Account</button>
      `;
      break;

    case "loans":
      content.innerHTML = `
        <h2>Loan Offers</h2>
        <ul>
          <li>🏠 Home Loan – 8.5% p.a – Up to ₹75L – <button>Apply</button></li>
          <li🎓> Education Loan – 9.25% p.a – ₹10L – <button>Apply</button></li>
          <li>🚗 Car Loan – 10% p.a – ₹20L – <button>Apply</button></li>
        </ul>
      `;
      break;

    case "cards":
      content.innerHTML = `
        <h2>Manage Your Cards</h2>
        <ul>
          <li>🛑 Block Lost/Stolen Card <button>Block</button></li>
          <li>🔐 Generate/Change PIN <button>Generate</button></li>
          <li>📄 View Card Statements <button>View</button></li>
          <li>💳 View Available Credit Cards <button>Explore</button></li>
        </ul>
      `;
      break;

    case "investments":
      content.innerHTML = `
        <h2>Investments</h2>
        <ul>
          <li>📊 Portfolio Overview</li>
          <li>🧮 SIP Calculator <button>Try Now</button></li>
          <li>💡 Investment Tips & Guidance</li>
          <li>📈 Market Watch (Live Data)</li>
          <li>➕ New Investment Offers</li>
        </ul>
      `;
      break;

    default:
      content.innerHTML = "<h2>404</h2><p>Tab not found.</p>";
  }
}

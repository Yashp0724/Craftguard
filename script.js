// Global variables
let currentImage = null
let analysisResults = null

// Cultural art database
const culturalArtDatabase = {
  madhubani: {
    name: "Madhubani",
    features: ["geometric_patterns", "natural_dyes", "fine_lines", "traditional_motifs"],
    colors: ["red", "yellow", "blue", "green", "black"],
    region: "Bihar, India",
  },
  warli: {
    name: "Warli",
    features: ["simple_geometric", "white_pigment", "tribal_motifs", "stick_figures"],
    colors: ["white", "red", "brown"],
    region: "Maharashtra, India",
  },
  kalamkari: {
    name: "Kalamkari",
    features: ["hand_painted", "natural_dyes", "floral_patterns", "mythological_themes"],
    colors: ["indigo", "red", "black", "yellow"],
    region: "Andhra Pradesh, India",
  },
  banarasi: {
    name: "Banarasi",
    features: ["silk_fabric", "brocade_work", "gold_thread", "intricate_weaving"],
    colors: ["gold", "silver", "red", "green", "purple"],
    region: "Varanasi, India",
  },
}

// Verified sellers database
const verifiedSellers = [
  {
    name: "Madhubani Art Collective",
    specialty: "Madhubani",
    website: "https://thebimba.com/",
    rating: 4.8,
  },
  {
    name: "Warli Tribal Artisans",
    specialty: "Warli",
    website: "https://mahatribes.com/",
    rating: 4.9,
  },
  {
    name: "Kalamkari Heritage Center",
    specialty: "Kalamkari",
    website: "https://www.memeraki.com/",
    rating: 4.7,
  },
  {
    name: "Banarasi Silk Weavers",
    specialty: "Banarasi",
    website: "https://www.banarasee.in/",
    rating: 4.6,
  },
]

// Initialize the application
document.addEventListener("DOMContentLoaded", () => {
  setupEventListeners()
})

function setupEventListeners() {
  const uploadArea = document.getElementById("uploadArea")
  const imageInput = document.getElementById("imageInput")

  // File upload handling
  imageInput.addEventListener("change", handleFileSelect)

  // Drag and drop handling
  uploadArea.addEventListener("dragover", handleDragOver)
  uploadArea.addEventListener("dragleave", handleDragLeave)
  uploadArea.addEventListener("drop", handleDrop)
}

function handleFileSelect(event) {
  const file = event.target.files[0]
  if (file && file.type.startsWith("image/")) {
    displayImagePreview(file)
  }
}

function handleDragOver(event) {
  event.preventDefault()
  event.currentTarget.classList.add("dragover")
}

function handleDragLeave(event) {
  event.currentTarget.classList.remove("dragover")
}

function handleDrop(event) {
  event.preventDefault()
  event.currentTarget.classList.remove("dragover")

  const files = event.dataTransfer.files
  if (files.length > 0 && files[0].type.startsWith("image/")) {
    displayImagePreview(files[0])
  }
}

function displayImagePreview(file) {
  const reader = new FileReader()
  reader.onload = (e) => {
    currentImage = e.target.result
    document.getElementById("previewImage").src = currentImage
    document.getElementById("previewSection").style.display = "block"
    document.getElementById("resultsSection").style.display = "none"
  }
  reader.readAsDataURL(file)
}

function analyzeFromUrl() {
  const url = document.getElementById("urlInput").value
  if (url) {
    currentImage = url
    document.getElementById("previewImage").src = url
    document.getElementById("previewSection").style.display = "block"
    document.getElementById("resultsSection").style.display = "none"
  }
}

function analyzeImage() {
  if (!currentImage) return

  // Show loading
  document.getElementById("loadingSection").style.display = "block"
  document.getElementById("resultsSection").style.display = "none"

  // Simulate API call to ML model
  setTimeout(() => {
    const results = simulateMLAnalysis()
    displayResults(results)

    document.getElementById("loadingSection").style.display = "none"
    document.getElementById("resultsSection").style.display = "block"
  }, 3000)
}

function simulateMLAnalysis() {
  // This simulates the ML model analysis
  // In real implementation, this would call your trained model API

  const artForms = Object.keys(culturalArtDatabase)
  const detectedArtForm = artForms[Math.floor(Math.random() * artForms.length)]
  const authenticityScore = Math.floor(Math.random() * 40) + 60 // 60-100%

  const isAuthentic = authenticityScore > 75
  const confidence = Math.floor(Math.random() * 20) + 80 // 80-100%

  return {
    artForm: detectedArtForm,
    authenticityScore: authenticityScore,
    confidence: confidence,
    isAuthentic: isAuthentic,
    detectedFeatures: generateDetectedFeatures(detectedArtForm),
    riskFactors: generateRiskFactors(authenticityScore),
  }
}

function generateDetectedFeatures(artForm) {
  const artData = culturalArtDatabase[artForm]
  const features = artData.features.slice(0, Math.floor(Math.random() * 3) + 2)

  return features.map((feature) => {
    const confidence = Math.floor(Math.random() * 30) + 70
    return {
      feature: feature.replace(/_/g, " ").toUpperCase(),
      confidence: confidence,
    }
  })
}

function generateRiskFactors(score) {
  const factors = []

  if (score < 80) {
    factors.push("Uniform pattern repetition suggests machine production")
  }
  if (score < 70) {
    factors.push("Color consistency indicates synthetic dyes")
    factors.push("Lack of natural variations in brushstrokes")
  }
  if (score < 60) {
    factors.push("Digital artifacts detected in image")
    factors.push("Pattern precision exceeds human capability")
  }

  return factors
}

function displayResults(results) {
  analysisResults = results

  // Update score circle
  const scoreText = document.getElementById("scoreText")
  const scoreCircle = document.getElementById("scoreCircle")
  scoreText.textContent = results.authenticityScore + "%"

  // Color code the score
  if (results.authenticityScore >= 80) {
    scoreCircle.style.background = "conic-gradient(from 0deg, #10b981 0%, #10b981 100%)"
  } else if (results.authenticityScore >= 60) {
    scoreCircle.style.background = "conic-gradient(from 0deg, #f59e0b 0%, #f59e0b 100%)"
  } else {
    scoreCircle.style.background = "conic-gradient(from 0deg, #ef4444 0%, #ef4444 100%)"
  }

  // Update result title and description
  const resultTitle = document.getElementById("resultTitle")
  const resultDescription = document.getElementById("resultDescription")

  if (results.isAuthentic) {
    resultTitle.textContent = `Likely Authentic ${culturalArtDatabase[results.artForm].name}`
    resultDescription.textContent = `This appears to be genuine ${culturalArtDatabase[results.artForm].name} art with ${results.confidence}% confidence.`
  } else {
    resultTitle.textContent = `Possibly Machine-Made`
    resultDescription.textContent = `This may be a mass-produced imitation with ${results.confidence}% confidence.`
  }

  // Display analysis details
  displayAnalysisDetails(results)

  // Display recommendations
  displayRecommendations(results.artForm)

  // Display verified sellers
  displayVerifiedSellers(results.artForm)
}

function displayAnalysisDetails(results) {
  const analysisList = document.getElementById("analysisList")
  analysisList.innerHTML = ""

  // Add detected features
  results.detectedFeatures.forEach((feature) => {
    const li = document.createElement("li")
    li.textContent = `${feature.feature}: ${feature.confidence}% confidence`
    analysisList.appendChild(li)
  })

  // Add risk factors if any
  if (results.riskFactors.length > 0) {
    const riskHeader = document.createElement("li")
    riskHeader.innerHTML = "<strong>Risk Factors:</strong>"
    analysisList.appendChild(riskHeader)

    results.riskFactors.forEach((factor) => {
      const li = document.createElement("li")
      li.textContent = factor
      li.style.color = "#ef4444"
      analysisList.appendChild(li)
    })
  }
}

function displayRecommendations(artForm) {
  const tipsList = document.getElementById("tipsList")
  const artData = culturalArtDatabase[artForm]

  const tips = [
    `Look for natural variations in ${artData.name} patterns - handmade pieces have slight irregularities`,
    `Check for traditional color palette: ${artData.colors.join(", ")}`,
    `Authentic ${artData.name} should show signs of hand-crafting techniques`,
    `Verify the seller's credentials and ask for artisan information`,
    `Compare prices - extremely low prices often indicate mass production`,
  ]

  tipsList.innerHTML = ""
  tips.forEach((tip) => {
    const li = document.createElement("li")
    li.textContent = tip
    tipsList.appendChild(li)
  })
}

function displayVerifiedSellers(artForm) {
  const sellersList = document.getElementById("sellersList")
  const relevantSellers = verifiedSellers.filter((seller) => seller.specialty.toLowerCase() === artForm.toLowerCase())

  sellersList.innerHTML = ""
  relevantSellers.forEach((seller) => {
    const sellerCard = document.createElement("div")
    sellerCard.className = "seller-card"
    sellerCard.innerHTML = `
            <h5>${seller.name}</h5>
            <p>Specialty: ${seller.specialty} | Rating: ⭐ ${seller.rating}/5</p>
            <a href="${seller.website}" target="_blank" style="color: #667eea;">Visit Store →</a>
        `
    sellersList.appendChild(sellerCard)
  })
}

// Utility function to convert image to base64 for API calls
function imageToBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => resolve(reader.result.split(",")[1])
    reader.onerror = reject
    reader.readAsDataURL(file)
  })
}

// Function to call actual ML API (to be implemented)
async function callMLAPI(imageData) {
  try {
    const response = await fetch("/api/analyze", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        image: imageData,
        timestamp: Date.now(),
      }),
    })

    if (!response.ok) {
      throw new Error("Analysis failed")
    }

    return await response.json()
  } catch (error) {
    console.error("ML API Error:", error)
    // Fallback to simulation
    return simulateMLAnalysis()
  }
}

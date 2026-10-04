const fileInput = document.getElementById("fileInput");
const uploadBtn = document.getElementById("uploadBtn");
const askBtn = document.getElementById("askBtn");
const questionInput = document.getElementById("question");
const extractedText = document.getElementById("extractedText");
const answerBox = document.getElementById("answer");
const sourcesBox = document.getElementById("sources");
const documentInfo = document.getElementById("documentInfo");
const statusBox = document.getElementById("status");
const language = document.getElementById("language");

function setStatus(text) {
  if (statusBox) {
    statusBox.textContent = text;
  }
}

/* ---------------------------------
   SHOW SELECTED FILE IMMEDIATELY
---------------------------------- */

fileInput.addEventListener("change", function () {
  if (fileInput.files.length === 0) {
    documentInfo.textContent = "No document loaded";
    return;
  }

  const file = fileInput.files[0];

  documentInfo.textContent = `${file.name} • ${(file.size / 1024).toFixed(1)} KB`;

  setStatus("Document selected");
});

/* ---------------------------------
   PROCESS DOCUMENT / OCR
---------------------------------- */

uploadBtn.addEventListener("click", async function () {
  if (!fileInput.files || fileInput.files.length === 0) {
    alert("Please select a Panchayat record image first.");
    return;
  }

  const file = fileInput.files[0];

  const formData = new FormData();
  formData.append("file", file);

  if (language) {
    formData.append("languages", language.value);
  }

  setStatus("Processing OCR...");

  uploadBtn.disabled = true;
  uploadBtn.textContent = "Processing...";

  try {
    const response = await fetch("/upload", {
      method: "POST",
      body: formData,
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.error || "Upload failed.");
    }

    /* Show extracted OCR text */
    extractedText.textContent = data.text || "No text extracted.";

    /* Show document information */
    documentInfo.textContent = `${data.filename} • ${data.chunk_count} searchable chunks`;

    /* Reset answer area */
    answerBox.innerHTML =
      '<p class="muted">Document indexed. Ask a question above.</p>';

    sourcesBox.innerHTML = "";

    setStatus("Document ready");
  } catch (error) {
    console.error(error);

    setStatus("Error");

    alert("OCR processing failed.\n\n" + error.message);
  } finally {
    uploadBtn.disabled = false;
    uploadBtn.textContent = "Process Document";
  }
});

/* ---------------------------------
   ASK QUESTION
---------------------------------- */

async function askQuestion() {
  const question = questionInput.value.trim();

  if (!question) {
    alert("Please enter a question.");
    return;
  }

  setStatus("Searching...");

  askBtn.disabled = true;
  askBtn.textContent = "Searching...";

  try {
    const response = await fetch("/query", {
      method: "POST",

      headers: {
        "Content-Type": "application/json",
      },

      body: JSON.stringify({
        question: question,
      }),
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.error || "Question failed.");
    }

    /* Display answer */
    answerBox.textContent = data.answer || "No answer generated.";

    /* Display sources */
    sourcesBox.innerHTML = "<strong>Retrieved Sources</strong>";

    if (data.sources && data.sources.length > 0) {
      data.sources.forEach((source, index) => {
        const div = document.createElement("div");

        div.className = "source";

        div.textContent = `Source ${index + 1} • similarity ${source.score.toFixed(3)}
${source.chunk}`;

        sourcesBox.appendChild(div);
      });
    } else {
      sourcesBox.innerHTML += "<p>No sources found.</p>";
    }

    setStatus("Answer generated");
  } catch (error) {
    console.error(error);

    setStatus("Error");

    alert(error.message);
  } finally {
    askBtn.disabled = false;
    askBtn.textContent = "Ask";
  }
}

askBtn.addEventListener("click", askQuestion);

/* Press Enter to ask */
questionInput.addEventListener("keydown", function (event) {
  if (event.key === "Enter") {
    askQuestion();
  }
});

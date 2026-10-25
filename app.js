const forms = [
  {
    formId: "regression-form",
    resultId: "regression-result",
    metricId: "regression-metric",
    endpoint: "/predict/regression",
    format: (data) => `Estimated votes: ${Number(data.prediction).toLocaleString("en-US")} · Test R² ${Number(data.r2).toFixed(3)}`,
  },
  {
    formId: "classification-form",
    resultId: "classification-result",
    metricId: "classification-metric",
    endpoint: "/predict/classification",
    format: (data) => `Predicted primary genre: ${data.prediction} · Confidence ${(Number(data.confidence) * 100).toFixed(1)}%`,
  },
];

for (const config of forms) {
  const form = document.getElementById(config.formId);
  const result = document.getElementById(config.resultId);
  const button = form.querySelector("button");

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    result.hidden = true;
    result.classList.remove("error");
    button.disabled = true;
    button.textContent = "Running model…";

    const features = Object.fromEntries(new FormData(form).entries());
    for (const [key, value] of Object.entries(features)) {
      features[key] = Number(value);
    }

    try {
      const response = await fetch(config.endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ features }),
      });
      const data = await response.json();
      if (!response.ok) {
        const detail = Array.isArray(data.detail)
          ? data.detail.map((error) => `${error.loc.at(-1)}: ${error.msg}`).join(", ")
          : data.detail;
        throw new Error(detail || "The prediction could not be completed.");
      }
      result.textContent = config.format(data);
      document.getElementById(config.metricId).textContent =
        config.endpoint.endsWith("regression")
          ? `R² ${Number(data.r2).toFixed(3)}`
          : `Accuracy ${(Number(data.accuracy) * 100).toFixed(1)}%`;
    } catch (error) {
      result.textContent = error.message || "Could not connect to the prediction service.";
      result.classList.add("error");
    } finally {
      result.hidden = false;
      button.disabled = false;
      button.innerHTML = `${config.endpoint.endsWith("regression") ? "Predict vote count" : "Predict genre"} <span aria-hidden="true">→</span>`;
    }
  });
}

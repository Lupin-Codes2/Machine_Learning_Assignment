const modelStatus = document.getElementById("model-status");
const modelUrl = new URL("./model-parameters.json", document.currentScript.src);

function predictVotes(features, model) {
  const [rankingCoefficient, yearCoefficient] = model.coefficients;
  const prediction =
    rankingCoefficient * features.ranking +
    yearCoefficient * features.year +
    model.intercept;

  return Math.max(0, Math.round(prediction));
}

function predictGenre(features, model) {
  const inputs = [features.ranking, features.year, features.votes];
  const scaled = inputs.map((value, index) => (value - model.mean[index]) / model.scale[index]);
  const scores = model.coefficients.map((coefficients, classIndex) =>
    coefficients.reduce(
      (score, coefficient, featureIndex) => score + coefficient * scaled[featureIndex],
      model.intercepts[classIndex],
    ),
  );
  const maxScore = Math.max(...scores);
  const exponentials = scores.map((score) => Math.exp(score - maxScore));
  const total = exponentials.reduce((sum, value) => sum + value, 0);
  const classId = scores.indexOf(maxScore);

  return {
    genre: model.classes[classId],
    confidence: exponentials[classId] / total,
  };
}

function readFeatures(form) {
  const formData = new FormData(form);
  const features = Object.fromEntries(
    [...formData]
      .filter(([name]) => name !== "game")
      .map(([name, value]) => [name, Number(value)]),
  );

  if (Object.values(features).some((value) => !Number.isFinite(value))) {
    throw new Error("Enter a valid number for every field.");
  }

  return { ...features, game: String(formData.get("game") || "").trim() };
}

function showError(result, error) {
  result.textContent = error.message || "The prediction could not be completed.";
  result.classList.add("error");
  result.hidden = false;
}

function connectForm({ formId, resultId, metricId, predict, format, metric }) {
  const form = document.getElementById(formId);
  const result = document.getElementById(resultId);

  form.addEventListener("submit", (event) => {
    event.preventDefault();
    result.hidden = true;
    result.classList.remove("error");

    try {
      const features = readFeatures(form);
      const output = predict(features);
      result.textContent = format(output, features);
      document.getElementById(metricId).textContent = metric();
      result.hidden = false;
    } catch (error) {
      showError(result, error);
    }
  });
}

async function initialize() {
  const response = await fetch(modelUrl);
  if (!response.ok) {
    throw new Error(`Model parameters could not be loaded (${response.status}).`);
  }
  const models = await response.json();
  const classificationForm = document.getElementById("classification-form");
  const gameInput = classificationForm.elements.game;
  const gameOptions = document.getElementById("game-options");
  const gamesByName = new Map(
    models.games.map((game) => [game.name.toLocaleLowerCase(), game]),
  );

  gameOptions.replaceChildren(
    ...models.games.map((game) => {
      const option = document.createElement("option");
      option.value = game.name;
      return option;
    }),
  );

  gameInput.addEventListener("change", () => {
    const game = gamesByName.get(gameInput.value.trim().toLocaleLowerCase());
    if (!game) {
      return;
    }

    classificationForm.elements.ranking.value = game.ranking;
    classificationForm.elements.year.value = game.year;
    classificationForm.elements.votes.value = game.votes;
  });

  connectForm({
    formId: "regression-form",
    resultId: "regression-result",
    metricId: "regression-metric",
    predict: (features) => predictVotes(features, models.regression),
    format: (prediction) =>
      `Estimated votes: ${prediction.toLocaleString("en-US")} · Test R² ${models.regression.r2.toFixed(3)}`,
    metric: () => `R² ${models.regression.r2.toFixed(3)}`,
  });

  connectForm({
    formId: "classification-form",
    resultId: "classification-result",
    metricId: "classification-metric",
    predict: (features) => predictGenre(features, models.classification),
    format: ({ genre, confidence }, features) => {
      const gameContext = features.game
        ? ` for ${features.game} (#${features.ranking}, ${features.year}, ${features.votes.toLocaleString("en-US")} votes)`
        : "";
      return `Predicted primary genre${gameContext}: ${genre} · Confidence ${(confidence * 100).toFixed(1)}%`;
    },
    metric: () => `Accuracy ${(models.classification.accuracy * 100).toFixed(1)}%`,
  });

  modelStatus.textContent = "Models ready";
}

initialize().catch((error) => {
  modelStatus.textContent = "Models unavailable";
  document.querySelectorAll("form button").forEach((button) => {
    button.disabled = true;
  });
  document.querySelectorAll(".result").forEach((result) => {
    showError(result, error);
  });
});

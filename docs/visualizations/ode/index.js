// --- Parameters ---
let L = 5;
let gamma = 1;
let dt = 0.01;
let steps = 10000;
let noise = 0.01; // Noise magnitude

// --- D3 Controls ---
d3.select("body").html(""); // Clear body

const controls = d3.select("body").append("div").style("margin-bottom", "20px");

controls.append("label").text("Bitstring Length (L): ");
controls.append("input")
  .attr("type", "range")
  .attr("id", "L")
  .attr("min", 2)
  .attr("max", 8)
  .attr("step", 1)
  .attr("value", L)
  .style("width", "120px");
controls.append("span").attr("id", "L_val").text(L);

controls.append("span").style("margin-left", "30px");
controls.append("label").text("Gamma: ");
controls.append("input")
  .attr("type", "range")
  .attr("id", "gamma")
  .attr("min", -5)
  .attr("max", 5)
  .attr("step", 0.01)
  .attr("value", gamma)
  .style("width", "120px");
controls.append("span").attr("id", "gamma_val").text(gamma);

controls.append("span").style("margin-left", "30px");
controls.append("label").text("Noise: ");
controls.append("input")
  .attr("type", "range")
  .attr("id", "noise")
  .attr("min", 0)
  .attr("max", 0.2)
  .attr("step", 0.001)
  .attr("value", noise)
  .style("width", "120px");
controls.append("span").attr("id", "noise_val").text(noise);

controls.append("button")
  .attr("id", "restart")
  .style("margin-left", "30px")
  .text("Restart");

// --- Utility Functions ---
function intToBitstring(i, L) {
  return i.toString(2).padStart(L, '0').split('').map(Number);
}
function sumBits(bits) {
  return bits.reduce((a, b) => a + b, 0);
}
function hamming(a, b) {
  let d = 0;
  for (let i = 0; i < a.length; ++i) d += a[i] !== b[i] ? 1 : 0;
  return d;
}

// --- ODE System ---
function makeODEs(L, gamma) {
  const nLang = 1 << L;
  const bitstrings = Array.from({length: nLang}, (_, i) => intToBitstring(i, L));
  // Precompute hamming distances and sumBits
  const sumBitsArr = bitstrings.map(sumBits);
  const hammingMat = Array.from({length: nLang}, (_, i) =>
    Array.from({length: nLang}, (_, j) => hamming(bitstrings[i], bitstrings[j]))
  );
  return {nLang, sumBitsArr, hammingMat};
}

function stepODE(P, ode, gamma, dt, noise) {
  const {nLang, sumBitsArr, hammingMat} = ode;
  const Pnext = new Array(nLang).fill(0);
  const totalP = P.reduce((a, b) => a + b, 0);
  for (let i = 0; i < nLang; ++i) {
    let growth = sumBitsArr[i] * P[i];
    let penalty = 0;
    for (let j = 0; j < nLang; ++j) {
      penalty += hammingMat[i][j] * P[i] * P[j];
    }
    penalty *= gamma;
    // Add noise term: Gaussian noise with stddev = noise
    let noiseTerm = noise * (Math.random() * 2 - 1); // Uniform in [-noise, noise]
    Pnext[i] = P[i] + dt * (growth - penalty) + noiseTerm * Math.sqrt(dt);
    if (Pnext[i] < 0) Pnext[i] = 0;
  }
  // Normalize if needed
  const norm = Pnext.reduce((a, b) => a + b, 0);
  if (norm > 0) for (let i = 0; i < nLang; ++i) Pnext[i] /= norm;
  return Pnext;
}

// --- Simulation ---
function runODESystem(L, gamma, dt, steps, noise) {
  const ode = makeODEs(L, gamma);
  // Start with uniform initial condition
  let P = Array.from({length: ode.nLang}, () => 1 / ode.nLang);

  // Store time series
  let history = [P.slice()];
  for (let t = 0; t < steps; ++t) {
    P = stepODE(P, ode, gamma, dt, noise);
    history.push(P.slice());
  }
  console.log("sumBitsArr:", ode.sumBitsArr);
console.log("finalP:", history[history.length - 1]);
  return {history, ode};
}

// --- Plotting ---
function plotTopLanguagesOverTime(history, ode, L) {
  d3.select("#lang_time_plot").remove();
  d3.select("#lang_bar_plot").remove();

  const nLang = ode.nLang;
  const steps = history.length;
  // Find the top 10 at the final time
  const finalP = history[history.length - 1];
  const top = finalP.map((v, i) => ({idx: i, freq: v}))
    .sort((a, b) => b.freq - a.freq)
    .slice(0, 10);

  // --- Line plot of top 10 over time ---
  const w = 700, h = 320;
  const svg = d3.select("body").append("svg")
    .attr("id", "lang_time_plot")
    .attr("width", w)
    .attr("height", h)
    .style("border", "1px solid #ccc")
    .style("margin-bottom", "20px");

  const x = d3.scaleLinear().domain([0, steps-1]).range([60, w-20]);
  const y = d3.scaleLinear().domain([0, d3.max(top, d => d.freq)]).range([h-40, 30]);

  svg.append("g")
    .attr("transform", `translate(0,${h-40})`)
    .call(d3.axisBottom(x));
  svg.append("g")
    .attr("transform", `translate(60,0)`)
    .call(d3.axisLeft(y));

  const color = d3.scaleOrdinal(d3.schemeCategory10);

  top.forEach((lang, i) => {
    const line = d3.line()
      .x((d, t) => x(t))
      .y((d, t) => y(history[t][lang.idx]));
    svg.append("path")
      .datum(history)
      .attr("fill", "none")
      .attr("stroke", color(i))
      .attr("stroke-width", 2)
      .attr("d", line);

    // Label
    svg.append("text")
      .attr("x", w-10)
      .attr("y", y(history[history.length-1][lang.idx]))
      .attr("dy", "0.35em")
      .attr("text-anchor", "end")
      .attr("fill", color(i))
      .text(intToBitstring(lang.idx, L).join(''));
  });

  svg.append("text")
    .attr("x", w/2)
    .attr("y", 20)
    .attr("text-anchor", "middle")
    .text("Top 10 Languages Over Time");

  // --- Bar chart of top 10 at final time ---
  const w2 = 700, h2 = 220;
  const svg2 = d3.select("body").append("svg")
    .attr("id", "lang_bar_plot")
    .attr("width", w2)
    .attr("height", h2)
    .style("border", "1px solid #ccc");

  const x2 = d3.scaleBand().domain(top.map(d => d.idx)).range([60, w2-20]).padding(0.2);
  const y2 = d3.scaleLinear().domain([0, d3.max(top, d => d.freq)]).range([h2-40, 30]);

  svg2.append("g")
    .attr("transform", `translate(0,${h2-40})`)
    .call(d3.axisBottom(x2).tickFormat(i => intToBitstring(i, L).join('')));
  svg2.append("g")
    .attr("transform", `translate(60,0)`)
    .call(d3.axisLeft(y2));

  svg2.selectAll("rect")
    .data(top)
    .enter()
    .append("rect")
    .attr("x", d => x2(d.idx))
    .attr("y", d => y2(d.freq))
    .attr("width", x2.bandwidth())
    .attr("height", d => h2-40 - y2(d.freq))
    .attr("fill", (d, i) => color(i));

  svg2.selectAll("text.freq")
    .data(top)
    .enter()
    .append("text")
    .attr("class", "freq")
    .attr("x", d => x2(d.idx) + x2.bandwidth()/2)
    .attr("y", d => y2(d.freq) - 5)
    .attr("text-anchor", "middle")
    .text(d => d.freq.toFixed(3));

  svg2.append("text")
    .attr("x", w2/2)
    .attr("y", 20)
    .attr("text-anchor", "middle")
    .text("Top 10 Languages at Final Time");
}

// --- UI Events ---
function updateAndRun() {
  L = +d3.select("#L").property("value");
  gamma = +d3.select("#gamma").property("value");
  noise = +d3.select("#noise").property("value");
  d3.select("#L_val").text(L);
  d3.select("#gamma_val").text(gamma);
  d3.select("#noise_val").text(noise);
  const {history, ode} = runODESystem(L, gamma, dt, steps, noise);
  plotTopLanguagesOverTime(history, ode, L);
}

d3.select("#L").on("input", updateAndRun);
d3.select("#gamma").on("input", updateAndRun);
d3.select("#noise").on("input", updateAndRun);
d3.select("#restart").on("click", updateAndRun);

// --- Initial Run ---
updateAndRun();
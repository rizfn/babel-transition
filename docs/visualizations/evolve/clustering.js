// --- Parameters and Defaults ---
let gamma = -1;
let N = 100;
let L = 10;
let N_rounds = 20;
let mu = 0.01;
let children_per_success = 5; // integer >= 2

let population = [];
let fitness_history = [];
let unique_history = [];
let cluster_history = [];

let generation = 0;

// --- D3 Controls ---
d3.select("body").html(""); // Clear body

const layout = d3.select("body")
    .append("div")
    .style("display", "flex")
    .style("gap", "30px");

const sidebar = layout.append("div")
    .attr("id", "sidebar")
    .style("min-width", "220px")
    .style("padding", "10px")
    .style("border-right", "1px solid #ccc");

const controls = layout.append("div")
    .attr("id", "controls")
    .style("flex", "1");

// --- Sidebar fields ---
sidebar.append("label").text("Population (N): ")
    .append("input")
    .attr("type", "number")
    .attr("id", "N")
    .attr("min", 10)
    .attr("max", 500)
    .attr("step", 1)
    .attr("value", N)
    .style("width", "70px");
sidebar.append("br");

sidebar.append("label").text("Rounds/Gen: ")
    .append("input")
    .attr("type", "number")
    .attr("id", "N_rounds")
    .attr("min", 1)
    .attr("max", 100)
    .attr("step", 1)
    .attr("value", N_rounds)
    .style("width", "70px");
sidebar.append("br");

sidebar.append("label").text("Children per Success: ")
    .append("input")
    .attr("type", "number")
    .attr("id", "children_per_success")
    .attr("min", 2)
    .attr("max", 20)
    .attr("step", 1)
    .attr("value", children_per_success)
    .style("width", "70px");
sidebar.append("br");

sidebar.append("label").text("Bitstring Length (L): ")
    .append("input")
    .attr("type", "range")
    .attr("id", "L")
    .attr("min", 2)
    .attr("max", 20)
    .attr("step", 1)
    .attr("value", L)
    .style("width", "100px");
sidebar.append("span")
    .attr("id", "L_val")
    .text(L);
sidebar.append("br");

sidebar.append("button")
    .attr("id", "pause")
    .style("margin-top", "10px")
    .text("Pause");

sidebar.append("button")
    .attr("id", "restart")
    .style("margin-top", "10px")
    .text("Restart Simulation");

sidebar.append("div")
    .attr("id", "success_ratio_display")
    .style("margin-top", "10px")
    .text(`Success Ratio: ${(1 / children_per_success).toFixed(3)}`);

// --- Top controls (sliders for gamma, mu) ---
function addSlider(label, id, min, max, step, value) {
    const row = controls.append("div").style("margin-bottom", "8px");
    row.append("label")
        .attr("for", id)
        .text(label + ": ");
    row.append("input")
        .attr("type", "range")
        .attr("id", id)
        .attr("min", min)
        .attr("max", max)
        .attr("step", step)
        .attr("value", value)
        .style("width", "200px");
    row.append("span")
        .attr("id", id + "_val")
        .text(value);
}

addSlider("Gamma", "gamma", -10, 1, 0.1, gamma);
addSlider("Mutation Rate (mu)", "mu", 0, 0.1, 0.001, mu);

// SVGs for plots
controls.append("svg")
    .attr("id", "fitness_plot")
    .attr("width", 600)
    .attr("height", 300)
    .style("border", "1px solid #ccc")
    .style("margin-top", "10px");

controls.append("svg")
    .attr("id", "unique_plot")
    .attr("width", 600)
    .attr("height", 150)
    .style("border", "1px solid #ccc")
    .style("margin-top", "10px");

controls.append("svg")
    .attr("id", "languages_plot")
    .attr("width", 600)
    .attr("height", 300)
    .style("border", "1px solid #ccc")
    .style("margin-top", "10px");

controls.append("svg")
    .attr("id", "pca_plot")
    .attr("width", 600)
    .attr("height", 300)
    .style("border", "1px solid #ccc")
    .style("margin-top", "10px");

// --- Utility Functions ---
function randomBitstring(L) {
    return Array(L).fill(0);
}
function hamming(a, b) {
    return a.reduce((acc, v, i) => acc + (v !== b[i] ? 1 : 0), 0);
}
function mutate(bitstring, mu) {
    return bitstring.map(b => Math.random() < mu ? 1 - b : b);
}
function sumBits(bits) {
    return bits.reduce((a, b) => a + b, 0);
}

// --- Simulation Functions ---
function initializePopulation() {
    // Ensure N is divisible by children_per_success
    if (N % children_per_success !== 0) {
        N = Math.round(N / children_per_success) * children_per_success;
        d3.select("#N").property("value", N);
    }
    population = Array.from({ length: N }, () => ({
        language: randomBitstring(L),
        fitness: 0,
        fitnesses: []
    }));
    fitness_history = [];
    generation = 0;
}

function runGeneration() {
    population.forEach(p => { p.fitness = 0; p.fitnesses = []; });

    for (let round = 0; round < N_rounds; round++) {
        let indices = [...Array(N).keys()].sort(() => Math.random() - 0.5);
        for (let i = 0; i < N; i += 2) {
            if (i + 1 >= N) break;
            let a = population[indices[i]];
            let b = population[indices[i + 1]];
            let fit_a = sumBits(a.language) - gamma * hamming(a.language, b.language);
            let fit_b = sumBits(b.language) - gamma * hamming(b.language, a.language);
            a.fitnesses.push(fit_a);
            b.fitnesses.push(fit_b);
        }
    }
    population.forEach(p => {
        p.fitness = p.fitnesses.reduce((a, b) => a + b, 0) / p.fitnesses.length;
    });

    let max_fit = Math.max(...population.map(p => p.fitness));
    fitness_history.push(max_fit);

    // Count unique languages
    let unique = new Set(population.map(p => p.language.join(""))).size;
    unique_history.push(unique);

    // Calculate number of successful agents
    let n_success = N / children_per_success;
    let sorted = [...population].sort((a, b) => b.fitness - a.fitness);
    let next_gen = [];
    for (let i = 0; i < n_success; i++) {
        for (let c = 0; c < children_per_success; c++) {
            let child_lang = mutate(sorted[i].language, mu);
            next_gen.push({ language: child_lang, fitness: 0, fitnesses: [] });
        }
    }
    population = next_gen;
    generation++;
}

// --- D3 Plotting ---
function plotFitness() {
    const svg = d3.select("#fitness_plot");
    svg.selectAll("*").remove();
    const w = +svg.attr("width"), h = +svg.attr("height");
    const x = d3.scaleLinear().domain([0, fitness_history.length - 1]).range([40, w - 10]);
    const y = d3.scaleLinear().domain([0, d3.max(fitness_history) || 1]).range([h - 30, 10]);
    const line = d3.line()
        .x((d, i) => x(i))
        .y(d => y(d));
    svg.append("g").attr("transform", `translate(0,${h - 30})`)
        .call(d3.axisBottom(x).ticks(10));
    svg.append("g").attr("transform", `translate(40,0)`)
        .call(d3.axisLeft(y).ticks(10));
    svg.append("path")
        .datum(fitness_history)
        .attr("fill", "none")
        .attr("stroke", "steelblue")
        .attr("stroke-width", 2)
        .attr("d", line);
    svg.append("text")
        .attr("x", w / 2).attr("y", 20)
        .attr("text-anchor", "middle")
        .text("Max Fitness Over Time");
}

function plotLanguages() {
    const svg = d3.select("#languages_plot");
    svg.selectAll("*").remove();
    const w = +svg.attr("width"), h = +svg.attr("height");

    // Sort population by bitstring value (lexicographically)
    const sortedPop = [...population].sort((a, b) => {
        const aStr = a.language.join("");
        const bStr = b.language.join("");
        return aStr.localeCompare(bStr);
    });

    const cellW = Math.max(1, (w - 60) / sortedPop.length);
    const cellH = Math.max(1, (h - 40) / L);

    // Draw each individual's bitstring as a column
    svg.selectAll("rect")
        .data(sortedPop.flatMap((p, i) => p.language.map((b, j) => ({ col: i, row: j, val: b }))))
        .enter()
        .append("rect")
        .attr("x", d => d.col * cellW + 50)
        .attr("y", d => d.row * cellH + 30)
        .attr("width", cellW - 1)
        .attr("height", cellH - 1)
        .attr("stroke-width", 0)
        .attr("fill", d => d.val ? "#222" : "#eee");

    svg.append("text")
        .attr("x", w / 2)
        .attr("y", 20)
        .attr("text-anchor", "middle")
        .text("Languages (each agent in a column)");
}

function plotClusterCount() {
    const svg = d3.select("#unique_plot");
    svg.selectAll("*").remove();
    const w = +svg.attr("width"), h = +svg.attr("height");
    const x = d3.scaleLinear().domain([0, cluster_history.length - 1]).range([40, w - 10]);
    const y = d3.scaleLinear().domain([0, d3.max(cluster_history) || 1]).range([h - 30, 10]);
    const line = d3.line()
        .x((d, i) => x(i))
        .y(d => y(d));
    svg.append("g").attr("transform", `translate(0,${h - 30})`)
        .call(d3.axisBottom(x).ticks(10));
    svg.append("g").attr("transform", `translate(40,0)`)
        .call(d3.axisLeft(y).ticks(5));
    svg.append("path")
        .datum(cluster_history)
        .attr("fill", "none")
        .attr("stroke", "darkgreen")
        .attr("stroke-width", 2)
        .attr("d", line);
    svg.append("text")
        .attr("x", w / 2).attr("y", 20)
        .attr("text-anchor", "middle")
        .text("Number of Clusters (Languages) Over Time");
}


function plotUMAPAndClusters() {
    const svg = d3.select("#pca_plot");
    svg.selectAll("*").remove();
    const w = +svg.attr("width"), h = +svg.attr("height");
    if (population.length === 0) return;
    const data = population.map(p => p.language);

    // Run UMAP
    const umap = new UMAP({ nComponents: 2, nNeighbors: 15, minDist: 0.1 });
    const embedding = umap.fit(data);

    // Run DBSCAN from density-clustering
    const dbscan = new DBSCAN();
    const clusters = dbscan.run(embedding, 0.5, 3); // eps, minPts

    // Assign cluster labels for each point
    const labels = Array(embedding.length).fill(-1);
    clusters.forEach((cluster, i) => {
        cluster.forEach(idx => {
            labels[idx] = i;
        });
    });

    // Count clusters (ignore noise, i.e., label -1)
    const nClusters = new Set(labels.filter(l => l !== -1)).size;
    cluster_history.push(nClusters);

    // Color palette
    const palette = d3.schemeCategory10.concat(d3.schemeSet3);

    // Find min/max for scaling
    const xs = embedding.map(d => d[0]);
    const ys = embedding.map(d => d[1]);
    const x = d3.scaleLinear().domain([Math.min(...xs), Math.max(...xs)]).range([40, w - 20]);
    const y = d3.scaleLinear().domain([Math.min(...ys), Math.max(...ys)]).range([h - 30, 30]);

    svg.append("g").attr("transform", `translate(0,${h - 30})`).call(d3.axisBottom(x).ticks(8));
    svg.append("g").attr("transform", `translate(40,0)`).call(d3.axisLeft(y).ticks(8));
    svg.selectAll("circle")
        .data(embedding)
        .enter()
        .append("circle")
        .attr("cx", (d) => x(d[0]))
        .attr("cy", (d) => y(d[1]))
        .attr("r", 4)
        .attr("fill", (d, i) => {
            const label = labels[i];
            if (label === -1) return "#bbb"; // noise
            return palette[label % palette.length];
        })
        .attr("opacity", 0.85);
    svg.append("text")
        .attr("x", w / 2)
        .attr("y", 20)
        .attr("text-anchor", "middle")
        .text("UMAP of Languages (colored by cluster)");
}

// --- Controls and Loop ---
function updateSidebarFields() {
    d3.select("#L_val").text(d3.select("#L").property("value"));
    d3.select("#success_ratio_display").text(`Success Ratio: ${(1 / children_per_success).toFixed(3)}`);
}

d3.select("#L").on("input", function () {
    L = +this.value;
    d3.select("#L_val").text(L);
});

d3.select("#gamma").on("input", function () {
    gamma = +this.value;
    d3.select("#gamma_val").text(gamma);
});
d3.select("#mu").on("input", function () {
    mu = +this.value;
    d3.select("#mu_val").text(mu);
});

// Only update these on restart
d3.select("#restart").on("click", () => {
    // Read sidebar fields
    let newN = +d3.select("#N").property("value");
    let newRounds = +d3.select("#N_rounds").property("value");
    let newChildren = +d3.select("#children_per_success").property("value");
    let newL = +d3.select("#L").property("value");

    // Ensure N is divisible by children_per_success
    if (newN % newChildren !== 0) {
        newN = Math.round(newN / newChildren) * newChildren;
        newN = Math.max(10, Math.min(newN, 500));
        d3.select("#N").property("value", newN);
    }
    // Clamp children_per_success if needed
    if (newChildren > newN) {
        newChildren = newN;
        d3.select("#children_per_success").property("value", newChildren);
    }
    // If still not a divisor, pick closest valid divisor
    if (newN % newChildren !== 0) {
        let divisors = [];
        for (let i = 2; i <= Math.min(20, newN); i++) {
            if (newN % i === 0) divisors.push(i);
        }
        if (divisors.length > 0) {
            newChildren = divisors.reduce((prev, curr) =>
                Math.abs(curr - newChildren) < Math.abs(prev - newChildren) ? curr : prev
            );
            d3.select("#children_per_success").property("value", newChildren);
        }
    }

    N = newN;
    N_rounds = newRounds;
    children_per_success = newChildren;
    L = newL;

    updateSidebarFields();
    initializePopulation();
    plotFitness();
    plotLanguages();
    cluster_history = [];
    plotUMAPAndClusters();
    plotClusterCount();
});

let running = true;
let timeoutHandle = null;

function step() {
    if (!running) return;
    runGeneration();
    plotFitness();
    plotLanguages();
    plotUMAPAndClusters(); // replaces plotPCA
    plotClusterCount(); // replaces plotUniqueLanguages
    timeoutHandle = setTimeout(step, 200);
}

// Pause/Resume button logic
d3.select("#pause").on("click", function () {
    running = !running;
    d3.select(this).text(running ? "Pause" : "Resume");
    if (running) step();
});



// --- Start ---
updateSidebarFields();
initializePopulation();
plotFitness();
plotLanguages();
step();
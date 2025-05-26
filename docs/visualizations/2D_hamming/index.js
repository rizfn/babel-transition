// Parameters
const L = 20; // lattice size
const N = 8;  // language length
let gamma = 0.1;  // similarity penalty
let mutationRate = 0.01;
const T = 3.5; // logistic map parameter
const cellSize = 25;
const width = L * cellSize, height = L * cellSize;

if (typeof document !== "undefined") {
    const gammaSlider = document.getElementById("gammaSlider");
    const gammaValue = document.getElementById("gammaValue");
    gammaSlider.addEventListener("input", function () {
        gamma = parseFloat(this.value);
        gammaValue.textContent = gamma;
    });

    // Mutation rate slider
    const mutationSlider = document.getElementById("mutationSlider");
    const mutationValue = document.getElementById("mutationValue");
    mutationSlider.addEventListener("input", function () {
        mutationRate = parseFloat(this.value);
        mutationValue.textContent = mutationRate;
    });
}
// Helper: random binary string
function randomLang(n) {
    return Array.from({ length: n }, () => Math.random() < 0.5 ? '0' : '1').join('');
}

// Helper: hamming distance
function hamming(a, b) {
    let d = 0;
    for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) d++;
    return d;
}

// Initialize lattice
let lattice = [];
for (let y = 0; y < L; y++) {
    let row = [];
    for (let x = 0; x < L; x++) {
        row.push({
            lang: randomLang(N),
            pop: Math.random() * 0.5 + 0.25 // initial population
        });
    }
    lattice.push(row);
}

// D3 setup
const svg = d3.select("#lattice")
    .attr("width", width)
    .attr("height", height);

// Draw cells
svg.selectAll("g")
    .data(lattice)
    .join("g")
    .selectAll("rect")
    .data(d => d)
    .join("rect")
    .attr("class", "cell")
    .attr("x", (d, i, nodes) => (nodes[i].parentNode.__data__.indexOf(d)) * cellSize)
    .attr("y", (d, i, nodes) => lattice.indexOf(nodes[i].parentNode.__data__) * cellSize)
    .attr("width", cellSize)
    .attr("height", cellSize)
    .on("click", function (event, d) {
        d3.select("#info").text(`Language: ${d.lang}, Population: ${d.pop.toFixed(3)}`);
    });

function langToHue(lang) {
    let hash = 0;
    for (let i = 0; i < lang.length; i++) {
        hash = (hash * 31 + lang.charCodeAt(i)) % 360;
    }
    return hash / 360;
}

function mutateLang(lang, mutationRate = 0.01) {
    let arr = lang.split('');
    for (let i = 0; i < arr.length; i++) {
        if (Math.random() < mutationRate) {
            arr[i] = arr[i] === '0' ? '1' : '0';
        }
    }
    return arr.join('');
}

// Update function
function update() {
    // Deep copy for synchronous update
    let newLattice = lattice.map(row => row.map(cell => ({ ...cell })));

    for (let y = 0; y < L; y++) {
        for (let x = 0; x < L; x++) {
            let cell = lattice[y][x];
            // Logistic growth, simpler language grows faster
            let complexity = cell.lang.split('').filter(c => c === '0').length;
            let growthRate = T - (complexity / N) * 2; // Simpler = faster
            let pop = cell.pop;
            let growth = growthRate * pop * (1 - pop);

            // Competition: penalize by similarity to neighbors
            let penalty = 0;
            let neighbors = [
                [y - 1, x], [y + 1, x], [y, x - 1], [y, x + 1]
            ].filter(([ny, nx]) => ny >= 0 && ny < L && nx >= 0 && nx < L);

            for (let [ny, nx] of neighbors) {
                let neighbor = lattice[ny][nx];
                let ham = hamming(cell.lang, neighbor.lang);
                penalty += gamma * (N - ham) / N; // more similar = higher penalty
            }

            // Mutation: weaker (lower pop) languages mutate more
            let effectiveMutation = mutationRate / Math.max(cell.pop, 0.05); // clamp to avoid huge rates
            newLattice[y][x].lang = mutateLang(newLattice[y][x].lang, effectiveMutation);

            newLattice[y][x].pop = Math.max(0, Math.min(1, pop + growth - penalty * pop));
        }
    }
    lattice = newLattice;

    // Update colors
    svg.selectAll("g")
        .data(lattice)
        .selectAll("rect")
        .data(d => d)
        .attr("fill", d => {
            let hue = langToHue(d.lang);
            let alpha = d.pop;
            let rgb = d3.hsl(hue * 360, 0.8, 0.5).rgb();
            return `rgba(${rgb.r},${rgb.g},${rgb.b},${alpha})`;
        });
}
// Animation loop
setInterval(update, 200);

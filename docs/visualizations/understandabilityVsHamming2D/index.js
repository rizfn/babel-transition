class LatticeSimulation {
    constructor() {
        this.L = 64;
        this.B = 16;
        this.gamma = 1;
        this.alpha = 1;
        this.mu = 0.001;
        this.step = 0;
        this.isRunning = false;
        this.selectedLanguage = null;
        
        this.lattice = this.initializeLattice();
        this.colorMap = this.createBitstringColorMap();
        
        this.setupUI();
        this.setupVisualization();
        this.render();
    }
    
    initializeLattice() {
        const lattice = [];
        for (let i = 0; i < this.L; i++) {
            const row = [];
            for (let j = 0; j < this.L; j++) {
                row.push({
                    language: new Array(this.B).fill(0),
                    fitness: 0,
                    immune: false
                });
            }
            lattice.push(row);
        }
        return lattice;
    }
    
    createBitstringColorMap() {
        const colorMap = new Map();
        const numBitstrings = Math.pow(2, this.B);
        
        const allBitstrings = [];
        for (let i = 0; i < numBitstrings; i++) {
            const bits = [];
            for (let b = 0; b < this.B; b++) {
                bits.push((i >> b) & 1);
            }
            allBitstrings.push(bits.join(''));
        }
        
        const colors = [];
        for (let i = 0; i < numBitstrings; i++) {
            const hue = (i / (numBitstrings - 1)) * 360;
            colors.push(`hsl(${hue}, 70%, 50%)`);
        }
        
        for (let i = colors.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [colors[i], colors[j]] = [colors[j], colors[i]];
        }
        
        allBitstrings.forEach((bitstring, index) => {
            colorMap.set(bitstring, colors[index]);
        });
        
        return colorMap;
    }
    
    getUniqueLanguages() {
        const languageSet = new Set();
        const languageCounts = new Map();
        
        for (let i = 0; i < this.L; i++) {
            for (let j = 0; j < this.L; j++) {
                const langStr = this.lattice[i][j].language.join('');
                languageSet.add(langStr);
                languageCounts.set(langStr, (languageCounts.get(langStr) || 0) + 1);
            }
        }
        
        const sortedLanguages = Array.from(languageSet).sort((a, b) => {
            const weightA = a.split('').reduce((sum, bit) => sum + parseInt(bit), 0);
            const weightB = b.split('').reduce((sum, bit) => sum + parseInt(bit), 0);
            if (weightA !== weightB) return weightA - weightB;
            return a.localeCompare(b);
        });
        
        return { languages: sortedLanguages, counts: languageCounts };
    }
    
    communicability(lang1, lang2) {
        let count = 0;
        for (let i = 0; i < this.B; i++) {
            count += (lang1[i] & lang2[i]);
        }
        return count;
    }
    
    meanFieldDistance(language, meanField) {
        let distance = 0;
        for (let i = 0; i < this.B; i++) {
            distance += Math.abs(language[i] - meanField[i]);
        }
        return distance / this.B;
    }
    
    findWeakestNeighbor(cx, cy) {
        const neighbors = [
            [(cx + 1) % this.L, cy],
            [(cx - 1 + this.L) % this.L, cy],
            [cx, (cy + 1) % this.L],
            [cx, (cy - 1 + this.L) % this.L]
        ];
        
        let minFitness = Infinity;
        const weakestNeighbors = [];
        
        for (const [nx, ny] of neighbors) {
            const neighbor = this.lattice[nx][ny];
            if (!neighbor.immune && neighbor.fitness < minFitness) {
                minFitness = neighbor.fitness;
            }
        }
        
        for (const [nx, ny] of neighbors) {
            const neighbor = this.lattice[nx][ny];
            if (!neighbor.immune && neighbor.fitness === minFitness) {
                weakestNeighbors.push([nx, ny]);
            }
        }
        
        if (weakestNeighbors.length > 0) {
            const idx = Math.floor(Math.random() * weakestNeighbors.length);
            return weakestNeighbors[idx];
        }
        
        return [-1, -1];
    }
    
    update() {
        const meanField = new Array(this.B).fill(0);
        const totalAgents = this.L * this.L;
        
        for (let i = 0; i < this.L; i++) {
            for (let j = 0; j < this.L; j++) {
                for (let b = 0; b < this.B; b++) {
                    meanField[b] += this.lattice[i][j].language[b];
                }
            }
        }
        
        for (let b = 0; b < this.B; b++) {
            meanField[b] /= totalAgents;
        }
        
        for (let i = 0; i < this.L; i++) {
            for (let j = 0; j < this.L; j++) {
                const agent = this.lattice[i][j];
                agent.fitness = 0;
                
                const globalFitness = this.gamma * this.meanFieldDistance(agent.language, meanField);
                agent.fitness += globalFitness;
                
                const neighbors = [
                    [(i + 1) % this.L, j],
                    [(i - 1 + this.L) % this.L, j],
                    [i, (j + 1) % this.L],
                    [i, (j - 1 + this.L) % this.L]
                ];
                
                let localFitness = 0;
                for (const [ni, nj] of neighbors) {
                    const neighbor = this.lattice[ni][nj];
                    const comm = this.communicability(agent.language, neighbor.language);
                    localFitness += (this.alpha / 4.0) * (comm / this.B);
                }
                
                agent.fitness += localFitness;
            }
        }
        
        const positions = [];
        for (let i = 0; i < this.L; i++) {
            for (let j = 0; j < this.L; j++) {
                positions.push([i, j]);
            }
        }
        
        for (let i = positions.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [positions[i], positions[j]] = [positions[j], positions[i]];
        }
        
        for (let i = 0; i < this.L; i++) {
            for (let j = 0; j < this.L; j++) {
                this.lattice[i][j].immune = false;
            }
        }
        
        const trials = Math.floor((this.L * this.L) / 2);
        for (let trial = 0; trial < trials && trial < positions.length; trial++) {
            const [i, j] = positions[trial];
            
            if (this.lattice[i][j].immune) continue;
            
            const [wi, wj] = this.findWeakestNeighbor(i, j);
            
            if (wi === -1 && wj === -1) continue;
            
            if (this.lattice[i][j].fitness > this.lattice[wi][wj].fitness) {
                this.lattice[wi][wj].language = [...this.lattice[i][j].language];
                this.lattice[i][j].immune = true;
                this.lattice[wi][wj].immune = true;
            }
        }
        
        for (let i = 0; i < this.L; i++) {
            for (let j = 0; j < this.L; j++) {
                for (let b = 0; b < this.B; b++) {
                    if (Math.random() < this.mu) {
                        this.lattice[i][j].language[b] = 1 - this.lattice[i][j].language[b];
                    }
                }
            }
        }
        
        this.step++;
    }
    
    setupUI() {
        const body = d3.select('body');
        
        const mainContainer = body.append('div')
            .attr('class', 'main-container');
        
        const controlsContainer = mainContainer.append('div')
            .attr('class', 'controls-container');
        
        this.stepDisplay = controlsContainer.append('div')
            .attr('class', 'step-display')
            .text('Step: 0');
        
        this.startButton = controlsContainer.append('button')
            .attr('class', 'control-button')
            .text('Start')
            .on('click', () => this.toggleSimulation());
        
        controlsContainer.append('button')
            .attr('class', 'control-button')
            .text('Step')
            .on('click', () => this.singleStep());
        
        controlsContainer.append('button')
            .attr('class', 'control-button')
            .text('Reset')
            .on('click', () => this.reset());
        
        // Gamma control
        const gammaContainer = controlsContainer.append('div')
            .attr('class', 'parameter-container');
        
        gammaContainer.append('label')
            .attr('class', 'parameter-label')
            .text('Gamma:');
        
        const gammaInput = gammaContainer.append('input')
            .attr('class', 'parameter-input')
            .attr('type', 'range')
            .attr('min', '0')
            .attr('max', '5')
            .attr('step', '0.1')
            .attr('value', this.gamma)
            .on('input', (event) => {
                this.gamma = parseFloat(event.target.value);
                gammaDisplay.text(this.gamma.toFixed(1));
            });
        
        const gammaDisplay = gammaContainer.append('span')
            .attr('class', 'parameter-display')
            .text(this.gamma.toFixed(1));
        
        // Alpha control
        const alphaContainer = controlsContainer.append('div')
            .attr('class', 'parameter-container');
        
        alphaContainer.append('label')
            .attr('class', 'parameter-label')
            .text('Alpha:');
        
        const alphaInput = alphaContainer.append('input')
            .attr('class', 'parameter-input')
            .attr('type', 'range')
            .attr('min', '0')
            .attr('max', '5')
            .attr('step', '0.1')
            .attr('value', this.alpha)
            .on('input', (event) => {
                this.alpha = parseFloat(event.target.value);
                alphaDisplay.text(this.alpha.toFixed(1));
            });
        
        const alphaDisplay = alphaContainer.append('span')
            .attr('class', 'parameter-display')
            .text(this.alpha.toFixed(1));
        
        // Mu control
        const muContainer = controlsContainer.append('div')
            .attr('class', 'parameter-container');
        
        muContainer.append('label')
            .attr('class', 'parameter-label')
            .text('Mu:');
        
        const muInput = muContainer.append('input')
            .attr('class', 'parameter-input')
            .attr('type', 'range')
            .attr('min', '0')
            .attr('max', '0.01')
            .attr('step', '0.0001')
            .attr('value', this.mu)
            .on('input', (event) => {
                this.mu = parseFloat(event.target.value);
                muDisplay.text(this.mu.toFixed(4));
            });
        
        const muDisplay = muContainer.append('span')
            .attr('class', 'parameter-display')
            .text(this.mu.toFixed(4));
        
        this.latticeContainer = mainContainer.append('div')
            .attr('class', 'visualization-container');
        
        this.heatmapContainer = mainContainer.append('div')
            .attr('class', 'visualization-container');
    }
    
    setupVisualization() {
        const windowHeight = window.innerHeight;
        const windowWidth = window.innerWidth;
        const frWidth = (windowWidth - windowWidth * 0.1) / 2;
        
        const availableHeight = windowHeight - 60;
        const availableWidth = frWidth - 20;
        const maxLatticeSize = Math.min(availableHeight, availableWidth);
        const cellSize = Math.floor(maxLatticeSize / this.L);
        const latticeSize = cellSize * this.L;
        
        this.latticeContainer.append('h3')
            .attr('class', 'visualization-title')
            .text('Lattice');
        
        this.svg = this.latticeContainer
            .append('svg')
            .attr('width', latticeSize)
            .attr('height', latticeSize);
        
        this.gridData = [];
        for (let i = 0; i < this.L; i++) {
            for (let j = 0; j < this.L; j++) {
                this.gridData.push({
                    x: i,
                    y: j,
                    id: `cell-${i}-${j}`
                });
            }
        }
        
        this.cells = this.svg.selectAll('.cell')
            .data(this.gridData)
            .enter()
            .append('rect')
            .attr('class', 'lattice-cell')
            .attr('x', d => d.x * cellSize)
            .attr('y', d => d.y * cellSize)
            .attr('width', cellSize)
            .attr('height', cellSize)
            .on('click', (event, d) => {
                const agent = this.lattice[d.x][d.y];
                const langStr = agent.language.join('');
                this.selectLanguage(langStr);
            });
        
        this.heatmapContainer.append('h3')
            .attr('class', 'visualization-title')
            .text('Top 20 Languages');
        
        const maxHeatmapHeight = maxLatticeSize;
        const rowHeight = Math.floor(maxHeatmapHeight / 21);
        const bitWidth = Math.floor((frWidth - 100) / this.B);
        
        this.heatmapSvg = this.heatmapContainer
            .append('svg')
            .attr('width', this.B * bitWidth + 100)
            .attr('height', 21 * rowHeight + 40);
        
        this.heatmapGroup = this.heatmapSvg.append('g');
        
        this.heatmapDimensions = {
            rowHeight: rowHeight,
            bitWidth: bitWidth,
            margin: { top: 20, right: 20, bottom: 20, left: 60 }
        };
    }
    
    selectLanguage(langStr) {
        if (this.selectedLanguage === langStr) {
            this.selectedLanguage = null;
        } else {
            this.selectedLanguage = langStr;
        }
        this.renderSelection();
    }
    
    renderHeatmap() {
        const { languages, counts } = this.getUniqueLanguages();
        const { rowHeight, bitWidth, margin } = this.heatmapDimensions;
        
        const sortedByCount = languages.sort((a, b) => counts.get(b) - counts.get(a));
        const top20Languages = sortedByCount.slice(0, 20);
        
        let displayLanguages = [...top20Languages];
        if (this.selectedLanguage && !top20Languages.includes(this.selectedLanguage)) {
            displayLanguages.push(this.selectedLanguage);
        }
        
        const languageRows = this.heatmapGroup
            .selectAll('.language-row')
            .data(displayLanguages.map(lang => ({
                langStr: lang,
                bits: lang.split('').map(b => parseInt(b)),
                count: counts.get(lang) || 0,
                color: this.colorMap.get(lang),
                isSelected: lang === this.selectedLanguage
            })), d => d.langStr);
        
        const enterRows = languageRows.enter()
            .append('g')
            .attr('class', 'language-row')
            .on('click', (event, d) => {
                this.selectLanguage(d.langStr);
            });
        
        enterRows.append('text')
            .attr('class', 'count-label')
            .attr('x', margin.left - 10)
            .attr('text-anchor', 'end')
            .attr('dy', '0.35em');
        
        enterRows.each(function(d) {
            const rowGroup = d3.select(this);
            rowGroup.selectAll('.bit-cell')
                .data(d.bits.map((bit, i) => ({
                    bit: bit,
                    bitIndex: i,
                    langStr: d.langStr,
                    color: d.color,
                    isSelected: d.isSelected
                })))
                .enter()
                .append('rect')
                .attr('class', 'bit-cell')
                .attr('width', bitWidth)
                .attr('height', rowHeight);
        });
        
        const updateRows = languageRows.merge(enterRows);
        
        updateRows
            .attr('transform', (d, i) => `translate(0, ${margin.top + i * rowHeight})`);
        
        updateRows.select('.count-label')
            .text(d => d.count)
            .attr('y', rowHeight / 2)
            .classed('selected', d => d.isSelected);
        
        updateRows.each(function(d) {
            const rowGroup = d3.select(this);
            const bitCells = rowGroup.selectAll('.bit-cell')
                .data(d.bits.map((bit, i) => ({
                    bit: bit,
                    bitIndex: i,
                    langStr: d.langStr,
                    color: d.color,
                    isSelected: d.isSelected
                })));
            
            bitCells
                .attr('x', (bitData, i) => margin.left + i * bitWidth)
                .attr('y', 0)
                .attr('fill', bitData => bitData.bit === 1 ? bitData.color : '#ffffff00')
                .classed('selected', bitData => bitData.isSelected);
        });
        
        languageRows.exit().remove();
    }
    
    renderSelection() {
        this.cells
            .classed('selected', d => {
                const agent = this.lattice[d.x][d.y];
                const langStr = agent.language.join('');
                return langStr === this.selectedLanguage;
            });
        
        this.renderHeatmap();
    }
    
    render() {
        this.stepDisplay.text(`Step: ${this.step}`);
        
        this.cells
            .datum((d, i) => {
                const agent = this.lattice[d.x][d.y];
                const bitstringKey = agent.language.join('');
                return {
                    ...d,
                    color: this.colorMap.get(bitstringKey) || '#000000'
                };
            })
            .attr('fill', d => d.color);
        
        this.renderHeatmap();
        this.renderSelection();
    }
    
    toggleSimulation() {
        if (this.isRunning) {
            this.stop();
        } else {
            this.start();
        }
    }
    
    start() {
        this.isRunning = true;
        this.startButton.text('Stop');
        this.runLoop();
    }
    
    stop() {
        this.isRunning = false;
        this.startButton.text('Start');
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
        }
    }
    
    runLoop() {
        if (!this.isRunning) return;
        
        this.update();
        this.render();
        
        setTimeout(() => {
            this.animationId = requestAnimationFrame(() => this.runLoop());
        }, 100);
    }
    
    singleStep() {
        if (!this.isRunning) {
            this.update();
            this.render();
        }
    }
    
    reset() {
        this.stop();
        this.step = 0;
        this.selectedLanguage = null;
        this.lattice = this.initializeLattice();
        this.render();
    }
}

document.addEventListener('DOMContentLoaded', () => {
    new LatticeSimulation();
});
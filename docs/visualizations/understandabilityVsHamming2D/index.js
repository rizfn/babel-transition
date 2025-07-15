class LatticeSimulation {
    constructor() {
        this.L = 64; // Smaller lattice for web performance
        this.B = 16; // Bitstring length
        this.gamma = 1;
        this.alpha = 1;
        this.mu = 0.001;
        this.step = 0;
        this.isRunning = false;
        this.selectedLanguage = null; // Track selected language
        this.baseFontSize = 1.5; // Base font size in rem - change this to scale all text
        
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
        
        // Generate all possible bitstrings
        const allBitstrings = [];
        for (let i = 0; i < numBitstrings; i++) {
            const bits = [];
            for (let b = 0; b < this.B; b++) {
                bits.push((i >> b) & 1);
            }
            allBitstrings.push(bits.join(''));
        }
        
        // Create shuffled colors from rainbow colormap
        const colors = [];
        for (let i = 0; i < numBitstrings; i++) {
            const hue = (i / (numBitstrings - 1)) * 360;
            colors.push(`hsl(${hue}, 70%, 50%)`);
        }
        
        // Shuffle colors using Fisher-Yates algorithm
        for (let i = colors.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [colors[i], colors[j]] = [colors[j], colors[i]];
        }
        
        // Map bitstrings to colors
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
        
        // Sort languages by Hamming weight (number of 1s), then lexicographically
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
        
        // Find minimum fitness among non-immune neighbors
        for (const [nx, ny] of neighbors) {
            const neighbor = this.lattice[nx][ny];
            if (!neighbor.immune && neighbor.fitness < minFitness) {
                minFitness = neighbor.fitness;
            }
        }
        
        // Collect all non-immune neighbors with minimum fitness
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
        // 1. Calculate mean-field bitstring
        const meanField = new Array(this.B).fill(0);
        const totalAgents = this.L * this.L;
        
        for (let i = 0; i < this.L; i++) {
            for (let j = 0; j < this.L; j++) {
                for (let b = 0; b < this.B; b++) {
                    meanField[b] += this.lattice[i][j].language[b];
                }
            }
        }
        
        // Normalize to get probabilities
        for (let b = 0; b < this.B; b++) {
            meanField[b] /= totalAgents;
        }
        
        // 2. Fitness evaluation
        for (let i = 0; i < this.L; i++) {
            for (let j = 0; j < this.L; j++) {
                const agent = this.lattice[i][j];
                agent.fitness = 0;
                
                // Global interaction with mean field
                const globalFitness = this.gamma * this.meanFieldDistance(agent.language, meanField);
                agent.fitness += globalFitness;
                
                // Local interactions with neighbors
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
        
        // 3. Reproduction: stochastic invasion trials
        const positions = [];
        for (let i = 0; i < this.L; i++) {
            for (let j = 0; j < this.L; j++) {
                positions.push([i, j]);
            }
        }
        
        // Shuffle positions
        for (let i = positions.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [positions[i], positions[j]] = [positions[j], positions[i]];
        }
        
        // Reset immunity
        for (let i = 0; i < this.L; i++) {
            for (let j = 0; j < this.L; j++) {
                this.lattice[i][j].immune = false;
            }
        }
        
        // Perform invasion trials
        const trials = Math.floor((this.L * this.L) / 2);
        for (let trial = 0; trial < trials && trial < positions.length; trial++) {
            const [i, j] = positions[trial];
            
            if (this.lattice[i][j].immune) continue;
            
            const [wi, wj] = this.findWeakestNeighbor(i, j);
            
            if (wi === -1 && wj === -1) continue;
            
            if (this.lattice[i][j].fitness > this.lattice[wi][wj].fitness) {
                // Invade: clone current agent into weakest neighbor position
                this.lattice[wi][wj].language = [...this.lattice[i][j].language];
                this.lattice[i][j].immune = true;
                this.lattice[wi][wj].immune = true;
            }
        }
        
        // 4. Mutation
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
        const body = d3.select('body')
            .style('margin', '0')
            .style('padding', '0')
            .style('background-color', 'transparent');
        
        // Create main container with grid layout
        const mainContainer = body.append('div')
            .style('display', 'grid')
            .style('grid-template-columns', '10% 1fr 1fr')
            .style('height', '100vh')
            .style('width', '100vw')
            .style('margin', '0')
            .style('padding', '0')
            .style('box-sizing', 'border-box');
        
        // Controls container (10% width)
        const controlsContainer = mainContainer.append('div')
            .style('display', 'flex')
            .style('flex-direction', 'column')
            .style('gap', '10px')
            .style('padding', '10px')
            .style('box-sizing', 'border-box');
        
        // Step counter
        this.stepDisplay = controlsContainer.append('div')
            .style('font-size', `${this.baseFontSize * 1.3}rem`)
            .style('font-weight', 'bold')
            .text('Step: 0');
        
        // Control buttons
        this.startButton = controlsContainer.append('button')
            .text('Start')
            .style('padding', '8px 16px')
            .style('font-size', `${this.baseFontSize}rem`)
            .on('click', () => this.toggleSimulation());
        
        controlsContainer.append('button')
            .text('Step')
            .style('padding', '8px 16px')
            .style('font-size', `${this.baseFontSize}rem`)
            .on('click', () => this.singleStep());
        
        controlsContainer.append('button')
            .text('Reset')
            .style('padding', '8px 16px')
            .style('font-size', `${this.baseFontSize}rem`)
            .on('click', () => this.reset());
        
        // Parameter controls
        const gammaContainer = controlsContainer.append('div')
            .style('display', 'flex')
            .style('flex-direction', 'column')
            .style('gap', '5px');
        
        gammaContainer.append('label')
            .text('Gamma:')
            .style('font-weight', 'bold')
            .style('font-size', `${this.baseFontSize}rem`);
        
        const gammaInput = gammaContainer.append('input')
            .attr('type', 'range')
            .attr('min', '0')
            .attr('max', '5')
            .attr('step', '0.1')
            .attr('value', this.gamma)
            .style('width', '100%')
            .on('input', (event) => {
                this.gamma = parseFloat(event.target.value);
                gammaDisplay.text(this.gamma.toFixed(1));
            });
        
        const gammaDisplay = gammaContainer.append('span')
            .text(this.gamma.toFixed(1))
            .style('text-align', 'center')
            .style('font-size', `${this.baseFontSize * 0.9}rem`);
        
        const alphaContainer = controlsContainer.append('div')
            .style('display', 'flex')
            .style('flex-direction', 'column')
            .style('gap', '5px');
        
        alphaContainer.append('label')
            .text('Alpha:')
            .style('font-weight', 'bold')
            .style('font-size', `${this.baseFontSize}rem`);
        
        const alphaInput = alphaContainer.append('input')
            .attr('type', 'range')
            .attr('min', '0')
            .attr('max', '5')
            .attr('step', '0.1')
            .attr('value', this.alpha)
            .style('width', '100%')
            .on('input', (event) => {
                this.alpha = parseFloat(event.target.value);
                alphaDisplay.text(this.alpha.toFixed(1));
            });
        
        const alphaDisplay = alphaContainer.append('span')
            .text(this.alpha.toFixed(1))
            .style('text-align', 'center')
            .style('font-size', `${this.baseFontSize * 0.9}rem`);
        
        const muContainer = controlsContainer.append('div')
            .style('display', 'flex')
            .style('flex-direction', 'column')
            .style('gap', '5px');
        
        muContainer.append('label')
            .text('Mu:')
            .style('font-weight', 'bold')
            .style('font-size', `${this.baseFontSize}rem`);
        
        const muInput = muContainer.append('input')
            .attr('type', 'range')
            .attr('min', '0')
            .attr('max', '0.01')
            .attr('step', '0.0001')
            .attr('value', this.mu)
            .style('width', '100%')
            .on('input', (event) => {
                this.mu = parseFloat(event.target.value);
                muDisplay.text(this.mu.toFixed(4));
            });
        
        const muDisplay = muContainer.append('span')
            .text(this.mu.toFixed(4))
            .style('text-align', 'center')
            .style('font-size', `${this.baseFontSize * 0.9}rem`);
        
        // Store containers for later use
        this.latticeContainer = mainContainer.append('div')
            .style('display', 'flex')
            .style('flex-direction', 'column')
            .style('align-items', 'center')
            .style('padding', '10px')
            .style('box-sizing', 'border-box');
        
        this.heatmapContainer = mainContainer.append('div')
            .style('display', 'flex')
            .style('flex-direction', 'column')
            .style('align-items', 'center')
            .style('padding', '10px')
            .style('box-sizing', 'border-box');
    }
    
    setupVisualization() {
        // Calculate dynamic sizing
        const windowHeight = window.innerHeight;
        const windowWidth = window.innerWidth;
        
        // Calculate actual 1fr width (45% of remaining space after 10% for controls)
        const frWidth = (windowWidth - windowWidth * 0.1) / 2; // Each 1fr column gets half the remaining width
        
        // Lattice size: maximize available space
        const availableHeight = windowHeight - 60; // Account for title and padding
        const availableWidth = frWidth - 20; // 1fr width minus padding
        const maxLatticeSize = Math.min(availableHeight, availableWidth);
        const cellSize = Math.floor(maxLatticeSize / this.L);
        const latticeSize = cellSize * this.L;
        
        // Lattice visualization
        this.latticeContainer.append('h3')
            .text('Lattice')
            .style('margin', '0 0 10px 0')
            .style('font-size', `${this.baseFontSize * 1.2}rem`);
        
        this.svg = this.latticeContainer
            .append('svg')
            .attr('width', latticeSize)
            .attr('height', latticeSize);
        
        // Create grid data
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
        
        // Create cells
        this.cells = this.svg.selectAll('.cell')
            .data(this.gridData)
            .enter()
            .append('rect')
            .attr('class', 'cell')
            .attr('x', d => d.x * cellSize)
            .attr('y', d => d.y * cellSize)
            .attr('width', cellSize)
            .attr('height', cellSize)
            .attr('stroke', '#333')
            .attr('stroke-width', 0.5)
            .style('cursor', 'pointer')
            .on('click', (event, d) => {
                const agent = this.lattice[d.x][d.y];
                const langStr = agent.language.join('');
                this.selectLanguage(langStr);
            });
        
        // Language heatmap
        this.heatmapContainer.append('h3')
            .text('Top 20 Languages')
            .style('margin', '0 0 10px 0')
            .style('font-size', `${this.baseFontSize * 1.2}rem`);
        
        // Calculate heatmap dimensions to fit 21 rows
        const maxHeatmapHeight = maxLatticeSize;
        const rowHeight = Math.floor(maxHeatmapHeight / 21);
        const bitWidth = Math.floor((frWidth - 100) / this.B);
        
        this.heatmapSvg = this.heatmapContainer
            .append('svg')
            .attr('width', this.B * bitWidth + 100)
            .attr('height', 21 * rowHeight + 40);
        
        this.heatmapGroup = this.heatmapSvg.append('g');
        
        // Store dimensions for later use
        this.heatmapDimensions = {
            rowHeight: rowHeight,
            bitWidth: bitWidth,
            margin: { top: 20, right: 20, bottom: 20, left: 60 }
        };
    }
    
    selectLanguage(langStr) {
        if (this.selectedLanguage === langStr) {
            // Deselect if clicking the same language
            this.selectedLanguage = null;
        } else {
            // Select new language
            this.selectedLanguage = langStr;
        }
        this.renderSelection();
    }
    
    renderHeatmap() {
        const { languages, counts } = this.getUniqueLanguages();
        const { rowHeight, bitWidth, margin } = this.heatmapDimensions;
        
        // Sort languages by count (descending) and take top 20
        const sortedByCount = languages.sort((a, b) => counts.get(b) - counts.get(a));
        const top20Languages = sortedByCount.slice(0, 20);
        
        // If a language is selected and not in top 20, add it
        let displayLanguages = [...top20Languages];
        if (this.selectedLanguage && !top20Languages.includes(this.selectedLanguage)) {
            displayLanguages.push(this.selectedLanguage);
        }
        
        // Bind data to language rows
        const languageRows = this.heatmapGroup
            .selectAll('.language-row')
            .data(displayLanguages.map(lang => ({
                langStr: lang,
                bits: lang.split('').map(b => parseInt(b)),
                count: counts.get(lang) || 0,
                color: this.colorMap.get(lang),
                isSelected: lang === this.selectedLanguage
            })), d => d.langStr);
        
        // Enter selection
        const enterRows = languageRows.enter()
            .append('g')
            .attr('class', 'language-row')
            .style('cursor', 'pointer')
            .on('click', (event, d) => {
                this.selectLanguage(d.langStr);
            });
        
        // Add background rectangle for highlighting
        enterRows.append('rect')
            .attr('class', 'row-background')
            .attr('x', margin.left - 5)
            .attr('width', this.B * bitWidth + 10)
            .attr('height', rowHeight)
            .attr('fill', 'none')
            .attr('stroke', 'none');
        
        // Add count labels
        enterRows.append('text')
            .attr('class', 'count-label')
            .attr('x', margin.left - 10)
            .attr('text-anchor', 'end')
            .attr('font-size', `${this.baseFontSize * 0.85}rem`)
            .attr('fill', '#666')
            .attr('dy', '0.35em');
        
        // Add bit cells for new rows
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
        
        // Update selection
        const updateRows = languageRows.merge(enterRows);
        
        updateRows
            .attr('transform', (d, i) => `translate(0, ${margin.top + i * rowHeight})`);
        
        updateRows.select('.count-label')
            .text(d => d.count)
            .attr('y', rowHeight / 2)
            .style('font-weight', d => d.isSelected ? 'bold' : 'normal');
        
        // Update bit cells - this is the key fix
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
                .attr('stroke', bitData => bitData.isSelected ? '#222' : '#333')
                .attr('stroke-width', bitData => bitData.isSelected ? 4 : 0.5);
        });
        
        // Exit selection
        languageRows.exit().remove();
    }
    
    renderSelection() {
        // Update lattice cell strokes
        this.cells
            .attr('stroke', d => {
                const agent = this.lattice[d.x][d.y];
                const langStr = agent.language.join('');
                return langStr === this.selectedLanguage ? '#fff' : '#333';
            })
            .attr('stroke-width', d => {
                const agent = this.lattice[d.x][d.y];
                const langStr = agent.language.join('');
                return langStr === this.selectedLanguage ? 3 : 0.5;
            });
        
        // Update heatmap (will be handled in renderHeatmap)
        this.renderHeatmap();
    }
    
    render() {
        // Update step display
        this.stepDisplay.text(`Step: ${this.step}`);
        
        // Update cell colors using D3's data join
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
        
        // Update heatmap
        this.renderHeatmap();
        
        // Update selection highlighting
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
        
        // Use setTimeout for controlled speed instead of requestAnimationFrame
        setTimeout(() => {
            this.animationId = requestAnimationFrame(() => this.runLoop());
        }, 100); // 100ms delay between steps
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

// Initialize the simulation when the page loads
document.addEventListener('DOMContentLoaded', () => {
    new LatticeSimulation();
});
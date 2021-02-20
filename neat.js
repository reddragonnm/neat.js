function sigmoid(x) {
    return 1 / (1 + Math.exp(-x));
}
function linear(x) {
    return x;
}
function sum(arr) {
    return arr.reduce((a, b) => a + b, 0);
}
function randUniform() {
    return Math.random() * 2 - 1;
}
function randChoice(arr) {
    return arr[Math.floor(Math.random() * arr.length)];
}
class Innovation {
    constructor(innov, newConn, fr, to, nodeId) {
        this.innov = innov;
        this.newConn = newConn;
        this.fr = fr;
        this.to = to;
        this.nodeId = nodeId;
    }
}
class InnovTable {
    static setNodeId(nodeId) {
        this.nodeId = Math.max(this.nodeId, nodeId);
    }
    static createInnov(fr, to, newConn) {
        let nodeId = null;
        if (!newConn) {
            nodeId = this.nodeId++;
        }
        const innovation = new Innovation(this.innov++, newConn, fr, to, nodeId);
        this.history.push(innovation);
        return innovation;
    }
    static getInnov(fr, to, newConn = true) {
        for (let innovation of this.history) {
            if (innovation.newConn == newConn &&
                innovation.fr == fr &&
                innovation.to == to) {
                return innovation;
            }
        }
        return InnovTable.createInnov(fr, to, newConn);
    }
}
InnovTable.history = [];
InnovTable.innov = 0;
InnovTable.nodeId = 0;
class Options {
}
Options.fitnessThreshold = Infinity;
Options.maxNodes = Infinity;
Options.activation = sigmoid;
Options.aggregation = sum;
Options.excessCoeff = 1;
Options.disjointCoeff = 1;
Options.weightCoeff = 0.5;
Options.addNodeProb = 0.07;
Options.addConnProb = 0.2;
Options.weightMutateProb = 0.1;
Options.newWeightProb = 0.1;
Options.weightInitRange = 1;
Options.weightMutatePower = 0.5;
Options.featureSelection = false;
Options.compatThresh = 3;
Options.dynamicCompatThresh = true;
Options.targetSpecies = 20;
Options.dropoffAge = 15;
Options.survivalRate = 0.2;
Options.speciesElitism = true;
Options.crossoverRate = 1;
Options.triesTournSel = 3;
Options.youngAgeThresh = 10;
Options.youngAgeFitnessBonus = 1.3;
Options.oldAgeThresh = 50;
Options.oldAgeFitnessPenalty = 0.7;
var NodeState;
(function (NodeState) {
    NodeState["input"] = "input";
    NodeState["hidden"] = "hidden";
    NodeState["bias"] = "bias";
    NodeState["output"] = "output";
})(NodeState || (NodeState = {}));
class NodeGen {
    constructor(id, state, x, y) {
        this.id = id;
        this.state = state;
        this.x = x;
        this.y = y;
        this.val = 0;
    }
    copy() {
        return new NodeGen(this.id, this.state, this.x, this.y);
    }
}
class ConnectionGen {
    constructor(fr, to, innov, weight = null, enabled = true) {
        this.fr = fr;
        this.to = to;
        this.weight =
            weight === null ? randUniform() * Options.weightInitRange : weight;
        this.enabled = enabled;
        this.innov = innov;
    }
    copy() {
        return new ConnectionGen(this.fr, this.to, this.innov, this.weight, this.enabled);
    }
}
class Brain {
    constructor(id, nodes = null, connections = null) {
        this.id = id;
        this.fitness = 0;
        this.nodes = nodes;
        this.connections = connections;
        if (nodes !== null) {
            this.nodes.sort((a, b) => a.id - b.id);
            return;
        }
        const inputPosX = 1 / (Options.numInputs + 1);
        const outputPosX = 1 / Options.numOutputs;
        let nodeId = 0;
        const inputNodes = [];
        const outputNodes = [];
        const biasNodes = [
            new NodeGen(nodeId++, NodeState.bias, 0.5 * inputPosX, 0),
        ];
        for (let i = 0; i < Options.numInputs; i++) {
            inputNodes.push(new NodeGen(nodeId++, NodeState.input, (i + 1.5) * inputPosX, 0));
        }
        for (let i = 0; i < Options.numOutputs; i++) {
            outputNodes.push(new NodeGen(nodeId++, NodeState.output, (i + 0.5) * outputPosX, 1));
        }
        this.nodes = biasNodes.concat(inputNodes.concat(outputNodes));
        InnovTable.setNodeId(nodeId);
        this.connections = [];
        if (Options.featureSelection) {
            const inp = randChoice(biasNodes.concat(inputNodes));
            const out = randChoice(outputNodes);
            this.connections.push(new ConnectionGen(inp.id, out.id, InnovTable.getInnov(inp.id, out.id).innov));
        }
        else {
            for (let node1 of biasNodes.concat(inputNodes)) {
                for (let node2 of outputNodes) {
                    this.connections.push(new ConnectionGen(node1.id, node2.id, InnovTable.getInnov(node1.id, node2.id).innov));
                }
            }
        }
    }
    getDrawInfo() {
        const info = {
            nodes: {
                input: [],
                hidden: [],
                output: [],
                bias: [],
            },
            connections: {
                enabled: [],
                disabled: [],
            },
        };
        for (let node of this.nodes) {
            info.nodes[node.state].push({
                id: node.id,
                pos: [node.x, node.y],
            });
        }
        for (let conn of this.connections) {
            const fr = this.getNode(conn.fr);
            const to = this.getNode(conn.to);
            const s = conn.enabled ? "enabled" : "disabled";
            info.connections[s].push({
                fr: [fr.x, fr.y],
                to: [to.x, to.y],
                weight: conn.weight,
            });
        }
        return info;
    }
    addConn() {
        const valid = [];
        for (let node1 of this.nodes) {
            for (let node2 of this.nodes) {
                if (this.validConn(node1, node2)) {
                    valid.push([node1.id, node2.id]);
                }
            }
        }
        if (valid.length) {
            let [node1Id, node2Id] = randChoice(valid);
            this.connections.push(new ConnectionGen(node1Id, node2Id, InnovTable.getInnov(node1Id, node1Id).innov));
        }
    }
    addNode() {
        const valid = this.connections.filter((x) => x.enabled && x.fr != 0);
        if (!valid.length) {
            return;
        }
        const conn = randChoice(valid);
        const fr = this.getNode(conn.fr);
        const to = this.getNode(conn.to);
        const nodeId = InnovTable.getInnov(conn.fr, conn.to, false).nodeId;
        conn.enabled = false;
        this.nodes.push(new NodeGen(nodeId, NodeState.hidden, (fr.x + to.x) / 2, (fr.y + to.y) / 2));
        this.connections = this.connections.concat([
            new ConnectionGen(conn.fr, nodeId, InnovTable.getInnov(conn.fr, nodeId).innov, 1),
            new ConnectionGen(nodeId, conn.to, InnovTable.getInnov(nodeId, conn.to).innov, conn.weight),
        ]);
    }
    mutate() {
        if (Math.random() < Options.addNodeProb &&
            this.nodes.length < Options.maxNodes) {
            this.addNode();
        }
        if (Math.random() < Options.addConnProb) {
            this.addConn();
        }
        for (let conn of this.connections) {
            if (Math.random() < Options.weightMutatePower) {
                if (Math.random() < Options.newWeightProb) {
                    conn.weight = randUniform() * Options.weightInitRange;
                }
                else {
                    conn.weight += randUniform() * Options.weightMutatePower;
                }
            }
        }
    }
    getInputConnections(nodeId) {
        return this.connections.filter((x) => x.to == nodeId);
    }
    getNode(nodeId) {
        return this.nodes.filter((x) => x.id == nodeId)[0];
    }
    validConn(node1, node2) {
        return (!this.connections.filter((x) => x.fr == node1.id && x.to == node2.id)
            .length &&
            node1.id != node2.id &&
            [NodeState.input, NodeState.hidden, NodeState.bias].includes(node1.state) &&
            [NodeState.hidden, NodeState.output].includes(node2.state) &&
            node1.y <= node2.y);
    }
    predict(inputs) {
        if (inputs.length != Options.numInputs) {
            console.error("Number of inputs do not match with the number declared in Options");
        }
        const depth = new Set(this.nodes.map((x) => x.y)).size;
        for (let node of this.nodes) {
            node.val = 0;
        }
        for (let i = 0; i < depth; i++) {
            let inpNum = 0;
            for (let node of this.nodes) {
                if (node.state == NodeState.input) {
                    node.val = inputs[inpNum++];
                }
                else if (node.state == NodeState.bias) {
                    node.val = 1;
                }
                else {
                    const values = [];
                    for (let conn of this.getInputConnections(node.id)) {
                        if (conn.enabled) {
                            values.push(conn.weight * this.getNode(conn.fr).val);
                        }
                    }
                    node.val = Options.activation(Options.aggregation(values));
                }
            }
        }
        return this.nodes
            .filter((x) => x.state == NodeState.output)
            .map((x) => x.val);
    }
    static crossover(mum, dad, babyId) {
        const nMum = mum.connections.length;
        const nDad = dad.connections.length;
        let better;
        if (mum.fitness == dad.fitness) {
            if (nMum == nDad) {
                better = randChoice([mum, dad]);
            }
            else if (nMum < nDad) {
                better = mum;
            }
            else {
                better = dad;
            }
        }
        else if (mum.fitness > dad.fitness) {
            better = mum;
        }
        else {
            better = dad;
        }
        const babyNodes = [];
        const babyConnections = [];
        const nodeIds = [];
        let iMum = 0;
        let iDad = 0;
        while (iMum < nMum || iDad < nDad) {
            const mumGene = iMum < nMum ? mum.connections[iMum] : null;
            const dadGene = iDad < nDad ? dad.connections[iDad] : null;
            let selectedGene;
            let selectedBrain;
            if (mumGene !== null && dadGene !== null) {
                if (mumGene.innov == dadGene.innov) {
                    [selectedGene, selectedBrain] = randChoice([
                        [mumGene, mum],
                        [dadGene, dad],
                    ]);
                    iMum++;
                    iDad++;
                }
                else if (dadGene.innov < mumGene.innov) {
                    if (better === dad) {
                        selectedGene = dad.connections[iDad];
                        selectedBrain = dad;
                    }
                    iDad++;
                }
                else if (mumGene.innov < dadGene.innov) {
                    if (better === mum) {
                        selectedGene = mumGene;
                        selectedBrain = mum;
                    }
                    iMum++;
                }
            }
            else if (mumGene === null && dadGene !== null) {
                if (better === dad) {
                    selectedGene = dad.connections[iDad];
                    selectedBrain = dad;
                }
                iDad++;
            }
            else if (mumGene !== null && dadGene == null) {
                if (better === mum) {
                    selectedGene = mumGene;
                    selectedBrain = mum;
                }
                iMum++;
            }
            if (selectedGene && selectedBrain) {
                babyConnections.push(selectedGene.copy());
                let node;
                if (!nodeIds.includes(selectedGene.fr)) {
                    node = selectedBrain.getNode(selectedGene.fr);
                    if (node) {
                        babyNodes.push(node.copy());
                        nodeIds.push(selectedGene.fr);
                    }
                }
                if (!nodeIds.includes(selectedGene.to)) {
                    node = selectedBrain.getNode(selectedGene.to);
                    if (node) {
                        babyNodes.push(node.copy());
                        nodeIds.push(selectedGene.to);
                    }
                }
            }
        }
        return new Brain(babyId, babyNodes, babyConnections);
    }
}
class Species {
    constructor(id, member) {
        this.id = id;
        this.best = member;
        this.pool = [member];
        this.age = 0;
        this.stagnation = 0;
        this.spawnsReq = 0;
        this.maxFitness = 0;
        this.avgFitness = 0;
    }
    purge() {
        this.age++;
        this.stagnation++;
        this.pool = [];
    }
    getBrain() {
        let best = this.pool[0];
        for (let i = 0; i < Math.min(this.pool.length, Options.triesTournSel); i++) {
            const b = randChoice(this.pool);
            if (b.fitness > best.fitness) {
                best = b;
            }
        }
        return best;
    }
    cull() {
        this.pool = this.pool.slice(0, Math.max(1, Math.round(this.pool.length * Options.survivalRate)));
    }
    adjustFitnesses() {
        this.avgFitness = 0;
        for (let b of this.pool) {
            let fitness = b.fitness;
            if (this.age < Options.youngAgeThresh) {
                fitness *= Options.youngAgeFitnessBonus;
            }
            if (this.age > Options.oldAgeThresh) {
                fitness *= Options.oldAgeFitnessPenalty;
            }
            this.avgFitness += fitness / this.pool.length;
        }
    }
    makeLeader() {
        this.pool.sort((a, b) => b.fitness - a.fitness);
        this.best = this.pool[0];
        if (this.best.fitness > this.maxFitness) {
            this.stagnation = 0;
            this.maxFitness = this.best.fitness;
        }
    }
    static compatDist(genome1, genome2) {
        let nMatch = 0;
        let nDisjoint = 0;
        let nExcess = 0;
        let weightDiff = 0;
        const nG1 = genome1.connections.length;
        const nG2 = genome2.connections.length;
        let iG1 = 0;
        let iG2 = 0;
        while (iG1 < nG1 || iG2 < nG2) {
            if (iG1 == nG1) {
                nExcess++;
                iG2++;
                continue;
            }
            if (iG2 == nG2) {
                nExcess++;
                iG1++;
                continue;
            }
            const conn1 = genome1.connections[iG1];
            const conn2 = genome2.connections[iG2];
            if (conn1.innov == conn2.innov) {
                nMatch++;
                iG1++;
                iG2++;
                weightDiff += Math.abs(conn1.weight - conn2.weight);
                continue;
            }
            if (conn1.innov < conn2.innov) {
                nDisjoint++;
                iG1++;
                continue;
            }
            if (conn1.innov > conn2.innov) {
                nDisjoint++;
                iG2++;
                continue;
            }
        }
        nMatch++;
        return ((Options.excessCoeff * nExcess + Options.disjointCoeff * nDisjoint) /
            Math.max(nG1, nG2) +
            (Options.weightCoeff * weightDiff) / nMatch);
    }
    sameSpecies(brain) {
        return Species.compatDist(brain, this.best) <= Options.compatThresh;
    }
}
class Population {
    constructor() {
        this.pool = Array(Options.popSize)
            .fill(0)
            .map((_, index) => new Brain(index));
        this.best = this.pool[0];
        this.species = [];
        this.gen = 0;
        this.brainId = this.pool.length;
        this.speciesId = 0;
    }
    data() {
        return `${this.gen}, ${this.best.fitness}`;
    }
    evaluate(evalFunc, numGen = Infinity, report = true) {
        while (true) {
            evalFunc(this.pool);
            this.epoch();
            if (report) {
                console.log(this.data());
            }
            if (this.best.fitness >= Options.fitnessThreshold) {
                return [this.best, true];
            }
            else if (this.gen >= numGen) {
                return [this.best, false];
            }
        }
    }
    speciate() {
        for (let brain of this.pool) {
            let added = false;
            for (let sp of this.species) {
                if (sp.sameSpecies(brain)) {
                    sp.pool.push(brain);
                    added = true;
                    break;
                }
            }
            if (!added) {
                this.species.push(new Species(this.speciesId++, brain));
            }
        }
        this.species = this.species.filter((x) => x.pool.length > 0);
    }
    calcSpawns() {
        const total = Math.max(1, this.species.reduce((a, b) => a + b.avgFitness, 0));
        for (let sp of this.species) {
            sp.spawnsReq = (Options.popSize * sp.avgFitness) / total;
        }
    }
    reproduce() {
        this.pool = [];
        for (let s of this.species) {
            const newPool = [];
            if (Options.speciesElitism) {
                newPool.push(s.best);
            }
            while (newPool.length < s.spawnsReq) {
                const brain1 = s.getBrain();
                let child;
                if (Math.random() < Options.crossoverRate) {
                    const brain2 = s.getBrain();
                    child = Brain.crossover(brain1, brain2, this.brainId++);
                }
                else {
                    child = Brain.crossover(brain1, brain1, this.brainId++);
                }
                child.mutate();
                newPool.push(child);
            }
            this.pool = this.pool.concat(newPool);
            s.purge();
        }
        while (this.pool.length < Options.popSize) {
            this.pool.push(new Brain(this.brainId++));
        }
    }
    sortPool() {
        this.pool.sort((a, b) => b.fitness - a.fitness);
        if (this.pool[this.pool.length - 1].fitness < 0) {
            console.error("Cannot handle negative values");
        }
        if (this.best.fitness < this.pool[0].fitness) {
            this.best = this.pool[0];
        }
    }
    adjustFitnesses() {
        for (let s of this.species) {
            s.makeLeader();
            s.adjustFitnesses();
        }
    }
    changeCompatThresh() {
        if (this.species.length < Options.targetSpecies) {
            Options.compatThresh *= 0.95;
        }
        else if (this.species.length > Options.targetSpecies) {
            Options.compatThresh *= 1.05;
        }
    }
    resetAndKill() {
        const newSpecies = [];
        for (let sp of this.species) {
            if (sp.stagnation > Options.dropoffAge || sp.spawnsReq == 0) {
                continue;
            }
            sp.cull();
            newSpecies.push(sp);
        }
        this.species = newSpecies;
    }
    epoch() {
        this.sortPool();
        this.speciate();
        if (Options.dynamicCompatThresh) {
            this.changeCompatThresh();
        }
        this.adjustFitnesses();
        this.calcSpawns();
        this.resetAndKill();
        this.reproduce();
        this.gen++;
    }
}

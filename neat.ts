function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}

function linear(x: number): number {
  return x;
}

function sum(arr: number[]): number {
  return arr.reduce((a, b) => a + b, 0);
}

function randUniform(): number {
  return Math.random() * 2 - 1;
}

function randChoice<T>(arr: T[]): T {
  return arr[Math.floor(Math.random() * arr.length)];
}

class Innovation {
  innov: number;
  newConn: boolean;
  fr: number;
  to: number;
  nodeId: number;

  constructor(
    innov: number,
    newConn: boolean,
    fr: number,
    to: number,
    nodeId: number
  ) {
    this.innov = innov;
    this.newConn = newConn;
    this.fr = fr;
    this.to = to;
    this.nodeId = nodeId;
  }
}

class InnovTable {
  static history: Innovation[] = [];
  static innov = 0;
  static nodeId = 0;

  static setNodeId(nodeId: number): void {
    this.nodeId = Math.max(this.nodeId, nodeId);
  }

  static createInnov(fr: number, to: number, newConn: boolean): Innovation {
    let nodeId = null;
    if (!newConn) {
      nodeId = this.nodeId++;
    }

    const innovation = new Innovation(this.innov++, newConn, fr, to, nodeId);
    this.history.push(innovation);

    return innovation;
  }

  static getInnov(fr: number, to: number, newConn: boolean = true): Innovation {
    for (let innovation of this.history) {
      if (
        innovation.newConn == newConn &&
        innovation.fr == fr &&
        innovation.to == to
      ) {
        return innovation;
      }
    }

    return InnovTable.createInnov(fr, to, newConn);
  }
}

class Options {
  static numInputs: number;
  static numOutputs: number;
  static popSize: number;

  static fitnessThreshold = Infinity;
  static maxNodes = Infinity;

  static activation = sigmoid;
  static aggregation = sum;

  static excessCoeff = 1;
  static disjointCoeff = 1;
  static weightCoeff = 0.5;

  static addNodeProb = 0.07;
  static addConnProb = 0.2;

  static weightMutateProb = 0.1;
  static newWeightProb = 0.1;
  static weightInitRange = 1;
  static weightMutatePower = 0.5;

  static featureSelection = false;

  static compatThresh = 3;
  static dynamicCompatThresh = true;

  static targetSpecies = 20;
  static dropoffAge = 15;
  static survivalRate = 0.2;
  static speciesElitism = true;

  static crossoverRate = 1;
  static triesTournSel = 3;

  static youngAgeThresh = 10;
  static youngAgeFitnessBonus = 1.3;
  static oldAgeThresh = 50;
  static oldAgeFitnessPenalty = 0.7;
}

enum NodeState {
  input,
  hidden,
  bias,
  output,
}

class NodeGen {
  id: number;
  state: NodeState;
  x: number;
  y: number;
  val: number;

  constructor(id: number, state: NodeState, x: number, y: number) {
    this.id = id;
    this.state = state;

    this.x = x;
    this.y = y;

    this.val = 0;
  }

  copy(): NodeGen {
    return new NodeGen(this.id, this.state, this.x, this.y);
  }
}

class ConnectionGen {
  fr: number;
  to: number;
  weight: number;
  enabled: boolean;
  innov: number;

  constructor(
    fr: number,
    to: number,
    innov: number,
    weight: number = null,
    enabled: boolean = true
  ) {
    this.fr = fr;
    this.to = to;

    this.weight =
      weight === null ? randUniform() * Options.weightInitRange : weight;

    this.enabled = enabled;
    this.innov = innov;
  }

  copy(): ConnectionGen {
    return new ConnectionGen(
      this.fr,
      this.to,
      this.innov,
      this.weight,
      this.enabled
    );
  }
}

class Brain {
  id: number;
  fitness: number;
  nodes: NodeGen[];
  connections: ConnectionGen[];

  constructor(
    id: number,
    nodes: NodeGen[] = null,
    connections: ConnectionGen[] = null
  ) {
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

    const inputNodes: NodeGen[] = [];
    const outputNodes: NodeGen[] = [];

    const biasNodes = [
      new NodeGen(nodeId++, NodeState.bias, 0.5 * inputPosX, 0),
    ];

    for (let i = 0; i < Options.numInputs; i++) {
      inputNodes.push(
        new NodeGen(nodeId++, NodeState.input, (i + 1.5) * inputPosX, 0)
      );
    }

    for (let i = 0; i < Options.numOutputs; i++) {
      outputNodes.push(
        new NodeGen(nodeId++, NodeState.output, (i + 0.5) * outputPosX, 1)
      );
    }

    this.nodes = biasNodes.concat(inputNodes.concat(outputNodes));
    InnovTable.setNodeId(nodeId);
    this.connections = [];

    if (Options.featureSelection) {
      const inp = randChoice(biasNodes.concat(inputNodes));
      const out = randChoice(outputNodes);

      this.connections.push(
        new ConnectionGen(
          inp.id,
          out.id,
          InnovTable.getInnov(inp.id, out.id).innov
        )
      );
    } else {
      for (let node1 of biasNodes.concat(inputNodes)) {
        for (let node2 of outputNodes) {
          this.connections.push(
            new ConnectionGen(
              node1.id,
              node2.id,
              InnovTable.getInnov(node1.id, node2.id).innov
            )
          );
        }
      }
    }
  }

  addConn(): void {
    const valid: [number, number][] = [];

    for (let node1 of this.nodes) {
      for (let node2 of this.nodes) {
        if (this.validConn(node1, node2)) {
          valid.push([node1.id, node2.id]);
        }
      }
    }

    if (valid.length) {
      let [node1Id, node2Id] = randChoice(valid);

      this.connections.push(
        new ConnectionGen(
          node1Id,
          node2Id,
          InnovTable.getInnov(node1Id, node1Id).innov
        )
      );
    }
  }

  addNode(): void {
    const valid = this.connections.filter((x) => x.enabled && x.fr != 0);

    if (!valid.length) {
      return;
    }

    const conn = randChoice(valid);

    const fr = this.getNode(conn.fr);
    const to = this.getNode(conn.to);

    const nodeId = InnovTable.getInnov(conn.fr, conn.to, false).nodeId;
    conn.enabled = false;

    this.nodes.push(
      new NodeGen(
        nodeId,
        NodeState.hidden,
        (fr.x + to.x) / 2,
        (fr.y + to.y) / 2
      )
    );

    this.connections = this.connections.concat([
      new ConnectionGen(
        conn.fr,
        nodeId,
        InnovTable.getInnov(conn.fr, nodeId).innov,
        1
      ),

      new ConnectionGen(
        nodeId,
        conn.to,
        InnovTable.getInnov(nodeId, conn.to).innov,
        conn.weight
      ),
    ]);
  }

  mutate(): void {
    if (
      Math.random() < Options.addNodeProb &&
      this.nodes.length < Options.maxNodes
    ) {
      this.addNode();
    }

    if (Math.random() < Options.addConnProb) {
      this.addConn();
    }

    for (let conn of this.connections) {
      if (Math.random() < Options.weightMutatePower) {
        if (Math.random() < Options.newWeightProb) {
          conn.weight = randUniform() * Options.weightInitRange;
        } else {
          conn.weight += randUniform() * Options.weightMutatePower;
        }
      }
    }
  }

  getInputConnections(nodeId: number): ConnectionGen[] {
    return this.connections.filter((x) => x.to == nodeId);
  }

  getNode(nodeId: number): NodeGen {
    return this.nodes.filter((x) => x.id == nodeId)[0];
  }

  validConn(node1: NodeGen, node2: NodeGen): boolean {
    return (
      !this.connections.filter((x) => x.fr == node1.id && x.to == node2.id)
        .length &&
      node1.id != node2.id &&
      [NodeState.input, NodeState.hidden, NodeState.bias].includes(
        node1.state
      ) &&
      [NodeState.hidden, NodeState.output].includes(node2.state) &&
      node1.y <= node2.y
    );
  }

  predict(inputs: number[]): number[] {
    if (inputs.length != Options.numInputs) {
      console.error(
        "Number of inputs do not match with the number declared in Options"
      );
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
        } else if (node.state == NodeState.bias) {
          node.val = 1;
        } else {
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

  static crossover(mum: Brain, dad: Brain, babyId: number): Brain {
    const nMum = mum.connections.length;
    const nDad = dad.connections.length;

    let better: Brain;

    if (mum.fitness == dad.fitness) {
      if (nMum == nDad) {
        better = randChoice([mum, dad]);
      } else if (nMum < nDad) {
        better = mum;
      } else {
        better = dad;
      }
    } else if (mum.fitness > dad.fitness) {
      better = mum;
    } else {
      better = dad;
    }

    const babyNodes: NodeGen[] = [];
    const babyConnections: ConnectionGen[] = [];
    const nodeIds: number[] = [];

    let iMum = 0;
    let iDad = 0;

    while (iMum < nMum || iDad < nDad) {
      const mumGene = iMum < nMum ? mum.connections[iMum] : null;
      const dadGene = iDad < nDad ? dad.connections[iDad] : null;

      let selectedGene: ConnectionGen;
      let selectedBrain: Brain;

      if (mumGene !== null && dadGene !== null) {
        if (mumGene.innov == dadGene.innov) {
          [selectedGene, selectedBrain] = randChoice([
            [mumGene, mum],
            [dadGene, dad],
          ]);

          iMum++;
          iDad++;
        } else if (dadGene.innov < mumGene.innov) {
          if (better === dad) {
            selectedGene = dad.connections[iDad];
            selectedBrain = dad;
          }

          iDad++;
        } else if (mumGene.innov < dadGene.innov) {
          if (better === mum) {
            selectedGene = mumGene;
            selectedBrain = mum;
          }

          iMum++;
        }
      } else if (mumGene === null && dadGene !== null) {
        if (better === dad) {
          selectedGene = dad.connections[iDad];
          selectedBrain = dad;
        }
        iDad++;
      } else if (mumGene !== null && dadGene == null) {
        if (better === mum) {
          selectedGene = mumGene;
          selectedBrain = mum;
        }

        iMum++;
      }

      if (selectedGene && selectedBrain) {
        babyConnections.push(selectedGene.copy());
        let node: NodeGen;

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
  best: Brain;
  pool: Brain[];
  id: number;
  age: number;
  stagnation: number;
  spawnsReq: number;
  maxFitness: number;
  avgFitness: number;

  constructor(id: number, member: Brain) {
    this.id = id;
    this.best = member;
    this.pool = [member];

    this.age = 0;
    this.stagnation = 0;
    this.spawnsReq = 0;
    this.maxFitness = 0;
    this.avgFitness = 0;
  }

  purge(): void {
    this.age++;
    this.stagnation++;
    this.pool = [];
  }

  getBrain(): Brain {
    let best = this.pool[0];
    for (
      let i = 0;
      i < Math.min(this.pool.length, Options.triesTournSel);
      i++
    ) {
      const b = randChoice(this.pool);
      if (b.fitness > best.fitness) {
        best = b;
      }
    }

    return best;
  }

  cull(): void {
    this.pool = this.pool.slice(
      0,
      Math.max(1, Math.round(this.pool.length * Options.survivalRate))
    );
  }

  adjustFitnesses(): void {
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

  makeLeader(): void {
    this.pool.sort((a, b) => b.fitness - a.fitness);
    this.best = this.pool[0];

    if (this.best.fitness > this.maxFitness) {
      this.stagnation = 0;
      this.maxFitness = this.best.fitness;
    }
  }

  static compatDist(genome1: Brain, genome2: Brain): number {
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
    return (
      (Options.excessCoeff * nExcess + Options.disjointCoeff * nDisjoint) /
        Math.max(nG1, nG2) +
      (Options.weightCoeff * weightDiff) / nMatch
    );
  }

  sameSpecies(brain: Brain): boolean {
    return Species.compatDist(brain, this.best) <= Options.compatThresh;
  }
}

class Population {
  pool: Brain[];
  species: Species[];
  best: Brain;
  gen: number;
  brainId: number;
  speciesId: number;

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

  evaluate(
    evalFunc: (t: Brain[]) => void,
    numGen: number = Infinity,
    report: boolean = true
  ): [Brain, boolean] {
    while (true) {
      evalFunc(this.pool);
      this.epoch();

      if (report) {
        console.log(this.gen, this.best.fitness);
      }

      if (this.best.fitness >= Options.fitnessThreshold) {
        return [this.best, true];
      } else if (this.gen >= numGen) {
        return [this.best, false];
      }
    }
  }

  speciate(): void {
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

  calcSpawns(): void {
    const total = Math.max(
      1,
      this.species.reduce((a, b) => a + b.avgFitness, 0)
    );

    for (let sp of this.species) {
      sp.spawnsReq = (Options.popSize * sp.avgFitness) / total;
    }
  }

  reproduce(): void {
    this.pool = [];

    for (let s of this.species) {
      const newPool = [];

      if (Options.speciesElitism) {
        newPool.push(s.best);
      }

      while (newPool.length < s.spawnsReq) {
        const brain1 = s.getBrain();
        let child: Brain;

        if (Math.random() < Options.crossoverRate) {
          const brain2 = s.getBrain();
          child = Brain.crossover(brain1, brain2, this.brainId++);
        } else {
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

  sortPool(): void {
    this.pool.sort((a, b) => b.fitness - a.fitness);

    if (this.pool[this.pool.length - 1].fitness < 0) {
      console.error("Cannot handle negative values");
    }

    if (this.best.fitness < this.pool[0].fitness) {
      this.best = this.pool[0];
    }
  }

  adjustFitnesses(): void {
    for (let s of this.species) {
      s.makeLeader();
      s.adjustFitnesses();
    }
  }

  changeCompatThresh(): void {
    if (this.species.length < Options.targetSpecies) {
      Options.compatThresh *= 0.95;
    } else if (this.species.length > Options.targetSpecies) {
      Options.compatThresh *= 1.05;
    }
  }

  resetAndKill(): void {
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

  epoch(): void {
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

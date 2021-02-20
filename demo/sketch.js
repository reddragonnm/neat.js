Options.numInputs = 2;
Options.numOutputs = 1;
Options.popSize = 150;
Options.activation = sigmoid;
Options.fitnessThreshold = 3.9;
Options.weightMutateProb = 0.5;
Options.addNodeProb = 0.005;

const xorInp = [
  [0, 0],
  [0, 1],
  [1, 0],
  [1, 1],
];
const xorOut = [0, 1, 1, 0];

function evalFunc(nns) {
  for (let nn of nns) {
    nn.fitness = 4;

    for (let i = 0; i < 4; i++) {
      const output = nn.predict(xorInp[i])[0];
      nn.fitness -= (output - xorOut[i]) ** 2;
    }
  }
}

const p = new Population();

function setup() {
  createCanvas(600, 400);
}

function drawBrain(brain) {
  const info = brain.getDrawInfo();
  fill(255);
  strokeWeight(3);

  for (let conn of info.connections.enabled) {
    stroke(conn.weight > 0 ? "dodgerblue" : "coral");

    line(
      200 * conn.fr[1] + 200,
      200 * conn.fr[0] + 100,
      200 * conn.to[1] + 200,
      200 * conn.to[0] + 100
    );
  }

  let n = info.nodes;
  for (let node of n.input.concat(n.bias.concat(n.output.concat(n.hidden)))) {
    stroke(0);
    circle(200 * node[1] + 200, 200 * node[0] + 100, 30);
  }
}

function draw() {
  background("lightgreen");

  evalFunc(p.pool);
  p.epoch();

  drawBrain(p.best);

  console.log(p.data());

  fill(0);
  textSize(32);
  noStroke();
  text(`Generation: ${p.gen}`, 40, 350);
  text(`Max fitness: ${round(p.best.fitness, 3)}`, 300, 350);

  if (p.best.fitness > 3.9) {
    noLoop();
  }
}

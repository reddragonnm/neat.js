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

function drawBrain(brain, inputLabels, outputLabels) {
  const info = brain.getDrawInfo();

  strokeWeight(5);
  for (let conn of info.connections.enabled) {
    stroke(conn.weight > 0 ? "dodgerblue" : "coral");

    line(
      300 * conn.fr[1] + 150,
      300 * conn.fr[0] + 50,
      300 * conn.to[1] + 150,
      300 * conn.to[0] + 50
    );
  }

  strokeWeight(3);
  textSize(25);
  let n = info.nodes;
  for (let node of n.input.concat(n.bias.concat(n.output.concat(n.hidden)))) {
    const posX = 300 * node.pos[1] + 150;
    const posY = 300 * node.pos[0] + 50;

    stroke(0);
    fill(255);
    circle(posX, posY, 40);

    textAlign(CENTER, CENTER);
    noStroke();
    fill(0);

    if (node.id == 0) {
      text("Bias", posX - 60, posY);
    } else if (node.id <= Options.numInputs) {
      text(inputLabels[node.id - 1], posX - 60, posY);
    } else if (
      Options.numInputs < node.id &&
      node.id <= Options.numInputs + Options.numOutputs
    ) {
      text(outputLabels[node.id - Options.numInputs - 1], posX + 60, posY);
    }

    text(node.id.toString(), posX, posY);
  }
}

function draw() {
  background("lightgreen");

  evalFunc(p.pool);
  p.epoch();

  drawBrain(p.best, ["inp1", "inp2"], ["out"]);

  console.log(p.data());

  fill(0);
  textSize(32);
  noStroke();
  text(`Gen: ${p.gen}`, 150, 350);
  text(`Fitness: ${round(p.best.fitness, 3)}`, 350, 350);

  if (p.best.fitness > 3.9) {
    noLoop();
  }
}

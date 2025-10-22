import { DataLoader } from './data-loader.js';
import { buildModel, fitModel, predictOne, evaluateAccuracy } from './model.js';

const $ = (id) => document.getElementById(id);
const log = (msg) => {
  const t = new Date().toLocaleTimeString();
  const el = $('log');
  el.textContent = (el.textContent === '—' ? '' : el.textContent + '\n') + `[${t}] ${msg}`;
  el.scrollTop = el.scrollHeight;
};
const setStatus = (s) => $('status').textContent = `Status: ${s}`;
function setTfVer() { 
  $('tfver').textContent = `TF.js: ${tf?.version_core || 'unknown'}`; 
}

let LOADER = null, MODEL = null, READY = false;

async function onTrain() {
  try {
    disable(true);
    setStatus('loading data…'); log('Loading ./data/train.csv');
    if (!LOADER) LOADER = new DataLoader(log, setStatus);
    await LOADER.loadCSV('./data/train.csv');

    setStatus('preparing data…'); log('Encoding & scaling + split 80/20');
    const { featNames } = LOADER.prepareMatrices();
    log(`Features after encoding: ${featNames.length}`);

    // Get training settings from UI
    const numLayers = parseInt($('numLayers').value);
    const neuronsPerLayer = parseInt($('neuronsPerLayer').value);
    const epochs = parseInt($('epochs').value);
    const batchSize = parseInt($('batchSize').value);
    const learningRate = parseFloat($('learningRate').value);

    // Create model with settings
    setStatus('building model…'); log(`Building model: MLP`);
    MODEL?.dispose?.(); MODEL = buildModel('mlp', featNames.length, learningRate, numLayers, neuronsPerLayer);
    log(`Params: ${MODEL.countParams().toLocaleString()}`);

    setStatus('training…'); 
    log(`Start fit (epochs=${epochs}, batch=${batchSize}, lr=${learningRate})`);
    await fitModel(MODEL, LOADER.getTrain(), LOADER.getTrainY(), epochs, batchSize, log);

    setStatus('testing…'); log('Evaluate accuracy on full test (thr=0.5)');
    const acc = evaluateAccuracy(MODEL, LOADER.getTest(), LOADER.getTestY(), 0.5);
    $('testAcc').textContent = `${(acc * 100).toFixed(2)}%`;

    LOADER.buildSimulationForm($('simGrid'));
    $('simFs').disabled = false; $('simCard').style.opacity = '1';
    READY = true;
    setStatus('done'); log('Training finished. Simulation enabled.');
  } catch (e) {
    console.error(e); log('Error: ' + e.message); setStatus('error');
  } finally {
    disable(false);
  }
}

function disable(b) {
  $('btnTrain').disabled = b;
  $('numLayers').disabled = b;
  $('neuronsPerLayer').disabled = b;
  $('epochs').disabled = b;
  $('batchSize').disabled = b;
  $('learningRate').disabled = b;
}

function main() {
  setTfVer(); setStatus('ready');
  $('btnTrain').onclick = onTrain;
  disable(false);
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', main);
} else {
  main();
}

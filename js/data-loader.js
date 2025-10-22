// js/data-loader.js
export class DataLoader {
  constructor(logFn, statusFn) {
    this.log = logFn || console.log;
    this.setStatus = statusFn || (() => {});
    this.raw = null; // [{col:val,...}]
    this.schema = null; // { features: {name,type,values?,stats?}, target:'accident_risk' }
    this.encoders = {}; // {catKey: [values...]}
    this.scaler = { type: 'minmax', stats: {} }; // by feature index
    this.X = null;
    this.y = null;
    this.idx = { train: [], test: [] };
    this.featNames = [];
  }

  async loadCSV(path = './data/train.csv') {
    this.setStatus('loading data…');
    this.log(`Fetching ${path}`);
    const res = await fetch(path);
    if (!res.ok) throw new Error(`Failed to fetch ${path}: ${res.status}`);
    const text = await res.text();
    this.raw = this.#parseCSV(text);
    if (!this.raw.length) throw new Error('CSV is empty.');
    this.#inferSchema();
    this.setStatus('data loaded');
    this.log(`Loaded rows=${this.raw.length}`);
  }

  // ... (rest of data-loader.js as it was before)
}

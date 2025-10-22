export function buildModel(kind, inputLen, lr=0.001, numLayers=3, neuronsPerLayer=128){
  const m = tf.sequential();
  
  if (kind === 'mlp') {
    m.add(tf.layers.dense({units: neuronsPerLayer, activation: 'relu', inputShape: [inputLen]}));
    for (let i = 1; i < numLayers; i++) {
      m.add(tf.layers.dense({units: neuronsPerLayer, activation: 'relu'}));
    }
    m.add(tf.layers.dense({units: 1, activation: 'sigmoid'}));
  } else {
    throw new Error('Unknown model kind');
  }

  m.compile({ optimizer: tf.train.adam(lr), loss: 'meanSquaredError', metrics: ['mae'] });
  return m;
}

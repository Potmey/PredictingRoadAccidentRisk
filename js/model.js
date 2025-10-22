// js/model.js
export function buildModel(kind, inputLen, lr = 0.001, numLayers = 2, neuronsPerLayer = 128, dropout = 0.0) {
  const m = tf.sequential();
  
  // Логирование параметров
  console.log(`Building model with: ${kind} | Layers: ${numLayers} | Neurons: ${neuronsPerLayer} | Dropout: ${dropout}`);
  
  // Создаём модель на основе выбранных настроек
  if (kind === 'cnn1d') {
    m.add(tf.layers.reshape({ targetShape: [inputLen, 1], inputShape: [inputLen] }));
    for (let i = 0; i < numLayers; i++) {
      m.add(tf.layers.conv1d({
        filters: neuronsPerLayer, kernelSize: 3, activation: 'relu', padding: 'same'
      }));
      m.add(tf.layers.globalAveragePooling1d());
      if (dropout > 0) m.add(tf.layers.dropout({ rate: dropout }));
    }
    m.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));
  } else if (kind === 'mlp') {
    m.add(tf.layers.dense({ units: neuronsPerLayer, activation: 'relu', inputShape: [inputLen] }));
    for (let i = 1; i < numLayers; i++) {
      m.add(tf.layers.dense({ units: neuronsPerLayer, activation: 'relu' }));
      if (dropout > 0) m.add(tf.layers.dropout({ rate: dropout }));
    }
    m.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));
  } else {
    throw new Error('Unknown model kind');
  }

  m.compile({ optimizer: tf.train.adam(lr), loss: 'meanSquaredError', metrics: ['mae'] });
  return m;
}

export async function fitModel(model, Xtr, ytr, epochs = 10, batchSize = 256, logFn) {
  const xt = tf.tensor2d(Xtr), yt = tf.tensor2d(ytr);
  const h = await model.fit(xt, yt, {
    epochs, batchSize, validationSplit: 0.1,
    callbacks: {
      onEpochEnd: (ep, logs) => {
        const mae = (logs.mae ?? 0).toFixed(6);
        const vmae = (logs.val_mae ?? 0).toFixed(6);
        logFn?.(`ep ${ep + 1}/${epochs} loss=${logs.loss?.toFixed(6)} val_loss=${logs.val_loss?.toFixed(6)} mae=${mae} val_mae=${vmae}`);
      }
    }
  });
  xt.dispose();
  yt.dispose();
  return h;
}

// Evaluation function to return accuracy and metrics
export function evaluateAccuracy(model, X, y, thr = 0.5) {
  const xt = tf.tensor2d(X);
  const yp = model.predict(xt);
  const arr = yp.arraySync().map(a => a[0]);
  xt.dispose();
  yp.dispose?.();
  const yTrue = y.map(a => a[0]);
  let correct = 0;
  for (let i = 0; i < arr.length; i++) {
    const p = arr[i] >= thr ? 1 : 0;
    const t = yTrue[i] >= thr ? 1 : 0;
    if (p === t) correct++;
  }
  return correct / Math.max(1, arr.length);
}

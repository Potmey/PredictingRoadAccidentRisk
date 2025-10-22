// js/model.js  (adding parameters to the model, but no influence yet)
export function buildModel(kind, inputLen, lr=0.001, layers=2, neurons=128, activation='relu', optimizer='adam'){
  const m = tf.sequential();
  
  // Add the initial layer
  m.add(tf.layers.dense({units: neurons, activation, inputShape: [inputLen]}));
  
  // Add the requested number of layers
  for (let i = 1; i < layers; i++) {
    m.add(tf.layers.dense({units: neurons, activation}));
  }

  // Final output layer
  m.add(tf.layers.dense({units: 1, activation:'sigmoid'}));

  // Compile the model with the selected optimizer
  m.compile({
    optimizer: optimizer,
    loss: 'meanSquaredError',
    metrics: ['mae']
  });
  
  return m;
}

export async function fitModel(model, Xtr, ytr, epochs=10, batchSize=256, logFn){
  const xt=tf.tensor2d(Xtr), yt=tf.tensor2d(ytr);
  const h = await model.fit(xt, yt, {
    epochs, batchSize, validationSplit: 0.1,
    callbacks: {
      onEpochEnd: (ep, logs)=> {
        const loss = logs.loss?.toFixed(6);
        const vloss = (logs.val_loss??0).toFixed(6);
        const mae = (logs.mae??0).toFixed(6);
        const vmae = (logs.val_mae??0).toFixed(6);
        logFn?.(`ep ${ep+1}/${epochs} loss=${loss} val_loss=${vloss} mae=${mae} val_mae=${vmae}`);
      }
    }
  });
  xt.dispose(); yt.dispose();
  return h;
}

export function predictOne(model, xRow){
  const xt = tf.tensor2d([xRow]);
  const y = model.predict(xt);
  const v = y.arraySync()[0][0];
  xt.dispose(); y.dispose?.();
  return v;
}

export function evaluateAccuracy(model, X, y, thr=0.5){
  const xt = tf.tensor2d(X);
  const yp = model.predict(xt);
  const arr = yp.arraySync().map(a=>a[0]);
  xt.dispose(); yp.dispose?.();
  const yTrue = y.map(a=>a[0]);
  let correct=0;
  for (let i=0;i<arr.length;i++){
    const p = arr[i] >= thr ? 1 : 0;
    const t = yTrue[i] >= thr ? 1 : 0;
    if (p===t) correct++;
  }
  return correct/Math.max(1,arr.length);
}

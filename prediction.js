import * as tf from '@tensorflow/tfjs'

import Helpers from './helpers'

Date.prototype.addDays = days => {
    var date = new Date(this.valueOf());
    date.setDate(date.getDate() + days);
    return date;
}

const buildCnn = (data, timePortion) => {
    return new Promise((resolve, reject) => {
        if(timePortion <= 0 || !timePortion || !data)
            return reject(new Error("Bad parameters on buildCnn function"));
        
        const model = tf.sequential({
            layers: [
                tf.layers.inputLayer({
                    inputShape: [timePortion, 1],
                }),
                tf.layers.conv1d({
                    kernelSize: 2,
                    filters: 128,
                    strides: 1,
                    use_bias: true,
                    activation: 'relu',
                    kernelInitializer: 'VarianceScaling'                    
                }),
                tf.layers.averagePooling1d({
                    poolSize: [2],
                    strides: [1],
                }),
                tf.layers.conv1d({
                    kernelSize: 2,
                    filters: 64,
                    strides: 1,
                    use_bias: true,
                    activation: 'relu',
                    kernelInitializer: 'VarianceScaling'                    
                }),
                tf.layers.averagePooling1d({
                    poolSize: [2],
                    strides: [1],
                }),
                tf.layers.flatten(),
                tf.layers.dense({
                    units: 1,
                    kernelInitializer: 'VarianceScaling',
                    activation: 'linear'                    
                })
            ]
        });     
        
        return resolve({
            'model': model,
            'data': data
        })
    })
}

const cnn = (model, data, epochs) => {
    console.log("Sequential model's layers: ");
    model.summary();    

    return new Promise((resolve, reject) => {
        try {
            model.compile({ optimizer: 'adam', loss: 'meanSquaredError' });

            model.fit(data.tensorTrainX, data.tensorTrainY, {
                epochs: epochs
            }).then(result => {
                Helpers.print("Loss after last Epoch (" + result.epoch.length + ") is: " + result.history.loss[result.epoch.length-1]);
                resolve(model)                
            })
        } catch(err) {
            reject(ex);
        }
    })
}


const url = process.env.URL
const epochs = process.env.epochs
const timePortion = process.env.timePortion



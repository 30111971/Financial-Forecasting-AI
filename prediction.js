import * as tf from '@tensorflow/tfjs'
import unirest from 'unirest'

import Helpers from './helpers'

require('dotenv').config()

Date.prototype.addDays = days => {
    var date = new Date(this.valueOf());
    date.setDate(date.getDate() + days);
    return date;
}

var partialMessage;

const buildCnn = (data, timePortion) => {
    return new Promise((resolve, reject) => {
        if(parseInt(timePortion) <= 0 || !parseInt(timePortion) || !data)
            return reject(new Error("Bad parameters in buildCnn function"));
        
        const model = tf.sequential({
            layers: [
                tf.layers.inputLayer({
                    inputShape: [parseInt(timePortion), 1],
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

    return new Promise(async (resolve, reject) => {
        try {
            model.compile({ optimizer: process.env.optimizer, loss: process.env.loss });

            await model.fit(data.tensorTrainX, data.tensorTrainY, {
                epochs: epochs
            }).then(result => {
                Helpers.print(`Loss after last Epoch (${result.params.epochs.length}) is: ${result.history.loss[result.params.epochs.length-1]}`);
                resolve(model)                
            })
        } catch(err) {
            reject(err);
        }
    })
}


const url = process.env.URL
const epochs = process.env.epochs
const timePortion = process.env.timePortion
const enterprise = process.env.enterprise

const getHistoricalData = (frequency, period1, period2) => {
    return new Promise((resolve, reject) => {
        if(typeof(frequency) !== "string")
            reject(new Error("Frequency must be a string"));

            
            let req = unirest("GET", url);

            req.query({
                "frequency": frequency,
                "filter": "history",
                "period1": period1,
                "period2": period2,
                "symbol": enterprise
            });
            
            req.headers({
                "x-rapidapi-host": "apidojo-yahoo-finance-v1.p.rapidapi.com",
                "x-rapidapi-key": process.env.API_KEY
            });
            
            
            req.end(res => {
                if (res.error) throw new Error(res.error);
            
                resolve(res.body.prices)
            });        
    })    
}

const period1 = 1546448400
const period2 = 1562086800

const predict = () => {
    getHistoricalData("1d", period1, period2)
        .then(data => {                                
            Helpers.processData(data, parseInt(timePortion))
                .then(result => {                 
                    let nextDayPrediction = Helpers.generateNextDayPrediction(result.originalData, parseInt(result.timePortion));
                    
                    buildCnn(result, parseInt(timePortion))
                        .then(built => {
                            
                            let tensorData = {
                                tensorTrainX: tf.tensor1d(built.data.trainX).reshape([built.data.size, parseInt(built.data.timePortion), 1]),
                                tensorTrainY: tf.tensor1d(built.data.trainY)
                            };                            
                            
                            let max = built.data.max;
                            let min = built.data.min;
                            
                            cnn(built.model, tensorData, epochs).then(model => {
                                var predictedX = model.predict(tensorData.tensorTrainX);

                                let nextDayPredictionScaled = Helpers.minMaxScaler(nextDayPrediction, min, max);
                                
                                let tensorNextDayPrediction = tf.tensor1d(nextDayPredictionScaled.data).reshape([1, parseInt(built.data.timePortion), 1]);

                                let predictedValue = model.predict(tensorNextDayPrediction);
                                
                                predictedValue.data().then(predValue => {
                                    if(predValue.constructor === Float32Array)
                                        predValue = Array.prototype.slice.call(predValue);
                                                                        
                                    let inversePredictedValue = Helpers.minMaxInverseScaler(predValue, min, max);

                                    predictedX.data().then(pred => {
                                        if(pred.constructor !== Array)
                                            pred = Array.prototype.slice.call(pred);
                                        
                                        if(built.data.trainY.constructor !== Array)
                                            built.data.trainY = Array.prototype.slice.call(built.data.trainY);
                                        
                                        var predictedXInverse = Helpers.minMaxInverseScaler(pred, min, max);

                                        predictedXInverse.data = Array.prototype.slice.call(predictedXInverse.data);

                                        predictedXInverse.data[predictedXInverse.data.length] = inversePredictedValue.data[0];
                                        
                                        var trainYInverse = Helpers.minMaxInverseScaler(built.data.trainY, min, max);                                        
                                    })

                                    Helpers.print("Predicted Close Price of " + enterprise + " for next day is: " + inversePredictedValue.data[0].toFixed(3) + "$");
                                    
                                    let prediction = inversePredictedValue.data[0].toFixed(3);
                                    
                                })
                            })
                            .catch(err => {
                                console.error(err)
                            })
                        }).catch(err => {
                            console.error(err)
                        })
                })
                .catch(err => {
                    console.error(err)
                })
        })
        .catch(err => {
            console.error(err)
        })
}

predict();
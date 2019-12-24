export default class Helpers {
    static processData(data, timePortion) {
        return new Promise((resolve, reject) => {
            let trainX = [], trainY = [], size = data.length, features = [];        
    
            for(let i = 0; i < size; i++) {
                features.push(data[i]['close']);
            }
    
            var scaledData = minMaxScaler(features, getMin(features), getMax(features));
            let scaledFeatures = scaledData.data;
    
            try {
                for(let i = timePortion; i < size; i++) {
                    for(let j = (i - timePortion); j < i; j++) {
                        trainX.push(scaledFeatures[j])
                    }
    
                    trainY.push(scaledFeatures[i])
                }
            } catch(err) {
                reject(err);
            }
    
            return resolve({
                size: (size - timePortion),
                timePortion: timePortion,
                trainX: trainX,
                trainY: trainY,
                min: scaledData.min,
                max: scaledData.max,
                originalData: features
            })
        })
    }

    static generateNextDayPrediction(data, timePortion) {
        let size = data.length;
        let features = [];
    
        for (let i = (size - timePortion); i < size; i++) {
            features.push(data[i]);
        }
        
        return features;        
    }

    static minMaxScaler(data, min, max) {
        let scaledData = data.map(value => {
            return (value - min) / (max - min);
        })
    
        return {
            data: scaledData,
            min: min,
            max: max
        }
    }

    static minMaxInverseScaler(data, min, max) {
        let scaledData = data.map(value => {
            return value * (max - min) + min
        })
    
        return {
            data: scaledData,
            min: min,
            max: max
        }
    }

    static getMin(data) {
        return Math.min(...data);
    }

    static getMax(data) {
        return Math.max(...data);
    }

    static print(text) {
        console.log(text)
    }

    static clearPrint() {
        console.clear();
    }
}
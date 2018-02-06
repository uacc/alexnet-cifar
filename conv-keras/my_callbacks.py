import keras
import pdb
import numpy as np

class Histories(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.margins = []
        return
    
    def on_epoch_begin(self, epoch, logs={}):
        return

    # Compute margin on epoch end
    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.model.predict(self.validation_data[0])
        y_true = self.validation_data[1]

        y = np.argmax(y_true, axis = 1)
        i = np.argmax(y_pred, axis = 1)
        margin = 0 
        for k in range(0,len(y)): 
            if y[k] == i[k]:
                pdb.set_trace()
                y_pred[k][i[k]] = 0
                j = np.amax(y_pred[k])
                margin = margin + (y_pred[k][y[k]] - j)
            else:
                margin = margin + (y_pred[k][y[k]] - y_pred[k][i[k]])
        margin_ave = margin / len(y) 
        self.margins.append(margin_ave)
        return



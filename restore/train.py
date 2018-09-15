# -*- coding: utf-8 -*-
import os
from models import restore_cnn_model, restore_cnn_bn_model
from tifffile import imread, imsave
from datetime import datetime
import numpy as np
import keras.backend as K

from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

##
## Images
##

dataDirectory = r'C:\workspace\imageprocessing\plainColor_env\\'
resultDiretory = r'C:\workspace\imageprocessing\trainedresult\\'

def getRGBAImgName(index, directory=dataDirectory+'commondata'):
    prefix = str(index).zfill(4)
    return os.path.join(directory, prefix+".rgba.tiff")

def getAlbedoImgName(index, directory=dataDirectory+'commondata'):
    prefix = str(index).zfill(4)
    return os.path.join(directory, prefix+".albedo.tiff")

def getMatteIDImgName(index, directory=dataDirectory+'commondata'):
    prefix = str(index).zfill(4)
    return os.path.join(directory, prefix+".matteid.tiff")

def getFGDeepImgName(index, directory=dataDirectory+'fg'):
    prefix = str(index).zfill(4)
    return os.path.join(directory, r'data', prefix+".deep.tiff")

# TODO : if we generate new mask, we need to redefine this function
# now mask is as 0003.0.mask.tiff 0003.1.mask.tiff
def getFGMaskImgName(index, subindex, directory=dataDirectory+'fg'):
    prefix = str(index).zfill(4)
    return os.path.join(directory, r'data', prefix+"."+str(subindex)+".mask.tiff")

def getFGResultImgName(index, directory=dataDirectory+'fg'):
    prefix = str(index).zfill(4)
    return os.path.join(directory, r'result', prefix+".tiff")

def list_filenames(indexRange, subindexRange, images_dir=dataDirectory+'fg'):
    input_rgba_filepaths = []
    input_albedo_filepaths = []
    input_matteid_filepaths = []
    input_deep_filepaths = []
    input_mask_filepaths = []

    output_filepaths = []
    for index in range(indexRange[0], indexRange[1]+1):
        rgbafilename = getRGBAImgName(index)
        albedofilename = getAlbedoImgName(index)
        matteidfilename = getMatteIDImgName(index)
        deepfilename = getFGDeepImgName(index)
        resultfilename = getFGResultImgName(index)
        for subindex in range(subindexRange[0], subindexRange[1]+1):
            maskfilename = getFGMaskImgName(index, subindex)

            input_rgba_filepaths.append(rgbafilename)
            input_albedo_filepaths.append(albedofilename)
            input_matteid_filepaths.append(matteidfilename)
            input_deep_filepaths.append(deepfilename)
            input_mask_filepaths.append(maskfilename)

            output_filepaths.append(resultfilename)

    return input_rgba_filepaths, input_albedo_filepaths, input_matteid_filepaths, input_deep_filepaths, input_mask_filepaths, output_filepaths

def get_filenames():
    input_rgba_filepaths, input_albedo_filepaths, input_matteid_filepaths, input_deep_filepaths, input_mask_filepaths, output_filepaths = list_filenames([3,58], [0,0])
    return input_rgba_filepaths, input_albedo_filepaths, input_matteid_filepaths, input_deep_filepaths, input_mask_filepaths, output_filepaths

# TODO : change imread to tiffread
def load_image(filename):
    img = imread(filename)
    return np.asarray(img, dtype=K.floatx())

def get_images(filenames):
    return np.asarray([load_image(f) for f in filenames])

def get_image_shape():
    input_rgba_filepaths, input_albedo_filepaths, input_matteid_filepaths, input_deep_filepaths, input_mask_filepaths, output_filepaths = get_filenames()
    
    print (len(input_mask_filepaths))
    #print (input_mask_filepaths)
    
    trainSet_x = []
    trainSet_y = []
    
    for i in range(0, len(input_mask_filepaths)):
        print (input_mask_filepaths[i])
        rgbadata = load_image(input_rgba_filepaths[i])
        albedodata = load_image(input_albedo_filepaths[i])
        matteiddata = load_image(input_matteid_filepaths[i])
        deepdata = load_image(input_deep_filepaths[i])
        maskdata = load_image(input_mask_filepaths[i])
        resultdata = load_image(output_filepaths[i])
    
        print (rgbadata.shape)
    

        # TODO : concat all input to x
        x = np.concatenate((rgbadata,albedodata,matteiddata,deepdata,maskdata), axis=2)
        y = resultdata
        
        
        
        trainSet_x.append(x)
        trainSet_y.append(y)
    
    trainSet_x = np.asarray(trainSet_x)
    trainSet_y = np.asarray(trainSet_y)
    print ("==============================")
    print (trainSet_x.shape)
    print (trainSet_y.shape)
    print ("===============================")
    return trainSet_x[0].shape, trainSet_y[0].shape

def get_count():
    input_rgba_filepaths, input_albedo_filepaths, input_matteid_filepaths, input_deep_filepaths, input_mask_filepaths, output_filepaths = get_filenames()
    return len(input_rgba_filepaths)

##
## Iterator
##

def lr_hr_generator():
    return image_pair_generator()

def image_pair_generator():
    input_rgba_filepaths, input_albedo_filepaths, input_matteid_filepaths, input_deep_filepaths, input_mask_filepaths, output_filepaths = get_filenames()
    while 1:
        for rgba, albedo, matteid, deep, mask, result in zip(input_rgba_filepaths, input_albedo_filepaths, input_matteid_filepaths, input_deep_filepaths, input_mask_filepaths, output_filepaths):
            rgbadata = load_image(rgba)
            albedodata = load_image(albedo)
            matteiddata = load_image(matteid)
            deepdata = load_image(deep)
            maskdata = load_image(mask)
            resultdata = load_image(result)

            # TODO : concat all input to x
            x = np.concatenate((rgbadata,albedodata,matteiddata,deepdata,maskdata), axis=2)
            y = resultdata

            x = np.reshape(x, (1,) + x.shape)
            y = np.reshape(y, (1,) + y.shape)
            
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            print (x.shape)
            print (y.shape)
            yield(x, y)


def steps_for_batch_size( batch_size):
    total = get_count()
    return max(1, int(total/batch_size))

##
## Loss
##

def PSNRLoss(y_true, y_pred):
    """
    PSNR is Peek Signal to Noise Ratio, which is similar to mean squared error.

    It can be calculated as
    PSNR = 20 * log10(MAXp) - 10 * log10(MSE)

    For images, MAXp = 255, so 1st term is 20 * log(255) == 48.1308036087.
    """
    def log10(x):
        return K.log(x) / K.log(K.constant(10, dtype=K.floatx()))

    return 48.1308036087 + -10. * log10(K.mean(K.square(y_pred - y_true)))

class Pipeline():
    def __init__(self, network='espcnn'):
        self.network = network
        self.results_root_dir = resultDiretory
        self.results_dir = self._get_and_prepare_results_dir()

    def _model_name(self):
        # timestamp
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = "%s_%s" % (self.network, ts)
        return model_name

    def _get_and_prepare_results_dir(self):
        """
            results_dir
                /model.h5
                /weights.h5
                /logs/
                    ...
        """
        model_name = self._model_name()

        # make output dirs
        results_dir = "%s/%s/" % (self.results_root_dir, model_name)
        mkdir_p(results_dir)
        mkdir_p(results_dir + 'logs/')

        print("\n\n[TRAIN]    saving results to %s\n" % results_dir)
        return results_dir

    def get_callbacks(self):
        # callbacks -- tensorboard
        log_dir = self.results_dir + 'logs/'
        tensorboard = TensorBoard(log_dir=log_dir)

        # callbacks -- model weights
        weights_path = self.results_dir + 'weights.h5'
        model_checkpoint = ModelCheckpoint(monitor='loss', filepath=weights_path, save_best_only=True)

        # callbacks -- learning rate
        reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=5, min_lr=1e-5)
        return [tensorboard, model_checkpoint, reduce_lr]

    def run(self, scale=4, epochs=100, batch_size=32, save=True):
        # input shape
        input_shape, output_shape = get_image_shape()
        image_count = get_count()

        print("[TRAIN] LR %s ==> HR %s. (%s images)" % (input_shape, output_shape, image_count))

        # model
        if (self.network == 'srcnn'):
            model = create_srcnn_model(input_shape, scale=scale)
        elif (self.network == 'resnet_up'):
            model = create_resnet_up_model(input_shape, scale=scale)
        elif (self.network == 'espcnn_bn'):
            model = create_espcnn_bn_model(input_shape, scale=scale)
        else:
            model = create_espcnn_model(input_shape, scale=scale)

        model.compile(loss='mse', optimizer=Adam(lr=1e-3), metrics=[PSNRLoss])

        # callbacks
        callbacks = self.get_callbacks()

        # train
        gen = image_pair_generator()
        steps = steps_for_batch_size(batch_size)
        model.fit_generator(gen, steps, epochs=epochs, callbacks=callbacks)

        # save
        if (save):
            model_path = self.results_dir + "model.h5"
            model.save(model_path)

class RestorePipeline(Pipeline):

    def run(self, epochs=100, batch_size=32, save=True):
        input_shape, output_shape = get_image_shape()
        image_count = get_count()
        print("[TRAIN] orig %s ==> enhanced %s. (%s images)" % (input_shape, output_shape, image_count))

        # model
        if (self.network == "cnn_bn"):
            model = restore_cnn_bn_model(input_shape)
        else:
            model = restore_cnn_model(input_shape)

        model.compile(loss='mse', optimizer='adam', metrics=[PSNRLoss])

        # callbacks
        callbacks = self.get_callbacks()

        # train
        gen = image_pair_generator()
        steps = steps_for_batch_size(batch_size)
        model.fit_generator(gen, steps, epochs=epochs, callbacks=callbacks)

        # save
        if (save):
            model_path = self.results_dir + "model.h5"
            model.save(model_path)



if __name__ == '__main__':

    import argparse
    import timeit
    parser = argparse.ArgumentParser(description="Train RESTORE model.")
    #parser.add_argument("image_path", type=str, help="Path to input images, expects sub-directories /path/original/ and /path/enhanced/.")
    #parser.add_argument("--results", type=str, default="results/restore/", help="Results base dir, will create subdirectories e.g. /results/model_timestamp/")
    parser.add_argument("--network", type=str, default="cnn", help="Network architecture. Default=cnn")
    parser.add_argument("--epochs", type=int, default=100, help="Epochs. Default=100")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size. Default=32")

    args = parser.parse_args()

    image_path = ''
    results_path = ''
    network = args.network
    epochs = args.epochs
    batch_size = args.batch_size

    # training pipeline
    p = RestorePipeline(network=network)

    start_time = timeit.default_timer()
    p.run(epochs=epochs, batch_size=batch_size)
    duration = timeit.default_timer() - start_time
    print("[RESTORE Train] time taken: %s" % duration)
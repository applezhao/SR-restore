# SR-Restore

Super resolution and Image restoration.

## Install

Requires `tensorflow==1.1.0` and `keras==2.0.2`

    $ pip install -r ./requirements.txt



## Image Restore example


````

    # Train Restore model.
    # Expects images in /path/images/original/ and /path/images/enhanced/
    # Saves model weights, checkpoints, and tensorboard logs to path specified in results.
    $ python -m restore.train 
        --network cnn \
        --epochs 100 \
        --batch_size 32


    # Predict original images using trained model weights.
    # (Optional) use `--limit` option to limit num of images to perform on.
    $ python -m restore.predict data/images/enhanced/ \
        --output results/output/ \
        --weights results/cnn_20170701_190534/weights.h5 \
        --network cnn \
        --batch_size 32 \
        --limit 20

````

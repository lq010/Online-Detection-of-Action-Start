from models.model_c3d_TC_GAN import GAN
import argparse
import warnings


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the model ')
    parser.add_argument(
        '-id',
        dest='experiment_id',
        default='test',
        help='Experiment ID to track and not overwrite resulting models. (default: %(default)s)')
    parser.add_argument(
        '-w',
        dest='pretrained_weights',
        default='/media/lq/C13E-1ED0/dataset/UCF_Crimes/results/adam_temporal_f1/weights/weights.03.hdf5',
        help='The pretrained weights, the weights will be used to initialize the GAN model. (default: %(default)s)')
    args = parser.parse_args()
      
    gan = GAN(c3d_weights=args.pretrained_weights)
    if args.experiment_id == 'test':
        warnings.warn("You didn't set the experiment id, the default id is 0, previous result will be overwritten")
    gan.train(iterations=190, save_it=[150,180], id=args.experiment_id)

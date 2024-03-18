#Code taken from IsoNet and restructured
import tensorflow as tf
import logging
tf.get_logger().setLevel(logging.ERROR)
from tensorflow.keras.models import load_model
import mrcfile
from IsoCARE.preprocessing.img_processing import normalize
import numpy as np
import tensorflow.keras.backend as K
import os
from IsoCARE.util.toTile import reform3D
from tqdm import tqdm
import sys

def predict_even(settings):    
    # model = load_model('{}/model_iter{:0>2d}.h5'.format(settings.result_dir,settings.iter_count+1))
    strategy = tf.distribute.MirroredStrategy()
    if settings.ngpus >1:
        with strategy.scope():
            model = load_model('{}/model_iter{:0>2d}.h5'.format(settings.result_dir,settings.iter_count-1))
    else:
        model = load_model('{}/model_iter{:0>2d}.h5'.format(settings.result_dir,settings.iter_count-1))
    N = settings.predict_batch_size 
    num_batches = len(settings.mrc_list_even)
    if num_batches%N == 0:
        append_number = 0
    else:
        append_number = N - num_batches%N
    data = []
    ##### diese for loop für odd und even mrc_list laufen lassen
    for i,mrc in enumerate(list(settings.mrc_list_even) + list(settings.mrc_list_even[:append_number])):
        root_name = mrc.split('/')[-1].split('.')[0]
        with mrcfile.open(mrc) as mrcData:
            real_data = mrcData.data.astype(np.float32)*-1
        real_data=normalize(real_data, percentile = settings.normalize_percentile)

        cube_size = real_data.shape[0]
        pad_size1 = (settings.predict_cropsize - cube_size)//2
        pad_size2 = pad_size1+1 if (settings.predict_cropsize - cube_size)%2 !=0 else  pad_size1
        padi = (pad_size1,pad_size2)
        real_data = np.pad(real_data, (padi,padi,padi), 'symmetric')

        if (i+1)%N != 0:
            data.append(real_data)
        else:
            data.append(real_data)
            data = np.array(data)
            predicted=model.predict(data[:,:,:,:,np.newaxis], batch_size= settings.predict_batch_size,verbose=0) ######wichtig
            predicted = predicted.reshape(predicted.shape[0:-1])
            for j,outData in enumerate(predicted):
                count = i + j - N + 1
                if count < len(settings.mrc_list_even):
                    m_name = settings.mrc_list_even[count]

                    root_name = m_name.split('/')[-1].split('.')[0]
                    end_size = pad_size1+cube_size
                    outData1 = outData[pad_size1:end_size, pad_size1:end_size, pad_size1:end_size]
                    outData1 = normalize(outData1, percentile = settings.normalize_percentile)
                    ### here we predict the next batch of subtomograms for training. we put them in either odd or even directory, in order to train "both directions"
                    # if (settings.iter_count % 2 == 0):
                    with mrcfile.new('{}/{}_iter{:0>2d}.mrc'.format(settings.data_dir_even,root_name,settings.iter_count-1), overwrite=True) as output_mrc:
                        output_mrc.set_data(-outData1)
                    # else:
                    #     with mrcfile.new('{}/{}_even_iter{:0>2d}.mrc'.format(settings.data_dir_odd,root_name,settings.iter_count-1), overwrite=True) as output_mrc:
                    #         output_mrc.set_data(-outData1)
            data = []
    K.clear_session()
    
def predict_odd(settings):    
    # model = load_model('{}/model_iter{:0>2d}.h5'.format(settings.result_dir,settings.iter_count+1))
    strategy = tf.distribute.MirroredStrategy()
    if settings.ngpus >1:
        with strategy.scope():
            model = load_model('{}/model_iter{:0>2d}.h5'.format(settings.result_dir,settings.iter_count-1))
    else:
        model = load_model('{}/model_iter{:0>2d}.h5'.format(settings.result_dir,settings.iter_count-1))
    N = settings.predict_batch_size 
    num_batches = len(settings.mrc_list_odd)
    if num_batches%N == 0:
        append_number = 0
    else:
        append_number = N - num_batches%N
    data = []
    ##### diese for loop für odd und even mrc_list laufen lassen
    for i,mrc in enumerate(list(settings.mrc_list_odd) + list(settings.mrc_list_odd[:append_number])):
        root_name = mrc.split('/')[-1].split('.')[0]
        with mrcfile.open(mrc) as mrcData:
            real_data = mrcData.data.astype(np.float32)*-1
        real_data=normalize(real_data, percentile = settings.normalize_percentile)

        cube_size = real_data.shape[0]
        pad_size1 = (settings.predict_cropsize - cube_size)//2
        pad_size2 = pad_size1+1 if (settings.predict_cropsize - cube_size)%2 !=0 else  pad_size1
        padi = (pad_size1,pad_size2)
        real_data = np.pad(real_data, (padi,padi,padi), 'symmetric')

        if (i+1)%N != 0:
            data.append(real_data)
        else:
            data.append(real_data)
            data = np.array(data)
            predicted=model.predict(data[:,:,:,:,np.newaxis], batch_size= settings.predict_batch_size,verbose=0) ######wichtig
            predicted = predicted.reshape(predicted.shape[0:-1])
            for j,outData in enumerate(predicted):
                count = i + j - N + 1
                if count < len(settings.mrc_list_odd):
                    m_name = settings.mrc_list_odd[count]

                    root_name = m_name.split('/')[-1].split('.')[0]
                    end_size = pad_size1+cube_size
                    outData1 = outData[pad_size1:end_size, pad_size1:end_size, pad_size1:end_size]
                    outData1 = normalize(outData1, percentile = settings.normalize_percentile)
                    ### here we predict the next batch of subtomograms for training. we put them in either odd or even directory, in order to train "both directions"
                    # if (settings.iter_count % 2 == 0):
                    with mrcfile.new('{}/{}_iter{:0>2d}.mrc'.format(settings.data_dir_odd,root_name,settings.iter_count-1), overwrite=True) as output_mrc:
                        output_mrc.set_data(-outData1)
                    # else:
                    #     with mrcfile.new('{}/{}_odd_iter{:0>2d}.mrc'.format(settings.data_dir_even,root_name,settings.iter_count-1), overwrite=True) as output_mrc:
                    #         output_mrc.set_data(-outData1)
            data = []
    K.clear_session()
    
def predict_one(args,one_tomo,even_tomo,output_file=None):
    #predict one tomogram in mrc format INPUT: mrc_file string OUTPUT: output_file(str) or <root_name>_corrected.mrc
    import logging
    if args.ngpus >1:
        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            model = tf.keras.models.load_model(args.model)
    else:
        model = tf.keras.models.load_model(args.model)

    logging.info("Loaded model from disk")
    root_name = one_tomo.split('/')[-1].split('.')[0]

    if output_file is None:
        if os.path.isdir(args.output_file):
            output_file = args.output_file+'/'+root_name+'_corrected.mrc'
        else:
            output_file = root_name+'_corrected.mrc'

    logging.info('predicting:{}'.format(root_name))
    ########odd data
    with mrcfile.open(one_tomo,permissive=True) as mrcData:
        real_data = mrcData.data.astype(np.float32)*-1
        voxelsize = mrcData.voxel_size
    real_data = normalize(real_data,percentile=args.normalize_percentile)
    data = np.expand_dims(real_data,axis=-1)
    reform_ins = reform3D(data)
    data = reform_ins.pad_and_crop_new(args.cube_size,args.crop_size)
    #to_predict_data_shape:(n,cropsize,cropsize,cropsize,1)
    #imposing wedge to every cubes
    #data=wedge_imposing(data)
    logging.info('odd data loaded')
    ########even data
    with mrcfile.open(even_tomo,permissive=True) as mrcData:
        real_data_even = mrcData.data.astype(np.float32)*-1
        voxelsize = mrcData.voxel_size
    real_data_even = normalize(real_data_even,percentile=args.normalize_percentile)
    data_even=np.expand_dims(real_data_even,axis=-1)
    reform_ins_even = reform3D(data_even)
    data_even = reform_ins.pad_and_crop_new(args.cube_size,args.crop_size)
    logging.info('even data loaded')
    
    N = args.batch_size #* args.ngpus * 4 # 8*4*8
    num_patches = data.shape[0]
    if num_patches%N == 0:
        append_number = 0
    else:
        append_number = N - num_patches%N
    data_odd = np.append(data, data[0:append_number], axis = 0)
    data_even = np.append(data_even, data_even[0:append_number], axis = 0)
    num_big_batch = data_odd.shape[0]//N
    outData = np.zeros(data_odd.shape)
    logging.info("total batches: {}".format(num_big_batch))
    for i in tqdm(range(num_big_batch), file=sys.stdout):
        in_data_odd = data_odd[i*N:(i+1)*N]
        in_data_even = data_even[i*N:(i+1)*N]
        # Get predictions for each input
        pred_odd = model.predict([in_data_odd], verbose=0)
        pred_even = model.predict([in_data_even], verbose=0)
        # Average the predictions
        averaged_predictions = (pred_odd + pred_even) / 2
        outData[i*N:(i+1)*N] = averaged_predictions
    outData = outData[0:num_patches]

    outData=reform_ins.restore_from_cubes_new(outData.reshape(outData.shape[0:-1]), args.cube_size, args.crop_size)

    outData = normalize(outData,percentile=args.normalize_percentile)
    with mrcfile.new(output_file, overwrite=True) as output_mrc:
        output_mrc.set_data(-outData)
        output_mrc.voxel_size = voxelsize
    K.clear_session()
    logging.info('Done predicting')
    # predict(args.model,args.weight,args.mrc_file,args.output_file, cubesize=args.cubesize, cropsize=args.cropsize, batch_size=args.batch_size, gpuID=args.gpuID, if_percentile=if_percentile)

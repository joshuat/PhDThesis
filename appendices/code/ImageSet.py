###### Imports ######
from glob import glob
from os.path import isdir
from re import split
from multiprocessing import cpu_count, Pool
from functools import partial
from numpy import array, unravel_index, pad, zeros, round, floor, ceil
from PIL.Image import open
from scipy.signal import fftconvolve
from h5py import File

###### Cosntants ######
DEFAULT_HDF_NAME = 'Images.h5'
REGISTERED_NAME_SUFFIX = ' (registered)'
PGM_EXTENTION = '.pgm'


###### Functions ######
# Go through directories process images sets.
# Assumes that when images are found there are no sub directories in that dir.
def processDirs(hdf_name=DEFAULT_HDF_NAME, directory='./', register=True, \
                multi_process=True, reg_iterations=1, chunk_size=0, \
                max_deviation=(None, None), image_extention=PGM_EXTENTION):
    hdf = File(hdf_name)
    
    processImageSetSet(directory, hdf, register=register, \
                       multi_process=multi_process, \
                       reg_iterations=reg_iterations, \
                       chunk_size=chunk_size, \
                       max_deviation=max_deviation, \
                       image_extention=image_extention)
                       
    return hdf
                       
def processImageSetSet(directory, hdf, register=True, multi_process=True, \
                       reg_iterations=1, chunk_size=0, max_deviation=(None, None), \
                       image_extention=PGM_EXTENTION):
    stuff = glob(directory + '/*')

    for thing in stuff:
        if isdir(thing):
            group = hdf.require_group(dirToName(thing))
            
            processImageSetSet(thing, group, register=register, \
                                multi_process=multi_process, \
                                reg_iterations=reg_iterations, \
                                chunk_size=chunk_size, \
                                max_deviation=max_deviation, \
                                image_extention=image_extention)
        elif PGM_EXTENTION in thing:
                # Thar be images here.
                files = glob(directory + '/*' + image_extention)#[0:100]
                name = dirToName(directory)
                
                processImageSet(files, hdf, name, register=register, \
                                multi_process=multi_process, \
                                reg_iterations=reg_iterations, \
                                chunk_size=chunk_size, \
                                max_deviation=max_deviation)
                                
                break
        else:
            # Not something we need to process.
            pass
                
# Process Image Sets into HDFs
def processImageSet(files, hdf, name, register=True, multi_process=True, \
                    reg_iterations=1, chunk_size=0, max_deviation=(None, None)):
    fetched = fetchImageSet(files, register=register, multi_process=multi_process, \
                            reg_iterations=reg_iterations, chunk_size=chunk_size, \
                            max_deviation=max_deviation)
                            
    if register:
        average_image, registered_image = fetched
    else:
        average_image = fetched
    
    if name in hdf:
        del hdf[name]
    hdf.create_dataset(name, data=average_image)
    
    if register:
        reg_name = name + REGISTERED_NAME_SUFFIX
        if reg_name in hdf:
            del hdf[reg_name]
            
        hdf.create_dataset(reg_name, data=registered_image)
    

# 'Fetching' Image Sets
def fetchImageSet(files, register=True, multi_process=True, \
                  reg_iterations=1, chunk_size=0, max_deviation=(None, None)):
    images = loadImageSet(files, multi_process=multi_process)
    
    average_image = images.mean(axis=0)
    
    if register:
        registered_image = registerImageSet(images, multi_process=multi_process, \
                                            iterations=reg_iterations, chunk_size=chunk_size, \
                                             max_deviation=max_deviation)
        
        return average_image, registered_image
    else:
        return average_image
    
# Loading Images.
def loadImageSet(files, multi_process=True):
    # Prepare the function for the workers.
    load = partial(loadImageI, files=files)
    
    if multi_process:
        with Pool(processes=cpu_count()) as workers:
            ims = array(workers.map(load, range(len(files))))
    else:
        image = load(0)
		
        ims = zeros((len(files), image.shape[0], image.shape[1]))
		
        ims[0, :, :] = image
		
        for i in range(1, len(files)):
            ims[i, :, :] = load(i)
		
    return ims
    
def loadImageI(i, files):
    print('\tLoading image', i+1, 'of', len(files))

    return loadImage(files[i])

def loadImage(file):
    return removeTimestamp(array(open(file), dtype='float64'))
    
def removeTimestamp(image, mode='nearest'):
    # Images from PointGrey FlyCap2 can have timestamps which are the first 4 pixels.
    if mode=='nearest':
        image[0][0] = image[1][0]
        image[0][1] = image[1][1]
        image[0][2] = image[1][2]
        image[0][3] = image[1][3]
    elif mode=='mean':
        value = image.mean()
        
        image[0][0] = value
        image[0][1] = value
        image[0][2] = value
        image[0][3] = value
    elif mode=='zero':
        image[0][0] = 0
        image[0][1] = 0
        image[0][2] = 0
        image[0][3] = 0
    else:
        raise ValueError('Invalid mode supplied to ImageSet.removeTimestamp (' + str(mode) + ').')
        
    return image
    
# Registering Images
def registerImageSet(images, multi_process=True, reference_image=None, \
                     roi_centre=None, roi_width=400, iterations=1, chunk_size=0, \
                     max_deviation=(None, None)):
    if chunk_size==0 or images.shape[0]<=chunk_size:
        # Use correlation to determine the common centre
        centre_xs, centre_ys = correlateImageSet(images, multi_process=multi_process, \
                                                 reference_image=reference_image, \
                                                 roi_centre=roi_centre, \
                                                 roi_width=roi_width, \
                                                 max_deviation=max_deviation)
        
        if False:
            from matplotlib import pyplot as plt
            plt.figure()
            plt.subplot(2, 1, 1)
            plt.plot(centre_xs, label='x')
            diffs = []
            for i in range(centre_xs.size-1):
                dif = centre_xs[i]-centre_xs[i+1]
                diffs.append(dif)
                if abs(dif)>=20:
                    plt.plot(i+1, centre_xs[i+1], 'rx')
            plt.plot(diffs)
            plt.legend()
            
            plt.subplot(2, 1, 2)
            plt.plot(centre_ys, label='y')
            diffs = []
            for i in range(centre_ys.size-1):
                dif = centre_ys[i]-centre_ys[i+1]
                diffs.append(dif)
                if abs(dif)>=20:
                    plt.plot(i+1, centre_ys[i+1], 'rx')
            plt.plot(diffs)
            plt.legend()
            plt.show()
            
            exit()
        
        # Align the common centres.
        aligned_image = alignImageSet(images, centre_xs, centre_ys, multi_process=multi_process)
        
        if iterations>1:
            # Use the aligned_image as the new reference_image.
            
            # Increase the aligned image to the original size.
            d_size_x = images[0].shape[1] - aligned_image.shape[1]
            d_size_y = images[0].shape[0] - aligned_image.shape[0]
            
            new_reference_image = pad(aligned_image,
                                  ((d_size_x, d_size_x), (d_size_y, d_size_y)),
                                  mode='edge')
                                  
            return registerImageSet(images, multi_process=multi_process, \
                                    reference_image=new_reference_image, \
                                    roi_centre=roi_centre, roi_width=roi_width, \
                                    iterations=iterations-1, \
                                    max_deviation=max_deviation)
        else:
            return aligned_image
    else:
        chunks = [images[i:i+chunk_size] for i in range(0, images.shape[0], chunk_size)]
        
        padded_aligned_chunks = zeros((len(chunks), images.shape[1], images.shape[2]))
        for i, chunk in enumerate(chunks):
            aligned_chunk = array(registerImageSet(chunk, \
                                             multi_process=multi_process, \
                                             reference_image=reference_image, \
                                             roi_centre=roi_centre, roi_width=roi_width, \
                                             iterations=iterations, chunk_size=chunk_size, \
                                             max_deviation=max_deviation))
                                             
            # Increase the chunk to the original size.
            d_size_x = images[0].shape[1] - aligned_chunk.shape[1]
            d_size_y = images[0].shape[0] - aligned_chunk.shape[0]
            
            pad_left, pad_right = int(ceil(d_size_x/2)), int(floor(d_size_x/2))
            pad_bottom, pad_top = int(ceil(d_size_y/2)), int(floor(d_size_y/2))
            
            padded_aligned_chunk = pad(aligned_chunk,
                                       ((pad_bottom, pad_top), \
                                        (pad_left, pad_right)),
                                       mode='edge')
                  
            padded_aligned_chunks[i, :, :] = padded_aligned_chunk
        
        return registerImageSet(padded_aligned_chunks, multi_process=multi_process, \
                                reference_image=None, \
                                roi_centre=roi_centre, roi_width=roi_width, \
                                iterations=iterations, chunk_size=0, \
                                max_deviation=max_deviation)
    
def correlateImageSet(images, multi_process=True, reference_image=None, roi_centre=None, \
                      roi_width=50, max_deviation=(None, None)):
    if reference_image is None:
        #reference_image = images[0]
        #reference_image = images[0:10].mean(axis=0)
        reference_image = images.mean(axis=0)
    
    if roi_centre is None:
        roi_centre = getMaxCoordinates(reference_image)
        
        
    subimages = getSubimages(images, roi_centre[0], roi_centre[1], roi_width)
    #reference_subimage = getSubimage(reference_image, roi_centre[0], roi_centre[1], roi_width)
    
    p_correlate = partial(correlate, reference_image=reference_image)
                              
    if max_deviation[0] is None and max_deviation[1] is None and multi_process:
        with Pool(processes=cpu_count()) as workers:
                centre_xs, centre_ys = zip(*workers.map(p_correlate, enumerate(subimages)))
                
        centre_xs = array(centre_xs)
        centre_ys = array(centre_ys)
    else:
        centre_xs = zeros(subimages.shape[0])
        centre_ys = zeros(subimages.shape[0])
        
        for i, subimage in enumerate(subimages):
            centre_x, centre_y = p_correlate((i, subimage), \
                                             previous_centre=(centre_xs[i-1], centre_ys[i-1]) if i>0 else (None, None),
                                             max_deviation=max_deviation)
            
            centre_xs[i] = centre_x
            centre_ys[i] = centre_y
    
    # Shift to the full image coordinates
    start_x, stop_x, start_y, stop_y = getStartStop(roi_centre[0], roi_centre[1], roi_width, images[0].shape)
    centre_xs = centre_xs + start_x
    centre_ys = centre_ys + start_y
    
    return centre_xs, centre_ys
    
def correlate(enumerated_image, reference_image, previous_centre=(None,None), max_deviation=(None,None)):
    i, image = enumerated_image
    print('\tRegistering image', i+1)
    
    correlation = fftconvolve(image, reference_image[::-1, ::-1], mode='same')
    
    if previous_centre[0] is None or max_deviation[0] is None:
        start_x, stop_x = 0, correlation.shape[1]
    else:
        start_x = max(0, int(previous_centre[0] - max_deviation[0]))
        stop_x = min(correlation.shape[1], int(previous_centre[0] + max_deviation[0]))
        
    if previous_centre[1] is None or max_deviation[1] is None:
        start_y, stop_y = 0, correlation.shape[0]
    else:
        start_y = max(0, int(previous_centre[1] - max_deviation[1]))
        stop_y = min(correlation.shape[0], int(previous_centre[1] + max_deviation[1]))
    
    correlation = correlation[start_y:stop_y, start_x:stop_x]

    max_y, max_x = unravel_index(correlation.argmax(), correlation.shape)
    
    return max_x + start_x, max_y + start_y
    
def alignImageSet(images, centre_xs, centre_ys, multi_process=True):
    mean_x = round(centre_xs.mean())
    mean_y = round(centre_ys.mean())
    
    dxs = centre_xs - mean_x
    dys = centre_ys - mean_y

    min_dx = dxs.min()
    max_dx = dxs.max()
    min_dy = dys.min()
    max_dy = dys.max()
    
    p_align_image = partial(align_image, min_dx=min_dx, max_dx=max_dx,
                min_dy=min_dy, max_dy=max_dy)
    
    if multi_process:
        with Pool(processes=cpu_count()) as workers:
            output = workers.map(p_align_image, enumerate(zip(images, dxs, dys)))
            aligned_average_image = array(output).mean(axis=0)
    else:
        aligned_average_image = p_align_image((0, (images[0], dxs[0], dys[0])))
            
        for i in range(1, images.shape[0]):
            aligned_average_image += p_align_image((i, (images[i], dxs[i], dys[i])))
            
        aligned_average_image /= images.shape[0]

    return aligned_average_image
    
def align_image(tuple, min_dx, max_dx, min_dy, max_dy):
    i, (image, dx, dy) = tuple
    
    print('\tAligning image', i+1)
    
    x1 = int(-min_dx + dx)
    x2 = int(image.shape[1]-max_dx + dx)
    y1 = int(-min_dy + dy)
    y2 = int(image.shape[0]-max_dy + dy)

    return image[y1:y2, x1:x2]
    
# Utility Functions
def dirToName(dir):
    return split('/|\\\\', dir)[-1]
        # Split around / or \ (while escaping \'s).
    
def getStartStop2(x, y, length_x, length_y, shape):
    start_x = max(0, int(x - length_x/2))
    stop_x = min(shape[1], int(x + length_x/2))
    start_y = max(0, int(y - length_y/2))
    stop_y = min(shape[0], int(y + length_y/2))
        
    return start_x, stop_x, start_y, stop_y
    
def getStartStop(x, y, length, shape):
    return getStartStop2(x, y, length, length, shape)
    
def getSubimage(image, x, y, length):
    start_x, stop_x, start_y, stop_y = getStartStop(x, y, length, image.shape)
    
    return image[start_y:stop_y, start_x:stop_x]

def getSubimages(images, x, y, length):
    start_x, stop_x, start_y, stop_y = getStartStop(x, y, length, images[0].shape)
    
    return images[:, start_y:stop_y, start_x:stop_x]
    
def getMaxCoordinates(image):
    max_y, max_x = unravel_index(image.argmax(), image.shape)
    
    return max_x, max_y

    
##### Testing #####
if __name__=='__main__':
    # Some imports.
    from matplotlib import pyplot as plt
    
    from ImageUtility import plot_image
    
    # The script
    processDirs(directory='Test Images')
    
    with File(DEFAULT_HDF_NAME) as hdf:
        for key in hdf:
            arr = array(hdf[key])
            
            plt.figure(key)
            plot_image(arr, new_fig=False)
            
    plt.show()
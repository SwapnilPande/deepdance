import cv2
from tqdm import tqdm
import numpy as np
from alignment_by_row_channels import align as alignment_by_row


def align(fname1, fname2, write=False, outpath1='', outpath2='', offset=0):
    '''
    captures videos and aligns the
    :param fname1: filepath
    :param fname2: filepath
    :param write: save output videos or not
    :param outpath1: path for video1 if saving
    :param outpath2: path for video2 if saving
    :return: frames1, frames2, fps, shape1, shape2
    '''
    # delay = alignment_by_row(fname1, fname2, '..')
    delay = (0,0)
    cap1 = cv2.VideoCapture(fname1)
    cap2 = cv2.VideoCapture(fname2)
    fps = cap1.get(cv2.CAP_PROP_FPS)
    frame_width1 = cap1.get(3)
    frame_height1 = cap1.get(4)
    frame_width2 = cap2.get(3)
    frame_height2 = cap2.get(4)
    cap1.set(cv2.CAP_PROP_POS_MSEC, offset + delay[1] * 1000)
    cap2.set(cv2.CAP_PROP_POS_MSEC, offset + delay[0] * 1000)

    frames1 = []
    frames2 = []
    with tqdm(total=cap1.get(cv2.CAP_PROP_FRAME_COUNT), desc='Processing') as pbar:
        while 1:
            success1, frame1 = cap1.read()
            success2, frame2 = cap2.read()
            if success1 and success2:
                frames1.append(frame1)
                frames2.append(frame2)
                pbar.update(1)
            else:
                break
    if write:
        write_video(outpath1, frames1, fps, (int(frame_width1), int(frame_height1)))
        write_video(outpath2, frames2, fps, (int(frame_width2), int(frame_height2)))
    return frames1, frames2, fps, (int(frame_width1), int(frame_height1)), (int(frame_width2), int(frame_height2))

def write_video(fname, frames, fps, shape):
    api = cv2.CAP_FFMPEG
    code = cv2.VideoWriter.fourcc('m', 'p', '4', 'v')
    output = cv2.VideoWriter(fname, api, code, fps, shape)
    with tqdm(total=len(frames), desc='Writing') as pbar:
        for frame in frames:
            output.write(frame)
            pbar.update(1)
    output.release()

def check_alignment(frames1, frames2, fps, shape1, shape2, outpath):
    '''
    Pairs videos side-by-side and saves combined files to outpath
    :param frames1: list of aligned frames
    :param frames2: list of aligned frames
    :param fps: int number of fps for output video
    :param shape1: shape of frames1
    :param shape2: shape of frames2
    :param outpath: file path to output combined video
    :return: None
    '''
    shape = (int((shape1[0] + shape2[0])/2), int((shape1[1] + shape2[1]) / 4))
    frames = []
    for i in tqdm(range(len(frames1))):
        vis = np.concatenate((frames1[i], frames2[i]), axis=1)
        new_vis = cv2.resize(vis, shape)
        frames.append(new_vis)

    write_video(outpath, frames, fps, shape)

def check_alignment_from_files(fname1, fname2, outpath):
    cap1 = cv2.VideoCapture(fname1)
    cap2 = cv2.VideoCapture(fname2)
    fps = cap2.get(cv2.CAP_PROP_FPS)
    frame_width1 = cap1.get(3)
    frame_height1 = cap1.get(4)
    frame_width2 = cap2.get(3)
    frame_height2 = cap2.get(4)

    frames1 = []
    frames2 = []

    while 1:
        success1, frame1 = cap1.read()
        success2, frame2 = cap2.read()
        if success1 and success2:
            frames1.append(frame1)
            frames2.append(frame2)
        else:
            break
    check_alignment(frames1, frames2, fps, (int(frame_width1), int(frame_height1)), (int(frame_width2), int(frame_height2)), outpath)

if __name__ == '__main__':
    # frames1, frames2, fps, shape1, shape2 = align('videos/david-ymca.mp4', 'videos/ymca.mp4',
    #                                               write=True,
    #                                               outpath1='FF-align-david-ymca.mp4',
    #                                               outpath2='FF-align-caro-ymca.mp4',
    #                                               offset=7000)
    check_alignment_from_files('verification/david-hands-feet.mp4', 'verification/caro-hands-feet.mp4', 'verification/duet-hands-feet.mp4')
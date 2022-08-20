"""
    SORT: A Simple, Online and Realtime Tracker
    Copyright (C) 2016-2020 Alex Bewley alex@bewley.ai

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
from __future__ import print_function

import os
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage import io

import glob
import time
import argparse
from filterpy.kalman import KalmanFilter

np.random.seed(0)


def linear_assignment(cost_matrix):   # 掉包计算匈牙利算法二分图匹配
  try:
    import lap
    _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
    return np.array([[y[i],i] for i in x if i >= 0]) #
  except ImportError:
    from scipy.optimize import linear_sum_assignment
    x, y = linear_sum_assignment(cost_matrix)
    return np.array(list(zip(x, y)))


def iou_batch(bb_test, bb_gt):
  """
  From SORT: Computes IOU between two bboxes in the form [x1,y1,x2,y2]
  """
  bb_gt = np.expand_dims(bb_gt, 0)
  bb_test = np.expand_dims(bb_test, 1)
  
  xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
  yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
  xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
  yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
  w = np.maximum(0., xx2 - xx1)
  h = np.maximum(0., yy2 - yy1)
  wh = w * h
  o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])                                      
    + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)                                              
  return(o)  # np.array( [ [],[],.. ] )  计算所有m个 detection（bb_test) 与所有n个先验 tracker(bb_gt) 之间的 iou，因此是一个m×n矩阵


def convert_bbox_to_z(bbox):  # 传入的bbox为 np.array( [x1,y1,x2,y2,类别分数] )， 见114行。比宣称的 [x1,y1,x2,y2] 多了一项，但没关系
  """
  Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
    [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
    the aspect ratio
  """
  w = bbox[2] - bbox[0]
  h = bbox[3] - bbox[1]
  x = bbox[0] + w/2.
  y = bbox[1] + h/2.
  s = w * h    #scale is just area
  r = w / float(h)
  return np.array([x, y, s, r]).reshape((4, 1))   # np.array( [ [x],[y],[s],[r] ] )


def convert_x_to_bbox(x,score=None):
  """
  Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
    [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
  """
  w = np.sqrt(x[2] * x[3])
  h = x[2] / w
  if(score==None):
    return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.]).reshape((1,4))       # np.array( [ [x1,y1,x2,y2] ] )
  else:
    return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.,score]).reshape((1,5)) # np.array( [ [x1,y1,x2,y2,score] ] )


class KalmanBoxTracker(object):
  """
  This class represents the internal state of individual tracked objects observed as bbox.
  """
  count = 0
  def __init__(self,bbox):  # 传入的bbox为 np.array( [x1,y1,x2,y2,类别分数] )， 见240行
    """
    Initialises a tracker using initial bounding box.
    """
    #define constant velocity model
    self.kf = KalmanFilter(dim_x=7, dim_z=4)    # 这些 kf（KalmanFilter）的初始化文档见 https://filterpy.readthedocs.io/en/latest/kalman/KalmanFilter.html
    self.kf.F = np.array([[1,0,0,0,1,0,0],[0,1,0,0,0,1,0],[0,0,1,0,0,0,1],[0,0,0,1,0,0,0],  [0,0,0,0,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,1]]) # 状态转移矩阵
    self.kf.H = np.array([[1,0,0,0,0,0,0],[0,1,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,1,0,0,0]])   # 观测矩阵

    self.kf.R[2:,2:] *= 10.     # 观测噪声的协方差
    self.kf.P[4:,4:] *= 1000. #give high uncertainty to the unobservable initial velocities   # 初始后验状态的协方差
    self.kf.P *= 10.
    self.kf.Q[-1,-1] *= 0.01    # 过程噪声的协方差
    self.kf.Q[4:,4:] *= 0.01
                                            # 论文中的状态x为 [u, v, s, r, dot(u), dot(v), dot(s)]   其中宽高比r固定，不参与更新
    self.kf.x[:4] = convert_bbox_to_z(bbox) # bbox定义在99行的形参，为 np.array( [x1,y1,x2,y2,类别分数] )。本代码取观测值z为 [u, v, s, r]
    self.time_since_update = 0              # bbox：[x1,y1,x2,y2,类别分数] 经过 convert_bbox_to_z 变成 [u, v, s, r]
    self.id = KalmanBoxTracker.count
    KalmanBoxTracker.count += 1
    self.history = []
    self.hits = 0
    self.hit_streak = 0
    self.age = 0

  def update(self,bbox):    # bbox即dets[i]，即 np.array( [x1,y1,x2,y2,类别分数] )，见240行
    """
    Updates the state vector with observed bbox.
    """
    self.time_since_update = 0
    self.history = []
    self.hits += 1          # 命中
    self.hit_streak += 1    # 连续命中
    self.kf.update(convert_bbox_to_z(bbox)) # 关键行：卡尔曼滤波器更新，滤波器内的x修正为下一次的后验状态
                                            # 方法签名：update(z, R=None, H=None)  详见 https://filterpy.readthedocs.io/en/latest/kalman/KalmanFilter.html#filterpy.kalman.KalmanFilter.update
  def predict(self):
    """
    Advances the state vector and returns the predicted bounding box estimate.
    """
    if((self.kf.x[6]+self.kf.x[2])<=0):
      self.kf.x[6] *= 0.0
    self.kf.predict() # 关键行：卡尔曼滤波器预测先验状态
    self.age += 1     # predict(u=None, B=None, F=None, Q=None) Predict next state (prior) using the Kalman filter state propagation equations. 详见 https://filterpy.readthedocs.io/en/latest/kalman/KalmanFilter.html#filterpy.kalman.KalmanFilter.predict
    if(self.time_since_update>0):
      self.hit_streak = 0
    self.time_since_update += 1
    self.history.append(convert_x_to_bbox(self.kf.x)) # convert_x_to_bbox(self.kf.x) 返回 np.array( [ [x1,y1,x2,y2] ] )
    return self.history[-1]     # 返回下一次的先验状态 np.array( [ [x1,y1,x2,y2] ] )

  def get_state(self):  # 返回当前状态x（bbox表示）：np.array(  [  [x1,y1,x2,y2]  ]  )
    """
    Returns the current bounding box estimate.
    """
    return convert_x_to_bbox(self.kf.x)


def associate_detections_to_trackers(detections,trackers,iou_threshold = 0.3):
  """
  Assigns detections to tracked object (both represented as bounding boxes)
  detections 和 trackers 格式相同，均为 np.array( [  [x1,y1,x2,y2,类别分数],[],.. ] )
  Returns 3 lists of matches, unmatched_detections and unmatched_trackers
  """
  if(len(trackers)==0):
    return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,5),dtype=int)

  iou_matrix = iou_batch(detections, trackers)            # 第一步：计算所有m个 detection 与所有n个先验 tracker 之间的 iou，因此是一个m×n矩阵

  if min(iou_matrix.shape) > 0:                           # 第二步：对iou矩阵用匈牙利算法进行m个 detection 与 n个先验 tracker 之间的匹配
    a = (iou_matrix > iou_threshold).astype(np.int32)
    if a.sum(1).max() == 1 and a.sum(0).max() == 1:
        matched_indices = np.stack(np.where(a), axis=1)
    else:
      matched_indices = linear_assignment(-iou_matrix)    # 关键行：匈牙利算法将传入的cost矩阵做整体cost最小的匹配，而iou越大越应该匹配，因此cost:=-iou
  else:                                                   # 输出为匹配出的detection与相应tracker的序号对 matched_indices，
    matched_indices = np.empty(shape=(0,2))               # matched_indices 格式：np.array( [ [det_i,trk_j],[],..   ] )  i不一定等于j，0<=i<=m, 0<=j<=n

  unmatched_detections = []                               # 第三步：将不在 matched_indices 中的 detection 和 tracker 分别建表 unmatched_detections 和 unmatched_trackers
  for d, det in enumerate(detections):
    if(d not in matched_indices[:,0]):
      unmatched_detections.append(d)
  unmatched_trackers = []
  for t, trk in enumerate(trackers):
    if(t not in matched_indices[:,1]):
      unmatched_trackers.append(t)

  #filter out matched with low IOU
  matches = []                                            # 第四步：将序号对 matched_indices 再查表iou矩阵通过 iou阈值 过滤
  for m in matched_indices:                               # 符合阈值要求的建表 matches： np.array(  [  [det_i,trk_j],[],..  ]  )
    if(iou_matrix[m[0], m[1]]<iou_threshold):             # 不符合的各自加入第三步的 unmatched_detections 或 unmatched_trackers
      unmatched_detections.append(m[0])
      unmatched_trackers.append(m[1])
    else:
      matches.append(m.reshape(1,2))
  if(len(matches)==0):
    matches = np.empty((0,2),dtype=int)
  else:
    matches = np.concatenate(matches,axis=0)

  return matches, np.array(unmatched_detections), np.array(unmatched_trackers)  # 统一unmatched_detections和unmatched_trackers的格式：np.array( [d/t, d/t, ..] )


class Sort(object):
  def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
    """
    Sets key parameters for SORT
    """
    self.max_age = max_age
    self.min_hits = min_hits
    self.iou_threshold = iou_threshold
    self.trackers = []                    # self.trackers[]中存储 KalmanBoxTracker对象，见240行
    self.frame_count = 0

  def update(self, dets=np.empty((0, 5))):  # dets：np.array( [ [x1,y1,x2,y2,score],[],.. ] ) 302行
    """
    Params:
      dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
    Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 5)) for frames without detections).
    Returns the a similar array, where the last column is the object ID.

    NOTE: The number of objects returned may differ from the number of detections provided.
    """
    self.frame_count += 1
    # get predicted locations from existing trackers.               # 第一步：使用匈牙利算法将当前次观测值detections与当前次先验trackers进行匹配
    trks = np.zeros((len(self.trackers), 5))
    to_del = []
    ret = []
    for t, trk in enumerate(trks):
      pos = self.trackers[t].predict()[0]           # 关键步：通过卡尔曼滤波使用前一次的后验trackers生成当前次先验trackers    predict() 返回 np.array( [ [x1,y1,x2,y2] ] )   pos: np.array( [x1,y1,x2,y2] )
      trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]  # trk[:]: np.array( [x1,y1,x2,y2,0] )    trks: np.array( [ [x1,y1,x2,y2,0],[],.. ] )
      if np.any(np.isnan(pos)):
        to_del.append(t)
    trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
    for t in reversed(to_del):  # self.trackers是list，下面可能需要从中pop(index)，故从后往前遍历，t就是index
      self.trackers.pop(t)
    matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets,trks, self.iou_threshold)   # 关键步：调用匈牙利算法

    # update matched trackers with assigned detections              # 第二步：使用卡尔曼滤波将成功匹配的先验trackers和detections融合成当前次后验trackers
    for m in matched:
      self.trackers[m[1]].update(dets[m[0], :])   # self.trackers[m[1]]是 KalmanBoxTracker对象，见240行

    # create and initialise new trackers for unmatched detections   # 第三步：给未成功匹配的detections创建并初始化新的trackers
    for i in unmatched_dets:
        trk = KalmanBoxTracker(dets[i,:])   # 关键步：创建并初始化新的trackers   dets[i,:]即dets[i]，即 np.array( [x1,y1,x2,y2,类别分数] )
        self.trackers.append(trk)
    i = len(self.trackers)
    for trk in reversed(self.trackers): # self.trackers是list，下面可能需要从中pop(index)，故从后往前遍历，上一行的i就是index+1
        d = trk.get_state()[0]  # get_state：返回当前状态x（bbox表示）：np.array(  [  [x1,y1,x2,y2]  ]  )
        if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
          ret.append(np.concatenate((d,[trk.id+1])).reshape(1,-1)) # +1 as MOT benchmark requires positive  # append的元素：np.array(  [  [x1,y1,x2,y2,展示id]  ]  )
        i -= 1
        # remove dead tracklet
        if(trk.time_since_update > self.max_age):
          self.trackers.pop(i)
    if(len(ret)>0):                                                 # 第四步：返回所有新trackers（当前次更新过的和新增的）集合
      return np.concatenate(ret)  # 返回新展示框集合：np.array( [ [x1,y1,x2,y2,展示id],[],.. ] )    展示id=id+1
    return np.empty((0,5))

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='SORT demo')
    parser.add_argument('--display', dest='display', help='Display online tracker output (slow) [False]',action='store_true')
    parser.add_argument("--seq_path", help="Path to detections.", type=str, default='data')
    parser.add_argument("--phase", help="Subdirectory in seq_path.", type=str, default='train')
    parser.add_argument("--max_age", 
                        help="Maximum number of frames to keep alive a track without associated detections.", 
                        type=int, default=1)
    parser.add_argument("--min_hits", 
                        help="Minimum number of associated detections before track is initialised.", 
                        type=int, default=3)
    parser.add_argument("--iou_threshold", help="Minimum IOU for match.", type=float, default=0.3)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
  # all train
  args = parse_args()
  display = args.display
  phase = args.phase
  total_time = 0.0
  total_frames = 0
  colours = np.random.rand(32, 3) #used only for display
  if(display):
    if not os.path.exists('mot_benchmark'):
      print('\n\tERROR: mot_benchmark link not found!\n\n    Create a symbolic link to the MOT benchmark\n    (https://motchallenge.net/data/2D_MOT_2015/#download). E.g.:\n\n    $ ln -s /path/to/MOT2015_challenge/2DMOT2015 mot_benchmark\n\n')
      exit()
    plt.ion()
    fig = plt.figure()
    ax1 = fig.add_subplot(111, aspect='equal')

  if not os.path.exists('output'):
    os.makedirs('output')
  pattern = os.path.join(args.seq_path, phase, '*', 'det', 'det.txt')   # 'data/train/*/det/det.txt'
  for seq_dets_fn in glob.glob(pattern):                                # 比如：seq_dets_fn：'data/train/ADL-Rundle-6/det/det.txt'
    mot_tracker = Sort(max_age=args.max_age,                # 一个视频建立一个Sort对象
                       min_hits=args.min_hits,
                       iou_threshold=args.iou_threshold) #create instance of the SORT tracker
    seq_dets = np.loadtxt(seq_dets_fn, delimiter=',')                   # 比如：np.array( [ [1,-1,1691.97,381.048,152.23,352.617,0.995616,-1,-1,-1],[],.. ] )
    seq = seq_dets_fn[pattern.find('*'):].split(os.path.sep)[0]         # 比如：'ADL-Rundle-6'
    
    with open(os.path.join('output', '%s.txt'%(seq)),'w') as out_file:  # 比如：'output/ADL-Rundle-6.txt'
      print("Processing %s."%(seq))                                     # 比如：Processing ADL-Rundle-6.
      for frame in range(int(seq_dets[:,0].max())):                     # 比如：frame：0   range(int(seq_dets[:,0].max()))：range(0, 525)
        frame += 1 #detection and frame numbers begin at 1
        dets = seq_dets[seq_dets[:, 0]==frame, 2:7]                     # 比如：      dets: np.array( [ [1691.97, 381.048, 152.23,  352.617, 0.995616],[],.. ] )
        dets[:, 2:4] += dets[:, 0:2] #convert to [x1,y1,w,h] to [x1,y1,x2,y2] # 比如：dets: np.array( [ [1691.97, 381.048, 1844.20, 733.665, 0.995616],[],.. ] )
        total_frames += 1

        if(display):
          fn = os.path.join('mot_benchmark', phase, seq, 'img1', '%06d.jpg'%(frame))
          im =io.imread(fn)
          ax1.imshow(im)
          plt.title(seq + ' Tracked Targets')

        start_time = time.time()
        trackers = mot_tracker.update(dets)     # 使用卡尔曼滤波用新的观测值dets生成当前帧的后验trackers
        cycle_time = time.time() - start_time   # 比如：dets: np.array( [ [1691.97, 381.048, 1844.20, 733.665, 0.995616],[],.. ] ) 302行  update返回新后验展示框集合：np.array( [ [x1,y1,x2,y2,展示id],[],.. ] )    展示id=id+1
        total_time += cycle_time

        for d in trackers:
          print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1'%(frame,d[4],d[0],d[1],d[2]-d[0],d[3]-d[1]),file=out_file)
          if(display):
            d = d.astype(np.int32)
            ax1.add_patch(patches.Rectangle((d[0],d[1]),d[2]-d[0],d[3]-d[1],fill=False,lw=3,ec=colours[d[4]%32,:]))

        if(display):
          fig.canvas.flush_events()
          plt.draw()
          ax1.cla()

  print("Total Tracking took: %.3f seconds for %d frames or %.1f FPS" % (total_time, total_frames, total_frames / total_time))

  if(display):
    print("Note: to get real runtime results run without the option: --display")

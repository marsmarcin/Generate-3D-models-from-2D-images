# im2Avatar ——generate 3D model from 2D image
import tensorflow as tf
import numpy as np
import os
import h5py
import sys
import cv2
sys.path.append('./utils')
sys.path.append('./models')
import model_shape as model
from scipy import misc
from mayavi.mlab import quiver3d, draw
from mayavi import mlab

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'




FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('train_dir', './train_shape',
                           """Directory where to write summaries and checkpoint.""")
tf.app.flags.DEFINE_string('base_dir', './data/ShapeNetCore_im2avatar',
                           """The path containing all the samples.""")
tf.app.flags.DEFINE_string('cat_id', '02958343',
                           """The category id for each category: 02958343, 03001627, 03467517, 04379243""")
tf.app.flags.DEFINE_string('data_list_path', './data_list',
                          """The path containing data lists.""")
tf.app.flags.DEFINE_string('output_dir', './output_shape',
                           """Directory to save generated volume.""")

tf.app.flags.DEFINE_string('img', '', '''''')
TRAIN_DIR = os.path.join(FLAGS.train_dir, FLAGS.cat_id)
OUTPUT_DIR = os.path.join(FLAGS.output_dir, 'test')
img_path = 'M://im2avatar//data//test//10.jpeg'



if not os.path.exists(OUTPUT_DIR):
  os.makedirs(OUTPUT_DIR)

BATCH_SIZE = 1

IM_DIM = 128
VOL_DIM = 64

def inference():
  is_train_pl = tf.placeholder(tf.bool)
  img_pl, _, = model.placeholder_inputs(BATCH_SIZE, IM_DIM, VOL_DIM)
  pred = model.get_model(img_pl, is_train_pl)
  pred = tf.sigmoid(pred)

  config = tf.ConfigProto(device_count={'CPU': 1})
  with tf.Session(config=config) as sess:
    model_path = os.path.join(TRAIN_DIR, "trained_models")
    ckpt = tf.train.get_checkpoint_state(model_path)
    restorer = tf.train.Saver()
    restorer.restore(sess, ckpt.model_checkpoint_path)

    img_1 = np.array(misc.imread(img_path) / 255.0)
    img_1 = img_1.reshape((1, 128, 128, 3))
    feed_dict = {img_pl: img_1, is_train_pl: False}
    pred_res = sess.run(pred, feed_dict=feed_dict)

    vol_ = pred_res[0] # (vol_dim, vol_dim, vol_dim, 1)
    name_ = '001' # FLAGS.img.strip().split('.')[0] # xx.xxx.png

    save_path = OUTPUT_DIR
    if not os.path.exists(save_path):
      os.makedirs(save_path)

    save_path_name = os.path.join(save_path, name_+".h5")
    if os.path.exists(save_path_name):
      os.remove(save_path_name)

    h5_fout = h5py.File(save_path_name)
    h5_fout.create_dataset(
            'data', data=vol_,
            compression='gzip', compression_opts=4,
            dtype='float32')
    h5_fout.close()

    print(name_+'.h5 is predicted into %s' % (save_path_name))


if __name__ == '__main__':
  inference()


# Visualization
tf.app.flags.DEFINE_string('mode', 'cube', '''''')
tf.app.flags.DEFINE_float('thresh', '0.6', '''''')
path_shape = 'F:/python/mayaya/output_shape/test/001.h5'
with h5py.File(path_shape, 'r') as f:
	voxel = f['data'][:].reshape(64, 64, 64)
for i in range(64):
	for j in range(64):
		for k in range(64):
			if voxel[i, j, k] >= FLAGS.thresh:
				voxel[i, j, k] = 1
			else:
				voxel[i, j, k] = 0
x, y, z = np.where(voxel == 1)
xx = np.ones(len(x))
yy = np.zeros(len(x))
zz = np.zeros(len(x))
scalars = np.arange(len(x)) 
pts = quiver3d(x, y, z, xx, yy, zz, scalars=scalars, mode=FLAGS.mode)
mlab.show()
# input_img = cv2.imread(r'M://im2avatar//data//test//10.jpeg')
# cv2.imshow('2D',input_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
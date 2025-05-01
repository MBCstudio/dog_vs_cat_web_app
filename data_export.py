import os, shutil

#env path PYTHONUNBUFFERED=1;LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/marcin/miniconda3/envs/tf/lib/:/home/marcin/miniconda3/envs/tf/lib/python3.9/site-packages/nvidia/cudnn/lib);XLA_FLAGS=--xla_gpu_cuda_data_dir=/home/marcin/miniconda3/envs/tf/lib/)

orginal_data_path = '/home/marcin/PycharmProjects/pythonProject2/archive/dogs-vs-cats'

#making dirs for data
base_dir = '/home/marcin/PycharmProjects/pythonProject2/cats_dog_small_dir'
os.mkdir(base_dir)

train_dir = os.path.join(base_dir, 'train')
os.mkdir(train_dir)
validation_dir = os.path.join(base_dir, 'validation')
os.mkdir(validation_dir)
test_dir = os.path.join(base_dir, 'test')
os.mkdir(test_dir)

train_dir_cat = os.path.join(train_dir, 'cat')
os.mkdir(train_dir_cat)
train_dir_dog = os.path.join(train_dir, 'dog')
os.mkdir(train_dir_dog)

validation_dir_cat = os.path.join(validation_dir, 'cat')
os.mkdir(validation_dir_cat)
validation_dir_dog = os.path.join(validation_dir, 'dog')
os.mkdir(validation_dir_dog)

test_dir_cat = os.path.join(test_dir, 'cat')
os.mkdir(test_dir_cat)
test_dir_dog = os.path.join(test_dir, 'dog')
os.mkdir(test_dir_dog)

#adding data to the dirs
fnames = ['cat.{}.jpg'.format(i) for i in range(1,1001)]
for fname in fnames:
    src = os.path.join(orginal_data_path + '/cat', fname)
    dst = os.path.join(train_dir_cat, fname)
    shutil.copyfile(src, dst)

fnames = ['cat.{}.jpg'.format(i) for i in range(1001,1501)]
for fname in fnames:
    src = os.path.join(orginal_data_path + '/cat', fname)
    dst = os.path.join(validation_dir_cat, fname)
    shutil.copyfile(src, dst)

fnames = ['cat.{}.jpg'.format(i) for i in range(1501,2001)]
for fname in fnames:
    src = os.path.join(orginal_data_path + '/cat', fname)
    dst = os.path.join(test_dir_cat, fname)
    shutil.copyfile(src, dst)

fnames = ['dog.{}.jpg'.format(i) for i in range(1,1001)]
for fname in fnames:
    src = os.path.join(orginal_data_path + '/dog', fname)
    dst = os.path.join(train_dir_dog, fname)
    shutil.copyfile(src, dst)

fnames = ['dog.{}.jpg'.format(i) for i in range(1001,1501)]
for fname in fnames:
     src = os.path.join(orginal_data_path + '/dog', fname)
     dst = os.path.join(validation_dir_dog, fname)
     shutil.copyfile(src, dst)

fnames = ['dog.{}.jpg'.format(i) for i in range(1501,2001)]
for fname in fnames:
    src = os.path.join(orginal_data_path+'/dog', fname)
    dst = os.path.join(test_dir_dog, fname)
    shutil.copyfile(src, dst)



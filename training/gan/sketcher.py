import tensorflow as tf
from gan.models import Generator, Discriminator
import gan.cloud_paths as cloud_paths

import os
import time

import argparse

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--headless', action='store_true')
arg_parser.add_argument('--cloud', action='store_true')
arg_parser.add_argument('--skip-training', action='store_true')
arg_parser.add_argument('--skip-restore', action='store_true')
arg_parser.add_argument('--use-train-for-test', action='store_true')
arg_parser.add_argument('--show_untrained', action='store_true')
# gcloud ai-platform automatically passes this arg
arg_parser.add_argument('--job-dir')
arg_parser.add_argument(
    '--dataset',
    default='facades',
    choices=['facades', 'paintings']
)
arg_parser.add_argument('--epochs', type=int, default=450)
args = arg_parser.parse_args()

if not args.headless:
    from matplotlib import pyplot as plt
    from IPython import display



# !pip install -U tensorboard

"""## Load the dataset

You can download this dataset and similar datasets from [here](https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets). As mentioned in the [paper](https://arxiv.org/abs/1611.07004), apply random jittering and mirroring to the training dataset.

* In random jittering, the image is resized to `286 x 286` and then randomly cropped to `256 x 256`
* In random mirroring, the image is randomly flipped horizontally i.e left to right.
"""
PATH = ''

if args.cloud:
    PATH = cloud_paths.DATASET_PATH + args.dataset + '/'
    CHECKPOINT_DIR = cloud_paths.CHECKPOINT_PATH + args.dataset
else:
    if args.dataset == 'facades':
        _URL = 'https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/facades.tar.gz'

        path_to_zip = tf.keras.utils.get_file('facades.tar.gz',
                                              origin=_URL,
                                              extract=True)
        PATH = os.path.join(os.path.dirname(path_to_zip), 'facades/')
    elif args.dataset == 'paintings':
        PATH = '../images/generated/'
    CHECKPOINT_DIR = os.path.join('./training_checkpoints', args.dataset)
print(PATH)

TRAIN_PATTERN = PATH+'train/*.jpg'
TEST_PATTERN = PATH+'test/*.jpg'
if args.use_train_for_test:
    TEST_PATTERN = TRAIN_PATTERN

BUFFER_SIZE = 400
BATCH_SIZE = 1
IMG_WIDTH = 256
IMG_HEIGHT = 256

def load(image_file):
  print('loading {}'.format(image_file))
  image = tf.io.read_file(image_file)
  image = tf.image.decode_jpeg(image)

  w = tf.shape(image)[1]

  w = w // 2
  real_image = image[:, :w, :]
  input_image = image[:, w:, :]

  input_image = tf.cast(input_image, tf.float32)
  real_image = tf.cast(real_image, tf.float32)

  return input_image, real_image

def resize(input_image, real_image, height, width):
  input_image = tf.image.resize(input_image, [height, width],
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  real_image = tf.image.resize(real_image, [height, width],
                               method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

  return input_image, real_image

def random_crop(input_image, real_image):
  stacked_image = tf.stack([input_image, real_image], axis=0)
  cropped_image = tf.image.random_crop(
      stacked_image, size=[2, IMG_HEIGHT, IMG_WIDTH, 3])

  return cropped_image[0], cropped_image[1]

# normalizing the images to [-1, 1]

def normalize(input_image, real_image):
  input_image = (input_image / 127.5) - 1
  real_image = (real_image / 127.5) - 1

  return input_image, real_image

@tf.function()
def random_jitter(input_image, real_image):
  # resizing to 286 x 286 x 3
  input_image, real_image = resize(input_image, real_image, 286, 286)

  # randomly cropping to 256 x 256 x 3
  input_image, real_image = random_crop(input_image, real_image)

  if tf.random.uniform(()) > 0.5:
    # random mirroring
    input_image = tf.image.flip_left_right(input_image)
    real_image = tf.image.flip_left_right(real_image)

  return input_image, real_image

def load_image_train(image_file):
  input_image, real_image = load(image_file)
  input_image, real_image = random_jitter(input_image, real_image)
  input_image, real_image = normalize(input_image, real_image)

  return input_image, real_image

def load_image_test(image_file):
  input_image, real_image = load(image_file)
  input_image, real_image = resize(input_image, real_image,
                                   IMG_HEIGHT, IMG_WIDTH)
  input_image, real_image = normalize(input_image, real_image)

  return input_image, real_image

"""## Input Pipeline"""

train_dataset = tf.data.Dataset.list_files(TRAIN_PATTERN)
train_dataset = train_dataset.map(load_image_train,
                                  num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.batch(BATCH_SIZE)

test_dataset = tf.data.Dataset.list_files(TEST_PATTERN)
test_dataset = test_dataset.map(load_image_test)
test_dataset = test_dataset.batch(BATCH_SIZE)


"""* **Generator loss**
  * It is a sigmoid cross entropy loss of the generated images and an **array of ones**.
  * The [paper](https://arxiv.org/abs/1611.07004) also includes L1 loss which is MAE (mean absolute error) between the generated image and the target image.
  * This allows the generated image to become structurally similar to the target image.
  * The formula to calculate the total generator loss = gan_loss + LAMBDA * l1_loss, where LAMBDA = 100. This value was decided by the authors of the [paper](https://arxiv.org/abs/1611.07004).

The training procedure for the generator is shown below:
"""

LAMBDA = 100

def generator_loss(disc_generated_output, gen_output, target):
  gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

  # mean absolute error
  l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

  total_gen_loss = gan_loss + (LAMBDA * l1_loss)

  return total_gen_loss, gan_loss, l1_loss


"""**Discriminator loss**
  * The discriminator loss function takes 2 inputs; **real images, generated images**
  * real_loss is a sigmoid cross entropy loss of the **real images** and an **array of ones(since these are the real images)**
  * generated_loss is a sigmoid cross entropy loss of the **generated images** and an **array of zeros(since these are the fake images)**
  * Then the total_loss is the sum of real_loss and the generated_loss

"""

loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(disc_real_output, disc_generated_output):
  real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)

  generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

  total_disc_loss = real_loss + generated_loss

  return total_disc_loss



"""## Generate Images

Write a function to plot some images during training.

* Pass images from the test dataset to the generator.
* The generator will then translate the input image into the output.
* Last step is to plot the predictions and **voila!**

Note: The `training=True` is intentional here since
you want the batch statistics while running the model
on the test dataset. If you use training=False, you get
the accumulated statistics learned from the training dataset
(which you don't want)
"""

def generate_images(model, test_input, tar):
    prediction = model(test_input, training=True)
    if not args.headless:
        plt.figure(figsize=(15, 15))

    display_list = [test_input[0], tar[0], prediction[0]]
    title = ['Input Image', 'Ground Truth', 'Predicted Image']
    
    if not args.headless:
        for i in range(3):
            plt.subplot(1, 3, i+1)
            plt.title(title[i])
            # getting the pixel values between [0, 1] to plot it.
            plt.imshow(display_list[i] * 0.5 + 0.5)
            plt.axis('off')
        plt.show()



"""## Training

* For each example input generate an output.
* The discriminator receives the input_image and the generated image as the first input. The second input is the input_image and the target_image.
* Next, calculate the generator and the discriminator loss.
* Then, calculate the gradients of loss with respect to both the generator and the discriminator variables(inputs) and apply those to the optimizer.
* Then log the losses to TensorBoard.
"""

EPOCHS = args.epochs

import datetime
log_dir="logs/"

summary_writer = tf.summary.create_file_writer(
  log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

@tf.function
def train_step(input_image, target, epoch):
  with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
    gen_output = generator(input_image, training=True)

    disc_real_output = discriminator([input_image, target], training=True)
    disc_generated_output = discriminator([input_image, gen_output], training=True)

    gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target)
    disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

  generator_gradients = gen_tape.gradient(gen_total_loss,
                                          generator.trainable_variables)
  discriminator_gradients = disc_tape.gradient(disc_loss,
                                               discriminator.trainable_variables)

  generator_optimizer.apply_gradients(zip(generator_gradients,
                                          generator.trainable_variables))
  discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                              discriminator.trainable_variables))

  with summary_writer.as_default():
    tf.summary.scalar('gen_total_loss', gen_total_loss, step=epoch)
    tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=epoch)
    tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=epoch)
    tf.summary.scalar('disc_loss', disc_loss, step=epoch)

"""The actual training loop:

* Iterates over the number of epochs.
* On each epoch it clears the display, and runs `generate_images` to show it's progress.
* On each epoch it iterates over the training dataset, printing a '.' for each example.
* It saves a checkpoint every 20 epochs.
"""

def fit(train_ds, epochs, test_ds):
  for epoch in range(epochs):
    start = time.time()
    if not args.headless:
        display.clear_output(wait=True)

    #for example_input, example_target in test_ds.take(1):
    #  generate_images(generator, example_input, example_target)
    print("Epoch: ", epoch)

    # Train
    for n, (input_image, target) in train_ds.enumerate():
      print('.', end='')
      if (n+1) % 100 == 0:
        print()
      train_step(input_image, target, epoch)
    print()

    # saving (checkpoint) the model every 20 epochs
    if epoch % 20 == 0:
      checkpoint.save(file_prefix=checkpoint_prefix)

    print ('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                        time.time()-start))
  checkpoint.save(file_prefix=checkpoint_prefix)

"""This training loop saves logs you can easily view in TensorBoard to monitor the training progress. Working locally you would launch a separate tensorboard process. In a notebook, if you want to monitor with TensorBoard it's easiest to launch the viewer before starting the training.

To launch the viewer paste the following into a code-cell:
"""

  
if __name__ == '__main__':
    generator = Generator()
    discriminator = Discriminator()
    if not args.headless and False:
        tf.keras.utils.plot_model(generator, show_shapes=True, dpi=64)

        gen_output = generator(inp[tf.newaxis, ...], training=False)
        plt.imshow(gen_output[0, ...])
        
        tf.keras.utils.plot_model(discriminator, show_shapes=True, dpi=64)

        disc_out = discriminator([inp[tf.newaxis, ...], gen_output], training=False)
        plt.imshow(disc_out[0, ..., -1], vmin=-20, vmax=20, cmap='RdBu_r')
        plt.colorbar()


    """The training procedure for the discriminator is shown below.

    To learn more about the architecture and the hyperparameters you can refer the [paper](https://arxiv.org/abs/1611.07004).

    ![Discriminator Update Image](https://github.com/tensorflow/docs/blob/master/site/en/tutorials/generative/images/dis.png?raw=1)

    ## Define the Optimizers and Checkpoint-saver
    """

    generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

    checkpoint_dir = CHECKPOINT_DIR
    print('checkpoint_dir {}'.format(checkpoint_dir))
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                     discriminator_optimizer=discriminator_optimizer,
                                     generator=generator,
                                     discriminator=discriminator)

    if args.show_untrained:
        for example_input, example_target in test_dataset.take(1):
            generate_images(generator, example_input, example_target)

    """Now run the training loop:"""
    if not args.skip_restore:
        checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
    if not args.skip_training:
        fit(train_dataset, EPOCHS, test_dataset)
    if not args.headless:
        display.IFrame(
            src="https://tensorboard.dev/experiment/lZ0C6FONROaUMfjYkVyJqw",
            width="100%",
            height="1000px")

    """Interpreting the logs from a GAN is more subtle than a simple classification or regression model. Things to look for::

    * Check that neither model has "won". If either the `gen_gan_loss` or the `disc_loss` gets very low it's an indicator that this model is dominating the other, and you are not successfully training the combined model.
    * The value `log(2) = 0.69` is a good reference point for these losses, as it indicates a perplexity of 2: That the discriminator is on average equally uncertain about the two options.
    * For the `disc_loss` a value below `0.69` means the discriminator is doing better than random, on the combined set of real+generated images.
    * For the `gen_gan_loss` a value below `0.69` means the generator is doing better than random at fooling the descriminator.
    * As training progresses the `gen_l1_loss` should go down.

    ## Restore the latest checkpoint and test
    """

    # !ls {checkpoint_dir}

    # restoring the latest checkpoint in checkpoint_dir
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

    """## Generate using test dataset"""

    # Run the trained model on a few examples from the test dataset
    for inp, tar in test_dataset.take(5):
        generate_images(generator, inp, tar)
    

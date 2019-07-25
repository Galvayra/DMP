from DMP.utils.arg_training import *
from DMP.learning.score import MyScore
import os
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets
import numpy as np
from PIL import Image
import shutil

ALIVE_DIR = 'alive'
DEATH_DIR = 'death'


class TransferLearner(MyScore):
    def __init__(self, is_cross_valid=True):
        super().__init__()
        self.__name_of_log = str()
        self.__name_of_tensor = str()
        self.__is_cross_valid = is_cross_valid

    @property
    def name_of_log(self):
        return self.__name_of_log

    @name_of_log.setter
    def name_of_log(self, name):
        self.__name_of_log = name

    @property
    def name_of_tensor(self):
        return self.__name_of_tensor

    @name_of_tensor.setter
    def name_of_tensor(self, name):
        self.__name_of_tensor = name

    @property
    def is_cross_valid(self):
        return self.__is_cross_valid

    def __set_name_of_log(self):
        name_of_log = self.name_of_log + "fold_" + str(self.num_of_fold)

        if self.is_cross_valid:
            os.mkdir(name_of_log)

        if DO_SHOW:
            print("======== Directory for Saving ========")
            print("   Log File -", name_of_log)

    def __set_name_of_tensor(self):
        name_of_tensor = self.name_of_tensor + "fold_" + str(self.num_of_fold)

        if self.is_cross_valid:
            os.mkdir(name_of_tensor)

        if DO_SHOW:
            print("Tensor File -", name_of_tensor, "\n\n\n")

    @staticmethod
    def list_images(directory):
        """
        Get all the images and labels in directory/label/*.jpg
        """
        labels = os.listdir(directory)
        # Sort the labels so that training and validation get them in the same order
        labels.sort()

        files_and_labels = []
        for label in labels:
            for f in os.listdir(os.path.join(directory, label)):
                files_and_labels.append((os.path.join(directory, label, f), label))

        filenames, labels = zip(*files_and_labels)
        filenames = list(filenames)
        labels = list(labels)

        label_to_int = {
            ALIVE_DIR: 0,
            DEATH_DIR: 1
        }

        labels = [label_to_int[l] for l in labels]

        return filenames, labels

    # Standard preprocessing for VGG on ImageNet taken from here:
    # https://github.com/tensorflow/models/blob/master/research/slim/preprocessing/vgg_preprocessing.py
    # Also see the VGG paper for more details: https://arxiv.org/pdf/1409.1556.pdf

    # Preprocessing (for both training and validation):
    # (1) Decode the image from jpg format
    # (2) Resize the image so its smaller side is 256 pixels long
    @staticmethod
    def parse_function(filename, label):
        image_string = tf.read_file(filename)
        image_decoded = tf.image.decode_jpeg(image_string, channels=3)  # (1)
        image = tf.cast(image_decoded, tf.float32)

        smallest_side = 256.0
        height, width = tf.shape(image)[0], tf.shape(image)[1]
        height = tf.to_float(height)
        width = tf.to_float(width)

        scale = tf.cond(tf.greater(height, width),
                        lambda: smallest_side / width,
                        lambda: smallest_side / height)
        new_height = tf.to_int32(height * scale)
        new_width = tf.to_int32(width * scale)

        resized_image = tf.image.resize_images(image, [new_height, new_width])  # (2)
        return resized_image, label

    # Preprocessing (for training)
    # (3) Take a random 224x224 crop to the scaled image
    # (4) Horizontally flip the image with probability 1/2
    # (5) Substract the per color mean `VGG_MEAN`
    # Note: we don't normalize the data here, as VGG was trained without normalization
    @staticmethod
    def training_preprocess(image, label):
        crop_image = tf.random_crop(image, [224, 224, 3])                       # (3)
        flip_image = tf.image.random_flip_left_right(crop_image)                # (4)

        means = tf.reshape(tf.constant(VGG_MEAN), [1, 1, 3])
        centered_image = flip_image - means                                     # (5)

        return centered_image, label

    # Preprocessing (for validation)
    # (3) Take a central 224x224 crop to the scaled image
    # (4) Substract the per color mean `VGG_MEAN`
    # Note: we don't normalize the data here, as VGG was trained without normalization
    @staticmethod
    def val_preprocess(image, label):
        crop_image = tf.image.resize_image_with_crop_or_pad(image, 224, 224)    # (3)

        means = tf.reshape(tf.constant(VGG_MEAN), [1, 1, 3])
        centered_image = crop_image - means                                     # (4)

        return centered_image, label

    @staticmethod
    def check_accuracy(sess, correct_prediction, is_training, dataset_init_op):
        """
        Check the accuracy of the model on either train or val (depending on dataset_init_op).
        """
        # Initialize the correct dataset
        sess.run(dataset_init_op)
        num_correct, num_samples = 0, 0
        while True:
            try:
                correct_pred = sess.run(correct_prediction, {is_training: False})
                num_correct += correct_pred.sum()
                num_samples += correct_pred.shape[0]
            except tf.errors.OutOfRangeError:
                break

        # Return the fraction of datapoints that were correctly classified
        acc = float(num_correct) / num_samples
        return acc

    # @staticmethod
    # def __set_image_from_path(path):
    #     img = Image.open(path)
    #     img.load()
    #     new_img = np.asarray(img, dtype='int32')
    #
    #     for i, j in enumerate(new_img):
    #         print(i, j)

    def transfer_learning(self):
        print(1325)
        exit(-1)
        # Get the list of filenames and corresponding list of labels for training et validation
        train_filenames, train_labels = self.list_images(args.train_dir)
        val_filenames, val_labels = self.list_images(args.val_dir)
        assert set(train_labels) == set(val_labels), \
            "Train and val labels don't correspond:\n{}\n{}".format(set(train_labels),
                                                                    set(val_labels))

        num_classes = len(set(train_labels))

        # define the computation graph with all the necessary operations: loss, training op, accuracy...
        graph = tf.Graph()
        with graph.as_default():
            # ----------------------------------------------------------------------
            # DATASET CREATION using tf.contrib.data.Dataset
            # https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/data

            # The tf.contrib.data.Dataset framework uses queues in the background to feed in
            # data to the model.
            # We initialize the dataset with a list of filenames and labels, and then apply
            # the preprocessing functions described above.
            # Behind the scenes, queues will load the filenames, preprocess them with multiple
            # threads and apply the preprocessing in parallel, and then batch the data

            # Training dataset
            train_dataset = tf.data.Dataset.from_tensor_slices((train_filenames, train_labels))
            train_dataset = train_dataset.map(self.parse_function)
            train_dataset = train_dataset.map(self.training_preprocess)
            train_dataset = train_dataset.shuffle(buffer_size=10000)  # don't forget to shuffle
            batched_train_dataset = train_dataset.batch(args.batch_size)

            # Validation dataset
            val_dataset = tf.data.Dataset.from_tensor_slices((val_filenames, val_labels))
            val_dataset = val_dataset.map(self.parse_function)
            val_dataset = val_dataset.map(self.val_preprocess)
            batched_val_dataset = val_dataset.batch(args.batch_size)

            # Now we define an iterator that can operator on either dataset.
            # The iterator can be reinitialized by calling:
            #     - sess.run(train_init_op) for 1 epoch on the training set
            #     - sess.run(val_init_op)   for 1 epoch on the valiation set
            # Once this is done, we don't need to feed any value for images and labels
            # as they are automatically pulled out from the iterator queues.

            # A reinitializable iterator is defined by its structure. We could use the
            # `output_types` and `output_shapes` properties of either `train_dataset`
            # or `validation_dataset` here, because they are compatible.

            iterator = tf.data.Iterator.from_structure(batched_train_dataset.output_types,
                                                       batched_train_dataset.output_shapes)
            images, labels = iterator.get_next()

            train_init_op = iterator.make_initializer(batched_train_dataset)
            val_init_op = iterator.make_initializer(batched_val_dataset)

            # Indicates whether we are in training or in test mode
            is_training = tf.placeholder(tf.bool)

            # ---------------------------------------------------------------------
            # Now that we have set up the data, it's time to set up the model.
            # For this example, we'll use VGG-16 pretrained on ImageNet. We will remove the
            # last fully connected layer (fc8) and replace it with our own, with an
            # output size num_classes=8
            # We will first train the last layer for a few epochs.
            # Then we will train the entire model on our dataset for a few epochs.

            # Get the pretrained model, specifying the num_classes argument to create a new
            # fully connected replacing the last one, called "vgg_16/fc8"
            # Each model has a different architecture, so "vgg_16/fc8" will change in another model.
            # Here, logits gives us directly the predicted scores we wanted from the images.
            # We pass a scope to initialize "vgg_16/fc8" weights with he_initializer
            vgg = tf.contrib.slim.nets.vgg
            with slim.arg_scope(vgg.vgg_arg_scope(weight_decay=args.weight_decay)):
                logits, _ = vgg.vgg_16(images, num_classes=num_classes, is_training=is_training,
                                       dropout_keep_prob=args.dropout_keep_prob)

            # Specify where the model checkpoint is (pretrained weights).
            model_path = args.model_path
            assert (os.path.isfile(model_path))

            # Restore only the layers up to fc7 (included)
            # Calling function `init_fn(sess)` will load all the pretrained weights.
            variables_to_restore = tf.contrib.framework.get_variables_to_restore(exclude=['vgg_16/fc8'])
            init_fn = tf.contrib.framework.assign_from_checkpoint_fn(model_path, variables_to_restore)

            # Initialization operation from scratch for the new "fc8" layers
            # `get_variables` will only return the variables whose name starts with the given pattern
            fc8_variables = tf.contrib.framework.get_variables('vgg_16/fc8')
            fc8_init = tf.variables_initializer(fc8_variables)

            # ---------------------------------------------------------------------
            # Using tf.losses, any loss is added to the tf.GraphKeys.LOSSES collection
            # We can then call the total loss easily
            tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
            loss = tf.losses.get_total_loss()

            # First we want to train only the reinitialized last layer fc8 for a few epochs.
            # We run minimize the loss only with respect to the fc8 variables (weight and bias).
            fc8_optimizer = tf.train.GradientDescentOptimizer(args.learning_rate1)
            fc8_train_op = fc8_optimizer.minimize(loss, var_list=fc8_variables)

            # Then we want to finetune the entire model for a few epochs.
            # We run minimize the loss only with respect to all the variables.
            full_optimizer = tf.train.GradientDescentOptimizer(args.learning_rate2)
            full_train_op = full_optimizer.minimize(loss)

            # Evaluation metrics
            prediction = tf.to_int32(tf.argmax(logits, 1))
            correct_prediction = tf.equal(prediction, labels)
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            tf.get_default_graph().finalize()

        # --------------------------------------------------------------------------
        # Now that we have built the graph and finalized it, we define the session.
        # The session is the interface to *run* the computational graph.
        # We can call our training operations with `sess.run(train_op)` for instance
        with tf.Session(graph=graph) as sess:
            init_fn(sess)  # load the pretrained weights
            sess.run(fc8_init)  # initialize the new fc8 layer

            merged_summary = tf.summary.merge_all()
            train_writer = tf.summary.FileWriter(self.name_of_log + "/train", sess.graph)
            val_writer = tf.summary.FileWriter(self.name_of_log + "/val", sess.graph)
            # # Update only the last layer for a few epochs.
            # for epoch in range(args.num_epochs1):
            #     # Run an epoch over the training data.
            #     print('Starting epoch1 %d / %d' % (epoch + 1, args.num_epochs1))
            #     # Here we initialize the iterator with the training set.
            #     # This means that we can go through an entire epoch until the iterator becomes empty.
            #     sess.run(train_init_op)
            #     while True:
            #         try:
            #             _ = sess.run(fc8_train_op, {is_training: True})
            #         except tf.errors.OutOfRangeError:
            #             break
            #
            #     # Check accuracy on the train and val sets every epoch.
            #     train_acc = self.check_accuracy(sess, correct_prediction, is_training, train_init_op)
            #     val_acc = self.check_accuracy(sess, correct_prediction, is_training, val_init_op)
            #     print('Train accuracy: %f' % train_acc)
            #     print('Val accuracy: %f\n' % val_acc)

            # Train the entire model for a few more epochs, continuing with the *same* weights.
            for epoch in range(args.num_epochs2):
                print('Starting epoch2 %d / %d' % (epoch + 1, args.num_epochs2))
                sess.run(train_init_op)
                while True:
                    try:
                        _ = sess.run(full_train_op, {is_training: True})
                    except tf.errors.OutOfRangeError:
                        break

                # Check accuracy on the train and val sets every epoch
                train_acc = self.check_accuracy(sess, correct_prediction, is_training, train_init_op)
                val_acc = self.check_accuracy(sess, correct_prediction, is_training, val_init_op)
                print('Train accuracy: %f' % train_acc)
                print('Val accuracy: %f\n' % val_acc)

    # def transfer_learning(self):
    #     # Get the list of filenames and corresponding list of labels for training et validation
    #     train_filenames, train_labels = self.list_images(args.train_dir)
    #     val_filenames, val_labels = self.list_images(args.val_dir)
    #     assert set(train_labels) == set(val_labels), \
    #         "Train and val labels don't correspond:\n{}\n{}".format(set(train_labels),
    #                                                                 set(val_labels))
    #
    #     num_classes = len(set(train_labels))
    #
    #     # --------------------------------------------------------------------------
    #     # In TensorFlow, you first want to define the computation graph with all the
    #     # necessary operations: loss, training op, accuracy...
    #     # Any tensor created in the `graph.as_default()` scope will be part of `graph`
    #     graph = tf.Graph()
    #     with graph.as_default():
    #         # ----------------------------------------------------------------------
    #         # DATASET CREATION using tf.contrib.data.Dataset
    #         # https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/data
    #
    #         # The tf.contrib.data.Dataset framework uses queues in the background to feed in
    #         # data to the model.
    #         # We initialize the dataset with a list of filenames and labels, and then apply
    #         # the preprocessing functions described above.
    #         # Behind the scenes, queues will load the filenames, preprocess them with multiple
    #         # threads and apply the preprocessing in parallel, and then batch the data
    #
    #         # # Training dataset
    #         # train_dataset = tf.contrib.data.Dataset.from_tensor_slices((train_filenames, train_labels))
    #         # train_dataset = train_dataset.map(_parse_function,
    #         #                                   num_threads=args.num_workers, output_buffer_size=args.batch_size)
    #         # train_dataset = train_dataset.map(training_preprocess,
    #         #                                   num_threads=args.num_workers, output_buffer_size=args.batch_size)
    #         # train_dataset = train_dataset.shuffle(buffer_size=10000)  # don't forget to shuffle
    #         # batched_train_dataset = train_dataset.batch(args.batch_size)
    #         #
    #         # # Validation dataset
    #         # val_dataset = tf.contrib.data.Dataset.from_tensor_slices((val_filenames, val_labels))
    #         # val_dataset = val_dataset.map(_parse_function,
    #         #                               num_threads=args.num_workers, output_buffer_size=args.batch_size)
    #         # val_dataset = val_dataset.map(val_preprocess,
    #         #                               num_threads=args.num_workers, output_buffer_size=args.batch_size)
    #         # batched_val_dataset = val_dataset.batch(args.batch_size)
    #
    #         # Training dataset
    #         train_dataset = tf.data.Dataset.from_tensor_slices((train_filenames, train_labels))
    #         train_dataset = train_dataset.map(self.parse_function)
    #         train_dataset = train_dataset.map(self.training_preprocess)
    #         train_dataset = train_dataset.shuffle(buffer_size=10000)  # don't forget to shuffle
    #         batched_train_dataset = train_dataset.batch(args.batch_size)
    #
    #         # Validation dataset
    #         val_dataset = tf.data.Dataset.from_tensor_slices((val_filenames, val_labels))
    #         val_dataset = val_dataset.map(self.parse_function)
    #         val_dataset = val_dataset.map(self.val_preprocess)
    #         batched_val_dataset = val_dataset.batch(args.batch_size)
    #
    #         # Now we define an iterator that can operator on either dataset.
    #         # The iterator can be reinitialized by calling:
    #         #     - sess.run(train_init_op) for 1 epoch on the training set
    #         #     - sess.run(val_init_op)   for 1 epoch on the valiation set
    #         # Once this is done, we don't need to feed any value for images and labels
    #         # as they are automatically pulled out from the iterator queues.
    #
    #         # A reinitializable iterator is defined by its structure. We could use the
    #         # `output_types` and `output_shapes` properties of either `train_dataset`
    #         # or `validation_dataset` here, because they are compatible.
    #
    #         # iterator = tf.contrib.data.Iterator.from_structure(batched_train_dataset.output_types,
    #         #                                                    batched_train_dataset.output_shapes)
    #
    #         iterator = tf.data.Iterator.from_structure(batched_train_dataset.output_types,
    #                                                    batched_train_dataset.output_shapes)
    #         images, labels = iterator.get_next()
    #
    #         train_init_op = iterator.make_initializer(batched_train_dataset)
    #         val_init_op = iterator.make_initializer(batched_val_dataset)
    #
    #         # Indicates whether we are in training or in test mode
    #         is_training = tf.placeholder(tf.bool)
    #
    #         # ---------------------------------------------------------------------
    #         # Now that we have set up the data, it's time to set up the model.
    #         # For this example, we'll use VGG-16 pretrained on ImageNet. We will remove the
    #         # last fully connected layer (fc8) and replace it with our own, with an
    #         # output size num_classes=8
    #         # We will first train the last layer for a few epochs.
    #         # Then we will train the entire model on our dataset for a few epochs.
    #
    #         # Get the pretrained model, specifying the num_classes argument to create a new
    #         # fully connected replacing the last one, called "vgg_16/fc8"
    #         # Each model has a different architecture, so "vgg_16/fc8" will change in another model.
    #         # Here, logits gives us directly the predicted scores we wanted from the images.
    #         # We pass a scope to initialize "vgg_16/fc8" weights with he_initializer
    #         vgg = tf.contrib.slim.nets.vgg
    #         with slim.arg_scope(vgg.vgg_arg_scope(weight_decay=args.weight_decay)):
    #             logits, _ = vgg.vgg_16(images, num_classes=num_classes, is_training=is_training,
    #                                    dropout_keep_prob=args.dropout_keep_prob)
    #
    #         # Specify where the model checkpoint is (pretrained weights).
    #         model_path = args.model_path
    #         assert (os.path.isfile(model_path))
    #
    #         # Restore only the layers up to fc7 (included)
    #         # Calling function `init_fn(sess)` will load all the pretrained weights.
    #         variables_to_restore = tf.contrib.framework.get_variables_to_restore(exclude=['vgg_16/fc8'])
    #         init_fn = tf.contrib.framework.assign_from_checkpoint_fn(model_path, variables_to_restore)
    #
    #         # Initialization operation from scratch for the new "fc8" layers
    #         # `get_variables` will only return the variables whose name starts with the given pattern
    #         fc8_variables = tf.contrib.framework.get_variables('vgg_16/fc8')
    #         fc8_init = tf.variables_initializer(fc8_variables)
    #
    #         # ---------------------------------------------------------------------
    #         # Using tf.losses, any loss is added to the tf.GraphKeys.LOSSES collection
    #         # We can then call the total loss easily
    #         tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    #         loss = tf.losses.get_total_loss()
    #
    #         # First we want to train only the reinitialized last layer fc8 for a few epochs.
    #         # We run minimize the loss only with respect to the fc8 variables (weight and bias).
    #         fc8_optimizer = tf.train.GradientDescentOptimizer(args.learning_rate1)
    #         fc8_train_op = fc8_optimizer.minimize(loss, var_list=fc8_variables)
    #
    #         # Then we want to finetune the entire model for a few epochs.
    #         # We run minimize the loss only with respect to all the variables.
    #         full_optimizer = tf.train.GradientDescentOptimizer(args.learning_rate2)
    #         full_train_op = full_optimizer.minimize(loss)
    #
    #         # Evaluation metrics
    #         prediction = tf.to_int32(tf.argmax(logits, 1))
    #         correct_prediction = tf.equal(prediction, labels)
    #         accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    #
    #         tf.get_default_graph().finalize()
    #
    #     # --------------------------------------------------------------------------
    #     # Now that we have built the graph and finalized it, we define the session.
    #     # The session is the interface to *run* the computational graph.
    #     # We can call our training operations with `sess.run(train_op)` for instance
    #     with tf.Session(graph=graph) as sess:
    #         init_fn(sess)  # load the pretrained weights
    #         sess.run(fc8_init)  # initialize the new fc8 layer
    #
    #         # Update only the last layer for a few epochs.
    #         for epoch in range(args.num_epochs1):
    #             # Run an epoch over the training data.
    #             print('Starting epoch %d / %d' % (epoch + 1, args.num_epochs1))
    #             # Here we initialize the iterator with the training set.
    #             # This means that we can go through an entire epoch until the iterator becomes empty.
    #             sess.run(train_init_op)
    #             while True:
    #                 try:
    #                     _ = sess.run(fc8_train_op, {is_training: True})
    #                 except tf.errors.OutOfRangeError:
    #                     break
    #
    #             # Check accuracy on the train and val sets every epoch.
    #             train_acc = self.check_accuracy(sess, correct_prediction, is_training, train_init_op)
    #             val_acc = self.check_accuracy(sess, correct_prediction, is_training, val_init_op)
    #             print('Train accuracy: %f' % train_acc)
    #             print('Val accuracy: %f\n' % val_acc)
    #
    #         # Train the entire model for a few more epochs, continuing with the *same* weights.
    #         for epoch in range(args.num_epochs2):
    #             print('Starting epoch %d / %d' % (epoch + 1, args.num_epochs2))
    #             sess.run(train_init_op)
    #             while True:
    #                 try:
    #                     _ = sess.run(full_train_op, {is_training: True})
    #                 except tf.errors.OutOfRangeError:
    #                     break
    #
    #             # Check accuracy on the train and val sets every epoch
    #             train_acc = self.check_accuracy(sess, correct_prediction, is_training, train_init_op)
    #             val_acc = self.check_accuracy(sess, correct_prediction, is_training, val_init_op)
    #             print('Train accuracy: %f' % train_acc)
    #             print('Val accuracy: %f\n' % val_acc)

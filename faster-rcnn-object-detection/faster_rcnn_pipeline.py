import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

print("TensorFlow version:", tf.__version__)

# Load CIFAR-10 dataset

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

print("Train shape:", x_train.shape)
print("Test shape:", x_test.shape)

# Normalize pixel values (0–255 → 0–1)

x_train = x_train.astype("float32") / 255.0
x_test  = x_test.astype("float32") / 255.0

print("Normalization done ✅")

class_names = ['airplane','automobile','bird','cat','deer',
               'dog','frog','horse','ship','truck']

plt.figure(figsize=(8,4))

for i in range(8):
    plt.subplot(2,4,i+1)
    plt.imshow(x_train[i])
    plt.title(class_names[int(y_train[i])])
    plt.axis("off")

plt.show()

def build_backbone():
    """
    This CNN will act as feature extractor (Backbone).
    Output will be feature maps instead of final classification.
    """

    inputs = keras.Input(shape=(32,32,3))

    x = layers.Conv2D(32, (3,3), padding='same', activation='relu')(inputs)
    x = layers.MaxPooling2D((2,2))(x)

    x = layers.Conv2D(64, (3,3), padding='same', activation='relu')(x)
    x = layers.MaxPooling2D((2,2))(x)

    x = layers.Conv2D(128, (3,3), padding='same', activation='relu')(x)

    model = keras.Model(inputs, x, name="Backbone_CNN")

    return model


backbone = build_backbone()
backbone.summary()

# Pass one image through backbone

sample_image = x_train[:1]   # take 1 image
feature_map = backbone(sample_image)

print("Feature map shape:", feature_map.shape)

def build_rpn(feature_map, num_anchors=3):
    """
    RPN takes feature map and predicts:
    - Objectness score
    - Bounding box regression values
    """

    # Shared convolution layer
    x = layers.Conv2D(256, (3,3), padding='same', activation='relu')(feature_map)

    # Objectness score (1 value per anchor)
    objectness = layers.Conv2D(num_anchors, (1,1), activation='sigmoid', name="rpn_objectness")(x)

    # Bounding box regression (4 values per anchor)
    bbox_regression = layers.Conv2D(num_anchors * 4, (1,1), name="rpn_bbox")(x)

    return objectness, bbox_regression

# Build full backbone + RPN pipeline

inputs = keras.Input(shape=(32,32,3))

# Backbone feature map
features = backbone(inputs)

# RPN outputs
objectness, bbox_reg = build_rpn(features)

rpn_model = keras.Model(inputs, [objectness, bbox_reg], name="Backbone_RPN")

rpn_model.summary()

# Test RPN output on one image

obj_out, bbox_out = rpn_model(x_train[:1])

print("Objectness shape:", obj_out.shape)
print("BBox regression shape:", bbox_out.shape)

def generate_proposals(batch_size=1, num_proposals=16):
    """
    Generate random region proposals (normalized coordinates)
    Format: [y1, x1, y2, x2] between 0 and 1
    """

    proposals = tf.random.uniform(
        (batch_size, num_proposals, 4),
        minval=0,
        maxval=1
    )

    # Ensure y2 > y1 and x2 > x1
    y1 = tf.minimum(proposals[:,:,0], proposals[:,:,2])
    y2 = tf.maximum(proposals[:,:,0], proposals[:,:,2])
    x1 = tf.minimum(proposals[:,:,1], proposals[:,:,3])
    x2 = tf.maximum(proposals[:,:,1], proposals[:,:,3])

    proposals = tf.stack([y1, x1, y2, x2], axis=-1)

    return proposals

def roi_align(feature_map, proposals, pool_size=(7,7)):
    """
    ROI Align using tf.image.crop_and_resize
    """

    batch_size = tf.shape(feature_map)[0]
    num_rois = tf.shape(proposals)[1]

    pooled_features = []

    for i in range(batch_size):
        boxes = proposals[i]
        box_indices = tf.zeros((num_rois,), dtype=tf.int32)

        pooled = tf.image.crop_and_resize(
            feature_map[i:i+1],
            boxes,
            box_indices,
            pool_size
        )

        pooled_features.append(pooled)

    return tf.concat(pooled_features, axis=0)

# Get feature map from backbone
features = backbone(x_train[:1])

# Generate proposals
proposals = generate_proposals(batch_size=1, num_proposals=16)

# ROI Align
roi_features = roi_align(features, proposals)

print("ROI features shape:", roi_features.shape)

def detection_head(roi_features, num_classes=10):
    """
    Takes ROI features and predicts:
    - Class scores
    - Bounding box regression
    """

    x = layers.Flatten()(roi_features)

    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)

    # Classification output
    class_logits = layers.Dense(num_classes, name="class_logits")(x)

    # Bounding box output (4 values per ROI)
    bbox_deltas = layers.Dense(4, name="bbox_deltas")(x)

    return class_logits, bbox_deltas

# Run detection head on ROI features

class_out, bbox_out = detection_head(roi_features)

print("Class output shape:", class_out.shape)
print("BBox output shape:", bbox_out.shape)

# -------------------------------------------------------
# Proposal Generator Layer
# Generates random region proposals (for demo purposes)
# In real Faster R-CNN, proposals come from trained RPN
# -------------------------------------------------------

class ProposalLayer(layers.Layer):
    def __init__(self, num_proposals=16):
        super().__init__()
        self.num_proposals = num_proposals

    def call(self, feature_map):

        batch_size = tf.shape(feature_map)[0]

        # Generate random bounding boxes (normalized 0–1)
        proposals = tf.random.uniform(
            (batch_size, self.num_proposals, 4),
            0, 1
        )

        # Ensure y2 > y1 and x2 > x1

# -------------------------------------------------------
# ROI Align Layer
# Extracts fixed-size feature maps (7x7) for each proposal
# -------------------------------------------------------

class ROILayer(layers.Layer):
    def __init__(self, pool_size=(7,7)):
        super().__init__()
        self.pool_size = pool_size

    def call(self, inputs):

        feature_map, proposals = inputs

        # Remove batch dimension (since batch=1)
        boxes = proposals[0]

        num_rois = tf.shape(boxes)[0]
        box_indices = tf.zeros((num_rois,), dtype=tf.int32)

        pooled = tf.image.crop_and_resize(
            feature_map,
            boxes,
            box_indices,
            self.pool_size
        )

        return pooled

def build_faster_rcnn(num_classes=10):

    inputs = keras.Input(shape=(32,32,3))

    # Backbone
    feature_map = backbone(inputs)

    # RPN
    obj_score, bbox_reg = build_rpn(feature_map)

    # Detection head directly on feature map (educational shortcut)
    x = layers.GlobalAveragePooling2D()(feature_map)
    x = layers.Dense(256, activation='relu')(x)

    class_logits = layers.Dense(num_classes, name="class_logits")(x)
    bbox_deltas = layers.Dense(4, name="bbox_deltas")(x)

    model = keras.Model(inputs, [obj_score, bbox_reg, class_logits, bbox_deltas])

    return model


faster_rcnn_model = build_faster_rcnn()
faster_rcnn_model.summary()

# Take one test image
test_img = x_test[0:1]

# Run model
obj_score, bbox_reg, class_logits, bbox_deltas = faster_rcnn_model.predict(test_img)

print("Objectness shape:", obj_score.shape)
print("BBox reg shape:", bbox_reg.shape)
print("Class logits shape:", class_logits.shape)
print("BBox deltas shape:", bbox_deltas.shape)

# Get predicted class
pred_class = np.argmax(class_logits)

print("Predicted class:", class_names[pred_class])

# Create demo bounding box (centered)
h, w, _ = x_test[0].shape

x1, y1 = int(w*0.25), int(h*0.25)
x2, y2 = int(w*0.75), int(h*0.75)

img = x_test[0]

plt.figure(figsize=(4,4))
plt.imshow(img)
plt.axis("off")

# Draw bounding box
plt.gca().add_patch(
    plt.Rectangle((x1,y1), x2-x1, y2-y1,
                  fill=False, edgecolor='red', linewidth=2)
)

plt.title(f"Predicted: {class_names[pred_class]}")
plt.show()

plt.figure(figsize=(8,8))

for i in range(4):

    test_img = x_test[i:i+1]

    obj, bbox, cls, box = faster_rcnn_model.predict(test_img)

    pred_class = np.argmax(cls)

    img = x_test[i]

    h,w,_ = img.shape
    x1,y1 = int(w*0.25), int(h*0.25)
    x2,y2 = int(w*0.75), int(h*0.75)

    plt.subplot(2,2,i+1)
    plt.imshow(img)
    plt.axis("off")

    plt.gca().add_patch(
        plt.Rectangle((x1,y1), x2-x1, y2-y1,
                      fill=False, edgecolor='red', linewidth=2)
    )

    plt.title(class_names[pred_class])

plt.show()

"""Now we will test on the video"""

import cv2

import matplotlib.pyplot as plt
from IPython.display import clear_output
import time
#done on the airport video
video_path = "/content/12908023_3840_2160_30fps.mp4"   # change if your filename is different
cap = cv2.VideoCapture(video_path)

plt.figure(figsize=(5,5))

while cap.isOpened():

    ret, frame = cap.read()
    if not ret:
        break

    # Convert BGR → RGB for matplotlib
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Resize frame for CIFAR model
    resized = cv2.resize(frame_rgb, (32,32))
    img = resized.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    # Model prediction
    obj, bbox, cls, box = faster_rcnn_model.predict(img, verbose=0)
    pred_class = np.argmax(cls)
    label = class_names[pred_class]

    # Demo bounding box (center)
    h, w, _ = frame_rgb.shape
    x1, y1 = int(w*0.25), int(h*0.25)
    x2, y2 = int(w*0.75), int(h*0.75)

    # Draw box using matplotlib
    clear_output(wait=True)
    plt.imshow(frame_rgb)
    plt.axis("off")

    plt.gca().add_patch(
        plt.Rectangle((x1,y1), x2-x1, y2-y1,
                      fill=False, edgecolor='red', linewidth=2)
    )

    plt.title(f"Predicted: {label}")
    plt.show()

    time.sleep(0.03)   # controls playback speed

cap.release()


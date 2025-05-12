

PROJECT_NAME = "Remote Sensing Change Detection"
RANDOM_SEED = 42


DATA_PATH = " "         # Folder with training images
VAL_DATA_PATH = ""       # Folder with validation images
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
NUM_CHANNELS = 6
NUM_CLASSES = 1

INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS)


BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 1e-4
LOSS_FUNCTION = "combo_loss"
OPTIMIZER = "adam"


MODEL_SAVE_PATH = "checkpoints/best_model.keras"
LOG_CSV_PATH = "logs/training_log.csv"
EARLY_STOPPING_PATIENCE = 10
LR_REDUCE_PATIENCE = 5
LR_REDUCE_FACTOR = 0.5
MONITOR_METRIC = "val_iou_metric"
MONITOR_MODE = "max"


TEST_DATA_PATH = "data/test_images/"
EVAL_BATCH_SIZE = 8
SAVE_PREDICTIONS_PATH = "outputs/predictions/"

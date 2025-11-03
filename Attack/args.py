

ATTACK_BATCH_SIZE=8
ATTACK_EPOCHS=50
ATTACK_LR=1e-3
#CROP_SIZE=128 # for crop training defense

LAMBDA=0.005 # weight for adversarial regularisation
REG_EPOCHS=1 # number of epochs for optimising adversarial regularisation

MAX_GRAD_NORM=2.0

OUTPUT_CHANNELS=2# 9 for Duke dataset and 2 for UMN


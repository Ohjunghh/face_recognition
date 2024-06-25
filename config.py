BATCH_SIZE = 512
SAVE_FREQ = 1
TEST_FREQ = 1
TOTAL_EPOCH = 70

RESUME = ''
SAVE_DIR = './model'
MODEL_PRE = 'CASIA_B512_'


CASIA_DATA_DIR = '/home/xiaocc/Documents/caffe_project/sphereface/train/data'
LFW_DATA_DIR = '/home/xiaocc/Documents/caffe_project/sphereface/test/data'

# 3단계: 데이터 전처리
# MobileFaceNet은 SphereFace 전처리 방법을 따릅니다. CASIA-WebFace와 LFW 데이터셋의 정렬된 이미지를 다운로드합니다.

# Align-CASIA-WebFace@BaiduDrive
# Align-LFW@BaiduDrive
# 다운로드한 데이터셋을 config.py 파일에서 지정된 경로로 설정합니다.

GPU = 0, 1


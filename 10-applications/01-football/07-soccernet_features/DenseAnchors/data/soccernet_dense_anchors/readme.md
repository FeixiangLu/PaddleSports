# Copy source files and config files
    
    rsync -abviuzP paddlevideo/ $PADDLEVIDEO_SOURCE_FOLDER/

    rsync config or data files

# Prepare training with class and event time labels

Generate label_mapping.txt (for category to category index map) and dense.list files.

    python data/soccernet_dense_anchors/generate_dense_anchors_labels.py \
    --clips_folder /mnt/storage/gait-0/xin/dataset/soccernet_456x256 \
    --output_folder ./

Split into train, val, test

    python data/soccernet/split_annotation_into_train_val_test.py \
    --annotation_file dense.list \
    --clips_folder ./ \
    --mode json

# Inference on whole video files

## Convert video input into lower resolution

This generates a sample script that converts all of the Soccernet videos.

    python data/soccernet_inference/convert_video_to_lower_resolution_for_inference.py \
    --input_folder /mnt/big/multimodal_sports/SoccerNet_HQ/raw_data \
    --output_folder /mnt/storage/gait-0/xin/dataset/soccernet_456x256_inference > \
    data/soccernet_inference/convert_video_to_lower_resolution_for_inference.sh

## Parallelize resolution conversion

Each 45 min video files takes about 10 min to convert to lower resolution. So we parallelize to 100 such jobs.

    for i in {0..99};
    do
    sed -n ${i}~100p data/soccernet_inference/convert_video_to_lower_resolution_for_inference.sh > data/soccernet_inference/convert_video_to_lower_resolution_for_inference_parallel/${i}.sh;
    done

Run the parallel jobs on a cluster, slurm based for example.

    for i in {0..99};
    do
    sbatch -p 1080Ti,2080Ti,TitanXx8  --gres=gpu:1 --cpus-per-task 4 -n 1 --wrap \
    "echo no | bash data/soccernet_inference/convert_video_to_lower_resolution_for_inference_parallel/${i}.sh" \
    --output="data/soccernet_inference/convert_video_to_lower_resolution_for_inference_parallel/${i}.log"
    done

# Train command

    python -u -B -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" --log_dir=logs/soccernet_videoswin_k400_dense_lr_0.001_sgd_60 main.py --validate -c data/soccernet/experiments/soccernet_videoswin_k400_dense_lr_0.001_sgd_60.yaml -w pretrained_weights/swin_base_patch4_window7_224.pdparams

## Generate inference json job files

    python data/soccernet_dense_anchors/generate_whole_video_inference_jsons.py \
    --videos_folder /mnt/storage/gait-0/xin/dataset/soccernet_456x256_inference \
    --output_folder /mnt/storage/gait-0/xin/dataset/soccernet_456x256_inference_json_lists

## Sample inference command

    INFERENCE_WEIGHT_FILE=output/ppTimeSformer_dense_event_lr_100/ppTimeSformer_dense_event_lr_100_epoch_00007.pdparams
    INFERENCE_JSON_CONFIG=/mnt/storage/gait-0/xin/dataset/soccernet_456x256_inference_json_lists/spain_laliga.2016-2017.2017-05-21_-_21-00_Malaga_0_-_2_Real_Madrid.2_LQ.mkv
    INFERENCE_DIR_ROOT=/mnt/storage/gait-0/xin/soccernet_features
    SHORTNAME=`basename "$INFERENCE_JSON_CONFIG" .mkv`
    INFERENCE_DIR=$INFERENCE_DIR_ROOT/$SHORTNAME
    echo $INFERENCE_DIR

    mkdir $INFERENCE_DIR

    python3.7 -B -m paddle.distributed.launch --gpus="0" --log_dir=log_videoswin_test  main.py  --test -c data/soccernet_inference/soccernet_pptimesformer_k400_videos_dense_event_lr_50_one_file_inference.yaml -w $INFERENCE_WEIGHT_FILE -o inference_dir=$INFERENCE_DIR -o DATASET.test.file_path=$INFERENCE_JSON_CONFIG 

# List of changed files and corresponding changes.

- Label files processing are changed and labels of category and event_times are composed into dicts to send into the pipeline. Class names are added into the init.
    
        paddlevideo/loader/dataset/video_dense_anchors.py

        paddlevideo/loader/dataset/__init__.py

    Added temporal coordinate embedding to inputs. Removed event time loss for background class. Added parser for one video file list.

        paddlevideo/loader/dataset/video_dense_anchors_one_file_inference.py

- Added EventSampler

        paddlevideo/loader/pipelines/sample.py

        paddlevideo/loader/pipelines/__init__.py

    Added sampling one whole video file.

        paddlevideo/loader/pipelines/sample_one_file.py
    
    Added decoder for just one file 

        paddlevideo/loader/pipelines/decode.py

- Multitask losses.

        paddlevideo/modeling/losses/dense_anchor_loss.py
        
        paddlevideo/modeling/losses/__init__.py

- Changed head output. Class and event times.

        paddlevideo/modeling/heads/i3d_anchor_head.py

        paddlevideo/modeling/heads/pptimesformer_anchor_head.py

        paddlevideo/modeling/heads/__init__.py

- Input and output format in train_step, val step etc.

        paddlevideo/modeling/framework/recognizers/recognizer_transformer_features_inference.py

        paddlevideo/modeling/framework/recognizers/recognizer_transformer_dense_anchors.py
        
        paddlevideo/modeling/framework/recognizers/__init__.py

- Add a new mode to log both class loss and event time loss.

        paddlevideo/utils/record.py

- Added MODEL.head.name and MODEL.head.output_mode branch to process outputs of class scores and event_times. Also unified feature inference with simple classification mode.

        paddlevideo/tasks/test.py

- Lower generate lower resolution script.

        data/soccernet_inference/convert_video_to_lower_resolution_for_inference.py

- Balanced samples do not seem necessary 
    
        data/soccernet_dense_anchors/balance_class_samples.py

- Collate file to replace the current library file

        /mnt/home/xin/.conda/envs/paddle_soccernet_feature_extraction/lib/python3.7/site-packages/paddle/fluid/dataloader/collate.py

- Config files

        data/soccernet/soccernet_videoswin_k400_dense_one_file_inference.yaml

- Updated to support dense anchors

        data/soccernet/split_annotation_into_train_val_test.py

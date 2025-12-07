# nohup bash all.sh> all.log 2>&1 &
#!/bin/bash

# 定義數據集的順序
datasets=("dbpedia" "amazon" "yahoo" "agnews")

# 定義數據集與任務的映射
declare -A dataset_task_map
dataset_task_map["dbpedia"]="TC"
dataset_task_map["amazon"]="SC"
dataset_task_map["yahoo"]="TC"
dataset_task_map["agnews"]="TC"

# 基礎模型路徑
base_model="./initial_model/t5-large"

# 模型保存目錄
save_dir="./saved_models"

# 創建模型保存目錄（如果不存在）
mkdir -p $save_dir

for seed in 438 689 251; do
    echo "使用 seed: $seed"
    
    # 初始化模型路徑為空（首次訓練不使用預訓練模型）
    model_path=""
    first_run=true

    # 初始化測試數據文件和標籤文件的數組（每個 seed 可視情況重置或保留）
    test_data_files=()
    test_labels_files=()

    # 遍歷每個數據集進行訓練和評估
    for dataset in "${datasets[@]}"; do
        task=${dataset_task_map[$dataset]}
        echo "==============================="
        echo "訓練數據集: $dataset (任務: $task)"
        echo "==============================="

        # 定義訓練數據和標籤文件的路徑
        data_file="./CL_Benchmark/$task/$dataset/train.json"
        labels_file="./CL_Benchmark/$task/$dataset/labels.json"

        # 定義驗證數據和標籤文件的路徑
        eval_file="./CL_Benchmark/$task/$dataset/dev.json"
        eval_labels="./CL_Benchmark/$task/$dataset/labels.json"

        # 定義測試數據和標籤文件的路徑
        test_data="./CL_Benchmark/$task/$dataset/test.json"
        test_labels="./CL_Benchmark/$task/$dataset/labels.json"

        # 定義模型保存的輸出目錄
        output_dir="$save_dir/$dataset"
        mkdir -p $output_dir

        # 將測試數據和標籤文件添加到數組中
        test_data_files+=("$test_data")
        test_labels_files+=("$test_labels")

        echo "開始訓練 $dataset with seed $seed..."
        if [ "$first_run" = true ]; then
            # 首次訓練，不傳遞 --model_path
            python main.py \
                --data_file "$data_file" \
                --labels_file "$labels_file" \
                --output_dir "$output_dir" \
                --eval_file "$eval_file" \
                --eval_labels_files "$eval_labels" \
                --test_data_files "${test_data_files[@]}" \
                --test_labels_files "${test_labels_files[@]}" \
                --seed $seed
            first_run=false
        else
            # 後續訓練，傳遞 --model_path
            python main.py \
                --data_file "$data_file" \
                --labels_file "$labels_file" \
                --model_path "$model_path" \
                --output_dir "$output_dir" \
                --eval_file "$eval_file" \
                --eval_labels_files "$eval_labels" \
                --test_data_files "${test_data_files[@]}" \
                --test_labels_files "${test_labels_files[@]}" \
                --seed $seed
        fi

        echo "訓練完成: $dataset with seed $seed"

        # 更新模型路徑為最新的模型
        model_path="$output_dir"

        # 等待片刻以確保日誌寫入
        sleep 2
    done

    echo "Seed $seed 的所有數據集訓練和評估已完成。"
done

echo "order1 所有實驗已完成。"

# 定義數據集的順序
datasets=("dbpedia" "amazon" "agnews" "yahoo")

# 定義數據集與任務的映射
declare -A dataset_task_map
dataset_task_map["dbpedia"]="TC"
dataset_task_map["amazon"]="SC"
dataset_task_map["yahoo"]="TC"
dataset_task_map["agnews"]="TC"

# 基礎模型路徑
base_model="./initial_model/t5-large"

# 模型保存目錄
save_dir="./saved_models"

# 創建模型保存目錄（如果不存在）
mkdir -p $save_dir

for seed in 438 689 251; do
    echo "使用 seed: $seed"
    
    # 初始化模型路徑為空（首次訓練不使用預訓練模型）
    model_path=""
    first_run=true

    # 初始化測試數據文件和標籤文件的數組（每個 seed 可視情況重置或保留）
    test_data_files=()
    test_labels_files=()

    # 遍歷每個數據集進行訓練和評估
    for dataset in "${datasets[@]}"; do
        task=${dataset_task_map[$dataset]}
        echo "==============================="
        echo "訓練數據集: $dataset (任務: $task)"
        echo "==============================="

        # 定義訓練數據和標籤文件的路徑
        data_file="./CL_Benchmark/$task/$dataset/train.json"
        labels_file="./CL_Benchmark/$task/$dataset/labels.json"

        # 定義驗證數據和標籤文件的路徑
        eval_file="./CL_Benchmark/$task/$dataset/dev.json"
        eval_labels="./CL_Benchmark/$task/$dataset/labels.json"

        # 定義測試數據和標籤文件的路徑
        test_data="./CL_Benchmark/$task/$dataset/test.json"
        test_labels="./CL_Benchmark/$task/$dataset/labels.json"

        # 定義模型保存的輸出目錄
        output_dir="$save_dir/$dataset"
        mkdir -p $output_dir

        # 將測試數據和標籤文件添加到數組中
        test_data_files+=("$test_data")
        test_labels_files+=("$test_labels")

        echo "開始訓練 $dataset with seed $seed..."
        if [ "$first_run" = true ]; then
            # 首次訓練，不傳遞 --model_path
            python main.py \
                --data_file "$data_file" \
                --labels_file "$labels_file" \
                --output_dir "$output_dir" \
                --eval_file "$eval_file" \
                --eval_labels_files "$eval_labels" \
                --test_data_files "${test_data_files[@]}" \
                --test_labels_files "${test_labels_files[@]}" \
                --seed $seed
            first_run=false
        else
            # 後續訓練，傳遞 --model_path
            python main.py \
                --data_file "$data_file" \
                --labels_file "$labels_file" \
                --model_path "$model_path" \
                --output_dir "$output_dir" \
                --eval_file "$eval_file" \
                --eval_labels_files "$eval_labels" \
                --test_data_files "${test_data_files[@]}" \
                --test_labels_files "${test_labels_files[@]}" \
                --seed $seed
        fi

        echo "訓練完成: $dataset with seed $seed"

        # 更新模型路徑為最新的模型
        model_path="$output_dir"

        # 等待片刻以確保日誌寫入
        sleep 2
    done

    echo "Seed $seed 的所有數據集訓練和評估已完成。"
done

echo "order2 所有實驗已完成。"

# 定義數據集的順序
datasets=("yahoo" "amazon" "agnews" "dbpedia")

# 定義數據集與任務的映射
declare -A dataset_task_map
dataset_task_map["dbpedia"]="TC"
dataset_task_map["amazon"]="SC"
dataset_task_map["yahoo"]="TC"
dataset_task_map["agnews"]="TC"

# 基礎模型路徑
base_model="./initial_model/t5-large"

# 模型保存目錄
save_dir="./saved_models"

# 創建模型保存目錄（如果不存在）
mkdir -p $save_dir

for seed in 438 689 251; do
    echo "使用 seed: $seed"
    
    # 初始化模型路徑為空（首次訓練不使用預訓練模型）
    model_path=""
    first_run=true

    # 初始化測試數據文件和標籤文件的數組（每個 seed 可視情況重置或保留）
    test_data_files=()
    test_labels_files=()

    # 遍歷每個數據集進行訓練和評估
    for dataset in "${datasets[@]}"; do
        task=${dataset_task_map[$dataset]}
        echo "==============================="
        echo "訓練數據集: $dataset (任務: $task)"
        echo "==============================="

        # 定義訓練數據和標籤文件的路徑
        data_file="./CL_Benchmark/$task/$dataset/train.json"
        labels_file="./CL_Benchmark/$task/$dataset/labels.json"

        # 定義驗證數據和標籤文件的路徑
        eval_file="./CL_Benchmark/$task/$dataset/dev.json"
        eval_labels="./CL_Benchmark/$task/$dataset/labels.json"

        # 定義測試數據和標籤文件的路徑
        test_data="./CL_Benchmark/$task/$dataset/test.json"
        test_labels="./CL_Benchmark/$task/$dataset/labels.json"

        # 定義模型保存的輸出目錄
        output_dir="$save_dir/$dataset"
        mkdir -p $output_dir

        # 將測試數據和標籤文件添加到數組中
        test_data_files+=("$test_data")
        test_labels_files+=("$test_labels")

        echo "開始訓練 $dataset with seed $seed..."
        if [ "$first_run" = true ]; then
            # 首次訓練，不傳遞 --model_path
            python main.py \
                --data_file "$data_file" \
                --labels_file "$labels_file" \
                --output_dir "$output_dir" \
                --eval_file "$eval_file" \
                --eval_labels_files "$eval_labels" \
                --test_data_files "${test_data_files[@]}" \
                --test_labels_files "${test_labels_files[@]}" \
                --seed $seed
            first_run=false
        else
            # 後續訓練，傳遞 --model_path
            python main.py \
                --data_file "$data_file" \
                --labels_file "$labels_file" \
                --model_path "$model_path" \
                --output_dir "$output_dir" \
                --eval_file "$eval_file" \
                --eval_labels_files "$eval_labels" \
                --test_data_files "${test_data_files[@]}" \
                --test_labels_files "${test_labels_files[@]}" \
                --seed $seed
        fi

        echo "訓練完成: $dataset with seed $seed"

        # 更新模型路徑為最新的模型
        model_path="$output_dir"

        # 等待片刻以確保日誌寫入
        sleep 2
    done

    echo "Seed $seed 的所有數據集訓練和評估已完成。"
done

echo "order3 所有實驗已完成。"
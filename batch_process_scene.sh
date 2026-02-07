#!/bin/bash

# ElasticPanoramaDepthRefine 批量处理脚本
# 用法: ./batch_process_scene.sh <场景名称> [配置模板路径]
# 示例: c
#       ./batch_process_scene.sh BridgeB config.yaml

set -e  # 遇到错误立即退出
# 注意：算术表达式 ((...)) 在 set -e 下可能返回非零，需要特殊处理
set +e  # 暂时关闭，在循环中手动处理错误

# 检查参数
if [ $# -lt 1 ]; then
    echo "用法: $0 <场景名称> [配置模板路径]"
    echo "示例: $0 BridgeB"
    echo "      $0 BridgeB config.yaml"
    exit 1
fi

SCENE_NAME=$1
CONFIG_TEMPLATE=${2:-"config.yaml"}

# 获取脚本所在目录（项目根目录）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 检查配置模板是否存在
if [ ! -f "$CONFIG_TEMPLATE" ]; then
    echo "错误: 配置文件 '$CONFIG_TEMPLATE' 不存在"
    exit 1
fi

# 使用Python解析配置文件，提取路径信息
eval $(python3 << EOF
import yaml
import sys
import os
import re

scene_name = "$SCENE_NAME"

try:
    with open("$CONFIG_TEMPLATE", 'r') as f:
        config = yaml.safe_load(f)
    
    paths = config.get('paths', {})
    
    # 提取深度图路径
    depth_path = paths.get('depth_dap', '')
    if depth_path:
        # 从路径中提取基础目录（例如：/root/autodl-tmp/data/STAGE1_4x）
        depth_dir = os.path.dirname(depth_path)  # .../depth_npy
        depth_base = os.path.dirname(depth_dir)  # .../BridgeB
        depth_base_dir = os.path.dirname(depth_base)  # .../STAGE1_4x
        print(f"export DEPTH_BASE_DIR='{depth_base_dir}'")
    
    # 提取COLMAP路径
    anchor_gen = paths.get('anchor_generation', {})
    colmap_dir_path = anchor_gen.get('colmap_dir', '')
    if colmap_dir_path:
        # 提取到 colmap_STAGE1_4x 这一级（去掉场景名和sparse/0）
        # 例如: /root/autodl-tmp/data/colmap_STAGE1_4x/BridgeB/sparse/0
        # -> /root/autodl-tmp/data/colmap_STAGE1_4x
        parts = colmap_dir_path.split('/')
        # 找到 'colmap_STAGE1_4x' 的索引
        try:
            idx = parts.index('colmap_STAGE1_4x')
            colmap_base = '/'.join(parts[:idx+1])
        except ValueError:
            # 如果找不到，使用原来的方法
            colmap_base = os.path.dirname(os.path.dirname(colmap_dir_path))
        print(f"export COLMAP_BASE_DIR='{colmap_base}'")
    
    # 提取输出目录（需要根据场景名称替换）
    output_dir_template = paths.get('output_dir', '')
    if output_dir_template:
        # 替换场景名称部分
        # 例如: .../BridgeB/elastic_refined -> .../SCENE_NAME/elastic_refined
        output_dir = re.sub(r'/[^/]+/elastic_refined$', f'/{scene_name}/elastic_refined', output_dir_template)
        print(f"export OUTPUT_BASE_DIR='{output_dir}'")
    
    # 提取相机名称
    camera_name = anchor_gen.get('camera_name', 'pano_camera12')
    print(f"export CAMERA_NAME='{camera_name}'")
    
except Exception as e:
    print(f"# Error parsing config: {e}", file=sys.stderr)
    sys.exit(1)
EOF
)

# 如果Python解析失败，使用默认值
if [ -z "$DEPTH_BASE_DIR" ]; then
    DEPTH_BASE_DIR="/root/autodl-tmp/data/STAGE1_4x"
    echo "警告: 无法从配置中提取数据路径，使用默认路径: $DEPTH_BASE_DIR"
fi

if [ -z "$COLMAP_BASE_DIR" ]; then
    COLMAP_BASE_DIR="/root/autodl-tmp/data/colmap_STAGE1_4x"
    echo "警告: 无法从配置中提取COLMAP路径，使用默认路径: $COLMAP_BASE_DIR"
fi

if [ -z "$OUTPUT_BASE_DIR" ]; then
    OUTPUT_BASE_DIR="$DEPTH_BASE_DIR/$SCENE_NAME/elastic_refined"
    echo "警告: 无法从配置中提取输出路径，使用默认路径: $OUTPUT_BASE_DIR"
fi

if [ -z "$CAMERA_NAME" ]; then
    CAMERA_NAME="pano_camera12"
fi

# 构建路径
DEPTH_DIR="$DEPTH_BASE_DIR/$SCENE_NAME/depth_npy"
RGB_DIR="$DEPTH_BASE_DIR/$SCENE_NAME/backgrounds"

# 从配置文件中直接提取COLMAP相关路径，然后替换场景名称
eval $(python3 << EOF
import yaml
import re
import sys

scene_name = "$SCENE_NAME"

try:
    with open("$CONFIG_TEMPLATE", 'r') as f:
        config = yaml.safe_load(f)
    
    anchor_gen = config.get('paths', {}).get('anchor_generation', {})
    colmap_dir_template = anchor_gen.get('colmap_dir', '')
    fused_ply_template = anchor_gen.get('fused_ply', '')
    
    # 替换场景名称（假设路径中包含场景名）
    if colmap_dir_template:
        # 使用正则表达式替换场景名称部分
        # 例如: .../BridgeB/sparse/0 -> .../SCENE_NAME/sparse/0
        colmap_dir = re.sub(r'/[^/]+/sparse/0$', f'/{scene_name}/sparse/0', colmap_dir_template)
        print(f"export COLMAP_DIR='{colmap_dir}'")
    
    if fused_ply_template:
        # 替换场景名称
        fused_ply = re.sub(r'/[^/]+/fused\.ply$', f'/{scene_name}/fused.ply', fused_ply_template)
        print(f"export FUSED_PLY='{fused_ply}'")
    
except Exception as e:
    # 如果解析失败，使用拼接方式
    colmap_base = "$COLMAP_BASE_DIR"
    print(f"export COLMAP_DIR='{colmap_base}/{scene_name}/sparse/0'")
    print(f"export FUSED_PLY='{colmap_base}/{scene_name}/fused.ply'")
EOF
)

# 如果Python解析失败，使用拼接方式作为后备
if [ -z "$COLMAP_DIR" ]; then
    COLMAP_DIR="$COLMAP_BASE_DIR/$SCENE_NAME/sparse/0"
fi

if [ -z "$FUSED_PLY" ]; then
    FUSED_PLY="$COLMAP_BASE_DIR/$SCENE_NAME/fused.ply"
fi

# 检查目录是否存在
if [ ! -d "$DEPTH_DIR" ]; then
    echo "错误: 深度图目录不存在: $DEPTH_DIR"
    exit 1
fi

# 查找所有深度图文件
echo "=========================================="
echo "批量处理场景: $SCENE_NAME"
echo "=========================================="
echo "深度图目录: $DEPTH_DIR"
echo "RGB目录: $RGB_DIR"
echo "COLMAP目录: $COLMAP_DIR"
echo "输出目录: $OUTPUT_BASE_DIR"
echo ""

# 获取所有 .npy 文件（点位）
POINTS=($(find "$DEPTH_DIR" -name "*.npy" -type f | sort))

if [ ${#POINTS[@]} -eq 0 ]; then
    echo "错误: 在 $DEPTH_DIR 中未找到任何 .npy 文件"
    exit 1
fi

echo "找到 ${#POINTS[@]} 个点位:"
for point in "${POINTS[@]}"; do
    echo "  - $(basename "$point")"
done
echo ""

# 创建临时配置目录
TEMP_CONFIG_DIR=$(mktemp -d)
trap "rm -rf $TEMP_CONFIG_DIR" EXIT

# 处理每个点位
SUCCESS_COUNT=0
FAIL_COUNT=0
FAILED_POINTS=()

for depth_file in "${POINTS[@]}"; do
    POINT_NAME=$(basename "$depth_file" .npy)
    echo "=========================================="
    echo "处理点位: $POINT_NAME"
    echo "=========================================="
    
    # 构建文件路径
    RGB_FILE="$RGB_DIR/${POINT_NAME}.png"
    OUTPUT_DIR="$OUTPUT_BASE_DIR"
    
    # 检查RGB文件是否存在（可选）
    RGB_FILE_VALUE=""
    if [ -f "$RGB_FILE" ]; then
        RGB_FILE_VALUE="$RGB_FILE"
    else
        echo "警告: RGB文件不存在: $RGB_FILE (将使用None)"
    fi
    
    # 检查COLMAP目录和fused.ply是否存在
    if [ ! -d "$COLMAP_DIR" ]; then
        echo "错误: COLMAP目录不存在: $COLMAP_DIR"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        FAILED_POINTS+=("$POINT_NAME (COLMAP目录不存在)")
        continue
    fi
    
    if [ ! -f "$FUSED_PLY" ]; then
        echo "错误: fused.ply文件不存在: $FUSED_PLY"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        FAILED_POINTS+=("$POINT_NAME (fused.ply不存在)")
        continue
    fi
    
    # 为当前点位创建临时配置文件
    TEMP_CONFIG="$TEMP_CONFIG_DIR/${POINT_NAME}_config.yaml"
    
    # 使用Python生成配置文件（更可靠）
    python3 << EOF > "$TEMP_CONFIG"
import yaml
import sys

# 读取模板配置
with open("$CONFIG_TEMPLATE", 'r') as f:
    config = yaml.safe_load(f)

# 更新路径
config['paths']['depth_dap'] = "$depth_file"
if "$RGB_FILE_VALUE":
    config['paths']['rgb'] = "$RGB_FILE_VALUE"
else:
    config['paths']['rgb'] = None
config['paths']['output_dir'] = "$OUTPUT_DIR"
config['paths']['anchor_generation']['fused_ply'] = "$FUSED_PLY"
config['paths']['anchor_generation']['colmap_dir'] = "$COLMAP_DIR"
config['paths']['anchor_generation']['camera_name'] = "$CAMERA_NAME"

# 写入新配置
with open("$TEMP_CONFIG", 'w') as f:
    yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
EOF

    PYTHON_EXIT_CODE=$?
    if [ $PYTHON_EXIT_CODE -ne 0 ]; then
        echo "错误: 无法生成配置文件"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        FAILED_POINTS+=("$POINT_NAME (配置生成失败)")
        continue
    fi
    
    # 运行优化
    echo "运行优化..."
    # 捕获错误输出
    ERROR_OUTPUT=$(python3 main.py --config "$TEMP_CONFIG" 2>&1)
    PYTHON_EXIT_CODE=$?
    
    if [ $PYTHON_EXIT_CODE -eq 0 ]; then
        echo "✅ 点位 $POINT_NAME 处理成功"
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
        echo "❌ 点位 $POINT_NAME 处理失败"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        
        # 提取错误原因（从错误输出中提取最后一行错误信息）
        ERROR_REASON=$(echo "$ERROR_OUTPUT" | grep -E "(ValueError|FileNotFoundError|Error|错误)" | tail -1 | sed 's/^[[:space:]]*//' | cut -c1-80)
        if [ -z "$ERROR_REASON" ]; then
            ERROR_REASON="未知错误"
        fi
        
        # 将错误原因添加到失败点位信息中
        FAILED_POINTS+=("$POINT_NAME: $ERROR_REASON")
        
        # 显示简要错误信息
        echo "   错误: $ERROR_REASON"
    fi
    
    echo ""
done

# 输出总结
echo "=========================================="
echo "批量处理完成"
echo "=========================================="
echo "成功: $SUCCESS_COUNT / ${#POINTS[@]}"
echo "失败: $FAIL_COUNT / ${#POINTS[@]}"
echo ""

if [ $FAIL_COUNT -gt 0 ]; then
    echo "失败的点位及原因:"
    for point in "${FAILED_POINTS[@]}"; do
        echo "  - $point"
    done
    echo ""
    echo "提示: 如果错误是 '未找到全景图对应的 frame'，"
    echo "      说明该点位在 COLMAP 重建中没有对应的 frame，"
    echo "      这是数据问题，可以跳过该点位。"
    echo ""
fi

echo "输出目录: $OUTPUT_DIR"
echo "=========================================="

# 如果所有点位都失败，返回错误码
if [ $SUCCESS_COUNT -eq 0 ]; then
    exit 1
fi

exit 0

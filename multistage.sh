#!/bin/bash

# Initialize variables
confs=()
trans=()
output_path="out"
device=""
mock_batch_count=""
mock_epoch_count=""
run_name=""
replica_size=2
analysis_level=1
use_amp=0

args=$(getopt -o c:t:o:d:r:a: --long confs:,trans:,output-path:,device:,replica-size:,use-amp:,analysis-level:,mock-batch-count:,mock-epoch-count:,run-name: -n "$0" -- "$@")

# Flag variables to track which list is being processed
in_confs=false
in_trans=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    -c|--confs)
        in_confs=true
        in_trans=false
        shift
        ;;
    -t|--trans)
        in_confs=false
        in_trans=true
        shift
        ;;
    -d|--device)
        shift
        device="$1"
        shift
        ;;
    -r|--replica-size)
        shift
        replica_size="$1"
        shift
        ;;
    --use-amp)
        shift
        use_amp="$1"
        shift
        ;;
    -a|--analysis-level)
        shift
        analysis_level="$1"
        shift
        ;;
    --mock-batch-count)
        mock_batch_count="$2"
        shift 2
        ;;
    --mock-epoch-count)
        mock_epoch_count="$2"
        shift 2
        ;;
    --run-name)
        run_name="$2"
        shift 2
        ;;
    -o|--output-path)
        output_path="$2"
        shift 2
        ;;
    *)
        if $in_confs; then
            confs+=("$1")
        elif $in_trans; then
            trans+=("$1")
        fi
        shift
        ;;
    esac
done

if [ -z "$run_name" ]; then
    echo "run_name is required"
    exit 1
fi

# Determine the length of the longer list
length=${#confs[@]}
if [ ${#trans[@]} -gt $length ]; then
    length=${#trans[@]}
fi

if [ -z "$mock_batch_count" ] && [ -z "$mock_epoch_count" ]; then
    echo "Validating configurations"
    
    conf=""
    tran=""
    stage_run_name=""
    ckpt_path=""
    for ((i = 0; i < length; i++)); do

        if [ $i -lt ${#confs[@]} ]; then
            conf=${confs[i]}
        fi

        if [ $i -gt 0 ]; then
            if [ $((i-1)) -lt ${#trans[@]} ]; then
                tran=${trans[i-1]}
                ckpt_path="$output_path/$stage_run_name/ckpts/final.ckpt"
            fi
        fi

        stage_run_name="$run_name/stage$i"
        options=()
        if [ -n "$conf" ]; then
            options+=("-c" "$conf")
        fi
        if [ -n "$device" ]; then
            options+=("-d" "$device")
        fi
        if [ -n "$replica_size" ]; then
            options+=("-r" "$replica_size")
        fi
        if [ -n "$use_amp" ] && [ "$use_amp" -ne 0 ]; then
            options+=("--use-amp")
        fi
        if [ -n "$analysis_level" ]; then
            options+=("-a" "$analysis_level")
        fi
        if [ -n "$stage_run_name" ]; then
            options+=("--run-name" "$stage_run_name")
        fi
        if [ -n "$ckpt_path" ]; then
            options+=("--ckpt-path" "$ckpt_path")
        fi
        if [ -n "$tran" ]; then
            options+=("--ckpt-map-conf-path" "$tran")
        fi

        options+=("--mock-batch-count" "1")
        options+=("--mock-epoch-count" "1")

        python_command="python mt_pipe/singlestage.py ${options[@]}"
        
        echo ""
        echo "+----------------------------------"
        echo "| Invoking command \`$python_command\`"
        echo "+----------------------------------"
        echo ""
        eval "$python_command"

        if [ $? -ne 0 ]; then
            echo "Error: Stage$i failed "
            exit 1
        fi
    done

    echo "Configurations validation successful"
    echo ""
    echo ""
    echo "Running actual job"
fi


# start the job
conf=""
tran=""
stage_run_name=""
ckpt_path=""
for ((i = 0; i < length; i++)); do

    if [ $i -lt ${#confs[@]} ]; then
        conf=${confs[i]}
    fi

    if [ $i -gt 0 ]; then
        if [ $((i-1)) -lt ${#trans[@]} ]; then
            tran=${trans[i-1]}
            ckpt_path="$output_path/$stage_run_name/ckpts/final.ckpt"
        fi
    fi

    stage_run_name="$run_name/stage$i"
    options=()
    if [ -n "$conf" ]; then
        options+=("-c" "$conf")
    fi
    if [ -n "$device" ]; then
        options+=("-d" "$device")
    fi
    if [ -n "$replica_size" ]; then
        options+=("-r" "$replica_size")
    fi
    # if [ -n "$use_amp" ]; then
    #     if [ $use_amp -ne 0 ]; then
    #         options+=("--use-amp")
    #     fi
    # fi
    if [ -n "$use_amp" ] && [ "$use_amp" -ne 0 ]; then
        options+=("--use-amp")
    fi
    if [ -n "$analysis_level" ]; then
        options+=("-a" "$analysis_level")
    fi
    if [ -n "$stage_run_name" ]; then
        options+=("--run-name" "$stage_run_name")
    fi
    if [ -n "$ckpt_path" ]; then
        options+=("--ckpt-path" "$ckpt_path")
    fi
    if [ -n "$tran" ]; then
        options+=("--ckpt-map-conf-path" "$tran")
    fi
    if [ -n "$mock_batch_count" ]; then
        options+=("--mock-batch-count" "$mock_batch_count")
    fi
    if [ -n "$mock_epoch_count" ]; then
        options+=("--mock-epoch-count" "$mock_epoch_count")
    fi
    python_command="python mt_pipe/singlestage.py ${options[@]}"
    
    echo ""
    echo "+----------------------------------"
    echo "| Invoking command \`$python_command\`"
    echo "+----------------------------------"
    echo ""
    eval "$python_command"

    if [ $? -ne 0 ]; then
        echo "Error: Stage$i failed "
        exit 1
    fi
done

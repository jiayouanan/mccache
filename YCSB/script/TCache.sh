#!/usr/bin/bash
# -----------------------------------------------------------------------------
# Workload Generation Script for YCSB-TCache Experiment
#
# Get script path
SCRIPT_DIR=$(dirname "$0" 2>/dev/null)

# Set YCSB_HOME directory
[ -z "$YCSB_HOME" ] && YCSB_HOME=$(cd "$SCRIPT_DIR/.." || exit; pwd)
cd $YCSB_HOME
echo "Working under directory: $YCSB_HOME"

# Parse command line arguments
for ARGUMENT in "$@"
do
    KEY=$(echo $ARGUMENT | cut -f1 -d=)
    VALUE=$(echo $ARGUMENT | cut -f2 -d=)   

    case "$KEY" in
            recordcount)    recordcount=${VALUE} ;;
            fieldlength)    fieldlength=${VALUE} ;;
            distribution)   distribution=${VALUE} ;;
            distrconst)     distrconst=${VALUE} ;;
            synthetic)      synthetic=${VALUE} ;;
            multisize)      multisize=${VALUE} ;;
            datasetname)    datasetname=${VALUE} ;;
            txnsizefile)    txnsizefile=${VALUE} ;;    
            *)   
    esac    
done

if [ -z $datasetname ] ; then
    echo "Input for dataset name:"
    read datasetname
    DATASET_DIR="$YCSB_HOME/datasets/$datasetname"
    WORKLOAD_FILE="$DATASET_DIR/workload"
    LOAD_LOG_FILE="$DATASET_DIR/load.dat"
    RUN_LOG_FILE="$DATASET_DIR/transactions.dat"
    if [ -d $DATASET_DIR ] ; then
        echo "DATASET_DIR: $DATASET_DIR"
    else
        echo "[ERROR] DATASET_DIR $DATASET_DIR does not exist. Exiting."
        exit 1;
    fi
fi

# Determine command argument
if [ "config" = $1 ] ; then
    echo "command: $1, setup workload config file"
    WORKLOAD_TEMPLATE_FILE="$YCSB_HOME/script/workload_template"
    NEW_WORKLOAD_FILE="$YCSB_HOME/script/workload_new"
    cp $WORKLOAD_TEMPLATE_FILE $NEW_WORKLOAD_FILE
    # Detect and take input for not specified workload configurations
    for i in `grep '=$' $NEW_WORKLOAD_FILE`
    do
        VAR_NAME=`echo $i | cut -d= -f1`
        echo "Input for $VAR_NAME:"
        read tmp_var
        if [ "$VAR_NAME" = "querysizefile" ] || [ "$VAR_NAME" = "txnsizefile" ] ; then
            tmp_val="$datasetname/$tmp_var"
            echo "$VAR_NAME=$tmp_val"
            sed -i "/$VAR_NAME=.*/c\\$VAR_NAME=$tmp_val" $NEW_WORKLOAD_FILE
        else
            echo "$VAR_NAME=$tmp_var"
            sed -i "/$VAR_NAME=.*/c\\$VAR_NAME=$tmp_var" $NEW_WORKLOAD_FILE
        fi
    done
    # move workload_new configuration file to $DATASET_DIR/workload
    mv $NEW_WORKLOAD_FILE $WORKLOAD_FILE
    echo "WORKLOAD CONFIGURATION: " && cat $WORKLOAD_FILE
elif [ "load" = $1 ] ; then
    echo "load db"
    # Build only with PostgreSQL binding:
    mvn -T 1C -pl site.ycsb:postgrenosql-binding -am clean package
    # pg_ctl start -l $PGLOG -D $PGDATA
    # psql -U postgres -d ycsb -c 'DROP TABLE IF EXISTS usertable;'
    # psql -U postgres -d ycsb -c 'CREATE TABLE usertable (YCSB_KEY VARCHAR(255) PRIMARY KEY not NULL, YCSB_VALUE JSONB not NULL);'
    psql -U postgres -h 172.17.68.65 -d ycsb -c 'DROP TABLE IF EXISTS usertable;'
    psql -U postgres -h 172.17.68.65 -d ycsb -c 'CREATE TABLE usertable (YCSB_KEY VARCHAR(255) PRIMARY KEY not NULL, YCSB_VALUE JSONB not NULL);'
    # Load Data to DB layer
    ./bin/ycsb load postgrenosql -P $WORKLOAD_FILE -s > $LOAD_LOG_FILE
elif [ "run" = $1 ] ; then
    echo "generate transaction"
    ./bin/ycsb run postgrenosql -P $WORKLOAD_FILE -s > $RUN_LOG_FILE
else
    echo "[ERROR] Found unknown command '$1'"
    echo "[ERROR] Expected one of 'config', 'load', or 'run'. Exiting."
    exit 1;
fi

# TODO: Copy workload configuration template file, 
#       edit lines according to argument values
# Copy template workload file workload_template to workload_new

# if [ -n "$recordcount"] ; then
#     RECORD_COUNT_LINE="recordcount=$recordcount"
    
# WORKLOAD_NAME_LINE="workload=$FUNC"
# sed -i "/workload=/c\\$WORKLOAD_NAME_LINE" $YCSB_HOME/script/foo.txt
# DB_NAME_LINE="dbname=$NUM"
# sed -i "/dbname=/c\\$DB_NAME_LINE" $YCSB_HOME/script/foo.txt


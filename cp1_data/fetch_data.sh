download_from_gdrive() {
    file_id=$1
    file_name=$2 # first stage to get the warning html
    curl -L -o $file_name -c /tmp/cookies \
    "https://drive.google.com/uc?export=download&id=$file_id"
    if grep "Virus scan warning" $file_name > /dev/null;then
        # second stage to extract the download link from html above
        download_link=$(cat $file_name | \
        grep -Po 'uc-download-link" [^>]* href="\K[^"]*' | \
        sed 's/\&amp;/\&/g')
        if [ ! -z "$download_link" ];then
            curl -L -b /tmp/cookies \
            "https://drive.google.com$download_link" > $file_name
        fi
    fi
}

file_name=C1-P1_Train_Dev_fixed.rar
download_from_gdrive 1LFe2NhXLJ0FzStLvgjrh8aWE-PbN7BIz $file_name
rar x $file_name
python mv.py dev.csv C1-P1_Dev/
python mv.py train.csv C1-P1_Train/

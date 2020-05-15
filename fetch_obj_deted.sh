download_from_gdrive() {
    file_id=$1
    file_name=$2 # first stage to get the warning html
    curl -L -o $file_name -c /tmp/cookies \
    "https://drive.google.com/uc?export=download&id=$file_id"
    if grep "Virus scan warning" $file_name > /dev/null;then
        # second stage to extract the download link from html above
        download_link=$(cat $file_name | \
        grep -Eo 'uc-download-link" [^>]* href="[^\"]*' | sed 's/\&amp;/\&/g' | sed  's/.*href="\(.*\)/\1/')
        if [ ! -z "$download_link" ];then
            curl -L -b /tmp/cookies \
            "https://drive.google.com$download_link" > $file_name
        fi
    fi
}

file_name=obj_deted_data.rar
download_from_gdrive 1clP7FoZbpPFirwFzMru6TxJ7vcWt9iO3 $file_name
rar x $file_name

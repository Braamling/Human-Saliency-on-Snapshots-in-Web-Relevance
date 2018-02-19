# This bash script can be used as a cronjob to kill all firefox and geckodrivers if the process has been stuck for more than 30 minutes.
cd Human-Saliency-on-Snapshots-in-Web-Relevance/preprocessing/highlightGenerator/storage/highlights
lastfile=`ls . -t | head -1`
echo $lastfile

if test `find $lastfile -mmin +30`
then
    killall firefox
    killall geckodriver
fi
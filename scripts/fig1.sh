# read args
label=linear5v2
dur=5
shape="linear" #step, linear, exp
shapeF="none"
if [ $shape == "linear" ]; then
    shapeF=$shape
elif [ $shape == "exp" ]; then
    shapeF="exponential"
fi

#shapeF="none" #none, linear, exponential
seed=139563874
M=25
K=10
shp="2.0" # bigger => bigger mean (therefore smaller inverse mean, e.g., global event strength)
rte="1.0" # bigger => smaller mean, lower variance
lcl="1.0"
aph="0.01" # controls proportion; bigger means more even topics, smaller means less even
totalD=100 # total number of days

rm -rf $label
mkdir $label
mkdir $label/sim
mkdir $label/fit

# set up LDA infrastructure
#wget http://www.cs.princeton.edu/~blei/lda-c/lda-c-dist.tgz
#tar -xzvf lda-c-dist.tgz
#cd lda-c-dist; make; cd ..

## run capsule for each simulated dataset
#echo "***** RUNNING CAPSULE and baselines *****"
for id in `seq 1 $M`
do
    echo "****** dataset $id ******"
    r=$RANDOM
    echo " - simulating data with seed $r"
    # simualte a dataset
    mkdir $label/sim/$id
    cd $label/sim/$id
    python2.7 ../../../simulate_data.py $dur $shape $r $K $shp $rte $lcl $aph $totalD > log.out
    cd ../../../

    echo " - kicking off capsule fits"
    mkdir $label/fit/$id

    #echo "../src/capsule --K $K --batch --a_phi 0.1 --b_phi 0.1 --a_psi 0.1 --b_psi 0.1 --a_theta 0.1 --a_epsilon 0.1 --a_pi 0.1 --a_beta 0.1 --data $label/sim/$id --out $label/fit/$id/capsule --event_dur $dur --seed $seed --conv_freq 1 --min_iter 100 --max_iter 100 --event_decay $shapeF"
    #../src/capsule
    #echo ../src/capsule --K $K --batch --a_phi 0.1 --b_phi 0.1 --a_psi 0.1 --b_psi 0.1 --a_theta 0.1 --a_epsilon 0.1 --a_pi 0.1 --a_beta 0.1 --data $label/sim/$id --out $label/fit/$id/capsule --event_dur $dur --seed $seed --conv_freq 1 --min_iter 100 --max_iter 100 --event_decay $shapeF
    #echo "*******"
    #ls $label/sim/$id
    #pwd
    #ls $label/fit/$id
    (../src/capsule --K $K --batch --a_phi 0.1 --b_phi 0.1 --a_psi 0.1 --b_psi 0.1 --a_theta 0.1 --a_epsilon 0.1 --a_pi 0.1 --a_beta 0.1 --data $label/sim/$id --out $label/fit/$id/capsule --event_dur $dur --seed $seed --conv_freq 1 --min_iter 100 --max_iter 100 --event_decay $shapeF > $label/fit/"$id"capsule.out 2>$label/fit/"$id"capsule.err &)
    (../src/capsule --K $K --batch --a_phi 0.1 --b_phi 0.1 --a_psi 0.1 --b_psi 0.1 --a_theta 0.1 --a_epsilon 0.1 --a_pi 0.1 --a_beta 0.1 --data $label/sim/$id --out $label/fit/$id/entity_only --event_dur $dur --seed $seed --conv_freq 1 --min_iter 100 --max_iter 100 --event_decay $shapeF --entity_only > $label/fit/"$id"entity.out 2> $label/fit/"$id"entity.err &)
    (../src/capsule --K $K --batch --a_phi 0.1 --b_phi 0.1 --a_psi 0.1 --b_psi 0.1 --a_theta 0.1 --a_epsilon 0.1 --a_pi 0.1 --a_beta 0.1 --data $label/sim/$id --out $label/fit/$id/event_only --event_dur $dur --seed $seed --conv_freq 1 --min_iter 100 --max_iter 100 --event_decay $shapeF --event_only >$label/fit/"$id"event.out 2>$label/fit/"$id"event.err &)


    mkdir $label/fit/$id/baselines

    echo " - processing data to mult format"
    # process data to mult format
    python2.7 process_to_mult.py $label/sim/$id/train.tsv $label/sim/$id/mult.dat > $label/sim/$id/log.out

    # run LDA
    echo " - fitting LDA"
    lda-c-dist/lda est 0.1 $K lda-c-dist/settings.txt $label/sim/$id/mult.dat random $label/fit/$id/baselines/lda > $label/fit/$id/baselines/lda.log

    echo " - computing baselines"
    python2.7 baselines.py $label/sim/$id/ $label/fit/$id/baselines > $label/fit/$id/baselines.log
done

echo "*************************************"
ps ux
echo "*************************************"
#read -p "Please confirm all fits have completed:" response
sleep 10m
echo "*************************************"
#pr=`ps ux | grep capsule | wc -l`
#echo $pr
#while [ $pr != "1" ]
#do
#    echo "waiting for capsule to finish ($pr processes)"
#    sleep 1m
#    pr=`ps ux | grep capsule | wc -l`
#done

echo "***** EVALUATING ALL MODELS *****"
echo "data,method,value" > $label/results.csv
for id in `seq 1 $M`
do
    v=`python eval.py $label/sim/$id/event_strength.tsv $label/fit/$id/capsule/psi-final.dat inv`
    echo "$id,capsule,$v" >> $label/results.csv

    v=`python eval.py $label/sim/$id/event_strength.tsv $label/fit/$id/event_only/psi-final.dat inv`
    echo "$id,event_only,$v" >> $label/results.csv

    for base in ave_deviation total_deviation ave_deviation_entity lda_ent_MG_ave total_deviation_tfidf ave_deviation_entity_tfidf lda_ent_MG_max word_outlier ave_deviation_tfidf lda_MG_ave word_outlier_entity_ave ave_dev_lda lda_MG_max word_outlier_entity_ave_tfidf ave_ent_dev_lda max_deviation_entity word_outlier_entity_max doc_ent_outlier_lda max_deviation_entity_tfidf word_outlier_entity_max_tfidf doc_outlier_lda random word_outlier_tfidf
    do
        v=`python eval.py $label/sim/$id/event_strength.tsv $label/fit/$id/baselines/$base.dat n`
        echo "$id,$base,$v" >> $label/results.csv
    done
done

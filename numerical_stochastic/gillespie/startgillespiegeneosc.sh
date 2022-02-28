for ic in lo hi
do
    for am in {5..105..10}
    do
        nohup python gillespie_geneosc.py $ic gillespie/geneosc_v${V/./"-"}_${ic}_${am}.h5 aM=$am V=$V &
    done
done

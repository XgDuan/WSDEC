python evaluate.py --checkpoint final_models/final_model_0304.ckp --n_rand 15 --rand_seed 233 --alias rand_eval_15_233 &
sleep 1
python evaluate.py --checkpoint final_models/final_model_0304.ckp --n_rand 15 --rand_seed 346 --alias rand_eval_15_346 &
sleep 1
python evaluate.py --checkpoint final_models/final_model_0304.ckp --n_rand 15 --rand_seed 581 --alias rand_eval_15_581 &
sleep 1
python evaluate.py --checkpoint final_models/final_model_0304.ckp --n_rand 15 --rand_seed 910 --alias rand_eval_15_910 
wait
sleep 1
python evaluate.py --checkpoint final_models/final_model_0304.ckp --n_rand 15 --rand_seed 191 --alias rand_eval_15_191 &
sleep 1
python evaluate.py --checkpoint final_models/final_model_0304.ckp --n_rand 10 --rand_seed 233 --alias rand_eval_10_233 &
sleep 1
python evaluate.py --checkpoint final_models/final_model_0304.ckp --n_rand 10 --rand_seed 346 --alias rand_eval_10_346 &
sleep 1
python evaluate.py --checkpoint final_models/final_model_0304.ckp --n_rand 10 --rand_seed 581 --alias rand_eval_10_581
sleep 1
wait
python evaluate.py --checkpoint final_models/final_model_0304.ckp --n_rand 10 --rand_seed 910 --alias rand_eval_10_910 &
sleep 1
python evaluate.py --checkpoint final_models/final_model_0304.ckp --n_rand 10 --rand_seed 191 --alias rand_eval_10_191 &

sleep 1
python evaluate.py --checkpoint final_models/final_model_0304.ckp --n_rand 20 --rand_seed 233 --alias rand_eval_20_233 &
sleep 1
python evaluate.py --checkpoint final_models/final_model_0304.ckp --n_rand 20 --rand_seed 346 --alias rand_eval_20_246
sleep 1
wait
python evaluate.py --checkpoint final_models/final_model_0304.ckp --n_rand 20 --rand_seed 581 --alias rand_eval_20_581 &
sleep 1
python evaluate.py --checkpoint final_models/final_model_0304.ckp --n_rand 20 --rand_seed 910 --alias rand_eval_20_910 &
sleep 1
python evaluate.py --checkpoint final_models/final_model_0304.ckp --n_rand 20 --rand_seed 191 --alias rand_eval_20_191
sleep 1
wait

echo 'Start'
python kd_train.py -s 18 -t 2
python kd_train.py -s 18 -t 5
python kd_train.py -s 18 -t 10
python kd_train.py -s 34 -t 10
python kd_train.py -s 50 -t 10
echo 'Mischief Managed'

echo 'Start'
python after_amp.py -p False -n 0
python after_amp.py -p False -n 1
python after_amp.py -p False -n 2
python after_amp.py -p False -n 4
python after_amp.py -p True -n 0
python after_amp.py -p True -n 1
python after_amp.py -p True -n 2
python after_amp.py -p True -n 4

python before_amp.py -p False -n 0
python before_amp.py -p False -n 1
python before_amp.py -p False -n 2
python before_amp.py -p False -n 4
python before_amp.py -p True -n 0
python before_amp.py -p True -n 1
python before_amp.py -p True -n 2
python before_amp.py -p True -n 4
echo 'Finish'
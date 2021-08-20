echo 'Start'
python pruned_test.py -p 0
python pruned_test.py -p 1
python pruned_test.py -p 2
python pruned_test.py -p 3
python pruned_test.py -p 5
python pruned_test.py -p 10
python pruned_test.py -p 15
python pruned_test.py -p 25
python pruned_test.py -p 50
python pruned_test.py -p 75
python summary_result.py
echo 'Finish'
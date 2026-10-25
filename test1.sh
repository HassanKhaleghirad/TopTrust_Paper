cd ~/PycharmProjects/TopTrust

echo "===== 1. CLI OPTIONS ====="
python toptrust_revised4.py --help | grep -E 'report_test|intas_features|epsilon|rounds|seed|mode'

echo
echo "===== 2. TEST EVALUATION AND CLI SOURCE ====="
grep -nE 'report_test|TEST NOT EVALUATED|ArgumentParser|add_argument|select_threshold|test_logits|test_y|def main|VALIDATION' toptrust_revised4.py

echo
echo "===== 3. MAIN FUNCTION ====="
nl -ba toptrust_revised4.py | sed -n '880,1040p'
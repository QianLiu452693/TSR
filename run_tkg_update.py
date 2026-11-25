"""按新案件增量更新犯罪 TKG。

用法：
  py -3 run_tkg_update.py --demo
  py -3 run_tkg_update.py --input new_cases.csv
  py -3 run_tkg_update.py --schedule day --input new_cases.csv
  py -3 run_tkg_update.py --if-due --input new_cases.csv
"""

from tkg_model.update_tkg import main

if __name__ == "__main__":
    main()

import os
import time

def main():
    print("Unzipping plink files...")
    os.system("unzip -o chasm/plink/plink_linux_x86_64_20230116.zip -d chasm/plink")
    os.system("unzip -o chasm/plink/plink2_linux_avx2_20230825.zip -d chasm/plink")
    time.sleep(1)
    os.system("rm -rf chasm/plink/toy*")
    os.system("rm -rf chasm/plink/prettify*")
if __name__ == "__main__":
    main()

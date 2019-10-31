
from scripts.flavour_classifier import get_flavour
from scripts.color_classifier import get_color
import os

def run():
    while(True):
        img_p = input("Enter Next('exit' to quit):")
        if img_p == 'exit':
            break
        flavour = get_flavour(img_path=img_p)
        print(flavour)



def run_for_all():

    total = 0
    true_count = 0
    ROOT_DIR = './snack_cropped/'
    for the_directoy in os.listdir(ROOT_DIR):
        if the_directoy == ".DS_Store":
            continue
        for the_file in os.listdir(os.path.join(ROOT_DIR, the_directoy)):
            file_path = os.path.join(ROOT_DIR,the_directoy, the_file)
            if os.path.isfile(file_path) and (".jpg" in file_path or ".jpeg" in file_path or ".png" in file_path):
                flavour = get_flavour(img_path=file_path)
                flavour = flavour.replace('_',' ')
                total+=1
                if flavour in the_directoy:
                    true_count+=1
                else:
                    print(f"!!! saying {the_directoy} is {flavour} which is wrong")
                print(total)
    print("result:",(true_count/total))
    print("total: ",total)
    print("true_count",true_count)


print(get_color("im/yellowcap-027.jpeg"))
run()




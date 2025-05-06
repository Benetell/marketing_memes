from bing_image_downloader import downloader
import os
import random as rand

directory_path="/home/thetis/codes/test"
for root, dirs, files in os.walk(directory_path):
    for i,dir_name in enumerate(dirs):
        brand_name = dir_name + " memes"
        if(brand_name == "HP memes"):
            brand_name = "hp brand memes"
            print(f"Current brand: {brand_name}")
            
            brand_path = os.path.join("/home/thetis/codes/bing_images_only_memes/", brand_name)
            
            downloader.download(brand_name, limit=100, output_dir=brand_path, adult_filter_off=True)
            #save_path = os.path.join("/home/thetis/codes/company_memes", f"{brand_name}_image{rand}.jpg")
    

print("Image download completed.")


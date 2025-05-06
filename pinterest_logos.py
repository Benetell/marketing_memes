import os
import random

from selenium import webdriver
from selenium.webdriver import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.service import Service
from selenium.common.exceptions import StaleElementReferenceException
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

import time
import requests



# Initialize WebDriver
GECKO_DRIVER_PATH = "/home/thetis/codes/projekt/webscrape/linux/geckodriver"
service = Service(GECKO_DRIVER_PATH)
driver = webdriver.Firefox(service=service)
driver.implicitly_wait(10)  # Implicit wait set to 10 seconds

driver.get(f"https://hu.pinterest.com/ideas/")

directory_path = "home/thetis/codes/train"


def scrapeBrands(dir, any_driver, picPerBrand):
    i = 0
    for root, dirs, files in os.walk(dir):
        if i > 0:
            print(f"Current directory: {root}")
            name = root.split('/')[-1]
            print(name)
            try:
                wait = WebDriverWait(any_driver, 5)  # Maximum 5 seconds of explicit wait
                search = wait.until(
                    EC.presence_of_element_located((By.XPATH, "//input[@data-test-id='searchBoxAccessibleText']")))
                # Locate the search element inside each iteration
                #search = any_driver.find_element(By.XPATH, "//input[@data-test-id='search-box-input']")
                search = any_driver.find_element(By.XPATH, "//input[@data-test-id='searchBoxAccessibleText']")

                # Clear the search input and send keys
                search.send_keys(Keys.CONTROL + "a")
                search.send_keys(Keys.DELETE)
                search.send_keys(name + " memes", Keys.ENTER)
                time.sleep(1)  # You might want to adjust this sleep time based on your needs
                images = any_driver.find_elements(By.CLASS_NAME, "hCL")
                path = "/home/thetis/codes/company_memes"
                rand = random.randint(0,200)
                for j, image in enumerate(images[:picPerBrand], 1):
                    image_url = image.get_attribute("src")
                    if image_url:
                        # Download the image using requests
                        response = requests.get(image_url)
                        if response.status_code == 200:
                            rand = random.randint(0, 100)
                            with open(os.path.join(path, f"{name}_image{rand}_{j}.jpg"), 'wb') as f:
                                f.write(response.content)

                            print(f"Downloaded image{j}.jpg for {name}")

            except StaleElementReferenceException:
                # If the search element becomes stale, you might want to refresh the page or reinitialize the driver
                any_driver.refresh()  # Refresh the page to reset the state
                continue  # Skip to the next iteration
        i += 1
        print("------------------------")


scrapeBrands(directory_path, driver, 5)

driver.quit()

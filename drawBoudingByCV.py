# import cv2
# import pandas as pd

# img_path = "./static/04c64dd7683662188094816c7c2ca88d_LMLO.jpg"

# # Read CSV file using pandas
# csv_file_path = "data.csv"
# df = pd.read_csv(csv_file_path)

# df.info()

# df['name'] = df['name'].str.replace(' ', '')
# print(df['name'])

# def drawBoundingBoxCheck(img_path):
#     img = cv2.imread(img_path)

#     cv2.waitKey(0)

#     x_max = 712.4799805

#     x_min = 309.7139893

#     y_max = 2190.97998

#     y_min = 1761.420044

#     box_width = x_max - x_min
#     box_height = y_max - y_min

#     box_center_x = int(x_min + (box_width / 2))
#     box_center_y = int(y_min + (box_height / 2))

#     # Calculate the coordinates and size for the rectangle
#     w = int(box_width)
#     h = int(box_height)

#     # Calculate top-left corner coordinates
#     x = box_center_x - int(w / 2)
#     y = box_center_y - int(h / 2)

#     # Draw rectangle on the image
#     img_with_box = img.copy()
#     cv2.rectangle(img_with_box, (x, y), (x + w, y + h), (0, 255, 0), 15)

#     cv2.waitKey(0)

#     # Save the image with the drawn rectangle
#     output_path = "./static/LCC_with_box.jpg"
#     cv2.imwrite(output_path, img_with_box)


# drawBoundingBoxCheck(img_path)


import cv2
import pandas as pd

# img_path = "./static/ec2c69b6d06c24bf6b93e7323f66c257_LCC.jpg"

# Read CSV file using pandas
csv_file_path = "data.csv"
df = pd.read_csv(csv_file_path)

def drawBoundingBoxCheck(img_path, df):
    # Extract the image filename from the path
    img_filename = img_path.split("/")[-1]

    # Find the corresponding row in the DataFrame based on the 'name' column
    img_row = df[df['name'] == img_filename]

    if not img_row.empty:
        # Extract the xmin and ymin values from the DataFrame
        x_min = img_row['xmin'].values[0]
        x_max = img_row['xmax'].values[0]
        y_max = img_row['ymax'].values[0]
        y_min = img_row['ymin'].values[0]

        # Load the image
        img = cv2.imread(img_path)

        box_width = x_max - x_min
        box_height = y_max - y_min

        box_center_x = int(x_min + (box_width / 2))
        box_center_y = int(y_min + (box_height / 2))
        # Calculate the coordinates and size for the rectangle
        w = int(box_width)
        h = int(box_height)

        # Calculate top-left corner coordinates
        x = box_center_x - int(w / 2)
        y = box_center_y - int(h / 2)

        # Draw rectangle on the image
        img_with_box = img.copy()
        cv2.rectangle(img_with_box, (x, y), (x + w, y + h), (0, 255, 0), 15)

        # Save the image with the drawn rectangle
        output_path = "./static/LCC_with_box.jpg"
        cv2.imwrite(output_path, img_with_box)

        cv2.waitKey(0)
    else:
        print(f"No matching entry found for {img_filename} in the DataFrame.")

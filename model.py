from ultralytics import YOLO
import cv2
import numpy as np
import os
import math
import random

model = YOLO("yolov8n.pt")  # Load a pretrained model (e.g., yolov8n.pt)


def detect(image_path):
    """
    Perform object detection on an image, visualize the results with numbering and class names,
    save the image, and return a list of detected object numbers and class names.

    Args:
        image_path (str): Path to the input image.
        model_name (str): Name of the pretrained YOLO model.

    Returns:
        tuple: A tuple containing:
            - list: List of detection results from the YOLO model.
            - list: A list of tuples (object_number, class_name) for detected objects.
    """

    results = model(image_path)  # Run inference on the image

    detected_objects_info = []

    # Visualize results and save the image
    img = cv2.imread(image_path)
    for r in results:
        boxes = r.boxes
        class_names_dict = r.names  # Get the class names dictionary
        for i, box in enumerate(boxes):  # Enumerate to get numbering
            x1, y1, x2, y2 = box.xyxy[0]
            class_id = int(box.cls[0])
            class_name = class_names_dict[class_id]  # Get class name from dictionary

            detected_objects_info.append((i + 1, class_name))

            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            # Display numbering and class name
            label = f"{i+1}: {class_name}"  # Format as Number: ClassName
            cv2.putText(
                img,
                label,
                (int(x1), int(y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
            )

    # Construct the output filename
    base, ext = os.path.splitext(image_path)
    output_path = f"{base}_result{ext}"

    cv2.imwrite(output_path, img)
    print(f"Result image saved to {output_path}")

    return results, detected_objects_info, output_path


def predict(results, selected_candidate):
    # results[0]からResultsオブジェクトを取り出す
    result_object = results[0]

    # 画像データの大きさを取得 (width, height)
    img_width, img_height = result_object.orig_shape[::-1]
    img_area = img_width * img_height
    print(f"Image size: Width={img_width}, Height={img_height}, Area={img_area}")

    # バウンディングボックスの座標を取得
    bounding_boxes = result_object.boxes.xyxy

    # クラスIDを取得
    class_ids = result_object.boxes.cls

    # クラス名の辞書を取得
    class_names_dict = result_object.names

    # 計算に含める物体のクラス名をリストで指定
    selected_classes = []  # ここに計算に含めたいクラス名を追加/変更してください
    selected_classes.append(selected_candidate)

    # 検出された物体の合計面積を計算
    total_object_area = 0
    for box, class_id in zip(bounding_boxes, class_ids):
        class_name = class_names_dict[int(class_id)]
        if class_name in selected_classes:
            x1, y1, x2, y2 = box
            object_width = x2 - x1
            object_height = y2 - y1
            object_area = object_width * object_height
            total_object_area += object_area
            print(f"Object: {class_name}, Box coordinates: {box}, Area: {object_area}")

            # 物体の中心座標を計算
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            print(
                f"Object: {class_name}, Center coordinates: ({center_x:.2f}, {center_y:.2f})"
            )

        else:
            print(f"Object: {class_name} (excluded from calculation)")

    print(f"Total object area (selected classes): {total_object_area}")

    # 画像データに対する検出された物体の占める面積の割合を計算
    area_ratio = total_object_area / img_area
    print(f"Ratio of selected object area to image area: {area_ratio:.4f}")

    # 三分割したときの交点座標を計算する
    # Calculate the vertical division points
    v_third_1 = img_width / 3
    v_third_2 = 2 * img_width / 3

    # Calculate the horizontal division points
    h_third_1 = img_height / 3
    h_third_2 = 2 * img_height / 3

    # Calculate the coordinates of the four intersection points
    intersection_points = [
        (v_third_1, h_third_1),  # Top-left intersection
        (v_third_2, h_third_1),  # Top-right intersection
        (v_third_1, h_third_2),  # Bottom-left intersection
        (v_third_2, h_third_2),  # Bottom-right intersection
    ]

    print("Intersection points:")
    for point in intersection_points:
        print(f"({point[0]:.2f}, {point[1]:.2f})")

    # 被写体中心から交点までの距離を計算する
    # Calculate the center coordinates of the specified object (assuming only one specified object for simplicity)
    # Use the first bounding box as the specified object's bounding box
    if len(bounding_boxes) > 0:
        x1, y1, x2, y2 = bounding_boxes[0]
        object_center_x = (x1 + x2) / 2
        object_center_y = (y1 + y2) / 2
        object_center = (object_center_x, object_center_y)
        print(
            f"Specified object center coordinates: ({object_center[0]:.2f}, {object_center[1]:.2f})"
        )

        # Calculate the Euclidean distance to each intersection point
        distances = []
        for point in intersection_points:
            distance = np.sqrt(
                (object_center[0] - point[0]) ** 2 + (object_center[1] - point[1]) ** 2
            )
            distances.append(distance)
            print(
                f"Distance to intersection point ({point[0]:.2f}, {point[1]:.2f}): {distance:.2f}"
            )

        # Find the minimum distance and assign it to j1
        j1 = min(distances)
        print(f"Minimum distance (j1): {j1:.2f}")
    else:
        print("No objects detected to calculate center coordinates.")
        j1 = None

    # 画像中心から被写体中心までの距離を計算する
    # Calculate the center coordinates of the image
    image_center_x = img_width / 2
    image_center_y = img_height / 2
    image_center = (image_center_x, image_center_y)
    print(f"Image center coordinates: ({image_center[0]:.2f}, {image_center[1]:.2f})")

    # Calculate the Euclidean distance between the object center and the image center
    j2 = np.sqrt(
        (object_center[0] - image_center[0]) ** 2
        + (object_center[1] - image_center[1]) ** 2
    )
    print(f"Distance from object center to image center (j2): {j2:.2f}")

    # 対角線構図を確認する
    # 1. Identify the coordinates of the two points defining the diagonal from top-right to bottom-right.
    # Note: The instruction seems to have a typo in step 1 and 2. It should be top-left to bottom-right and top-right to bottom-left for diagonals of the *entire* image.
    # Assuming the intention was to calculate distances to the diagonals of the image from corner to corner:
    # Diagonal 1: Top-left (0, 0) to Bottom-right (img_width, img_height)
    # Diagonal 2: Top-right (img_width, 0) to Bottom-left (0, img_height)

    # Coordinates for Diagonal 1 (Top-left to Bottom-right)
    p1_d1 = (0, 0)
    p2_d1 = (img_width, img_height)

    # Coordinates for Diagonal 2 (Top-right to Bottom-left)
    p1_d2 = (img_width, 0)
    p2_d2 = (0, img_height)

    # 3. Find the equation of each line (Ax + By + C = 0)
    # For a line passing through (x1, y1) and (x2, y2), the equation is (y2 - y1)x - (x2 - x1)y + x2*y1 - y2*x1 = 0
    # So, A = y2 - y1, B = -(x2 - x1) = x1 - x2, C = x2*y1 - y2*x1

    # Equation for Diagonal 1 (A1*x + B1*y + C1 = 0)
    A1 = p2_d1[1] - p1_d1[1]  # img_height - 0 = img_height
    B1 = p1_d1[0] - p2_d1[0]  # 0 - img_width = -img_width
    C1 = p2_d1[0] * p1_d1[1] - p2_d1[1] * p1_d1[0]  # img_width * 0 - img_height * 0 = 0

    # Equation for Diagonal 2 (A2*x + B2*y + C2 = 0)
    A2 = p2_d2[1] - p1_d2[1]  # img_height - 0 = img_height
    B2 = p1_d2[0] - p2_d2[0]  # img_width - 0 = img_width
    C2 = (
        p2_d2[0] * p1_d2[1] - p2_d2[1] * p1_d2[0]
    )  # 0 * 0 - img_height * img_width = -img_height * img_width

    print(f"Diagonal 1 Equation: {A1}x + {B1}y + {C1} = 0")
    print(f"Diagonal 2 Equation: {A2}x + {B2}y + {C2} = 0")

    # 4. Calculate the distance from the object center (x0, y0) to each line
    # Distance = |A*x0 + B*y0 + C| / sqrt(A^2 + B^2)
    x0, y0 = object_center

    # Distance to Diagonal 1
    distance_d1 = abs(A1 * x0 + B1 * y0 + C1) / np.sqrt(A1**2 + B1**2)

    # Distance to Diagonal 2
    distance_d2 = abs(A2 * x0 + B2 * y0 + C2) / np.sqrt(A2**2 + B2**2)

    print(f"Distance to Diagonal 1: {distance_d1:.2f}")
    print(f"Distance to Diagonal 2: {distance_d2:.2f}")

    # 5. Assign the smaller of the two distances to j3
    j3 = min(distance_d1, distance_d2)

    print(f"Minimum distance to diagonals (j3): {j3:.2f}")

    # 画像の対角線の長さを計算
    diagonal_length = math.sqrt(img_width**2 + img_height**2)
    print(f"Diagonal length of the image: {diagonal_length:.2f}")

    # 基準線の長さの5%を計算
    threshold = diagonal_length * 0.05
    print(f"5% of diagonal length (threshold): {threshold:.2f}")

    # 構図の判定
    composition = "構図なし"
    suggestion = ""

    if j2 is not None and j2 < threshold:
        composition = "日の丸構図"
    elif j3 is not None and j3 < threshold:
        composition = "対角線構図"
    elif j1 is not None and j1 < threshold:
        composition = "三分割構図"

    if composition == "構図なし":
        # 画像全体に対する検出された物体の占める面積の割合を計算 (HmV6Bfd1U8rR セルで計算された area_ratio を使用)
        # area_ratio が計算されていない場合は、ここで計算するか、エラー処理を行います。
        # ここでは HmV6Bfd1U8rR で計算された area_ratio が利用可能であることを前提とします。
        # もし利用できない場合は、area_ratio = total_object_area / img_area の計算をここで行う必要があります。

        if "area_ratio" in locals() and area_ratio is not None:
            if area_ratio >= 0.50:
                suggestion = "検出された物体が画像の大部分を占めているため、「日の丸構図」を提案します。"
                composition = "日の丸構図 (提案)"
            else:
                suggested_compositions = ["三分割構図", "対角線構図"]
                random_suggestion = random.choice(suggested_compositions)
                suggestion = f"検出された物体の面積が比較的小さいため、「{random_suggestion}」を提案します。"
                composition = f"{random_suggestion} (提案)"
        else:
            suggestion = (
                "面積比率が計算されていないため、構図の提案ができませんでした。"
            )

    CLASS_TO_CATEGORY = {
        # 人物
        "person": "portrait",
        # 動物
        "cat": "animal",
        "dog": "animal",
        "bird": "animal",
        "horse": "animal",
        # 風景（自然・乗り物など含む）
        "bicycle": "landscape",
        "car": "landscape",
        "bus": "landscape",
        "train": "landscape",
        "boat": "landscape",
        "elephant": "landscape",
        "zebra": "landscape",
        "giraffe": "landscape",
        # 建物・建築物
        "bench": "architecture",
        "traffic light": "architecture",
        "fire hydrant": "architecture",
        "stop sign": "architecture",
        # 小物・食べ物・静物
        "banana": "still_life",
        "apple": "still_life",
        "sandwich": "still_life",
        "pizza": "still_life",
        "cake": "still_life",
        "bottle": "still_life",
        "cup": "still_life",
        "chair": "still_life",
        "book": "still_life",
        "clock": "still_life",
    }

    COMPOSITION_ADVICE = {
        "portrait": "三分割構図や対角線構図で人物を配置すると良いでしょう。視線の先に余白を作ると自然です。",
        "animal": "ローアングルで目線を合わせて三分割構図にするとおすすめです。日の丸構図でかわいさも強調できます。",
        "landscape": "三分割構図で水平線を意識してみましょう。リーディングラインや対角線を活かすと奥行きが出ます。",
        "architecture": "シンメトリー構図や対角線構図で迫力を強調できます。垂直・水平を意識しましょう。",
        "still_life": "俯瞰撮影＋三分割構図がおすすめです。日の丸で主役を強調し、余白や光の向きを活かしてみましょう。",
    }

    def get_category(label: str) -> str:
        return CLASS_TO_CATEGORY.get(label, "still_life")  # 未登録は静物扱い

    def get_advice(category: str) -> str:
        return COMPOSITION_ADVICE.get(category, "構図の工夫で魅力が増します。")

    category = get_category(selected_candidate)
    advice = get_advice(category)

    if suggestion:
        return suggestion, advice

    return composition, advice

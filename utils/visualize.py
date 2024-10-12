import cv2
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


def draw_matches(img1, kp1, img2, kp2, matches, orientation='horizontal'):
    # Convert images to RGB format if they are grayscale
    if len(img1.shape) == 2:  # Grayscale to RGB
        img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2RGB)
    if len(img2.shape) == 2:  # Grayscale to RGB
        img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)

    # Add text using OpenCV

    img1 = put_text(img1, "top_center", "Previous Frame")
    img2 = put_text(img2, "top_center", "Current Frame")

    # Stack images based on the orientation
    if orientation == 'vertical':
        combined_img = stack_images(img1, img2, orientation='vertical')
    elif orientation == 'horizontal':
        combined_img = stack_images(img1, img2, orientation='horizontal')
    else:
        raise ValueError("Orientation must be 'vertical' or 'horizontal'")

    # Draw matches on the combined image
    if orientation == 'vertical':
        height1 = img1.shape[0]
        adjusted_kp2 = [cv2.KeyPoint(kp.pt[0], kp.pt[1] + height1, kp.size) for kp in kp2]
    else:
        width1 = img1.shape[1]
        adjusted_kp2 = [cv2.KeyPoint(kp.pt[0] + width1, kp.pt[1], kp.size) for kp in kp2]

    for match in matches:
        pt1 = (int(kp1[match.queryIdx].pt[0]), int(kp1[match.queryIdx].pt[1]))
        pt2 = (int(adjusted_kp2[match.trainIdx].pt[0]), int(adjusted_kp2[match.trainIdx].pt[1]))

        # Draw lines between the matched keypoints
        cv2.line(combined_img, pt1, pt2, (0, 255, 0), 1)

        # Draw circles around keypoints
        cv2.circle(combined_img, pt1, 5, (255, 255, 255), -1)
        cv2.circle(combined_img, pt2, 5, (255, 255, 255), -1)

    return combined_img


def put_text(image, org, text, color=(0, 0, 255), font_scale=1, thickness=1, font=cv2.FONT_HERSHEY_DUPLEX):
    # to make sure it is writable
    image = image.copy()

    # Get the size of the text
    (label_width, label_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    w, h = image.shape[1], image.shape[0]

    place_h, place_w = org.split("_")

    org_w = 0
    org_h = 0

    # Calculate vertical position
    if place_h == "top":
        org_h = label_height + 10  # Add padding for top
    elif place_h == "bottom":
        org_h = h - label_height
    elif place_h == "center":
        org_h = h // 2 + label_height // 2

    # Calculate horizontal position
    if place_w == "left":
        org_w = 0
    elif place_w == "right":
        org_w = w - label_width
    elif place_w == "center":
        org_w = w // 2 - label_width // 2

    # Draw the text on the image using OpenCV
    cv2.putText(image, text, (org_w, org_h), font, font_scale, color, thickness, cv2.LINE_AA)
    return image


def stack_images(img1, img2, orientation='horizontal'):
    if orientation == 'vertical':
        combined_img = np.vstack((img1, img2))
    else:  # horizontal
        combined_img = np.hstack((img1, img2))

    return combined_img


def visualize_paths(gt_path, pred_path, title="VO exercises", file_out="plot.png"):
    # Convert to NumPy arrays
    gt_path = np.array(gt_path)
    pred_path = np.array(pred_path)

    # Extract x, y coordinates
    gt_x, gt_y = gt_path.T
    pred_x, pred_y = pred_path.T

    # Calculate error (Euclidean distance)
    diff = np.linalg.norm(gt_path - pred_path, axis=1)

    # Prepare data for paths
    path_df = pd.DataFrame({
        'x': np.concatenate([gt_x, pred_x]),
        'y': np.concatenate([gt_y, pred_y]),
        'Path': ['Ground Truth'] * len(gt_x) + ['Prediction'] * len(pred_x)
    })

    # Prepare data for errors
    error_df = pd.DataFrame({
        'Frame': np.arange(len(diff)),
        'Error': diff
    })

    # Set up the figure and axes
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    # Plot the paths using seaborn
    sns.lineplot(data=path_df, x='x', y='y', hue='Path', ax=axs[0], palette=['blue', 'green'], linewidth=2)
    axs[0].set_title('Ground Truth vs Predicted Path')
    axs[0].set_xlabel('x')
    axs[0].set_ylabel('y')
    axs[0].legend(title='Path')

    # Plot the error over frames using seaborn
    sns.lineplot(data=error_df, x='Frame', y='Error', ax=axs[1], color='red', linewidth=2)
    axs[1].set_title('Error per Frame')
    axs[1].set_xlabel('Frame')
    axs[1].set_ylabel('Error')

    # Adjust the layout and show the plots
    plt.suptitle(title)
    plt.tight_layout()

    # Save the plot
    plt.savefig(file_out)
    # plt.show()


def visualize_paths2(gt_path, pred_path, html_tile="", title="VO exercises", file_out="plot.html"):
    from bokeh.io import output_file, show
    from bokeh.plotting import figure, ColumnDataSource
    from bokeh.layouts import layout
    from bokeh.models import Div

    output_file(file_out, title=html_tile)

    gt_path = np.array(gt_path)
    pred_path = np.array(pred_path)

    tools = "pan,wheel_zoom,box_zoom,box_select,lasso_select,reset"

    gt_x, gt_y = gt_path.T
    pred_x, pred_y = pred_path.T
    xs = list(np.array([gt_x, pred_x]).T)
    ys = list(np.array([gt_y, pred_y]).T)

    diff = np.linalg.norm(gt_path - pred_path, axis=1)

    source = ColumnDataSource(
        data=dict(
            gtx=gt_path[:, 0], gty=gt_path[:, 1],
            px=pred_path[:, 0], py=pred_path[:, 1],
            diffx=np.arange(len(diff)), diffy=diff,
            disx=xs, disy=ys)
    )

    fig1 = figure(
        title="Paths",
        tools=tools,
        match_aspect=True,
        width_policy="max",
        toolbar_location="above",
        x_axis_label="x",
        y_axis_label="y"
    )

    fig1.circle("gtx", "gty", source=source, color="blue", hover_fill_color="firebrick", legend_label="GT")
    fig1.line("gtx", "gty", source=source, color="blue", legend_label="GT")

    fig1.circle("px", "py", source=source, color="green", hover_fill_color="firebrick", legend_label="Pred")
    fig1.line("px", "py", source=source, color="green", legend_label="Pred")

    fig1.multi_line("disx", "disy", source=source, legend_label="Error", color="red", line_dash="dashed")
    fig1.legend.click_policy = "hide"

    fig2 = figure(
        title="Error",
        tools=tools,
        width_policy="max",
        toolbar_location="above",
        x_axis_label="frame",
        y_axis_label="error"
    )

    fig2.circle("diffx", "diffy", source=source, hover_fill_color="firebrick", legend_label="Error")
    fig2.line("diffx", "diffy", source=source, legend_label="Error")

    show(layout([Div(text=f"<h1>{title}</h1>"),
                 Div(text="<h2>Paths</h1>"),
                 [fig1, fig2],
                 ], sizing_mode='scale_width'))


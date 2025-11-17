import argparse

from models.StreamVGGT.demo_gradio import gradio_demo, model   

def main():
    parser = argparse.ArgumentParser(description="Run StreamVGGT 3D reconstruction without Gradio UI")
    parser.add_argument("--target_dir", type=str, required=True,
                        help="Directory containing images/ folder")
    parser.add_argument("--conf_thres", type=float, default=3.0,
                        help="Confidence threshold")
    parser.add_argument("--frame_filter", type=str, default="All",
                        help="Frame filter (e.g., 'All' or '0:000001.png')")
    parser.add_argument("--mask_black_bg", action="store_true", help="Enable black background masking")
    parser.add_argument("--mask_white_bg", action="store_true", help="Enable white background masking")
    parser.add_argument("--show_cam", action="store_true", help="Show camera positions")
    parser.add_argument("--mask_sky", action="store_true", help="Filter out sky")
    parser.add_argument("--prediction_mode", type=str,
                        default="Pointmap Regression",
                        choices=["Pointmap Regression", "Depthmap and Camera Branch", "Pointmap Branch"],
                        help="Prediction mode")
    args = parser.parse_args()

    # ---------------- Run reconstruction ----------------
    print("Running reconstruction...")
    glb_path, log_msg, _ = gradio_demo(
        target_dir=args.target_dir,
        conf_thres=args.conf_thres,
        frame_filter=args.frame_filter,
        mask_black_bg=args.mask_black_bg,
        mask_white_bg=args.mask_white_bg,
        show_cam=args.show_cam,
        mask_sky=args.mask_sky,
        prediction_mode=args.prediction_mode,
    )

    print("--------------------------------------------------------")
    print("StreamVGGT CLI Reconstruction Finished")
    print("Log:", log_msg)
    if glb_path:
        print("Generated GLB:", glb_path)
    else:
        print("Failed. Check your target_dir.")
    print("--------------------------------------------------------")


if __name__ == "__main__":
    main()
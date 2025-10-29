import argparse

from ultralytics import YOLO

def train_model(args):
    # Create a new YOLO11n-OBB model from scratch
    model = YOLO(args.model_config)

    # Train the model on the random_poses_0.2ms dataset
    path_to_yaml = "datasets/random_poses_0.2ms/random_poses.yaml"
    if args.grayscale:
        path_to_yaml = "datasets/random_poses_0.2ms/random_poses_grayscale.yaml"
    save_dir = f"runs/obb/{args.model_config.split('.')[0]}_{'gray' if args.grayscale else 'color'}_e{args.epochs}_sz{args.imgsz}_b{args.batch}"
    results = model.train(data=path_to_yaml, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, save_dir=save_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_config", type=str, default="yolo11n-obb.yaml", help="Model configuration file name.")
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs.")
    parser.add_argument("--imgsz", type=int, default=1024, help="Image size for training.")
    parser.add_argument("--batch", type=int, default=16, help="Batch size for training.")
    parser.add_argument("--grayscale", action='store_true', help="Use grayscale images if set.")
    args = parser.parse_args()

    # Train!
    train_model(args)

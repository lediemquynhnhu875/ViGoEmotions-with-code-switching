from train_baseline import main


if __name__ == "__main__":
    import sys

    sys.argv.extend(["--model_key", "visobert"])
    main()

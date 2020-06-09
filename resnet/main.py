"""Main module to training classifier (ResNet)
"""
import config
import trainer

if __name__ == "__main__":
    args = config.get_config()
    print(args)

    trainer.train(args)
    print("Well Done.")

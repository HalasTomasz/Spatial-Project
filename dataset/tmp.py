from datasets import load_dataset
from PIL import Image
from torchvision import transforms

# Load the dataset
dataset = load_dataset("kraina/text2tile")

# Define a preprocessing function for images and captions
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize to 256x256
    transforms.ToTensor()          # Convert image to tensor
])

def preprocess(examples):
    processed_images = []
    processed_captions = []
    for image_file, caption in zip(examples["image"], examples["caption"]):
        # Open and process the image
        image = Image.open(image_file).convert("RGB")
        processed_images.append(transform(image))
        
        # Process the caption
        processed_captions.append(caption)
    
    return {"image": processed_images, "caption": processed_captions}

# Apply the preprocessing function to the dataset
processed_dataset = dataset.map(preprocess, batched=True)

sample = processed_dataset["train"][0]
print("Image shape:", sample["image"].shape)
print("Caption:", sample["caption"])

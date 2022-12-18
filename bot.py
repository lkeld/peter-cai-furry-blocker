import discord
import requests
import tensorflow as tf
import tensorflow_hub as hub

client = discord.Client()

# Load the pre-trained TensorFlow model
model = hub.load("https://tfhub.dev/tf2-preview/mobilenet_v2_100_224/classification/2")

@client.event
async def on_message(message):
    # Check if the message contains an attachment (such as an image)
    if message.attachments:
        # Download the attachment
        attachment = message.attachments[0]
        file_url = attachment.url
        response = requests.get(file_url)
        file_data = response.content

        # Use the pre-trained model to classify the attachment as containing furries or not
        image = tf.image.decode_jpeg(file_data, channels=3)
        image = tf.image.resize(image, (224, 224))
        image = image / 255.0
        image = tf.expand_dims(image, 0)
        predictions = model(image)
        classes = ["Contains furries", "Does not contain furries"]
        prediction = classes[tf.argmax(predictions, axis=-1)[0]]

        # If the model predicts that the attachment contains furries, delete the message and send a message to the user
        if prediction == "Contains furries":
            await message.delete()
            await message.channel.send(f"{message.author.mention}, your message was blocked because it contained an image of furries.")

client.run(bot_token)

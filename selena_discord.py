import discord
from discord.ext import commands
import os
from dotenv import load_dotenv
import json
import random
import torch
from model import NeuralNet
from nltk_utils import tokenize, stem, bow_vector


load_dotenv('token.env')
TOKEN = os.getenv("discord_token")


with open('intents.json', 'r') as f:
    intents = json.load(f)

device = torch.device('cpu')

# Load the saved state dictionary
model_data = torch.load('data.pth', map_location=device)
all_words = model_data['all_words']
tags = model_data['tags']
model_state = model_data['model_state']
input_size = model_data['input_size']
hidden_size = model_data['hidden_size']
output_size = model_data['output_size']
# print(tags)

# Create an instance of your model class
model = NeuralNet(input_size, hidden_size, output_size)

# Load the state dictionary into the model
model.load_state_dict(model_state)

# Set the model to evaluation mode
model.eval()

discord_intent = discord.Intents.default()
discord_intent.members = True
discord_intent.message_content = True

# client = discord.Client(intents=discord_intent)
client = commands.Bot(command_prefix='!', intents=discord_intent)


def predict_intent(sentence):
    tokens = tokenize(sentence)
    stemmed_tokens = [stem(token) for token in tokens]
    inputs = bow_vector(stemmed_tokens, all_words)
    outputs = model(inputs)
    _, predicted = torch.max(outputs, dim=1)
    intent = predicted.item()
    return intent


@client.event
async def on_ready():
    print(f'Logged in as {client.user}')


@client.event
async def on_message(message):
    if message.author == client.user:
        return
    # message receive
    print(f"{message.author.name}#{message.author.discriminator} said {message.content}")
    await client.process_commands(message)

    intent = predict_intent(message.content)
    responses = intents['intents'][intent]['responses']
    response = random.choice(responses)
    type_of_intent = tags[intent]
    # bot reply and type of intent
    print(f"{client.user} reply {response} : type= {type_of_intent}")
    await message.channel.send(response)

client.run(TOKEN)

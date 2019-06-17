from __future__ import division
from __future__ import print_function

import torch
import torchvision
import numpy as np
import scipy.misc as smp

from torch import nn
from torchvision import transforms
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

###
# Author Clay O'Neil
# Paper https://arxiv.org/abs/1511.06434
###

batch_size = 32
train_steps = 1000
latent_space = 100
epochs = 100

class G(torch.nn.Module):
  def __init__(self):
    super(G, self).__init__()
    self.l1 = nn.Sequential(
                        nn.ConvTranspose2d(in_channels=latent_space, out_channels=1024, kernel_size=5, stride=1, bias=False), 
                        nn.ReLU(inplace=True), 
                        torch.nn.BatchNorm2d(1024), 
                        nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=5, stride=2, bias=False),
                        nn.ReLU(inplace=True), 
                        torch.nn.BatchNorm2d(512), 
                        nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=5, stride=2, bias=False),
                        nn.ReLU(inplace=True), 
                        torch.nn.BatchNorm2d(256),
                        nn.ConvTranspose2d(in_channels=256, out_channels=3, kernel_size=4, stride=1, bias=False),  
                        nn.Tanh()
                        ) 
    self.l2 = lambda x : nn.functional.interpolate(input=x, size=(32, 32),  mode='nearest')

  def forward(self, z):
    conv1 = self.l1(z)
    return conv1


class D(torch.nn.Module):
  def __init__(self):
    super(D, self).__init__()
    self.l1 = nn.Sequential(nn.Conv2d(3, 128, 4, stride=2, padding=1, bias=False), # batch_size x 64 x 16 x 16
                            nn.LeakyReLU(0.2, inplace=True), 
                            nn.Conv2d(128, 256, 4, stride=2, padding=1, bias=False), # batch_size x 128 x 8 x 8
                            torch.nn.BatchNorm2d(256),
                            nn.LeakyReLU(0.2, inplace=True),
                            nn.Conv2d(256, 512, 4, stride=2, padding=1, bias=False), # batch_size x 256 x 4 x 4
                            torch.nn.BatchNorm2d(512),
                            nn.LeakyReLU(0.2, inplace=True),  
                            nn.Conv2d(512, 1, 4, stride=1, padding=0, bias=False), # batch_size x 1 x 1 x 1
                            nn.Sigmoid()
                            )
    self.l2 = lambda x : nn.Linear(x.size()[1], 1)(x)

  def forward(self, x):
    conv1 = self.l1(x)
    return conv1.view(-1)

def sample_noise(sampler):
  noise = sampler.sample()

  return noise

def save_photo(photo, name):
  img_x = smp.toimage(photo)
  smp.imsave('./images/' + name + '.jpg', img_x)

def init_weights(m):
  print(m)
  if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
    m.weight.data.normal_(0.0, 0.02)
    print(m.weight)
  elif type(m) == nn.BatchNorm2d :
    m.weight.data.normal_(1.0, 0.02)
    m.bias.data.fill_(0)


sampler = torch.distributions.uniform.Uniform(torch.zeros(batch_size, latent_space, 1, 1), torch.ones(batch_size, latent_space, 1, 1))

generator = G()
descriminator = D()

generator.apply(init_weights)
descriminator.apply(init_weights)

criterion = nn.BCELoss()
g_losses = []
d_losses = []

g_optimizer = torch.optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
d_optimizer = torch.optim.Adam(descriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))

data = torchvision.datasets.CIFAR10('./data', train=True, transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        ]), download=False)

loader = torch.utils.data.DataLoader(
                  dataset=data,
                  batch_size=batch_size,
                  shuffle=True,
                  drop_last=True,
                  pin_memory=False)

for epoch in range(epochs):
  for t, photos in enumerate(loader):

    # Prepare batch of data
    photos = photos[0].view(batch_size, 3, 32, 32)
    real_labels = torch.ones(batch_size)
    fake_labels = torch.zeros(batch_size)
    noise = sample_noise(sampler)

    # Update descriminator to maximize log(D(x)) + log(1 - D(G(z)))
    d_optimizer.zero_grad()

    d_real_probs = descriminator(photos)
    fake_photos = generator(noise)
    d_fake_probs = descriminator(fake_photos.detach())

    dlr = criterion(d_real_probs, real_labels)
    dlf = criterion(d_fake_probs, fake_labels)
    de = dlf + dlr
    de.backward()

    d_optimizer.step()
    d_losses.append(de.item())
  
    # Update generator to maximize log(D(G(z)))
    g_optimizer.zero_grad()

    g_fake_probs = descriminator(fake_photos)
    ge = criterion(g_fake_probs, real_labels) # We want the generator to learn to make the fake photos be "real"

    ge.backward()
    g_optimizer.step()
    g_losses.append(ge.item())

    # Logging
    print('epoch', epoch + 1, 'batch', t + 1, 'generator loss', ge.item())

    d_real_acc = d_real_probs.mean().item()
    d_fake_acc = 1 - d_fake_probs.mean().item()

    print('epoch', epoch + 1, 'batch', t + 1, 'descriminator loss', de.item(), dlr.item(), dlf.item(), 'real acc', d_real_acc, 'fake acc', d_fake_acc)

    if t % 25 == 0:
      save_photo(fake_photos.detach()[0].view(3,32,32), str(epoch) + '_' + str(t))
   
  t = np.linspace(0, len(loader) * (epoch + 1) , len(d_losses))
  y = np.cos(np.pi * (t / len(d_losses)))

  plt.scatter(t, d_losses, c=y, s=1)

  plt.xlabel('batches', fontsize=14, color='red')
  plt.ylabel('loss', fontsize=14, color='red')
  plt.savefig('d_loss' + str(epoch))

  plt.clf()

  t = np.linspace(0, len(loader) * (epoch + 1) , len(g_losses))
  y = np.cos(np.pi * (t / len(g_losses)))

  plt.scatter(t, g_losses, c=y, s=1)

  plt.xlabel('batches', fontsize=14, color='red')
  plt.ylabel('loss', fontsize=14, color='red')
  plt.savefig('g_loss' + str(epoch))

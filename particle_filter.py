#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import math
from probabilistic_lib.functions import angle_wrap, get_polar_line


# ===============================================================================
class ParticleFilter(object):
  '''
    Class to hold the whole particle filter.

    p_wei: weights of particles in array of shape (N,)
    p_ang: angle in radians of each particle with respect of world axis, shape (N,)
    p_xy : position in the world frame of the particles, shape (2,N)
    '''

  # ===========================================================================
  def __init__(self, room_map, num, odom_lin_sigma, odom_ang_sigma,
               meas_rng_noise, meas_ang_noise, x_init, y_init, theta_init):
    '''
        Initializes the particle filter
        room_map : an array of lines in the form [x1 y1 x2 y2]
        num      : number of particles to use
        odom_lin_sigma: odometry linear noise
        odom_ang_sigma: odometry angular noise
        meas_rng_noise: measurement linear noise
        meas_ang_noise: measurement angular noise
        '''

    # Copy parameters
    self.map = room_map
    self.num = num
    self.odom_lin_sigma = odom_lin_sigma
    self.odom_ang_sigma = odom_ang_sigma
    self.meas_rng_noise = meas_rng_noise
    self.meas_ang_noise = meas_ang_noise

    # Map
    map_xmin = np.min(self.map[:, 0])
    map_xmax = np.max(self.map[:, 0])
    map_ymin = np.min(self.map[:, 1])
    map_ymax = np.max(self.map[:, 1])

    # Particle initialization arround starting point
    self.p_wei = 1.0 / num * np.ones(num)
    self.p_ang = theta_init + 0.3 * np.random.rand(num)
    self.p_xy = np.vstack((x_init + 1 * np.random.rand(num) - 0.5, y_init + 1 * np.random.rand(num) - 0.5))
    # Flags for resampling
    self.moving = False
    self.n_eff = 0  # Initialize Efficent number as 0

  # ===========================================================================
  def predict(self, odom):
    '''
        Moves particles with the given odometry.
        odom: incremental odometry [delta_x delta_y delta_yaw] in the vehicle frame
        '''
    # Check if we have moved from previous reading.
    if odom[0] == 0 and odom[1] == 0 and odom[2] == 0:
      self.moving = False
    else:
      # TODO: code here!!
      # Add Gaussian noise to odometry measures
      # .... np.random.randn(...)
      # Create white gaussian noise of zero mean with the same size as number of particles
      noise_x = self.odom_lin_sigma * np.random.randn(self.num) 
      noise_y = self.odom_lin_sigma * np.random.randn(self.num) 
      noise_ang = self.odom_ang_sigma * np.random.randn(self.num)
 
      # Convert odometry reading to world frame, add noise to it, then add it to the previous state of all particles
      self.p_xy[0, :] += ((odom[0] * np.cos(self.p_ang[:])) + (odom[1] * -np.sin(self.p_ang[:])) + noise_x)
      self.p_xy[1, :] += ((odom[0] * np.sin(self.p_ang[:])) + (odom[1] * np.cos(self.p_ang[:])) + noise_y)

      # Add some noise to the odometry angle measurement and add the result to the angle of all particles
      self.p_ang[:] += (odom[2] + noise_ang)
      self.p_ang = angle_wrap(self.p_ang)
 
      # Update flag for resampling
      self.moving = True


  def weight(self, lines):
    '''
        Look for the lines seen from the robot and compare them to the given map.
        Lines expressed as [x1 y1 x2 y2].
        '''
    val_rng = 1.0 / (self.meas_rng_noise * np.sqrt(2 * np.pi))
    val_ang = 1.0 / (self.meas_ang_noise * np.sqrt(2 * np.pi))


    # create a numpy array to store the transformed lines of room map in polar coordinate
    map_lines = np.zeros((self.map.shape[0],2))

    for i in range(self.num):
      # create a numpy array of weights to store the weights 
      weight = np.zeros((lines.shape[0], self.map.shape[0]))
      # create a numpy array that stores the maximum weight
      max_weights = np.zeros(lines.shape[0])

      for j in range(lines.shape[0]):
        # convert a single line from the robot to polar coordinate [range, theta]
        measured_line = get_polar_line(lines[j,:])

        # calculate the length of the line: NEEDED FOR THE OPTIONAL STEP
        line_len = math.dist(lines[j,:2], lines[j,2:4])

        for k in range(self.map.shape[0]):
          # convert each line in room map to polar coordinate [range, theta]
          map_lines[k, :] = get_polar_line(self.map[k,:], [self.p_xy[0, i], self.p_xy[1, i], self.p_ang[i]])
          
          # calculate the probabilities of range and bearing for a single line observed by the robot against a single line from the map
          rng_wei = val_rng * np.exp(-pow(measured_line[0] - map_lines[k, 0],2) / (2*(self.meas_rng_noise)**2))
          ang_wei = val_ang * np.exp(-pow(measured_line[1] - map_lines[k, 1],2) / (2*(self.meas_ang_noise)**2))
          
          # multiply the two probabilities and store the result in weight
          weight[j, k] = rng_wei * ang_wei
          
          # calculate the length of each line map: PART OF THE OPTIONAL STEP
          map_line_len = math.dist(self.map[k,:2], self.map[k,2:4])
          
          # compare the length of a line observed by the robot against all the lines in map, do one at a time
          # if the length of the line is greater than length of map lines, set the weight at point to zero 
          # OPTIONAL STEP IMPLEMENTED
          if line_len > map_line_len:
            weight[j,k] = 0

        # for each comparison of a line from the robot against all the lines in the map, get the max weight obtained 
        # and store it max_weights:
        max_weights[j] = np.amax(weight[j,:])

      # multiply all the maximum weights obtained using the first particle with the prior weight of the particle
      #  and assign the new weight to the particle
      self.p_wei[i] *= np.prod(np.array(max_weights))
    
    # normalize the weights of each particle
    self.p_wei /= np.sum(self.p_wei)

    # TODO: Compute efficient number
    self.n_eff = 1.0/ np.sum(np.square(self.p_wei))




  def resample(self):
    '''
        Systematic resampling of the particles.
        '''

    # TODO: code here!!
    # Look for particles to replicate

    M = self.num
    W = np.sum(self.p_wei)
    r = np.random.uniform(0, W / M)
    c = self.p_wei[0]
    i = 0
    for m in range(M):
      u = r + (m) * (W / M)
      while u > c:
        i = i + 1
        c = c + self.p_wei[i]
      self.p_xy[:, m] = self.p_xy[:, i]
      self.p_ang[m] = self.p_ang[i]

    # reset the weight of all particles
    self.p_wei = 1.0 / self.num * np.ones(self.num)

  # ===========================================================================
  def get_mean_particle(self):
    '''
        Gets mean particle.
        '''
    # Weighted mean
    weig = np.vstack((self.p_wei, self.p_wei))
    mean = np.sum(self.p_xy * weig, axis=1) / np.sum(self.p_wei)

    ang = np.arctan2(
      np.sum(self.p_wei * np.sin(self.p_ang)) / np.sum(self.p_wei),
      np.sum(self.p_wei * np.cos(self.p_ang)) / np.sum(self.p_wei))

    return np.array([mean[0], mean[1], ang])


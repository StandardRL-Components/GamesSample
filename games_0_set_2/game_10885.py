import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T11:08:46.548953
# Source Brief: brief_00885.md
# Brief Index: 885
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque

class GameEnv(gym.Env):
    """
    A Gymnasium environment where the agent controls a train car.
    The goal is to collect fuel and synchronize speed with changing track
    elevations to complete 5 levels without derailing or running out of fuel.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Control a high-speed train, collecting fuel and matching speed to the track's "
        "elevation to avoid derailing. Complete all levels to win."
    )
    user_guide = (
        "Controls: ↑ to accelerate, ↓ to decelerate. Use ← and → to switch between lanes to collect fuel."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    
    # Colors
    COLOR_BG = (16, 16, 24)         # Dark blue-black
    COLOR_TRACK = (64, 64, 80)      # Mid-grey
    COLOR_TRAIN = (255, 64, 64)     # Bright red
    COLOR_TRAIN_GLOW = (255, 100, 100)
    COLOR_FUEL = (255, 255, 0)      # Bright yellow
    COLOR_FUEL_GLOW = (255, 255, 150)
    COLOR_UI_TEXT = (220, 220, 240)
    COLOR_UI_BAR_BG = (40, 40, 60)
    COLOR_UI_FUEL_BAR = (255, 180, 0)
    COLOR_PARTICLE_FUEL = (255, 220, 100)
    COLOR_PARTICLE_SMOKE = (180, 180, 190)

    # Game Parameters
    TRAIN_X_POS = 120
    NUM_LANES = 3
    LANE_WIDTH = 40
    TRACK_CENTER_Y = 250
    TRACK_SEGMENT_WIDTH = 20
    LEVEL_LENGTH_PIXELS = 4000
    MAX_LEVELS = 5
    MAX_STEPS = 5000

    # Physics
    MIN_SPEED = 2.0
    MAX_SPEED = 12.0
    SPEED_ACCEL = 0.2
    SPEED_DECEL = 0.2
    DERAIL_TOLERANCE = 2.5
    FUEL_CONSUMPTION_RATE = 0.005 # per pixel traveled
    FUEL_PICKUP_AMOUNT = 25.0

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        self.render_mode = render_mode

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("monospace", 16, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 48, bold=True)

        # Initialize state variables
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.level = 1
        self.level_progress = 0.0
        self.fuel = 100.0
        self.train_speed = 5.0
        self.train_target_speed = 5.0
        self.train_lane = 1
        self.train_target_lane = 1
        self.track_segments = deque()
        self.fuel_pickups = []
        self.particles = []
        
        # This will be populated in reset()
        self.current_track_y = self.TRACK_CENTER_Y
        self.current_track_ideal_speed = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.level = 1
        self.level_progress = 0.0
        self.fuel = 100.0
        self.train_speed = 5.0
        self.train_target_speed = 5.0
        self.train_lane = 1
        self.train_target_lane = 1
        
        self.track_segments.clear()
        self.fuel_pickups.clear()
        self.particles.clear()

        # Procedurally generate the initial track
        last_y = self.TRACK_CENTER_Y
        for i in range(self.SCREEN_WIDTH // self.TRACK_SEGMENT_WIDTH + 5):
            x = i * self.TRACK_SEGMENT_WIDTH
            last_y = self._generate_track_segment(x, last_y)

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0
        terminated = False
        truncated = False
        
        self._handle_input(action)
        self._update_train_state()
        self._update_world()
        
        collision_reward = self._check_collisions()
        reward += collision_reward

        self.steps += 1
        
        termination_reason, terminal_reward = self._check_termination()
        if termination_reason is not None:
            terminated = True
            reward += terminal_reward
            self.game_over = True
            if termination_reason == "win":
                self.score += 100
            elif termination_reason == "steps":
                truncated = True
                terminated = False
            else:
                self.score -= 100

        self.score += collision_reward

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, action):
        movement = action[0]

        # Action 1: Speed up, Action 2: Speed down
        if movement == 1:
            self.train_target_speed += self.SPEED_ACCEL
        elif movement == 2:
            self.train_target_speed -= self.SPEED_DECEL
        self.train_target_speed = np.clip(self.train_target_speed, self.MIN_SPEED, self.MAX_SPEED)

        # Action 3: Move left, Action 4: Move right
        if movement == 3:
            self.train_target_lane = max(0, self.train_lane - 1)
        elif movement == 4:
            self.train_target_lane = min(self.NUM_LANES - 1, self.train_lane + 1)
        
    def _update_train_state(self):
        # Smoothly interpolate speed
        self.train_speed += (self.train_target_speed - self.train_speed) * 0.1
        
        # For discrete lanes, we can just snap to the target lane
        self.train_lane = self.train_target_lane

    def _update_world(self):
        # Update progress and fuel
        self.level_progress += self.train_speed
        self.fuel -= self.train_speed * self.FUEL_CONSUMPTION_RATE

        # Move track and fuel pickups
        for seg in self.track_segments:
            seg['x'] -= self.train_speed
        for fuel in self.fuel_pickups:
            fuel['x'] -= self.train_speed
        for p in self.particles:
            p['x'] -= self.train_speed

        # Remove off-screen elements
        if self.track_segments and self.track_segments[0]['x'] < -self.TRACK_SEGMENT_WIDTH:
            self.track_segments.popleft()
        self.fuel_pickups = [f for f in self.fuel_pickups if f['x'] > -20]
        self.particles = [p for p in self.particles if p['life'] > 0]
        
        # Generate new track segments
        while self.track_segments[-1]['x'] < self.SCREEN_WIDTH + self.TRACK_SEGMENT_WIDTH:
            last_seg = self.track_segments[-1]
            self._generate_track_segment(last_seg['x'] + self.TRACK_SEGMENT_WIDTH, last_seg['y'])

        # Update train's vertical position and ideal speed based on the track below it
        for i in range(len(self.track_segments) - 1):
            seg1 = self.track_segments[i]
            seg2 = self.track_segments[i+1]
            if seg1['x'] <= self.TRAIN_X_POS < seg2['x']:
                # Linear interpolation for smooth vertical movement
                interp_factor = (self.TRAIN_X_POS - seg1['x']) / self.TRACK_SEGMENT_WIDTH
                self.current_track_y = seg1['y'] + (seg2['y'] - seg1['y']) * interp_factor
                self.current_track_ideal_speed = seg1['ideal_speed'] + (seg2['ideal_speed'] - seg1['ideal_speed']) * interp_factor
                break
        
        # Update particles
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            p['radius'] *= 0.98

        # Emit smoke particles from train
        if self.steps % (max(1, int(15 - self.train_speed))) == 0:
            self._create_particles(1, self.COLOR_PARTICLE_SMOKE, 
                                   [self.TRAIN_X_POS - 15, self.get_train_y() - 10], 
                                   vel_x_range=(-0.5, -0.2), vel_y_range=(-0.5, 0.5), 
                                   radius=random.uniform(3, 6), life=40)

    def _generate_track_segment(self, x, last_y):
        difficulty_factor = 0.1 + (self.level - 1) * 0.05
        max_dy = 15 * difficulty_factor
        
        new_y = last_y + self.np_random.uniform(-max_dy, max_dy)
        new_y = np.clip(new_y, self.TRACK_CENTER_Y - 120, self.TRACK_CENTER_Y + 120)
        
        # Ideal speed is higher when the track is higher (like cresting a hill)
        ideal_speed = self.MIN_SPEED + (self.MAX_SPEED - self.MIN_SPEED) * \
                      ((self.TRACK_CENTER_Y - new_y + 120) / 240)
        
        self.track_segments.append({'x': x, 'y': new_y, 'ideal_speed': ideal_speed})

        # Chance to spawn fuel
        if self.np_random.random() < 0.1: # 10% chance per segment
            lane = self.np_random.integers(0, self.NUM_LANES)
            self.fuel_pickups.append({'x': x, 'lane': lane, 'rect': None})

        return new_y

    def _check_collisions(self):
        reward = 0
        train_y = self.get_train_y()
        train_rect = pygame.Rect(self.get_train_x() - 15, train_y - 10, 30, 20)
        
        for fuel in self.fuel_pickups[:]:
            fuel_y = self.get_y_for_lane(fuel['x'], fuel['lane'])
            fuel_rect = pygame.Rect(fuel['x'] - 5, fuel_y - 5, 10, 10)
            if train_rect.colliderect(fuel_rect):
                self.fuel = min(100.0, self.fuel + self.FUEL_PICKUP_AMOUNT)
                reward += 10.0 # Changed from 0.1 to 10 for better scaling
                self.fuel_pickups.remove(fuel)
                # sfx: fuel pickup sound
                self._create_particles(20, self.COLOR_PARTICLE_FUEL, [fuel['x'], fuel_y], life=30, radius=4)
        return reward

    def _check_termination(self):
        # Win condition
        if self.level > self.MAX_LEVELS:
            return "win", 100.0
        
        # Level completion
        if self.level_progress >= self.LEVEL_LENGTH_PIXELS:
            self.level += 1
            self.level_progress = 0
            # sfx: level complete fanfare
            return None, 50.0 # Reward for completing a level

        # Derailment
        speed_diff = abs(self.train_speed - self.current_track_ideal_speed)
        if speed_diff > self.DERAIL_TOLERANCE:
            # sfx: crash/explosion sound
            self._create_particles(50, self.COLOR_TRAIN, [self.get_train_x(), self.get_train_y()], 
                                   vel_x_range=(-3, 3), vel_y_range=(-3, 3), life=60, radius=5)
            return "derail", -100.0

        # Out of fuel
        if self.fuel <= 0:
            # sfx: failure sound
            return "fuel", -100.0

        # Max steps
        if self.steps >= self.MAX_STEPS:
            return "steps", -50.0

        return None, 0

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "level": self.level,
            "fuel": self.fuel,
            "train_speed": self.train_speed,
            "ideal_speed": self.current_track_ideal_speed,
        }

    def _render_game(self):
        self._render_track()
        self._render_fuel_pickups()
        self._render_particles()
        self._render_train()

    def _render_track(self):
        points_by_lane = [[] for _ in range(self.NUM_LANES)]
        for seg in self.track_segments:
            for i in range(self.NUM_LANES):
                y = seg['y'] + (i - (self.NUM_LANES - 1) / 2) * 10 # Lane vertical separation
                points_by_lane[i].append((int(seg['x']), int(y)))
        
        for lane_points in points_by_lane:
            if len(lane_points) > 1:
                pygame.draw.aalines(self.screen, self.COLOR_TRACK, False, lane_points, 1)

    def _render_fuel_pickups(self):
        for fuel in self.fuel_pickups:
            y = self.get_y_for_lane(fuel['x'], fuel['lane'])
            pos = (int(fuel['x']), int(y))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 7, self.COLOR_FUEL_GLOW)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 5, self.COLOR_FUEL)

    def _render_train(self):
        if self.game_over and self._check_termination()[0] != "steps": return

        x = int(self.get_train_x())
        y = int(self.get_train_y())
        
        # Glow
        glow_radius = int(25 + abs(self.train_speed - self.current_track_ideal_speed) * 5)
        temp_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(temp_surf, (*self.COLOR_TRAIN_GLOW, 50), (glow_radius, glow_radius), glow_radius)
        self.screen.blit(temp_surf, (x - glow_radius, y - glow_radius), special_flags=pygame.BLEND_RGBA_ADD)

        # Body
        train_rect = pygame.Rect(x - 15, y - 10, 30, 20)
        pygame.draw.rect(self.screen, self.COLOR_TRAIN, train_rect, border_radius=3)
        pygame.draw.rect(self.screen, (255,255,255), (x - 10, y - 8, 8, 8), border_radius=2) # Window

    def _render_particles(self):
        for p in self.particles:
            alpha = int(255 * (p['life'] / p['max_life']))
            color = (*p['color'], alpha)
            pos = (int(p['pos'][0]), int(p['pos'][1]))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], int(p['radius']), color)

    def _render_ui(self):
        # Fuel Gauge
        fuel_bar_w = 200
        pygame.draw.rect(self.screen, self.COLOR_UI_BAR_BG, (10, 10, fuel_bar_w, 20))
        fuel_width = int(fuel_bar_w * (self.fuel / 100.0))
        pygame.draw.rect(self.screen, self.COLOR_UI_FUEL_BAR, (10, 10, max(0, fuel_width), 20))
        fuel_text = self.font_small.render("FUEL", True, self.COLOR_UI_TEXT)
        self.screen.blit(fuel_text, (15, 12))

        # Speed Indicator
        speed_bar_w = 200
        pygame.draw.rect(self.screen, self.COLOR_UI_BAR_BG, (10, 35, speed_bar_w, 10))
        # Ideal speed range
        ideal_min = self.current_track_ideal_speed - self.DERAIL_TOLERANCE
        ideal_max = self.current_track_ideal_speed + self.DERAIL_TOLERANCE
        ideal_start_px = int(speed_bar_w * (ideal_min / self.MAX_SPEED))
        ideal_width_px = int(speed_bar_w * ((ideal_max - ideal_min) / self.MAX_SPEED))
        pygame.draw.rect(self.screen, (0, 100, 0), (10 + ideal_start_px, 35, ideal_width_px, 10))
        # Current speed
        speed_pos = int(speed_bar_w * (self.train_speed / self.MAX_SPEED))
        pygame.draw.rect(self.screen, self.COLOR_UI_TEXT, (10 + speed_pos - 1, 32, 3, 16))

        # Level Indicator
        level_text = self.font_small.render(f"LEVEL: {min(self.level, self.MAX_LEVELS)} / {self.MAX_LEVELS}", True, self.COLOR_UI_TEXT)
        self.screen.blit(level_text, (self.SCREEN_WIDTH - level_text.get_width() - 10, 10))

        # Game Over Message
        if self.game_over:
            reason, _ = self._check_termination()
            msg = "MISSION COMPLETE!" if reason == "win" else "GAME OVER"
            if reason == "steps":
                msg = "TIME UP"
            color = (100, 255, 100) if reason == "win" else (255, 100, 100)
            end_text = self.font_large.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    # --- Utility Methods ---
    def get_train_x(self):
        base_x = self.TRAIN_X_POS
        return base_x

    def get_train_y(self):
        lane_offset = (self.train_lane - (self.NUM_LANES - 1) / 2) * 10
        return self.current_track_y + lane_offset

    def get_y_for_lane(self, x_pos, lane):
        for i in range(len(self.track_segments) - 1):
            seg1 = self.track_segments[i]
            seg2 = self.track_segments[i+1]
            if seg1['x'] <= x_pos < seg2['x']:
                interp_factor = (x_pos - seg1['x']) / self.TRACK_SEGMENT_WIDTH
                base_y = seg1['y'] + (seg2['y'] - seg1['y']) * interp_factor
                lane_offset = (lane - (self.NUM_LANES - 1) / 2) * 10
                return base_y + lane_offset
        return self.TRACK_CENTER_Y # Fallback

    def _create_particles(self, count, color, pos, vel_x_range=(-1, 1), vel_y_range=(-1, 1), life=20, radius=3):
        for _ in range(count):
            self.particles.append({
                'pos': list(pos),
                'vel': [self.np_random.uniform(*vel_x_range), self.np_random.uniform(*vel_y_range)],
                'color': color,
                'radius': radius,
                'life': life,
                'max_life': life,
                'x': pos[0] # for world movement
            })

    def close(self):
        pygame.quit()

# Example usage:
if __name__ == '__main__':
    # Un-comment the line below to run with a display window
    # os.environ.pop("SDL_VIDEODRIVER", None)
    
    env = GameEnv(render_mode="rgb_array")
    
    # --- Manual Play ---
    # This allows a human to play the game to test the experience
    obs, info = env.reset()
    done = False
    
    # Pygame window for human play
    try:
        pygame.display.set_caption("Train Conductor")
        screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    except pygame.error:
        print("No display available, running headlessly.")
        screen = None

    total_reward = 0
    
    # Map keys to actions
    key_map = {
        pygame.K_UP: 1,
        pygame.K_DOWN: 2,
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4,
    }
    
    movement_action = 0 # No-op
    
    running = True
    while running:
        if screen:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key in key_map:
                        movement_action = key_map[event.key]
                    if event.key == pygame.K_r: # Reset on 'r'
                        obs, info = env.reset()
                        total_reward = 0
                        done = False
                if event.type == pygame.KEYUP:
                    if event.key in key_map and movement_action == key_map[event.key]:
                        movement_action = 0
        
        if done: # If the episode is done, wait for a reset
            if screen:
                env.clock.tick(60)
            continue

        # Construct the MultiDiscrete action
        # We only care about the first element for this game
        action = [movement_action, 0, 0]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated
        
        if done:
            print(f"Episode finished. Total Reward: {total_reward}")
            print(f"Info: {info}")
            if screen:
                # Render final frame
                surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
                screen.blit(surf, (0, 0))
                pygame.display.flip()
            # The env will automatically handle game over state, but we can reset
            # after a small delay for the player to see the result.
            # pygame.time.wait(2000) # This would pause everything
            
        # Render the observation to the display window
        if screen:
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            env.clock.tick(60) # Run at 60 FPS for smooth human play
        else: # If running headlessly, just step
            if not running: break

    env.close()
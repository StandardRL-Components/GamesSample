import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T16:23:47.316237
# Source Brief: brief_00982.md
# Brief Index: 982
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque

class GameEnv(gym.Env):
    """
    Gymnasium environment for 'Droplet Descent'.

    The player controls wind gusts to guide a water droplet down a procedurally
    generated slope. The goal is to collect smaller droplets to grow in size
    and reach the bottom of the slope within a time limit.

    **Visuals:**
    - Minimalist 2D style with a sky gradient background.
    - Player is a vibrant blue droplet with a subtle glow.
    - Collectible droplets are a lighter blue.
    - The slope is a solid brown line.
    - Wind gusts are visualized with white particle effects.

    **Gameplay:**
    - Use left/right actions to apply wind force.
    - The droplet is affected by gravity, wind, and slope physics.
    - Collecting droplets increases the player's size and score.
    - The episode ends if the player reaches the bottom, loses too many
      droplets, or the timer runs out.

    **Action Space:** MultiDiscrete([5, 2, 2])
    - actions[0]: Movement (0=none, 1=up(unused), 2=down(unused), 3=left, 4=right)
    - actions[1]: Space button (unused)
    - actions[2]: Shift button (unused)

    **Observation Space:** Box(0, 255, (400, 640, 3), uint8)
    - An RGB image of the game screen.
    """
    metadata = {"render_modes": ["rgb_array"]}
    game_description = "Guide a water droplet down a slope by controlling the wind. Collect smaller droplets to grow and reach the bottom before time runs out."
    user_guide = "Controls: Use ←→ to create wind gusts and guide the droplet."
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 60
    MAX_STEPS = 1800  # 30 seconds at 60 FPS

    # Colors
    COLOR_BG_TOP = (135, 206, 235)  # Sky Blue
    COLOR_BG_BOTTOM = (200, 230, 255) # Lighter Sky
    COLOR_PLAYER = (0, 120, 255)
    COLOR_PLAYER_HIGHLIGHT = (100, 180, 255)
    COLOR_DROPLET = (100, 200, 255)
    COLOR_SLOPE = (139, 69, 19) # Brown
    COLOR_WIND = (255, 255, 255)
    COLOR_UI_TEXT = (30, 30, 30)
    COLOR_WIN_TEXT = (0, 150, 0)
    COLOR_LOSE_TEXT = (200, 0, 0)

    # Physics
    GRAVITY = 0.1
    WIND_FORCE = 0.2
    DAMPING = 0.98
    SLOPE_BOUNCE = 0.1
    DROPLET_GRAVITY_FACTOR = 1.5

    # Gameplay
    WIN_CONDITION_COLLECTED = 6
    MAX_LOST_DROPLETS = 2
    INITIAL_PLAYER_RADIUS = 15
    SMALL_DROPLET_RADIUS = 5
    SLOW_DESCENT_THRESHOLD = 0.1

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 28)
        self.font_game_over = pygame.font.Font(None, 72)
        
        # State variables are initialized in reset()
        self.player_pos = None
        self.player_vel = None
        self.player_radius = None
        self.small_droplets = None
        self.slope_points = None
        self.particles = None
        self.steps = None
        self.score = None
        self.collected_droplets = None
        self.lost_droplets = None
        self.base_slope_angle = None
        self.small_droplet_spawn_prob = None
        self.game_over = None
        self.win_message = ""

        # Pre-render background gradient for performance
        self.bg_surface = self._create_gradient_background()
        
        # self.reset() # reset is called by gym.make

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.collected_droplets = 0
        self.lost_droplets = 0
        self.game_over = False
        self.win_message = ""

        self.player_pos = np.array([100.0, 50.0])
        self.player_vel = np.array([0.0, 0.0])
        self.player_radius = self.INITIAL_PLAYER_RADIUS

        self.small_droplets = []
        self.particles = deque(maxlen=100) # Efficiently manage particles

        # Difficulty parameters
        self.base_slope_angle = math.radians(10)
        self.small_droplet_spawn_prob = 0.02

        self._generate_slope()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            # If the game is over, do nothing but return the final state
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        reward = 0
        self.steps += 1

        # --- Update Game Logic ---
        self._handle_input(movement)
        
        prev_y_vel = self.player_vel[1]
        self._update_player()
        
        collected_this_step = self._update_small_droplets()
        if collected_this_step > 0:
            # sfx: droplet collect sound
            reward += 0.1 * collected_this_step
            self.score += 10 * collected_this_step

        self._update_particles()
        self._update_difficulty()

        # --- Calculate Rewards ---
        if self.player_vel[1] < self.SLOW_DESCENT_THRESHOLD:
            reward -= 0.01

        # --- Check Termination Conditions ---
        terminated = bool(self._check_termination())
        if terminated:
            self.game_over = True
            win = self.player_pos[1] > self.SCREEN_HEIGHT and self.collected_droplets >= self.WIN_CONDITION_COLLECTED
            if win:
                # sfx: win jingle
                reward += 50
                self.score += 1000
                self.win_message = "YOU WIN!"
            else:
                # sfx: lose sound
                reward -= 50
                self.score -= 500
                if self.steps >= self.MAX_STEPS:
                    self.win_message = "TIME'S UP!"
                else:
                    self.win_message = "GAME OVER"

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    # --- Game Logic Sub-functions ---

    def _handle_input(self, movement):
        if movement == 3:  # Left
            self.player_vel[0] -= self.WIND_FORCE
            self._create_wind_particles(direction=-1)
        elif movement == 4:  # Right
            self.player_vel[0] += self.WIND_FORCE
            self._create_wind_particles(direction=1)

    def _update_player(self):
        # Apply gravity and damping
        self.player_vel[1] += self.GRAVITY
        self.player_vel[0] *= self.DAMPING

        # Update position
        self.player_pos += self.player_vel

        # Slope collision
        self._handle_slope_collision(self.player_pos, self.player_vel, self.player_radius)

        # Screen boundary collision (horizontal)
        if self.player_pos[0] < self.player_radius:
            self.player_pos[0] = self.player_radius
            self.player_vel[0] *= -0.5
        elif self.player_pos[0] > self.SCREEN_WIDTH - self.player_radius:
            self.player_pos[0] = self.SCREEN_WIDTH - self.player_radius
            self.player_vel[0] *= -0.5

    def _update_small_droplets(self):
        # Spawn new droplets
        if self.np_random.random() < self.small_droplet_spawn_prob:
            self._spawn_small_droplet()

        collected_this_step = 0
        droplets_to_remove = []
        for i, droplet in enumerate(self.small_droplets):
            # Update position (simple gravity)
            droplet['pos'][1] += self.GRAVITY * self.DROPLET_GRAVITY_FACTOR
            self._handle_slope_collision(droplet['pos'], np.array([0.0, 0.0]), droplet['radius'])

            # Check for collection
            dist_sq = np.sum((self.player_pos - droplet['pos'])**2)
            if dist_sq < (self.player_radius + droplet['radius'])**2:
                droplets_to_remove.append(i)
                self._grow_player(droplet['radius'])
                self.collected_droplets += 1
                collected_this_step += 1
                continue
            
            # Check for falling off-screen
            if droplet['pos'][1] > self.SCREEN_HEIGHT + droplet['radius']:
                droplets_to_remove.append(i)
                self.lost_droplets += 1
                # sfx: droplet lost sound

        # Remove droplets that were collected or fell off
        for i in sorted(droplets_to_remove, reverse=True):
            del self.small_droplets[i]
        
        return collected_this_step
    
    def _handle_slope_collision(self, pos, vel, radius):
        for i in range(len(self.slope_points) - 1):
            p1 = np.array(self.slope_points[i])
            p2 = np.array(self.slope_points[i+1])
            
            line_vec = p2 - p1
            point_vec = pos - p1
            line_len_sq = np.dot(line_vec, line_vec)

            if line_len_sq == 0: continue

            t = np.dot(point_vec, line_vec) / line_len_sq
            t = np.clip(t, 0, 1)

            closest_point = p1 + t * line_vec
            dist_vec = pos - closest_point
            dist_sq = np.dot(dist_vec, dist_vec)

            if dist_sq < radius**2:
                dist = math.sqrt(dist_sq)
                penetration = radius - dist
                
                # Push the droplet out of the slope
                if dist > 0:
                    pos += (dist_vec / dist) * penetration
                
                # Adjust velocity for sliding effect
                if np.any(vel): # only adjust player velocity
                    normal = dist_vec / dist
                    vel_dot_normal = np.dot(vel, normal)
                    
                    # Reflect velocity slightly for a bounce, then project for sliding
                    vel[:] = vel - (1 + self.SLOPE_BOUNCE) * vel_dot_normal * normal
    
    def _spawn_small_droplet(self):
        x = self.np_random.uniform(20, self.SCREEN_WIDTH - 20)
        y = self.np_random.uniform(-50, -10)
        self.small_droplets.append({'pos': np.array([x, y]), 'radius': self.SMALL_DROPLET_RADIUS})

    def _grow_player(self, other_radius):
        # Area = pi * r^2. New area is sum of old areas.
        new_radius_sq = self.player_radius**2 + other_radius**2
        self.player_radius = math.sqrt(new_radius_sq)

    def _create_wind_particles(self, direction):
        for _ in range(5):
            # Emerge from the opposite side of the droplet
            offset_angle = self.np_random.uniform(-math.pi/4, math.pi/4)
            offset_x = math.cos(offset_angle) * self.player_radius * -direction
            offset_y = math.sin(offset_angle) * self.player_radius
            
            start_pos = self.player_pos + np.array([offset_x, offset_y])
            
            vel_angle = self.np_random.uniform(-math.pi/8, math.pi/8)
            vel_mag = self.np_random.uniform(1, 3)
            vel = np.array([math.cos(vel_angle) * vel_mag * direction, math.sin(vel_angle) * vel_mag])
            
            life = self.np_random.integers(15, 30)
            self.particles.append([start_pos, vel, life])

    def _update_particles(self):
        for p in self.particles:
            p[0] += p[1] # Update position
            p[2] -= 1    # Decrease life

    def _update_difficulty(self):
        if self.steps > 0 and self.steps % 500 == 0:
            self.base_slope_angle = min(self.base_slope_angle + 0.01, math.radians(45))
        if self.steps > 0 and self.steps % 300 == 0:
            self.small_droplet_spawn_prob = min(self.small_droplet_spawn_prob + 0.005, 0.1)

    def _check_termination(self):
        time_up = self.steps >= self.MAX_STEPS
        too_many_lost = self.lost_droplets > self.MAX_LOST_DROPLETS
        reached_bottom = self.player_pos[1] > self.SCREEN_HEIGHT
        return time_up or too_many_lost or reached_bottom

    # --- State & Rendering ---

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "collected_droplets": self.collected_droplets,
            "lost_droplets": self.lost_droplets,
            "player_radius": self.player_radius,
        }

    def _get_observation(self):
        self.screen.blit(self.bg_surface, (0, 0))
        self._render_game()
        self._render_ui()
        if self.game_over:
            self._render_game_over()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        self._render_slope()
        self._render_small_droplets()
        self._render_particles()
        self._render_player()

    def _render_player(self):
        pos = (int(self.player_pos[0]), int(self.player_pos[1]))
        radius = int(self.player_radius)
        
        # Main body
        pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, self.COLOR_PLAYER)
        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, self.COLOR_PLAYER)
        
        # Highlight for a more liquid look
        highlight_radius = int(radius * 0.4)
        highlight_offset_x = int(radius * 0.3)
        highlight_offset_y = -int(radius * 0.3)
        pygame.gfxdraw.aacircle(self.screen, pos[0] + highlight_offset_x, pos[1] + highlight_offset_y, highlight_radius, self.COLOR_PLAYER_HIGHLIGHT)
        pygame.gfxdraw.filled_circle(self.screen, pos[0] + highlight_offset_x, pos[1] + highlight_offset_y, highlight_radius, self.COLOR_PLAYER_HIGHLIGHT)

    def _render_small_droplets(self):
        for droplet in self.small_droplets:
            pos = (int(droplet['pos'][0]), int(droplet['pos'][1]))
            radius = int(droplet['radius'])
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, self.COLOR_DROPLET)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, self.COLOR_DROPLET)

    def _render_slope(self):
        if len(self.slope_points) > 1:
            pygame.draw.lines(self.screen, self.COLOR_SLOPE, False, self.slope_points, 8)

    def _render_particles(self):
        for pos, vel, life in list(self.particles):
            if life > 0:
                alpha = int(255 * (life / 30.0))
                # Create a temporary surface for alpha blending
                temp_surf = pygame.Surface((abs(vel[0]*2)+1, abs(vel[1]*2)+1), pygame.SRCALPHA)
                start = (pos[0], pos[1])
                end = (pos[0] - vel[0]*2, pos[1] - vel[1]*2)
                
                # Adjust coordinates to be relative to the temp surface
                min_x = min(start[0], end[0])
                min_y = min(start[1], end[1])
                rel_start = (start[0] - min_x, start[1] - min_y)
                rel_end = (end[0] - min_x, end[1] - min_y)

                pygame.draw.aaline(temp_surf, (*self.COLOR_WIND, alpha), rel_start, rel_end)
                self.screen.blit(temp_surf, (min_x, min_y))


    def _render_ui(self):
        # Time remaining
        time_left = (self.MAX_STEPS - self.steps) / self.FPS
        time_text = f"Time: {time_left:.1f}s"
        time_surf = self.font_ui.render(time_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(time_surf, (10, 10))
        
        # Lost droplets
        lost_text = f"Lost: {self.lost_droplets} / {self.MAX_LOST_DROPLETS + 1}"
        lost_surf = self.font_ui.render(lost_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(lost_surf, (self.SCREEN_WIDTH - lost_surf.get_width() - 10, 10))

        # Collected droplets
        collected_text = f"Collected: {self.collected_droplets} / {self.WIN_CONDITION_COLLECTED}"
        collected_surf = self.font_ui.render(collected_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(collected_surf, (10, 35))

    def _render_game_over(self):
        color = self.COLOR_WIN_TEXT if "WIN" in self.win_message else self.COLOR_LOSE_TEXT
        text_surf = self.font_game_over.render(self.win_message, True, color)
        text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
        
        # Simple background for readability
        bg_rect = text_rect.inflate(20, 20)
        s = pygame.Surface(bg_rect.size, pygame.SRCALPHA)
        s.fill((0,0,0,128))
        self.screen.blit(s, bg_rect)
        
        self.screen.blit(text_surf, text_rect)

    # --- Helper/Setup Functions ---

    def _generate_slope(self):
        self.slope_points = []
        x, y = -10, self.np_random.uniform(150, 200)
        segment_length = 50
        
        while x < self.SCREEN_WIDTH + segment_length:
            self.slope_points.append((x, y))
            angle = self.base_slope_angle + self.np_random.uniform(-0.1, 0.1)
            x += math.cos(angle) * segment_length
            y += math.sin(angle) * segment_length
    
    def _create_gradient_background(self):
        bg = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        for y in range(self.SCREEN_HEIGHT):
            ratio = y / self.SCREEN_HEIGHT
            color = [
                int(self.COLOR_BG_TOP[i] * (1 - ratio) + self.COLOR_BG_BOTTOM[i] * ratio)
                for i in range(3)
            ]
            pygame.draw.line(bg, color, (0, y), (self.SCREEN_WIDTH, y))
        return bg

    def close(self):
        pygame.quit()


# Example of how to run the environment
if __name__ == '__main__':
    # This block will not run in the testing environment, but is useful for local debugging.
    # It requires a display, so we unset the dummy video driver.
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv()
    
    # --- Manual Play Loop ---
    obs, info = env.reset()
    done = False
    
    # Pygame setup for display
    display_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Droplet Descent")
    
    total_reward = 0
    
    while not done:
        action = [0, 0, 0] # Default: no action
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
        
        if keys[pygame.K_q]:
            done = True

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward

        # Render to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(env.FPS)

    print(f"Game Over. Final Score: {info.get('score', 0)}, Total Reward: {total_reward:.2f}")
    env.close()
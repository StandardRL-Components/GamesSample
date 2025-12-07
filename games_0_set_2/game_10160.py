import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T10:54:58.360280
# Source Brief: brief_00160.md
# Brief Index: 160
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
    Gymnasium environment where the player guides falling water droplets into reservoirs
    by tilting the landscape and slowing down time.
    """
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    game_description = (
        "Guide falling water droplets into reservoirs by tilting the landscape. "
        "Fill all reservoirs before time runs out, and use slow-motion to navigate tricky obstacles."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to tilt the landscape. Hold the 'space' bar to slow down time."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    
    # Colors
    COLOR_BG = (20, 30, 40)
    COLOR_WATER = (50, 150, 255)
    COLOR_WATER_LITE = (150, 200, 255)
    COLOR_LANDSCAPE = (40, 180, 100) # Not used directly, represents concept
    COLOR_RESERVOIR = (120, 130, 140)
    COLOR_RESERVOIR_FULL = (100, 220, 255)
    COLOR_OBSTACLE = (139, 69, 19)
    COLOR_UI_TEXT = (220, 220, 220)
    COLOR_WIN = (100, 255, 100)
    COLOR_LOSS = (255, 100, 100)
    COLOR_TILT_INDICATOR = (255, 255, 255)

    # Physics and Gameplay
    BASE_GRAVITY = 0.2
    TILT_STRENGTH = 0.3
    MAX_TILT = 1.0
    TILT_INCREMENT = 0.1
    DROPLET_TERMINAL_VELOCITY = 5.0
    INITIAL_RAINFALL_INTENSITY = 1.0 # droplets per second
    RAINFALL_INTENSITY_INCREASE = 0.05
    DIFFICULTY_INTERVAL_STEPS = 200
    OBSTACLE_START_STEP = 500
    OBSTACLE_INTERVAL_STEPS = 500
    MAX_TIME = 60.0
    MAX_STEPS = 1800 # 60 seconds at 30 FPS
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.render_mode = render_mode
        
        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_game_over = pygame.font.SysFont("Arial", 60, bold=True)

        # --- State Variables ---
        # These are initialized in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_remaining = 0.0
        self.landscape_tilt = np.array([0.0, 0.0]) # [x_tilt, y_tilt]
        self.droplets = []
        self.reservoirs = []
        self.obstacles = []
        self.rainfall_intensity = 0.0
        self.time_to_next_droplet = 0.0
        self.next_obstacle_spawn_step = 0
        self.obstacle_count_to_spawn = 0
        self.last_space_held = False # To control visual effects

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # --- Initialize Game State ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_remaining = self.MAX_TIME
        self.landscape_tilt = np.array([0.0, 0.0])
        
        self.droplets = []
        self.obstacles = []
        self.reservoirs = self._initialize_reservoirs()

        self.rainfall_intensity = self.INITIAL_RAINFALL_INTENSITY
        self.time_to_next_droplet = 1.0 / self.rainfall_intensity
        self.next_obstacle_spawn_step = self.OBSTACLE_START_STEP
        self.obstacle_count_to_spawn = 1
        self.last_space_held = False

        return self._get_observation(), self._get_info()

    def _initialize_reservoirs(self):
        reservoirs = []
        capacity = 150
        res_width, res_height = 100, 20
        y_pos = self.SCREEN_HEIGHT - res_height - 10
        
        positions = [
            (self.SCREEN_WIDTH * 0.2 - res_width / 2, y_pos),
            (self.SCREEN_WIDTH * 0.5 - res_width / 2, y_pos),
            (self.SCREEN_WIDTH * 0.8 - res_width / 2, y_pos)
        ]
        
        for x, y in positions:
            reservoirs.append({
                'rect': pygame.Rect(x, y, res_width, res_height),
                'capacity': capacity,
                'fill_level': 0,
                'is_full': False
            })
        return reservoirs

    def step(self, action):
        reward = 0.0
        terminated = False

        if self.game_over:
            return self._get_observation(), 0.0, True, False, self._get_info()

        # --- Unpack Action ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        self.last_space_held = space_held
        
        # --- Update Game State ---
        self.steps += 1
        game_speed = 0.1 if space_held else 1.0
        time_delta = (1.0 / self.metadata["render_fps"]) * game_speed

        self.time_remaining -= (1.0 / self.metadata["render_fps"])
        
        self._update_tilt(movement)
        self._update_difficulty()
        self._spawn_entities(time_delta)
        reward += self._update_droplets(game_speed)
        
        # --- Check for Reservoir Fills ---
        for res in self.reservoirs:
            if not res['is_full'] and res['fill_level'] >= res['capacity']:
                res['is_full'] = True
                reward += 5.0
                # sfx: reservoir_full_sound()

        # --- Check Termination Conditions ---
        win_condition = all(r['is_full'] for r in self.reservoirs)
        loss_condition = self.time_remaining <= 0
        max_steps_reached = self.steps >= self.MAX_STEPS

        if win_condition:
            reward += 50.0
            terminated = True
            self.win_message = "ALL RESERVOIRS FILLED!"
        elif loss_condition:
            reward -= 50.0
            terminated = True
            self.win_message = "TIME'S UP!"
        elif max_steps_reached:
            terminated = True
            self.win_message = "MAX STEPS REACHED"

        self.game_over = terminated
        self.score += reward
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _update_tilt(self, movement):
        if movement == 1: # Up
            self.landscape_tilt[1] -= self.TILT_INCREMENT
        elif movement == 2: # Down
            self.landscape_tilt[1] += self.TILT_INCREMENT
        elif movement == 3: # Left
            self.landscape_tilt[0] -= self.TILT_INCREMENT
        elif movement == 4: # Right
            self.landscape_tilt[0] += self.TILT_INCREMENT
        
        # Clamp tilt to prevent excessive force
        np.clip(self.landscape_tilt, -self.MAX_TILT, self.MAX_TILT, out=self.landscape_tilt)

    def _update_difficulty(self):
        # Increase rainfall
        if self.steps > 0 and self.steps % self.DIFFICULTY_INTERVAL_STEPS == 0:
            self.rainfall_intensity += self.RAINFALL_INTENSITY_INCREASE

        # Spawn obstacles
        if self.steps >= self.next_obstacle_spawn_step:
            for _ in range(self.obstacle_count_to_spawn):
                self._spawn_obstacle()
            self.next_obstacle_spawn_step += self.OBSTACLE_INTERVAL_STEPS
            self.obstacle_count_to_spawn += 1

    def _spawn_entities(self, time_delta):
        self.time_to_next_droplet -= time_delta
        if self.time_to_next_droplet <= 0:
            pos = np.array([random.uniform(20, self.SCREEN_WIDTH - 20), -10.0])
            vel = np.array([0.0, 0.0])
            radius = random.uniform(3, 6)
            self.droplets.append({'pos': pos, 'vel': vel, 'radius': radius})
            self.time_to_next_droplet = 1.0 / self.rainfall_intensity
            # sfx: droplet_spawn_sound()

    def _spawn_obstacle(self):
        w = random.randint(40, 120)
        h = random.randint(10, 20)
        x = random.randint(0, self.SCREEN_WIDTH - w)
        y = random.randint(50, self.SCREEN_HEIGHT - 100)
        
        new_rect = pygame.Rect(x, y, w, h)
        
        # Ensure it doesn't overlap with reservoirs
        for res in self.reservoirs:
            if new_rect.colliderect(res['rect']):
                return # Skip spawning if it's in a bad spot
        
        self.obstacles.append({'rect': new_rect})

    def _update_droplets(self, game_speed):
        reward = 0
        gravity_force = np.array([
            self.landscape_tilt[0] * self.TILT_STRENGTH,
            self.BASE_GRAVITY + self.landscape_tilt[1] * self.TILT_STRENGTH
        ])

        # --- Move and collide droplets ---
        for drop in self.droplets:
            drop['vel'] += gravity_force * game_speed
            # Clamp velocity
            vel_mag = np.linalg.norm(drop['vel'])
            if vel_mag > self.DROPLET_TERMINAL_VELOCITY:
                drop['vel'] = drop['vel'] / vel_mag * self.DROPLET_TERMINAL_VELOCITY
            
            drop['pos'] += drop['vel'] * game_speed

            # Wall collisions
            if drop['pos'][0] - drop['radius'] < 0 or drop['pos'][0] + drop['radius'] > self.SCREEN_WIDTH:
                drop['vel'][0] *= -0.7 # Dampen bounce
                drop['pos'][0] = np.clip(drop['pos'][0], drop['radius'], self.SCREEN_WIDTH - drop['radius'])

            # Obstacle collisions
            for obs in self.obstacles:
                if obs['rect'].collidepoint(drop['pos']):
                    # A simple vertical bounce for gameplay feel
                    drop['vel'][1] *= -0.5
                    drop['pos'][1] = obs['rect'].top - drop['radius'] - 1

        # --- Handle reservoir collection and droplet removal ---
        droplets_to_remove = []
        for i, drop in enumerate(self.droplets):
            # Remove if off-screen
            if drop['pos'][1] - drop['radius'] > self.SCREEN_HEIGHT:
                droplets_to_remove.append(i)
                continue
            
            # Check for reservoir collection
            for res in self.reservoirs:
                if not res['is_full'] and res['rect'].collidepoint(drop['pos']):
                    fill_amount = drop['radius'] # Volume is proportional to radius
                    res['fill_level'] += fill_amount
                    reward += 0.1
                    droplets_to_remove.append(i)
                    # sfx: droplet_collect_sound()
                    break
        
        # Remove droplets in reverse order to avoid index errors
        for i in sorted(list(set(droplets_to_remove)), reverse=True):
            del self.droplets[i]

        # --- Merge droplets ---
        self._merge_droplets()
        
        return reward

    def _merge_droplets(self):
        merged_indices = set()
        for i in range(len(self.droplets)):
            for j in range(i + 1, len(self.droplets)):
                if i in merged_indices or j in merged_indices:
                    continue
                
                d1 = self.droplets[i]
                d2 = self.droplets[j]
                
                dist_sq = np.sum((d1['pos'] - d2['pos'])**2)
                radius_sum_sq = (d1['radius'] + d2['radius'])**2
                
                if dist_sq < radius_sum_sq:
                    # Merge smaller into larger
                    if d1['radius'] < d2['radius']:
                        d1, d2 = d2, d1
                        merged_indices.add(j)
                    else:
                        merged_indices.add(j)

                    # Conserve area (pi*r^2)
                    new_radius_sq = d1['radius']**2 + d2['radius']**2
                    d1['radius'] = math.sqrt(new_radius_sq)
                    
                    # Momentum conservation is complex, so we just keep the larger droplet's velocity
                    # This is a "game feel" decision over realism
        
        # Remove merged droplets
        if merged_indices:
            self.droplets = [d for i, d in enumerate(self.droplets) if i not in merged_indices]

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_remaining": self.time_remaining,
            "reservoirs_filled": sum(1 for r in self.reservoirs if r['is_full']),
            "total_reservoirs": len(self.reservoirs),
        }

    def _get_observation(self):
        # This is also our main render function
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        # Apply time-slow visual effect
        if self.last_space_held and not self.game_over:
             overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
             overlay.fill((100, 150, 255, 30))
             self.screen.blit(overlay, (0, 0))

        if self.game_over:
            self._render_game_over()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Render obstacles
        for obs in self.obstacles:
            pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, obs['rect'])
        
        # Render reservoirs
        for res in self.reservoirs:
            color = self.COLOR_RESERVOIR_FULL if res['is_full'] else self.COLOR_RESERVOIR
            pygame.draw.rect(self.screen, color, res['rect'], 2, border_radius=3)
            
            fill_ratio = min(1.0, res['fill_level'] / res['capacity'])
            if fill_ratio > 0:
                fill_height = int(res['rect'].height * fill_ratio)
                fill_rect = pygame.Rect(
                    res['rect'].x, res['rect'].y + res['rect'].height - fill_height,
                    res['rect'].width, fill_height
                )
                pygame.draw.rect(self.screen, self.COLOR_WATER, fill_rect, border_bottom_left_radius=3, border_bottom_right_radius=3)

        # Render droplets
        for drop in self.droplets:
            pos = (int(drop['pos'][0]), int(drop['pos'][1]))
            radius = int(drop['radius'])
            if radius > 0:
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, self.COLOR_WATER)
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, self.COLOR_WATER)
                # Add a small highlight for a 3D feel
                highlight_radius = max(1, int(radius * 0.4))
                highlight_pos = (pos[0] - int(radius*0.3), pos[1] - int(radius*0.3))
                pygame.gfxdraw.filled_circle(self.screen, highlight_pos[0], highlight_pos[1], highlight_radius, self.COLOR_WATER_LITE)


    def _render_ui(self):
        # Render Time
        time_text = f"Time: {max(0, self.time_remaining):.1f}"
        time_surf = self.font_ui.render(time_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(time_surf, (self.SCREEN_WIDTH - time_surf.get_width() - 10, 10))
        
        # Render Reservoirs Filled
        filled_count = sum(1 for r in self.reservoirs if r['is_full'])
        res_text = f"Filled: {filled_count}/{len(self.reservoirs)}"
        res_surf = self.font_ui.render(res_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(res_surf, (10, 10))

        # Render Tilt Indicator
        indicator_center = (self.SCREEN_WIDTH / 2, 25)
        indicator_max_offset = 15
        indicator_pos = (
            int(indicator_center[0] + self.landscape_tilt[0] * indicator_max_offset),
            int(indicator_center[1] + self.landscape_tilt[1] * indicator_max_offset)
        )
        pygame.gfxdraw.aacircle(self.screen, int(indicator_center[0]), int(indicator_center[1]), indicator_max_offset, self.COLOR_UI_TEXT)
        pygame.gfxdraw.filled_circle(self.screen, indicator_pos[0], indicator_pos[1], 5, self.COLOR_TILT_INDICATOR)

    def _render_game_over(self):
        color = self.COLOR_WIN if "FILLED" in self.win_message else self.COLOR_LOSS
        text_surf = self.font_game_over.render(self.win_message, True, color)
        text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
        
        # Add a subtle background for readability
        overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 150))
        self.screen.blit(overlay, (0, 0))
        
        self.screen.blit(text_surf, text_rect)

    def render(self):
        # This method is not strictly required by the new API but is good practice
        # for human-based rendering.
        if self.render_mode == "rgb_array":
            return self._get_observation()
        return None # Other modes not supported

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game manually
    # To run, you will need to `pip install pygame`
    # You can remove the `os.environ` line at the top of the file to run in a window.
    
    # Check if we are in a headless environment
    if os.environ.get("SDL_VIDEODRIVER") == "dummy":
        print("Running in headless mode. Manual play is not available.")
        print("To play manually, comment out the line `os.environ.setdefault(\"SDL_VIDEODRIVER\", \"dummy\")`")
        # Run a short test loop instead of the interactive one
        env = GameEnv(render_mode="rgb_array")
        obs, info = env.reset()
        for _ in range(200):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                print(f"Game ended. Final info: {info}")
                obs, info = env.reset()
        env.close()
        exit()

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    running = True
    game_window = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Droplet Dynamics")
    clock = pygame.time.Clock()

    while running:
        movement = 0 # No-op
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            movement = 1
        elif keys[pygame.K_DOWN]:
            movement = 2
        elif keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
            
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Display the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        game_window.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Steps: {info['steps']}")
            pygame.time.wait(3000) # Pause for 3 seconds
            obs, info = env.reset()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        clock.tick(env.metadata['render_fps'])

    env.close()
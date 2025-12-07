import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T15:21:18.133292
# Source Brief: brief_00947.md
# Brief Index: 947
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class Particle:
    """A simple particle for visual effects."""
    def __init__(self, x, y, color):
        self.x = x
        self.y = y
        self.color = color
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(1, 4)
        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed
        self.lifetime = random.randint(15, 30)  # Lasts for 0.5 to 1 second

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.lifetime -= 1
        self.vx *= 0.95 # Air resistance
        self.vy *= 0.95

    def draw(self, surface):
        if self.lifetime > 0:
            alpha = max(0, int(255 * (self.lifetime / 30)))
            color_with_alpha = self.color + (alpha,)
            temp_surf = pygame.Surface((3, 3), pygame.SRCALPHA)
            pygame.draw.rect(temp_surf, color_with_alpha, (0, 0, 3, 3))
            surface.blit(temp_surf, (int(self.x) - 1, int(self.y) - 1))

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 30}

    game_description = (
        "Navigate a shifting maze to collect all checkpoints before time runs out. Your own trail slows you down, so plan your path carefully and use your boost to maintain momentum."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move and press SPACE to boost, reducing friction on your trail."
    )
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.CELL_SIZE = 20
        self.GRID_WIDTH = self.WIDTH // self.CELL_SIZE
        self.GRID_HEIGHT = self.HEIGHT // self.CELL_SIZE
        self.GAME_DURATION = 90  # seconds
        self.FPS = 30
        self.MAX_STEPS = self.GAME_DURATION * self.FPS
        self.NUM_CHECKPOINTS = 20
        self.WALL_SHIFT_INTERVAL = 20 # steps
        self.WALL_SHIFT_AMOUNT = int((self.GRID_WIDTH * self.GRID_HEIGHT) * 0.01) # 1% of total cells

        # --- Physics Constants ---
        self.ACCELERATION = 0.5
        self.MAX_SPEED = 6.0
        self.BASE_FRICTION = 0.95
        self.TRAIL_FRICTION_MULTIPLIER = 0.1
        self.BOOST_FRICTION_REDUCTION = 0.04 # Reduces friction when boosting
        self.TRAIL_DENSITY_INCREASE = 0.1
        self.TRAIL_DENSITY_DECAY = 0.999
        self.TRAIL_DENSITY_MAX = 1.0

        # --- Color Palette ---
        self.COLOR_BG = (10, 20, 30)
        self.COLOR_WALL = (40, 50, 60)
        self.COLOR_PLAYER = (0, 150, 255)
        self.COLOR_PLAYER_GLOW = (0, 200, 255, 50)
        self.COLOR_PLAYER_BOOST_GLOW = (255, 255, 0, 100)
        self.COLOR_CHECKPOINT = (0, 255, 100)
        self.COLOR_CHECKPOINT_GLOW = (100, 255, 150, 100)
        self.COLOR_TRAIL_START = (0, 80, 150)
        self.COLOR_TRAIL_END = (10, 20, 30)
        self.COLOR_UI_TEXT = (220, 220, 240)
        
        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.ui_font = pygame.font.SysFont("Consolas", 24, bold=True)
        self.render_mode = render_mode
        if self.render_mode == "human":
            self.human_screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
            pygame.display.set_caption("Momentum Maze")

        # --- State Variables ---
        self.player_pos = pygame.Vector2(0, 0)
        self.player_vel = pygame.Vector2(0, 0)
        self.maze = np.zeros((self.GRID_WIDTH, self.GRID_HEIGHT), dtype=int)
        self.trail_density = np.zeros((self.GRID_WIDTH, self.GRID_HEIGHT), dtype=float)
        self.checkpoints = []
        self.checkpoints_collected = []
        self.particles = []
        self.steps = 0
        self.score = 0
        self.timer = 0.0
        self.game_over = False
        self.last_boost_state = False

    def _generate_maze(self):
        self.maze = np.zeros((self.GRID_WIDTH, self.GRID_HEIGHT), dtype=int)
        # Create borders
        self.maze[0, :] = 1
        self.maze[-1, :] = 1
        self.maze[:, 0] = 1
        self.maze[:, -1] = 1
        
        # Add random internal walls
        num_walls = int(self.GRID_WIDTH * self.GRID_HEIGHT * 0.20)
        for _ in range(num_walls):
            x, y = self.np_random.integers(1, self.GRID_WIDTH - 1), self.np_random.integers(1, self.GRID_HEIGHT - 1)
            self.maze[x, y] = 1

    def _place_player_and_checkpoints(self):
        floor_tiles = np.argwhere(self.maze == 0)
        self.np_random.shuffle(floor_tiles)
        
        # Place Player
        player_start_idx = floor_tiles[0]
        self.player_pos = pygame.Vector2(
            player_start_idx[0] * self.CELL_SIZE + self.CELL_SIZE / 2,
            player_start_idx[1] * self.CELL_SIZE + self.CELL_SIZE / 2
        )
        # Ensure start area is clear
        self.maze[player_start_idx[0], player_start_idx[1]] = 0
        
        # Place Checkpoints
        self.checkpoints = []
        for i in range(1, min(self.NUM_CHECKPOINTS + 1, len(floor_tiles))):
            cp_idx = floor_tiles[i]
            self.checkpoints.append(pygame.Vector2(cp_idx[0], cp_idx[1]))
            self.maze[cp_idx[0], cp_idx[1]] = 0 # Ensure checkpoint is on a floor tile
        self.checkpoints_collected = [False] * len(self.checkpoints)

    def _shift_walls(self):
        wall_indices = np.argwhere(self.maze[1:-1, 1:-1] == 1) + 1
        floor_indices = np.argwhere(self.maze[1:-1, 1:-1] == 0) + 1
        
        num_to_shift = min(self.WALL_SHIFT_AMOUNT, len(wall_indices), len(floor_indices))
        if num_to_shift == 0: return

        self.np_random.shuffle(wall_indices)
        self.np_random.shuffle(floor_indices)

        for i in range(num_to_shift):
            wall_x, wall_y = wall_indices[i]
            floor_x, floor_y = floor_indices[i]
            self.maze[wall_x, wall_y] = 0
            self.maze[floor_x, floor_y] = 1
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.timer = self.GAME_DURATION
        self.game_over = False
        
        self.player_vel = pygame.Vector2(0, 0)
        self.particles.clear()
        
        self._generate_maze()
        self._place_player_and_checkpoints()
        
        self.trail_density = np.zeros((self.GRID_WIDTH, self.GRID_HEIGHT), dtype=float)
        self.last_boost_state = False

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        reward = 0.0

        # --- Reward calculation (Part 1: Proximity) ---
        old_dist_to_checkpoint = self._get_dist_to_nearest_checkpoint()

        # --- Game Logic Update ---
        # 1. Apply acceleration from input
        accel = pygame.Vector2(0, 0)
        if movement == 1: accel.y = -self.ACCELERATION # Up
        elif movement == 2: accel.y = self.ACCELERATION # Down
        elif movement == 3: accel.x = -self.ACCELERATION # Left
        elif movement == 4: accel.x = self.ACCELERATION # Right
        self.player_vel += accel

        # 2. Apply friction
        player_grid_pos = (int(self.player_pos.x / self.CELL_SIZE), int(self.player_pos.y / self.CELL_SIZE))
        trail_fric = 0.0
        if 0 <= player_grid_pos[0] < self.GRID_WIDTH and 0 <= player_grid_pos[1] < self.GRID_HEIGHT:
            trail_fric = self.trail_density[player_grid_pos] * self.TRAIL_FRICTION_MULTIPLIER
        
        friction_mod = self.BOOST_FRICTION_REDUCTION if space_held else 0.0
        # Placeholder for sound effect
        # if space_held and not self.last_boost_state: play_sound("boost_start")
        # if not space_held and self.last_boost_state: play_sound("boost_end")
        self.last_boost_state = space_held

        friction_factor = max(0, self.BASE_FRICTION - friction_mod - trail_fric)
        self.player_vel *= friction_factor

        # 3. Clamp speed
        if self.player_vel.length() > self.MAX_SPEED:
            self.player_vel.scale_to_length(self.MAX_SPEED)

        # 4. Update position and handle collisions
        self.player_pos.x += self.player_vel.x
        self._handle_collision('x')
        self.player_pos.y += self.player_vel.y
        self._handle_collision('y')

        # 5. Update trail
        old_density = self.trail_density[player_grid_pos]
        self.trail_density *= self.TRAIL_DENSITY_DECAY
        self.trail_density[player_grid_pos] = min(self.TRAIL_DENSITY_MAX, self.trail_density[player_grid_pos] + self.TRAIL_DENSITY_INCREASE)
        new_density = self.trail_density[player_grid_pos]

        # 6. Check for checkpoint collection
        collected_checkpoint = self._check_checkpoint_collection()

        # --- Reward Calculation (Part 2: Events & Penalties) ---
        new_dist_to_checkpoint = self._get_dist_to_nearest_checkpoint()
        if new_dist_to_checkpoint is not None and old_dist_to_checkpoint is not None:
            if new_dist_to_checkpoint < old_dist_to_checkpoint:
                reward += 0.1 # Reward for getting closer
        
        if new_density > old_density:
             reward -= 0.01 * (new_density - old_density) # Penalty for entering denser trail

        if collected_checkpoint:
            reward += 10.0
            self.score += 1
            # Placeholder for sound effect
            # play_sound("checkpoint_get")

        # --- Update Timers and State ---
        self.steps += 1
        self.timer -= 1.0 / self.FPS
        if self.steps % self.WALL_SHIFT_INTERVAL == 0:
            self._shift_walls()

        # --- Check Termination ---
        terminated = False
        truncated = False
        if self.timer <= 0:
            reward -= 50.0 # Penalty for running out of time
            terminated = True
            self.game_over = True
        
        if all(self.checkpoints_collected):
            reward += 100.0 # Big reward for winning
            terminated = True
            self.game_over = True

        if self.steps >= self.MAX_STEPS:
            truncated = True # Episode ends if it runs too long
            terminated = True # In gym API, truncated implies terminated
            self.game_over = True

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _get_dist_to_nearest_checkpoint(self):
        uncollected_indices = [i for i, collected in enumerate(self.checkpoints_collected) if not collected]
        if not uncollected_indices:
            return None
        
        min_dist = float('inf')
        for i in uncollected_indices:
            cp_pos = pygame.Vector2(
                self.checkpoints[i].x * self.CELL_SIZE + self.CELL_SIZE / 2,
                self.checkpoints[i].y * self.CELL_SIZE + self.CELL_SIZE / 2
            )
            dist = self.player_pos.distance_to(cp_pos)
            if dist < min_dist:
                min_dist = dist
        return min_dist

    def _handle_collision(self, axis):
        player_grid_x = int(self.player_pos.x / self.CELL_SIZE)
        player_grid_y = int(self.player_pos.y / self.CELL_SIZE)
        
        if not (0 <= player_grid_x < self.GRID_WIDTH and 0 <= player_grid_y < self.GRID_HEIGHT):
            return # Should not happen with border walls

        if self.maze[player_grid_x, player_grid_y] == 1:
            if axis == 'x':
                if self.player_vel.x > 0:
                    self.player_pos.x = player_grid_x * self.CELL_SIZE - 1
                else:
                    self.player_pos.x = (player_grid_x + 1) * self.CELL_SIZE + 1
                self.player_vel.x *= -0.5 # Bounce
            elif axis == 'y':
                if self.player_vel.y > 0:
                    self.player_pos.y = player_grid_y * self.CELL_SIZE - 1
                else:
                    self.player_pos.y = (player_grid_y + 1) * self.CELL_SIZE + 1
                self.player_vel.y *= -0.5 # Bounce
            # Placeholder for sound effect
            # play_sound("wall_hit")

    def _check_checkpoint_collection(self):
        player_grid_pos = (int(self.player_pos.x / self.CELL_SIZE), int(self.player_pos.y / self.CELL_SIZE))
        for i, cp_pos in enumerate(self.checkpoints):
            if not self.checkpoints_collected[i] and (player_grid_pos[0], player_grid_pos[1]) == (cp_pos.x, cp_pos.y):
                self.checkpoints_collected[i] = True
                for _ in range(30): # Create particle burst
                    self.particles.append(Particle(self.player_pos.x, self.player_pos.y, self.COLOR_CHECKPOINT))
                return True
        return False

    def _get_observation(self):
        # --- Render Game ---
        self.screen.fill(self.COLOR_BG)
        
        # Render trail
        for x in range(self.GRID_WIDTH):
            for y in range(self.GRID_HEIGHT):
                density = self.trail_density[x, y]
                if density > 0.01:
                    alpha = int(200 * density)
                    color = self.COLOR_TRAIL_START
                    trail_surf = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
                    trail_surf.fill(color + (alpha,))
                    self.screen.blit(trail_surf, (x * self.CELL_SIZE, y * self.CELL_SIZE))

        # Render walls
        for x in range(self.GRID_WIDTH):
            for y in range(self.GRID_HEIGHT):
                if self.maze[x, y] == 1:
                    pygame.draw.rect(self.screen, self.COLOR_WALL, (x * self.CELL_SIZE, y * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE))

        # Render checkpoints
        for i, cp_pos in enumerate(self.checkpoints):
            if not self.checkpoints_collected[i]:
                center_x = int(cp_pos.x * self.CELL_SIZE + self.CELL_SIZE / 2)
                center_y = int(cp_pos.y * self.CELL_SIZE + self.CELL_SIZE / 2)
                radius = int(self.CELL_SIZE * 0.3)
                # Glow
                pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, radius + 3, self.COLOR_CHECKPOINT_GLOW)
                # Core
                pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, radius, self.COLOR_CHECKPOINT)
                pygame.gfxdraw.aacircle(self.screen, center_x, center_y, radius, self.COLOR_CHECKPOINT)

        # Render particles
        for p in self.particles[:]:
            p.update()
            if p.lifetime <= 0:
                self.particles.remove(p)
            else:
                p.draw(self.screen)

        # Render player
        player_size = int(self.CELL_SIZE * 0.8)
        player_rect = pygame.Rect(0, 0, player_size, player_size)
        player_rect.center = (int(self.player_pos.x), int(self.player_pos.y))
        
        glow_color = self.COLOR_PLAYER_BOOST_GLOW if self.last_boost_state else self.COLOR_PLAYER_GLOW
        glow_surf = pygame.Surface((player_size * 2, player_size * 2), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, glow_color, (player_size, player_size), player_size)
        self.screen.blit(glow_surf, (player_rect.centerx - player_size, player_rect.centery - player_size), special_flags=pygame.BLEND_RGBA_ADD)
        
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=3)

        # --- Render UI ---
        # Checkpoints collected
        cp_text = self.ui_font.render(f"CHECKPOINTS: {self.score}/{self.NUM_CHECKPOINTS}", True, self.COLOR_UI_TEXT)
        self.screen.blit(cp_text, (10, 5))
        
        # Timer
        timer_text = self.ui_font.render(f"TIME: {max(0, self.timer):.1f}", True, self.COLOR_UI_TEXT)
        self.screen.blit(timer_text, (self.WIDTH - timer_text.get_width() - 10, 5))

        # --- Convert to numpy array ---
        arr = pygame.surfarray.array3d(self.screen)
        arr = np.transpose(arr, (1, 0, 2)).astype(np.uint8)
        
        if self.render_mode == "human":
            self.human_screen.blit(self.screen, (0, 0))
            pygame.event.pump()
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
            
        return arr
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "timer": self.timer,
            "checkpoints_remaining": self.NUM_CHECKPOINTS - self.score
        }

    def close(self):
        pygame.font.quit()
        pygame.quit()

if __name__ == '__main__':
    # Example of how to run the environment
    env = GameEnv(render_mode="human")
    obs, info = env.reset()
    done = False
    
    # --- Manual Control ---
    # Use arrow keys to move, SPACE to boost
    # ----------------------
    
    total_reward = 0
    while not done:
        keys = pygame.key.get_pressed()
        
        movement = 0 # No-op
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        # Or use random actions for testing the agent interface
        # action = env.action_space.sample()
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if terminated or truncated:
            done = True
            
        # Handle window close
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

    print(f"Game Over. Final Info: {info}")
    print(f"Total reward: {total_reward:.2f}")
    env.close()
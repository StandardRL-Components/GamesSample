import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T17:39:54.556890
# Source Brief: brief_02929.md
# Brief Index: 2929
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment where the player guides a recursively splitting
    falling block through a maze of moving obstacles.
    """
    metadata = {"render_modes": ["rgb_array"]}
    
    game_description = (
        "Guide a recursively splitting block through a maze of moving obstacles to reach the bottom."
    )
    user_guide = (
        "Controls: Use the ← and → arrow keys to rotate the falling block."
    )
    auto_advance = True


    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    COLOR_BG = (25, 28, 36)
    COLOR_GRID = (40, 44, 55)
    COLOR_OBSTACLE = (255, 120, 57) # Orange
    BLOCK_COLORS = [
        (80, 165, 255),  # Blue (Gen 0)
        (87, 255, 153),  # Green (Gen 1)
        (255, 220, 85),  # Yellow (Gen 2)
        (255, 80, 80),   # Red (Gen 3)
    ]
    COLOR_UI = (230, 230, 230)
    MAX_STEPS = 1000
    GRID_SIZE = 40
    ROTATION_SPEED = 0.08  # Radians per step
    MAX_GENERATION = 3

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        self.render_mode = render_mode

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        try:
            self.font = pygame.font.SysFont("Consolas", 24)
        except pygame.error:
            self.font = pygame.font.Font(None, 28)

        # --- Game State Variables ---
        self.steps = 0
        self.score = 0
        self.successful_splits = 0
        self.game_over = False
        self.blocks = []
        self.obstacles = []
        self.particles = []
        self.base_obstacle_speed = 1.0
        self.current_obstacle_speed = 1.0
        self.next_block_id = 0

        # Initialize state for the first time
        # self.reset() is called by the environment wrapper
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # --- Reset Game State ---
        self.steps = 0
        self.score = 0
        self.successful_splits = 0
        self.game_over = False
        self.next_block_id = 0

        # --- Reset Obstacles ---
        self.base_obstacle_speed = 1.0
        self.current_obstacle_speed = self.base_obstacle_speed
        self.obstacles = []
        num_obstacles = self.np_random.integers(5, 8)
        for i in range(num_obstacles):
            self._spawn_obstacle(i)

        # --- Reset Player Block ---
        self.blocks = []
        self._spawn_block(
            x=self.WIDTH / 2,
            y=60,
            size=60,
            generation=0
        )
        
        # --- Reset Particles ---
        self.particles = []

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        # space_held = action[1] == 1
        # shift_held = action[2] == 1
        
        reward = 0.0
        terminated = False
        
        # --- Update Game Logic ---
        self.steps += 1
        reward += 0.1  # Survival reward

        # 1. Handle Player Input (Rotation)
        rotation_dir = 0
        if movement == 3:  # Left
            rotation_dir = -1
        elif movement == 4:  # Right
            rotation_dir = 1
        
        for block in self.blocks:
            block['rotation'] += rotation_dir * self.ROTATION_SPEED

        # 2. Update Entity Positions
        self._update_blocks()
        self._update_obstacles()
        self._update_particles()
        
        # 3. Handle Collisions and Splits
        reward += self._handle_collisions()

        # 4. Check Termination Conditions
        if not self.blocks:
            terminated = True
            reward = -100.0  # Lose all blocks
            # sound: game_over_lose
        else:
            for block in self.blocks:
                if block['y'] - block['size'] / 2 > self.HEIGHT:
                    terminated = True
                    reward = 100.0  # A block reached the bottom
                    # sound: game_win
                    break
        
        if self.steps >= self.MAX_STEPS:
            terminated = True

        self.game_over = terminated
        truncated = self.steps >= self.MAX_STEPS
        
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _spawn_block(self, x, y, size, generation, vel=(0, 0)):
        if generation > self.MAX_GENERATION:
            return
            
        fall_speed = 1.0 + (self.MAX_GENERATION - generation) * 0.5

        new_block = {
            'id': self.next_block_id,
            'x': x, 'y': y,
            'size': size,
            'rotation': self.np_random.uniform(0, 2 * math.pi),
            'fall_speed': fall_speed,
            'color': self.BLOCK_COLORS[generation],
            'generation': generation,
            'vx': vel[0], 'vy': vel[1]
        }
        self.blocks.append(new_block)
        self.next_block_id += 1

    def _spawn_obstacle(self, index):
        radius = self.np_random.uniform(15, 25)
        # Ensure obstacles don't block the entire screen
        y_band = (index % 4 + 1) * (self.HEIGHT / 5)
        y = y_band + self.np_random.uniform(-20, 20)
        
        vx = self.np_random.choice([-1, 1]) * self.np_random.uniform(0.5, 1.0)
        
        self.obstacles.append({
            'x': self.np_random.uniform(radius, self.WIDTH - radius),
            'y': y,
            'radius': radius,
            'base_vx': vx,
            'vx': vx * self.current_obstacle_speed
        })
        
    def _update_blocks(self):
        for b in self.blocks:
            b['y'] += b['fall_speed'] + b['vy']
            b['x'] += b['vx']
            # Dampen initial split velocity
            b['vx'] *= 0.95
            b['vy'] *= 0.95

    def _update_obstacles(self):
        for o in self.obstacles:
            o['x'] += o['vx']
            if o['x'] - o['radius'] < 0 or o['x'] + o['radius'] > self.WIDTH:
                o['vx'] *= -1
                o['x'] += o['vx'] # prevent getting stuck

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['life'] -= 1
            p['size'] = max(0, p['size'] - 0.2)

    def _handle_collisions(self):
        split_reward = 0
        blocks_to_add = []
        blocks_to_remove = set()
        
        for block in self.blocks:
            if block['id'] in blocks_to_remove:
                continue
            for obstacle in self.obstacles:
                if self._check_poly_circle_collision(block, obstacle):
                    # sound: block_split
                    split_reward += 1.0
                    self.score += 1
                    self.successful_splits += 1
                    
                    blocks_to_remove.add(block['id'])
                    
                    # Spawn particles
                    for _ in range(20):
                        self._spawn_particle(block['x'], block['y'], block['color'])

                    # Spawn children
                    if block['generation'] < self.MAX_GENERATION:
                        for i in [-1, 1]:
                            angle = block['rotation'] + self.np_random.uniform(-math.pi/4, math.pi/4)
                            speed = self.np_random.uniform(1, 3)
                            vel = (i * speed * math.cos(angle), speed * math.sin(angle))
                            
                            blocks_to_add.append({
                                'x': block['x'], 'y': block['y'],
                                'size': block['size'] / math.sqrt(2),
                                'generation': block['generation'] + 1,
                                'vel': vel
                            })
                    break # Block can only split once per frame
        
        if blocks_to_remove:
            self.blocks = [b for b in self.blocks if b['id'] not in blocks_to_remove]
            for params in blocks_to_add:
                self._spawn_block(**params)

            # Difficulty scaling
            if self.successful_splits > 0 and self.successful_splits % 5 == 0:
                self.base_obstacle_speed += 0.05
                self.current_obstacle_speed = self.base_obstacle_speed
                for o in self.obstacles:
                    o['vx'] = o['base_vx'] * self.current_obstacle_speed

        return split_reward
        
    def _check_poly_circle_collision(self, block, circle):
        # Transform circle center to block's local coordinates
        cx, cy = circle['x'], circle['y']
        bx, by = block['x'], block['y']
        size = block['size']
        angle = -block['rotation'] # un-rotate
        
        local_cx = math.cos(angle) * (cx - bx) - math.sin(angle) * (cy - by)
        local_cy = math.sin(angle) * (cx - bx) + math.cos(angle) * (cy - by)
        
        # Find closest point on the AABB (now that block is un-rotated) to the local circle center
        half_size = size / 2
        clamped_x = max(-half_size, min(half_size, local_cx))
        clamped_y = max(-half_size, min(half_size, local_cy))
        
        dist_x = local_cx - clamped_x
        dist_y = local_cy - clamped_y
        
        return (dist_x**2 + dist_y**2) < (circle['radius']**2)

    def _spawn_particle(self, x, y, color):
        angle = self.np_random.uniform(0, 2 * math.pi)
        speed = self.np_random.uniform(1, 4)
        self.particles.append({
            'x': x, 'y': y,
            'vx': math.cos(angle) * speed,
            'vy': math.sin(angle) * speed,
            'size': self.np_random.uniform(3, 7),
            'life': self.np_random.integers(20, 40),
            'color': color
        })

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # 1. Render Grid
        for i in range(0, self.WIDTH, self.GRID_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (i, 0), (i, self.HEIGHT))
        for i in range(0, self.HEIGHT, self.GRID_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, i), (self.WIDTH, i))

        # 2. Render Obstacles
        for o in self.obstacles:
            pygame.gfxdraw.aacircle(self.screen, int(o['x']), int(o['y']), int(o['radius']), self.COLOR_OBSTACLE)
            pygame.gfxdraw.filled_circle(self.screen, int(o['x']), int(o['y']), int(o['radius']), self.COLOR_OBSTACLE)

        # 3. Render Particles
        for p in self.particles:
            alpha = int(255 * (p['life'] / 40))
            color = (*p['color'], alpha)
            temp_surf = pygame.Surface((p['size']*2, p['size']*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (p['size'], p['size']), p['size'])
            self.screen.blit(temp_surf, (p['x']-p['size'], p['y']-p['size']), special_flags=pygame.BLEND_RGBA_ADD)

        # 4. Render Blocks
        for b in self.blocks:
            self._render_rotated_square(b)

    def _render_rotated_square(self, block):
        x, y, size, rotation, color = block['x'], block['y'], block['size'], block['rotation'], block['color']
        half_size = size / 2
        points = []
        for i in range(4):
            angle = rotation + math.pi / 4 + i * math.pi / 2
            px = x + half_size * math.sqrt(2) * math.cos(angle)
            py = y + half_size * math.sqrt(2) * math.sin(angle)
            points.append((int(px), int(py)))
        
        # Draw a slightly larger, semi-transparent glow
        glow_color = (*color, 60)
        glow_points = []
        glow_size = half_size + 4
        for i in range(4):
            angle = rotation + math.pi / 4 + i * math.pi / 2
            px = x + glow_size * math.sqrt(2) * math.cos(angle)
            py = y + glow_size * math.sqrt(2) * math.sin(angle)
            glow_points.append((int(px), int(py)))

        pygame.gfxdraw.aapolygon(self.screen, glow_points, glow_color)
        pygame.gfxdraw.filled_polygon(self.screen, glow_points, glow_color)

        # Draw main polygon
        pygame.gfxdraw.aapolygon(self.screen, points, color)
        pygame.gfxdraw.filled_polygon(self.screen, points, color)


    def _render_ui(self):
        score_text = self.font.render(f"SCORE: {self.score}", True, self.COLOR_UI)
        self.screen.blit(score_text, (10, 10))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "blocks": len(self.blocks),
            "splits": self.successful_splits
        }
        
    def close(self):
        pygame.quit()

# --- Example Usage ---
if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    
    # --- Manual Play ---
    # Use arrow keys to control rotation
    
    obs, info = env.reset()
    terminated = False
    truncated = False
    
    # Pygame setup for rendering
    real_screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Recursive Fall")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        # Action mapping from keyboard
        movement = 0 # no-op
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        
        action = [movement, 0, 0] # No use for space/shift yet

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment.")
                obs, info = env.reset()
                total_reward = 0
                terminated = False
                truncated = False

        if not terminated and not truncated:
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if terminated or truncated:
                print(f"Episode finished! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")

        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        real_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Run at 30 FPS

    env.close()
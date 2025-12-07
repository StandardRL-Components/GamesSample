
# Generated: 2025-08-27T18:55:50.496654
# Source Brief: brief_01996.md
# Brief Index: 1996

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move your robot on the grid. Avoid the red lasers."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Navigate a robot through rotating laser grids to rescue a trapped kitten in this isometric puzzle game."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_WIDTH = 16
    GRID_HEIGHT = 10
    TILE_WIDTH_HALF = 20
    TILE_HEIGHT_HALF = 10

    # Colors
    COLOR_BG = (30, 35, 40)
    COLOR_GRID = (60, 65, 70)
    COLOR_ROBOT = (0, 168, 243)
    COLOR_ROBOT_GLOW = (0, 168, 243, 50)
    COLOR_KITTEN = (255, 165, 0)
    COLOR_LASER = (255, 20, 20)
    COLOR_LASER_GLOW = (255, 20, 20, 100)
    COLOR_EMITTER = (100, 100, 110)
    COLOR_TEXT = (240, 240, 240)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 48, bold=True)

        # Game state variables are initialized in reset()
        self.robot_pos = None
        self.kitten_pos = None
        self.laser_emitters = None
        self.laser_hits = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.particles = None
        self.victory = None
        
        # Calculate grid offset for centering
        self.grid_offset_x = (self.SCREEN_WIDTH - (self.GRID_WIDTH - self.GRID_HEIGHT) * self.TILE_WIDTH_HALF) / 2
        self.grid_offset_y = (self.SCREEN_HEIGHT - (self.GRID_WIDTH + self.GRID_HEIGHT) * self.TILE_HEIGHT_HALF) / 2 + 30

        self.reset()
        
        # self.validate_implementation() # Optional: Call for debugging

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.laser_hits = 0
        self.game_over = False
        self.victory = False
        self.particles = []

        # Place robot
        self.robot_pos = pygame.math.Vector2(1, self.GRID_HEIGHT - 2)

        # Place kitten, ensuring it's not too close to the start
        while True:
            self.kitten_pos = pygame.math.Vector2(
                self.np_random.integers(self.GRID_WIDTH // 2, self.GRID_WIDTH),
                self.np_random.integers(0, self.GRID_HEIGHT)
            )
            if self.robot_pos.distance_to(self.kitten_pos) > (self.GRID_WIDTH + self.GRID_HEIGHT) / 3:
                break
        
        # Define laser emitters: pos, initial_direction_idx, rotation_period (in steps)
        self.laser_emitters = [
            {'pos': pygame.math.Vector2(self.GRID_WIDTH // 4, self.GRID_HEIGHT // 2), 'dir_idx': 0, 'period': 4},
            {'pos': pygame.math.Vector2(self.GRID_WIDTH - 4, 3), 'dir_idx': 2, 'period': 3},
            {'pos': pygame.math.Vector2(self.GRID_WIDTH - 5, self.GRID_HEIGHT - 4), 'dir_idx': 1, 'period': 5},
            {'pos': pygame.math.Vector2(4, 2), 'dir_idx': 3, 'period': 6},
        ]
        
        # Ensure kitten is not on an emitter
        for emitter in self.laser_emitters:
            if self.kitten_pos == emitter['pos']:
                self.kitten_pos.x = (self.kitten_pos.x + 1) % self.GRID_WIDTH

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        reward = -0.01  # Small penalty for taking a step

        # --- 1. Update Robot Position ---
        prev_pos = self.robot_pos.copy()
        if movement == 1:  # Up (North)
            self.robot_pos.y -= 1
        elif movement == 2:  # Down (South)
            self.robot_pos.y += 1
        elif movement == 3:  # Left (West)
            self.robot_pos.x -= 1
        elif movement == 4:  # Right (East)
            self.robot_pos.x += 1
        
        # Boundary checks
        self.robot_pos.x = max(0, min(self.GRID_WIDTH - 1, self.robot_pos.x))
        self.robot_pos.y = max(0, min(self.GRID_HEIGHT - 1, self.robot_pos.y))
        
        # Prevent moving onto an emitter
        for emitter in self.laser_emitters:
            if self.robot_pos == emitter['pos']:
                self.robot_pos = prev_pos
                break
        
        self.steps += 1
        
        # --- 2. Check Laser Collisions ---
        if self._check_laser_collision():
            self.laser_hits += 1
            reward -= 1.0
            # sfx: laser_zap.wav
            self._create_particles(self.robot_pos, self.COLOR_LASER, 20)

        # --- 3. Check Win/Loss Conditions ---
        terminated = False
        if self.robot_pos == self.kitten_pos:
            reward += 10.0
            self.victory = True
            terminated = True
            self.game_over = True
            # sfx: victory.wav
            self._create_particles(self.kitten_pos, self.COLOR_KITTEN, 50)
        
        if self.laser_hits >= 5:
            reward -= 5.0 # Extra penalty for game over
            terminated = True
            self.game_over = True
            # sfx: game_over.wav
        
        if self.steps >= 1000:
            terminated = True
            self.game_over = True

        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _iso_to_screen(self, x, y):
        screen_x = self.grid_offset_x + (x - y) * self.TILE_WIDTH_HALF
        screen_y = self.grid_offset_y + (x + y) * self.TILE_HEIGHT_HALF
        return int(screen_x), int(screen_y)

    def _draw_iso_cube(self, surface, x, y, color, height=20):
        px, py = self._iso_to_screen(x, y)
        py -= height // 2
        
        # Points for the cube
        p = [
            (px, py - height // 2),
            (px + self.TILE_WIDTH_HALF, py - height // 2 + self.TILE_HEIGHT_HALF),
            (px, py - height // 2 + self.TILE_HEIGHT_HALF * 2),
            (px - self.TILE_WIDTH_HALF, py - height // 2 + self.TILE_HEIGHT_HALF),
        ]
        
        top_face = [p[0], p[1], (p[1][0], p[1][1] - self.TILE_HEIGHT_HALF), (p[0][0], p[0][1] - self.TILE_HEIGHT_HALF)]
        
        top_points = [
            (px, py - self.TILE_HEIGHT_HALF),
            (px + self.TILE_WIDTH_HALF, py),
            (px, py + self.TILE_HEIGHT_HALF),
            (px - self.TILE_WIDTH_HALF, py)
        ]
        
        # Draw faces
        darker_color = tuple(max(0, c - 40) for c in color[:3])
        darkest_color = tuple(max(0, c - 60) for c in color[:3])
        
        pygame.draw.polygon(surface, color, top_points) # Top
        pygame.draw.polygon(surface, darker_color, [top_points[3], top_points[2], (top_points[2][0], top_points[2][1] + height), (top_points[3][0], top_points[3][1] + height)]) # Left
        pygame.draw.polygon(surface, darkest_color, [top_points[2], top_points[1], (top_points[1][0], top_points[1][1] + height), (top_points[2][0], top_points[2][1] + height)]) # Right

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        
        self._render_grid()
        self._render_lasers()
        self._render_entities()
        self._update_and_render_particles()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_grid(self):
        for y in range(self.GRID_HEIGHT + 1):
            start = self._iso_to_screen(0, y)
            end = self._iso_to_screen(self.GRID_WIDTH, y)
            pygame.draw.line(self.screen, self.COLOR_GRID, start, end, 1)
        for x in range(self.GRID_WIDTH + 1):
            start = self._iso_to_screen(x, 0)
            end = self._iso_to_screen(x, self.GRID_HEIGHT)
            pygame.draw.line(self.screen, self.COLOR_GRID, start, end, 1)

    def _render_lasers(self):
        directions = [
            (0, -1), (1, -1), (1, 0), (1, 1),
            (0, 1), (-1, 1), (-1, 0), (-1, -1)
        ]
        
        for emitter in self.laser_emitters:
            # Draw emitter base
            self._draw_iso_cube(self.screen, emitter['pos'].x, emitter['pos'].y, self.COLOR_EMITTER, height=10)
            
            # Calculate current laser direction
            current_dir_idx = (emitter['dir_idx'] + (self.steps // emitter['period'])) % 8
            direction = pygame.math.Vector2(directions[current_dir_idx])

            # Trace laser path
            start_pos = emitter['pos'].copy()
            end_pos = start_pos.copy()
            while 0 <= end_pos.x < self.GRID_WIDTH and 0 <= end_pos.y < self.GRID_HEIGHT:
                end_pos += direction
            
            # Draw beam
            p_start = self._iso_to_screen(start_pos.x, start_pos.y)
            p_end = self._iso_to_screen(end_pos.x, end_pos.y)
            
            pygame.draw.line(self.screen, self.COLOR_LASER_GLOW, p_start, p_end, 5)
            pygame.draw.aaline(self.screen, self.COLOR_LASER, p_start, p_end, True)

    def _check_laser_collision(self):
        directions = [
            (0, -1), (1, -1), (1, 0), (1, 1),
            (0, 1), (-1, 1), (-1, 0), (-1, -1)
        ]
        for emitter in self.laser_emitters:
            current_dir_idx = (emitter['dir_idx'] + (self.steps // emitter['period'])) % 8
            direction = pygame.math.Vector2(directions[current_dir_idx])
            
            current_pos = emitter['pos'].copy()
            for _ in range(max(self.GRID_WIDTH, self.GRID_HEIGHT)):
                current_pos += direction
                if not (0 <= current_pos.x < self.GRID_WIDTH and 0 <= current_pos.y < self.GRID_HEIGHT):
                    break
                if current_pos == self.robot_pos:
                    return True
        return False

    def _render_entities(self):
        # Draw kitten
        self._draw_iso_cube(self.screen, self.kitten_pos.x, self.kitten_pos.y, self.COLOR_KITTEN, height=15)
        
        # Draw robot
        bob_offset = math.sin(self.steps * 0.5) * 3
        px, py = self._iso_to_screen(self.robot_pos.x, self.robot_pos.y)
        py += int(bob_offset)
        
        # Glow effect
        pygame.gfxdraw.filled_circle(self.screen, px, py, 18, self.COLOR_ROBOT_GLOW)
        pygame.gfxdraw.aacircle(self.screen, px, py, 18, self.COLOR_ROBOT_GLOW)
        
        # Main body
        self._draw_iso_cube(self.screen, self.robot_pos.x, self.robot_pos.y, self.COLOR_ROBOT, height=20)

    def _render_ui(self):
        # Hits display
        hits_text = self.font_small.render(f"HITS: {self.laser_hits}/5", True, self.COLOR_TEXT)
        self.screen.blit(hits_text, (10, 10))
        
        # Steps display
        steps_text = self.font_small.render(f"STEPS: {self.steps}", True, self.COLOR_TEXT)
        self.screen.blit(steps_text, (self.SCREEN_WIDTH - steps_text.get_width() - 10, 10))

        # Game over messages
        if self.game_over:
            if self.victory:
                msg = "KITTEN RESCUED!"
                color = self.COLOR_KITTEN
            else:
                msg = "GAME OVER"
                color = self.COLOR_LASER
            
            end_text = self.font_large.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def _create_particles(self, grid_pos, color, count):
        screen_pos = self._iso_to_screen(grid_pos.x, grid_pos.y)
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            velocity = pygame.math.Vector2(math.cos(angle) * speed, math.sin(angle) * speed)
            self.particles.append({
                'pos': pygame.math.Vector2(screen_pos),
                'vel': velocity,
                'life': self.np_random.integers(10, 20),
                'color': color
            })

    def _update_and_render_particles(self):
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)
            else:
                size = max(1, int(p['life'] * 0.2))
                pygame.draw.circle(self.screen, p['color'], p['pos'], size)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "laser_hits": self.laser_hits,
            "robot_pos": (self.robot_pos.x, self.robot_pos.y),
            "kitten_pos": (self.kitten_pos.x, self.kitten_pos.y),
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == '__main__':
    # Set this to "human" to see the game being played
    render_mode = "human" # "rgb_array" or "human"
    
    if render_mode == "human":
        # For human mode, we need a real display
        pygame.display.set_caption("Rescue Kitten")
        screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    terminated = False
    total_reward = 0
    
    # Map Pygame keys to actions
    key_to_action = {
        pygame.K_UP: 1,
        pygame.K_DOWN: 2,
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4,
    }
    
    running = True
    clock = pygame.time.Clock()
    
    while running:
        action = [0, 0, 0] # Default action is no-op
        
        if render_mode == "human":
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key in key_to_action:
                        action[0] = key_to_action[event.key]
                    if event.key == pygame.K_r: # Reset on 'r'
                        obs, info = env.reset()
                        total_reward = 0
                        terminated = False
                        action = [0, 0, 0]
                    if event.key == pygame.K_q: # Quit on 'q'
                        running = False
            
            # Since auto_advance is False, we only step on key presses
            if action[0] != 0 and not terminated:
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                print(f"Step: {info['steps']}, Action: {action[0]}, Reward: {reward:.2f}, Total Reward: {total_reward:.2f}, Hits: {info['laser_hits']}")
            
            # Draw the observation to the human-mode screen
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            
            clock.tick(30) # Limit human-mode FPS
        
        else: # rgb_array mode (for training)
            if terminated:
                print(f"Episode finished. Total Reward: {total_reward:.2f}, Steps: {info['steps']}")
                obs, info = env.reset()
                total_reward = 0
            
            # Replace with your agent's action
            action = env.action_space.sample() 
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

    env.close()
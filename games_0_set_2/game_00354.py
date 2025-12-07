
# Generated: 2025-08-27T13:24:10.887455
# Source Brief: brief_00354.md
# Brief Index: 354

        
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
        "Controls: ←→ to move the falling tile, ↓ to speed up its descent."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Build the tallest tower you can by placing falling tiles. Manage the tower's stability to prevent a collapse. Reach a height of 20 to win!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_COLS, self.GRID_ROWS = 10, 12
        self.CELL_SIZE = 32
        self.GRID_WIDTH = self.GRID_COLS * self.CELL_SIZE
        self.GRID_HEIGHT = self.GRID_ROWS * self.CELL_SIZE
        self.PLAY_AREA_X_OFFSET = (self.WIDTH - self.GRID_WIDTH) / 2
        self.PLAY_AREA_Y_OFFSET = self.HEIGHT - self.GRID_HEIGHT
        self.WIN_HEIGHT = 20
        self.MAX_STEPS = 1500

        # --- Colors ---
        self.COLOR_BG = (20, 30, 40)
        self.COLOR_GRID = (40, 50, 60)
        self.COLOR_TEXT = (220, 230, 240)
        self.COLOR_STABLE = (0, 255, 127)
        self.COLOR_UNSTABLE = (255, 69, 0)
        self.COLOR_WARN = (255, 215, 0)
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 72)
        
        # State variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.placed_tiles = []
        self.falling_tile = None
        self.tower_height = 0
        self.fall_speed = 0
        self.stability = 1.0
        self.particles = []
        self.collapse_timer = 0
        
        # Initialize state variables
        self.reset()

        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.placed_tiles = []
        self.tower_height = 0
        self.fall_speed = 1.0
        self.stability = 1.0
        self.particles = []
        self.collapse_timer = 0
        
        self._spawn_new_tile()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0
        self.steps += 1

        if not self.game_over:
            self._handle_action(action)
            reward += self._update_game_state()
        elif self.collapse_timer > 0:
            self.collapse_timer -= 1

        self.score += reward
        terminated = self._check_termination()
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_action(self, action):
        movement = action[0]
        
        if self.falling_tile:
            if movement == 3:  # Left
                self.falling_tile['gx'] = max(0, self.falling_tile['gx'] - 1)
            elif movement == 4:  # Right
                self.falling_tile['gx'] = min(self.GRID_COLS - 1, self.falling_tile['gx'] + 1)
            
            # Update pixel position from grid position
            self.falling_tile['x'] = self.PLAY_AREA_X_OFFSET + self.falling_tile['gx'] * self.CELL_SIZE

    def _update_game_state(self):
        reward = 0
        if not self.falling_tile:
            return reward

        # Difficulty scaling
        if self.steps > 0 and self.steps % 200 == 0:
            self.fall_speed += 0.1

        # Soft drop
        movement = self.action_space.sample()[0] if not hasattr(self, '_last_action') else self._last_action[0]
        current_fall_speed = self.fall_speed * 5 if movement == 2 else self.fall_speed

        self.falling_tile['y'] += current_fall_speed
        
        # Collision detection
        collision_y = self.HEIGHT
        for tile in self.placed_tiles:
            if self.falling_tile['gx'] == tile['gx']:
                collision_y = min(collision_y, tile['y'])

        if self.falling_tile['y'] + self.CELL_SIZE >= collision_y:
            # Place tile
            self.falling_tile['y'] = collision_y - self.CELL_SIZE
            self.falling_tile['gy'] = self._pixel_to_grid_y(self.falling_tile['y'])
            self.placed_tiles.append(self.falling_tile)
            
            # Sound placeholder: sfx_place_tile.play()
            self._create_particles(self.falling_tile['x'] + self.CELL_SIZE / 2, self.falling_tile['y'] + self.CELL_SIZE)
            
            reward += 0.1  # Reward for placing a tile

            # Update tower height
            new_height = self._calculate_tower_height()
            if new_height > self.tower_height:
                reward += 1.0 * (new_height - self.tower_height)
                self.tower_height = new_height

            self.falling_tile = None
            
            # Check stability and potential collapse
            stability_reward = self._update_stability()
            reward += stability_reward

            # Check win condition
            if self.tower_height >= self.WIN_HEIGHT:
                self.game_over = True
                reward += 100
                # Sound placeholder: sfx_win_game.play()

            if not self.game_over:
                self._spawn_new_tile()

        return reward

    def _calculate_tower_height(self):
        if not self.placed_tiles:
            return 0
        min_y_coord = min(tile['gy'] for tile in self.placed_tiles)
        return self.GRID_ROWS - min_y_coord

    def _update_stability(self):
        if len(self.placed_tiles) <= 1:
            self.stability = 1.0
            return 0

        # Find support base
        lowest_y = max(tile['gy'] for tile in self.placed_tiles)
        support_tiles = [t for t in self.placed_tiles if t['gy'] == lowest_y]
        if not support_tiles:
            self.stability = 0
            self.game_over = True
            self.collapse_timer = 60
            return -100

        base_min_gx = min(t['gx'] for t in support_tiles)
        base_max_gx = max(t['gx'] for t in support_tiles)
        
        base_min_x = self.PLAY_AREA_X_OFFSET + base_min_gx * self.CELL_SIZE
        base_max_x = self.PLAY_AREA_X_OFFSET + (base_max_gx + 1) * self.CELL_SIZE
        base_center_x = (base_min_x + base_max_x) / 2
        base_width = base_max_x - base_min_x

        # Calculate Center of Mass (CoM)
        total_weight = 0
        weighted_sum_x = 0
        for tile in self.placed_tiles:
            total_weight += tile['weight']
            tile_center_x = tile['x'] + self.CELL_SIZE / 2
            weighted_sum_x += tile['weight'] * tile_center_x
        
        com_x = weighted_sum_x / total_weight if total_weight > 0 else self.WIDTH / 2

        # Check for collapse
        if com_x < base_min_x or com_x > base_max_x:
            self.game_over = True
            self.collapse_timer = 60 # Animation frames
            # Sound placeholder: sfx_tower_collapse.play()
            return -100
        
        # Update stability metric
        if base_width > 0:
            self.stability = 1.0 - abs(com_x - base_center_x) / (base_width / 2.0)
            self.stability = max(0, min(1, self.stability))
        else: # Single tile support
            self.stability = 1.0

        return 0

    def _spawn_new_tile(self):
        weight = self.np_random.uniform(0.5, 1.5)
        start_gx = self.np_random.integers(0, self.GRID_COLS)
        color_val = int(100 + 100 * (weight - 0.5))
        self.falling_tile = {
            'gx': start_gx,
            'gy': 0,
            'x': self.PLAY_AREA_X_OFFSET + start_gx * self.CELL_SIZE,
            'y': self.PLAY_AREA_Y_OFFSET - self.CELL_SIZE,
            'weight': weight,
            'color': (color_val, color_val, color_val + 20)
        }

    def _check_termination(self):
        if self.game_over:
            return True
        if self.steps >= self.MAX_STEPS:
            return True
        return False
    
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_grid()
        self._render_placed_tiles()
        if self.falling_tile and not self.game_over:
            self._render_tile(self.falling_tile)
        self._update_and_render_particles()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "height": self.tower_height,
            "stability": self.stability,
        }

    def _render_grid(self):
        for row in range(self.GRID_ROWS + 1):
            y = self.PLAY_AREA_Y_OFFSET + row * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.PLAY_AREA_X_OFFSET, y), (self.PLAY_AREA_X_OFFSET + self.GRID_WIDTH, y))
        for col in range(self.GRID_COLS + 1):
            x = self.PLAY_AREA_X_OFFSET + col * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, self.PLAY_AREA_Y_OFFSET), (x, self.HEIGHT))

    def _render_placed_tiles(self):
        is_collapsing = self.game_over and self.collapse_timer > 0
        for tile in self.placed_tiles:
            wobble_x = 0
            wobble_y = 0
            
            if is_collapsing:
                # Dramatic collapse animation
                fall_progress = (60 - self.collapse_timer) / 60.0
                wobble_x = self.np_random.uniform(-1, 1) * 10 * fall_progress
                wobble_y = fall_progress * fall_progress * 200
            elif self.stability < 0.8:
                # Subtle wobble for instability
                wobble_intensity = (1 - self.stability) * 4
                wobble_x = math.sin(self.steps * 0.3 + tile['gy']) * wobble_intensity
            
            # Determine color based on stability
            if self.stability < 0.3:
                flash_alpha = int(128 + 127 * math.sin(self.steps * 0.5))
                flash_color = (*self.COLOR_UNSTABLE, flash_alpha)
                self._render_tile(tile, wobble_x, wobble_y, flash_color)
            else:
                self._render_tile(tile, wobble_x, wobble_y)

    def _render_tile(self, tile, ox=0, oy=0, flash_color=None):
        rect = pygame.Rect(
            int(tile['x'] + ox), int(tile['y'] + oy), 
            self.CELL_SIZE, self.CELL_SIZE
        )
        border_color = tuple(max(0, c - 40) for c in tile['color'])
        pygame.draw.rect(self.screen, tile['color'], rect)
        pygame.draw.rect(self.screen, border_color, rect, 2)

        if flash_color:
            s = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
            s.fill(flash_color)
            self.screen.blit(s, (rect.x, rect.y))

    def _render_ui(self):
        # Height display
        height_text = self.font_small.render(f"HEIGHT: {self.tower_height} / {self.WIN_HEIGHT}", True, self.COLOR_TEXT)
        self.screen.blit(height_text, (10, 10))

        # Score display
        score_text = self.font_small.render(f"SCORE: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 35))

        # Stability Bar
        stability_label = self.font_small.render("STABILITY", True, self.COLOR_TEXT)
        self.screen.blit(stability_label, (self.WIDTH - 160, 10))
        bar_bg_rect = pygame.Rect(self.WIDTH - 160, 35, 150, 20)
        pygame.draw.rect(self.screen, self.COLOR_GRID, bar_bg_rect)
        
        bar_width = 150 * self.stability
        bar_color = self.COLOR_STABLE
        if self.stability < 0.6: bar_color = self.COLOR_WARN
        if self.stability < 0.3: bar_color = self.COLOR_UNSTABLE
        
        bar_rect = pygame.Rect(self.WIDTH - 160, 35, bar_width, 20)
        pygame.draw.rect(self.screen, bar_color, bar_rect)
        pygame.draw.rect(self.screen, self.COLOR_TEXT, bar_bg_rect, 1)

        # Game Over Message
        if self.game_over and self.collapse_timer <= 0:
            msg = "YOU WON!" if self.tower_height >= self.WIN_HEIGHT else "TOWER COLLAPSED"
            end_text = self.font_large.render(msg, True, self.COLOR_TEXT)
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def _pixel_to_grid_y(self, y_pos):
        return round((y_pos - self.PLAY_AREA_Y_OFFSET) / self.CELL_SIZE)

    def _create_particles(self, x, y):
        for _ in range(10):
            particle = {
                'x': x,
                'y': y,
                'vx': self.np_random.uniform(-2, 2),
                'vy': self.np_random.uniform(-3, -1),
                'life': 20,
                'color': random.choice([(200,200,200), (150,150,150), (250,250,250)])
            }
            self.particles.append(particle)

    def _update_and_render_particles(self):
        for p in self.particles:
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['vy'] += 0.2 # Gravity
            p['life'] -= 1
            size = max(1, int(p['life'] / 5))
            pygame.draw.rect(self.screen, p['color'], (int(p['x']), int(p['y']), size, size))
        self.particles = [p for p in self.particles if p['life'] > 0]

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        self._last_action = test_action # for soft drop check in update
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # To run and play the game manually
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    running = True
    total_score = 0
    
    # Use a dictionary to track held keys for smoother controls
    keys_held = {
        pygame.K_LEFT: False,
        pygame.K_RIGHT: False,
        pygame.K_DOWN: False,
    }
    
    # Create a window to display the game
    pygame.display.set_caption("Tower Builder")
    display_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))

    while running:
        action = [0, 0, 0] # Default action: no-op, buttons released

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key in keys_held:
                    keys_held[event.key] = True
                if event.key == pygame.K_r: # Reset on 'r'
                    obs, info = env.reset()
                    total_score = 0
            if event.type == pygame.KEYUP:
                if event.key in keys_held:
                    keys_held[event.key] = False

        # Map held keys to MultiDiscrete action
        if keys_held[pygame.K_LEFT]:
            action[0] = 3
        elif keys_held[pygame.K_RIGHT]:
            action[0] = 4
        elif keys_held[pygame.K_DOWN]:
            action[0] = 2
        else:
            action[0] = 0 # No movement

        obs, reward, terminated, truncated, info = env.step(action)
        total_score += reward
        
        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Episode finished. Final Score: {info['score']:.2f}, Height: {info['height']}")
            # Wait for a moment before auto-resetting
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_score = 0

        env.clock.tick(30) # 30 FPS

    pygame.quit()
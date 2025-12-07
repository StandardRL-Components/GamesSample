import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T11:57:16.972063
# Source Brief: brief_00792.md
# Brief Index: 792
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    Musical Skyscraper Environment

    The agent controls a character who jumps between skyscraper columns.
    Landing on a column adds a colored tile (a "musical note").
    Matching a sequence of same-colored tiles builds the column higher,
    creating a new platform and earning points. The goal is to build
    and ascend as high as possible without falling.

    Action Space: MultiDiscrete([5, 2, 2])
    - actions[0]: Movement (0=none, 1=up/right, 2=down/left, 3=left, 4=right)
    - actions[1]: Jump button (0=released, 1=pressed)
    - actions[2]: Shift button (unused)

    Observation Space: Box(0, 255, (400, 640, 3), uint8)
    - An RGB image of the game screen.

    Reward Structure:
    - +1 for each tile in a successful match.
    - +5 for successfully jumping to a new platform.
    - +100 for reaching a new maximum height across all columns.
    - -100 for falling (jumping to a non-existent platform).
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Jump between skyscraper columns to place colored musical tiles. Match sequences of "
        "the same color to build the columns higher and ascend to new heights."
    )
    user_guide = (
        "Use ← and → arrow keys to select a target column. Press space to jump."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30
    MAX_STEPS = 5000

    # Gameplay Constants
    NUM_COLUMNS = 10
    COLUMN_WIDTH = SCREEN_WIDTH // NUM_COLUMNS
    TILE_HEIGHT = 15
    PLAYER_JUMP_SPEED = 0.08  # Progress per frame

    # Color Palette
    COLOR_BG = (15, 18, 32)
    COLOR_PLAYER = (255, 255, 0)
    COLOR_PLAYER_GLOW = (255, 255, 0)
    COLOR_SELECTOR = (255, 255, 255)
    CHORD_COLORS = [
        (255, 70, 70),   # C (Red)
        (70, 255, 70),   # G (Green)
        (70, 70, 255),   # F (Blue)
        (255, 165, 0),   # Dm (Orange)
        (128, 0, 128),   # Am (Purple)
        (0, 255, 255),   # Em (Cyan)
        (255, 192, 203)  # Bdim (Pink)
    ]

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
        self.font_game_over = pygame.font.SysFont("Consolas", 50, bold=True)

        # --- Game State Variables ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_pos = np.array([0.0, 0.0])
        self.player_column = 0
        self.camera_y = 0.0
        self.skyscraper_columns = []
        self.platform_heights = []
        self.next_tile_color_idx = 0
        self.max_height_reached = 0
        self.jump_target_column = 0
        self.jump_state = None  # None or {'start_pos', 'end_pos', 'progress'}
        self.particles = []
        self.tiles_to_match = 3
        self.num_chord_types = 2
        self.successful_jumps = 0
        self.last_space_held = False
        self.reward_this_step = 0

        # --- Background Stars ---
        self.stars = [
            (random.randint(0, self.SCREEN_WIDTH), random.randint(0, self.SCREEN_HEIGHT), random.randint(1, 2))
            for _ in range(150)
        ]

        if self.render_mode == "human":
            pygame.display.set_caption("Musical Skyscraper")
            self.screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        
        # self.validate_implementation() # Optional validation call

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False

        base_y = self.SCREEN_HEIGHT - self.TILE_HEIGHT * 3
        self.platform_heights = [base_y] * self.NUM_COLUMNS
        self.skyscraper_columns = [[] for _ in range(self.NUM_COLUMNS)]

        self.player_column = self.np_random.integers(0, self.NUM_COLUMNS)
        player_x = (self.player_column + 0.5) * self.COLUMN_WIDTH
        self.player_pos = np.array([player_x, self.platform_heights[self.player_column] - self.TILE_HEIGHT])

        self.jump_target_column = self.player_column
        self.camera_y = 0
        self.max_height_reached = base_y
        self.next_tile_color_idx = self.np_random.integers(0, self.num_chord_types)
        
        self.jump_state = None
        self.particles = []
        self.tiles_to_match = 3
        self.num_chord_types = 2
        self.successful_jumps = 0
        self.last_space_held = False

        return self._get_observation(), self._get_info()

    def step(self, action):
        self.reward_this_step = 0
        
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        self._handle_action(action)
        self._update_game_state()

        self.steps += 1
        terminated = self.game_over or self.steps >= self.MAX_STEPS
        
        reward = self.reward_this_step
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_action(self, action):
        if self.jump_state:  # Cannot act while jumping
            return

        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        space_pressed = space_held and not self.last_space_held
        self.last_space_held = space_held

        # --- Update Jump Target ---
        if movement in [1, 4]:  # Up or Right
            self.jump_target_column = (self.jump_target_column + 1) % self.NUM_COLUMNS
        elif movement in [2, 3]:  # Down or Left
            self.jump_target_column = (self.jump_target_column - 1 + self.NUM_COLUMNS) % self.NUM_COLUMNS

        # --- Execute Jump ---
        if space_pressed:
            target_platform_y = self.platform_heights[self.jump_target_column]
            
            if target_platform_y is None:
                # Fall and end game
                self.game_over = True
                self.reward_this_step = -100
                # Sound: Fall
                start_pos = self.player_pos.copy()
                end_pos = np.array([start_pos[0], start_pos[1] + 200])
                self.jump_state = {'start_pos': start_pos, 'end_pos': end_pos, 'progress': 0.0, 'fall': True}
            else:
                # Successful jump
                self.reward_this_step += 5
                start_pos = self.player_pos.copy()
                target_x = (self.jump_target_column + 0.5) * self.COLUMN_WIDTH
                end_pos = np.array([target_x, target_platform_y - self.TILE_HEIGHT])
                self.jump_state = {'start_pos': start_pos, 'end_pos': end_pos, 'progress': 0.0, 'fall': False}
                # Sound: Jump

    def _update_game_state(self):
        # --- Update Jump Animation ---
        if self.jump_state:
            self.jump_state['progress'] = min(1.0, self.jump_state['progress'] + self.PLAYER_JUMP_SPEED)
            
            # Interpolate position
            p = self.jump_state['progress']
            start = self.jump_state['start_pos']
            end = self.jump_state['end_pos']
            
            # Linear interpolation for X
            self.player_pos[0] = start[0] + (end[0] - start[0]) * p
            
            # Parabolic arc for Y
            mid_y_offset = -abs(end[0] - start[0]) * 0.3 - 30  # Higher arc for longer jumps
            self.player_pos[1] = start[1] + (end[1] - start[1]) * p + mid_y_offset * (1 - (2*p - 1)**2)

            # --- Handle Landing ---
            if self.jump_state['progress'] >= 1.0:
                if self.jump_state.get('fall', False):
                    self.game_over = True
                else:
                    self.player_pos = self.jump_state['end_pos'].copy()
                    self.player_column = self.jump_target_column
                    self._land_and_build()
                self.jump_state = None

        # --- Update Camera ---
        target_camera_y = self.player_pos[1] - self.SCREEN_HEIGHT * 0.6
        self.camera_y += (target_camera_y - self.camera_y) * 0.05

        # --- Update Particles ---
        self.particles = [p for p in self.particles if p['lifespan'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['vel'][1] += 0.2  # Gravity
            p['lifespan'] -= 1

    def _land_and_build(self):
        # Add the "carried" tile to the column
        self.skyscraper_columns[self.player_column].append(self.next_tile_color_idx)
        # Sound: Place tile

        # Check for a match and update score/platforms
        self._check_and_process_match(self.player_column)
        
        self.successful_jumps += 1
        self._update_difficulty()
        
        # Get next tile
        self.next_tile_color_idx = self.np_random.integers(0, self.num_chord_types)

    def _check_and_process_match(self, col_idx):
        column = self.skyscraper_columns[col_idx]
        if len(column) < self.tiles_to_match:
            return

        last_tiles = column[-self.tiles_to_match:]
        if len(set(last_tiles)) == 1:
            # --- Match Found ---
            # Sound: Match success
            matched_color_idx = last_tiles[0]
            
            self.reward_this_step += self.tiles_to_match
            self.score += self.tiles_to_match

            # Remove matched tiles
            self.skyscraper_columns[col_idx] = column[:-self.tiles_to_match]

            # Create new platform
            old_platform_y = self.platform_heights[col_idx]
            new_platform_y = old_platform_y - self.TILE_HEIGHT * 2
            self.platform_heights[col_idx] = new_platform_y
            
            # Teleport player to new platform
            self.player_pos[1] = new_platform_y - self.TILE_HEIGHT
            
            # Create particles
            for _ in range(30):
                angle = self.np_random.uniform(0, 2 * math.pi)
                speed = self.np_random.uniform(2, 5)
                self.particles.append({
                    'pos': self.player_pos.copy() + np.array([0, self.TILE_HEIGHT]),
                    'vel': np.array([math.cos(angle) * speed, math.sin(angle) * speed]),
                    'color': self.CHORD_COLORS[matched_color_idx],
                    'lifespan': self.np_random.integers(20, 40)
                })

            # Check for new max height
            if new_platform_y < self.max_height_reached:
                self.reward_this_step += 100
                self.score += 100
                self.max_height_reached = new_platform_y

    def _update_difficulty(self):
        # New chord type every 50 jumps
        if self.successful_jumps > 0 and self.successful_jumps % 50 == 0:
            if self.num_chord_types < len(self.CHORD_COLORS):
                self.num_chord_types += 1
        
        # Increase tiles to match every 100 jumps
        if self.successful_jumps > 0 and self.successful_jumps % 100 == 0:
            self.tiles_to_match += 1

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()

        if self.render_mode == "human":
            pygame.display.flip()
            self.clock.tick(self.FPS)

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # --- Draw Stars ---
        for star in self.stars:
            color_val = 50 + star[2] * 20
            star_color = (color_val, color_val, color_val)
            pos = (star[0], (star[1] - int(self.camera_y * 0.1)) % self.SCREEN_HEIGHT)
            self.screen.set_at(pos, star_color)

        # --- Draw Skyscraper ---
        for i, column in enumerate(self.skyscraper_columns):
            col_x = i * self.COLUMN_WIDTH
            platform_y = self.platform_heights[i]
            
            # Draw placed tiles
            for j, tile_color_idx in enumerate(column):
                tile_y = platform_y + j * self.TILE_HEIGHT
                tile_rect = pygame.Rect(col_x, tile_y - self.camera_y, self.COLUMN_WIDTH, self.TILE_HEIGHT)
                color = self.CHORD_COLORS[tile_color_idx]
                pygame.draw.rect(self.screen, color, tile_rect)
                pygame.draw.rect(self.screen, tuple(c*0.7 for c in color), tile_rect, 1)

            # Draw platforms
            if platform_y is not None:
                platform_rect = pygame.Rect(col_x - 2, platform_y - self.camera_y, self.COLUMN_WIDTH + 4, 5)
                pygame.draw.rect(self.screen, (200, 200, 220), platform_rect)

        # --- Draw Jump Target Selector ---
        if not self.jump_state:
            selector_x = (self.jump_target_column + 0.5) * self.COLUMN_WIDTH
            selector_y = self.platform_heights[self.jump_target_column]
            if selector_y is not None:
                selector_y -= self.camera_y + 25
                points = [
                    (selector_x, selector_y),
                    (selector_x - 10, selector_y - 10),
                    (selector_x + 10, selector_y - 10)
                ]
                pygame.draw.polygon(self.screen, self.COLOR_SELECTOR, points)

        # --- Draw Particles ---
        for p in self.particles:
            pos = (int(p['pos'][0]), int(p['pos'][1] - self.camera_y))
            alpha = max(0, min(255, int(255 * (p['lifespan'] / 40.0))))
            color_with_alpha = p['color'] + (alpha,)
            temp_surf = pygame.Surface((4, 4), pygame.SRCALPHA)
            pygame.draw.rect(temp_surf, color_with_alpha, (0, 0, 4, 4))
            self.screen.blit(temp_surf, (pos[0]-2, pos[1]-2))

        # --- Draw Player ---
        px, py = int(self.player_pos[0]), int(self.player_pos[1] - self.camera_y)
        # Glow effect
        for i in range(10, 0, -2):
            alpha = 80 - i * 8
            pygame.gfxdraw.filled_circle(self.screen, px, py, 8 + i, (*self.COLOR_PLAYER_GLOW, alpha))
        # Player core
        pygame.gfxdraw.filled_circle(self.screen, px, py, 8, self.COLOR_PLAYER)
        pygame.gfxdraw.aacircle(self.screen, px, py, 8, self.COLOR_PLAYER)

    def _render_ui(self):
        # --- Score Display ---
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, (255, 255, 255))
        self.screen.blit(score_text, (10, 10))

        # --- Height Display ---
        height_val = (self.SCREEN_HEIGHT - self.max_height_reached) // 10
        height_text = self.font_ui.render(f"HEIGHT: {height_val}m", True, (255, 255, 255))
        self.screen.blit(height_text, (10, 35))

        # --- Next Tile Preview ---
        preview_text = self.font_ui.render("NEXT:", True, (255, 255, 255))
        self.screen.blit(preview_text, (self.SCREEN_WIDTH - 120, 10))
        preview_rect = pygame.Rect(self.SCREEN_WIDTH - 50, 10, 40, 20)
        pygame.draw.rect(self.screen, self.CHORD_COLORS[self.next_tile_color_idx], preview_rect)
        pygame.draw.rect(self.screen, (255,255,255), preview_rect, 1)

        # --- Difficulty Info ---
        difficulty_text = self.font_ui.render(f"MATCH: {self.tiles_to_match}", True, (255, 255, 255))
        self.screen.blit(difficulty_text, (self.SCREEN_WIDTH - 120, 40))

        # --- Game Over Screen ---
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            game_over_text = self.font_game_over.render("GAME OVER", True, (255, 50, 50))
            text_rect = game_over_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(game_over_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "max_height_meters": (self.SCREEN_HEIGHT - self.max_height_reached) // 10,
            "successful_jumps": self.successful_jumps,
            "current_difficulty_level": self.tiles_to_match,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """
        Call this at the end of __init__ to verify implementation.
        """
        print("Validating implementation...")
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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == '__main__':
    env = GameEnv(render_mode="human")
    obs, info = env.reset()
    done = False
    total_reward = 0
    
    print("--- Musical Skyscraper ---")
    print("Controls:")
    print("  Left/Right Arrow Keys: Select target column")
    print("  Spacebar: Jump")
    print("  R: Reset environment")
    print("--------------------------")

    # Manual play loop
    while not done:
        action = [0, 0, 0] # Default action: no-op
        
        # Pygame event handling for manual control
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    action[0] = 3
                elif event.key == pygame.K_RIGHT:
                    action[0] = 4
                elif event.key == pygame.K_SPACE:
                    action[1] = 1
                elif event.key == pygame.K_r:
                    obs, info = env.reset()
                    total_reward = 0
                    print("Environment reset.")

        # If an action was taken, step the environment
        if any(a != 0 for a in action):
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            # print(f"Step: {info['steps']}, Action: {action}, Reward: {reward:.2f}, Total Reward: {total_reward:.2f}, Done: {terminated}")
            if terminated or truncated:
                print(f"Episode finished. Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
                # The loop will break on the next check of `done`
                
        # We need to render every frame, even if no action is taken, to see animations
        env._get_observation()

        if env.game_over:
            # Wait for reset key after game over
            wait_for_reset = True
            while wait_for_reset:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        wait_for_reset = False
                        done = True
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                        obs, info = env.reset()
                        total_reward = 0
                        print("Environment reset.")
                        wait_for_reset = False

    env.close()
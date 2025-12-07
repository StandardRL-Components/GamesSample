import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T15:41:32.013441
# Source Brief: brief_01139.md
# Brief Index: 1139
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class Tile:
    """A helper class to represent a tile on the grid."""
    def __init__(self, color_idx, size=2, anim_scale=0.0):
        self.color_idx = color_idx
        self.size = size
        self.anim_scale = anim_scale
        self.is_merging = False

    def start_merge_animation(self):
        """Initiates the visual effect for a tile that will be absorbed in a merge."""
        self.is_merging = True
        self.anim_scale = 1.0 # Will shrink to 0

    def start_grow_animation(self):
        """Initiates the visual effect for a tile that grows after a merge."""
        self.anim_scale = 0.5 # Will grow from half size to full size

    def update_animation(self):
        """Updates the animation scale for the tile over time."""
        if self.is_merging:
            self.anim_scale = max(0.0, self.anim_scale - 0.15)
        else:
            if self.anim_scale < 1.0:
                self.anim_scale = min(1.0, self.anim_scale + 0.15)

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}
    
    game_description = (
        "Place and merge tiles of the same color and size to create larger tiles. "
        "Reach the target tile size of 64 before time runs out!"
    )
    user_guide = (
        "Use the arrow keys (↑↓←→) to move the cursor and press space to place a tile."
    )
    auto_advance = True


    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- CRITICAL: GYMNASIUM SPACES ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- PYGAME & VISUALS SETUP ---
        pygame.init()
        pygame.font.init()
        self.WIDTH, self.HEIGHT = 640, 400
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("Arial", 24, bold=True)
        self.font_score = pygame.font.SysFont("Arial", 32, bold=True)
        self.font_tile = pygame.font.SysFont("Arial", 16, bold=True)

        # --- GAME CONSTANTS ---
        self.GRID_DIM = 7
        self.CELL_SIZE = 50
        self.GRID_WIDTH = self.GRID_DIM * self.CELL_SIZE
        self.GRID_X = 40
        self.GRID_Y = (self.HEIGHT - self.GRID_WIDTH) // 2
        self.FPS = 30
        self.MAX_TIME = 120.0  # seconds
        self.MAX_STEPS = int(self.MAX_TIME * self.FPS) # Safety stop
        self.WIN_TILE_SIZE = 64

        # --- COLORS ---
        self.COLOR_BG = (20, 30, 40)
        self.COLOR_GRID = (50, 60, 70)
        self.COLOR_CURSOR = (255, 255, 0)
        self.COLOR_TEXT = (220, 220, 230)
        self.TILE_COLORS = [
            (255, 80, 80),   # Red
            (80, 255, 80),   # Green
            (80, 120, 255),  # Blue
            (255, 220, 80),  # Yellow
            (255, 120, 255), # Magenta
            (80, 220, 220),  # Cyan
        ]
        self.color_sequence = list(range(4)) # Red, Green, Blue, Yellow cycle

        # --- GAME STATE (initialized in reset) ---
        self.grid = None
        self.cursor_pos = None
        self.next_tile_color_idx = None
        self.score = None
        self.steps = None
        self.time_remaining = None
        self.game_over = None
        self.win_condition_met = None
        self.previous_space_held = None
        
        self.reset()
        # self.validate_implementation() # Optional validation call

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_condition_met = False
        self.time_remaining = self.MAX_TIME
        
        self.cursor_pos = [self.GRID_DIM // 2, self.GRID_DIM // 2]
        random.shuffle(self.color_sequence)
        self.next_tile_color_idx = self.color_sequence[0]
        self.previous_space_held = False

        self._initialize_grid()
        
        return self._get_observation(), self._get_info()

    def _initialize_grid(self):
        self.grid = [[None for _ in range(self.GRID_DIM)] for _ in range(self.GRID_DIM)]
        # Place 3 starting tiles with distinct colors
        available_pos = [(r, c) for r in range(self.GRID_DIM) for c in range(self.GRID_DIM)]
        random.shuffle(available_pos)
        start_colors = random.sample(self.color_sequence, 3)
        for i in range(3):
            r, c = available_pos.pop()
            self.grid[r][c] = Tile(color_idx=start_colors[i], size=2, anim_scale=1.0)


    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # 1. Unpack action and update time
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        self.steps += 1
        self.time_remaining -= 1.0 / self.FPS
        
        reward = 0
        
        # 2. Handle input
        self._handle_movement(movement)
        
        placement_reward = 0
        merge_reward = 0
        if space_held and not self.previous_space_held:
            placed_tile, placement_reward, merge_reward = self._place_tile()
            # sfx: place_tile.wav if placed_tile else place_fail.wav

        self.previous_space_held = space_held
        
        # 3. Update game state (animations) and check for win
        self._update_animations_and_check_win()

        # 4. Calculate rewards
        continuous_reward = self._calculate_continuous_reward()
        total_reward_this_step = placement_reward + merge_reward + continuous_reward
        self.score += total_reward_this_step
        
        # 5. Check for termination
        terminated = self._check_termination()
        if terminated:
            self.game_over = True
            if self.win_condition_met:
                # sfx: win_game.wav
                total_reward_this_step += 100
                self.score += 100
            else: # Timeout or max steps
                # sfx: lose_game.wav
                total_reward_this_step += -50
                self.score += -50
        
        return (
            self._get_observation(),
            total_reward_this_step,
            terminated,
            False,
            self._get_info()
        )

    def _handle_movement(self, movement):
        if movement == 1: # Up
            self.cursor_pos[0] = (self.cursor_pos[0] - 1) % self.GRID_DIM
        elif movement == 2: # Down
            self.cursor_pos[0] = (self.cursor_pos[0] + 1) % self.GRID_DIM
        elif movement == 3: # Left
            self.cursor_pos[1] = (self.cursor_pos[1] - 1) % self.GRID_DIM
        elif movement == 4: # Right
            self.cursor_pos[1] = (self.cursor_pos[1] + 1) % self.GRID_DIM
        # sfx: cursor_move.wav if movement != 0

    def _place_tile(self):
        r, c = self.cursor_pos
        if self.grid[r][c] is None:
            self.grid[r][c] = Tile(self.next_tile_color_idx, size=2)
            
            # Advance to next color
            current_color_seq_idx = self.color_sequence.index(self.next_tile_color_idx)
            self.next_tile_color_idx = self.color_sequence[(current_color_seq_idx + 1) % len(self.color_sequence)]
            
            merge_reward = self._handle_merges((r, c))
            return True, 1.0, merge_reward
        return False, 0.0, 0.0

    def _handle_merges(self, start_pos):
        total_merge_reward = 0
        q = [start_pos]
        
        while q:
            r, c = q.pop(0)
            
            if self.grid[r][c] is None:
                continue

            current_tile = self.grid[r][c]
            
            # Find connected component of same-size, same-color tiles
            component = self._find_connected_component((r, c), current_tile.color_idx, current_tile.size)
            
            if len(component) > 1:
                # sfx: merge.wav
                
                # Calculate reward
                base_size = current_tile.size
                new_size = base_size * 2
                num_merged = len(component)
                
                # Reward for base merge: +10 * 2^(n-1) where new_size = 2^n
                # base_size = 2^(n-1), so reward is 10 * base_size
                base_reward = 10 * base_size
                # Bonus for chain reaction
                bonus_reward = base_reward * (num_merged - 2)
                total_merge_reward += base_reward + bonus_reward
                
                # Perform merge
                # Pick the first tile in the component as the survivor
                survivor_pos = component[0]
                
                # Set others to merge away
                for pos in component:
                    if pos != survivor_pos:
                        self.grid[pos[0]][pos[1]].start_merge_animation()

                # Upgrade survivor tile
                survivor_tile = self.grid[survivor_pos[0]][survivor_pos[1]]
                survivor_tile.size = new_size
                survivor_tile.start_grow_animation()
                
                # Add survivor back to queue to check for chain reactions
                q.append(survivor_pos)
        
        return total_merge_reward

    def _find_connected_component(self, start_pos, color_idx, size):
        component = []
        q = [start_pos]
        visited = {start_pos}
        
        while q:
            r, c = q.pop(0)
            component.append((r, c))
            
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r + dr, c + dc
                
                if 0 <= nr < self.GRID_DIM and 0 <= nc < self.GRID_DIM and (nr, nc) not in visited:
                    neighbor = self.grid[nr][nc]
                    if neighbor and neighbor.color_idx == color_idx and neighbor.size == size:
                        visited.add((nr, nc))
                        q.append((nr, nc))
        return component

    def _update_animations_and_check_win(self):
        new_grid = [[None for _ in range(self.GRID_DIM)] for _ in range(self.GRID_DIM)]
        for r in range(self.GRID_DIM):
            for c in range(self.GRID_DIM):
                tile = self.grid[r][c]
                if tile:
                    tile.update_animation()
                    if tile.is_merging and tile.anim_scale == 0.0:
                        # Animation finished, tile disappears
                        pass
                    else:
                        new_grid[r][c] = tile
                    
                    if tile.size >= self.WIN_TILE_SIZE:
                        self.win_condition_met = True
        self.grid = new_grid

    def _calculate_continuous_reward(self):
        empty_count = sum(row.count(None) for row in self.grid)
        return 0.1 * empty_count

    def _check_termination(self):
        if self.win_condition_met:
            return True
        if self.time_remaining <= 0:
            return True
        if self.steps >= self.MAX_STEPS:
            return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid lines
        for i in range(self.GRID_DIM + 1):
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.GRID_X, self.GRID_Y + i * self.CELL_SIZE), (self.GRID_X + self.GRID_WIDTH, self.GRID_Y + i * self.CELL_SIZE), 2)
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.GRID_X + i * self.CELL_SIZE, self.GRID_Y), (self.GRID_X + i * self.CELL_SIZE, self.GRID_Y + self.GRID_WIDTH), 2)

        # Draw tiles
        for r in range(self.GRID_DIM):
            for c in range(self.GRID_DIM):
                tile = self.grid[r][c]
                if tile:
                    center_x = int(self.GRID_X + c * self.CELL_SIZE + self.CELL_SIZE / 2)
                    center_y = int(self.GRID_Y + r * self.CELL_SIZE + self.CELL_SIZE / 2)
                    
                    # Size of tile grows with its value, capped at cell size
                    base_radius = min(self.CELL_SIZE / 2 - 4, 4 + math.log2(tile.size) * 2.5)
                    radius = int(base_radius * tile.anim_scale)
                    
                    color = self.TILE_COLORS[tile.color_idx % len(self.TILE_COLORS)]
                    
                    if radius > 0:
                        pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, radius, color)
                        pygame.gfxdraw.aacircle(self.screen, center_x, center_y, radius, color)
                    
                    # Draw tile value text
                    if tile.anim_scale > 0.8:
                        text = self.font_tile.render(str(tile.size), True, self.COLOR_BG)
                        text_rect = text.get_rect(center=(center_x, center_y))
                        self.screen.blit(text, text_rect)

        # Draw cursor
        cursor_r, cursor_c = self.cursor_pos
        cursor_rect = pygame.Rect(self.GRID_X + cursor_c * self.CELL_SIZE, self.GRID_Y + cursor_r * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, 3, border_radius=4)


    def _render_ui(self):
        # Score
        score_text = self.font_score.render(f"Score: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.GRID_X, 10))

        # Timer bar
        timer_width = self.WIDTH
        bar_height = 10
        progress = max(0, self.time_remaining / self.MAX_TIME)
        
        if progress < 0.25:
            bar_color = (255, 50, 50) # Red
        elif progress < 0.5:
            bar_color = (255, 180, 50) # Yellow
        else:
            bar_color = (50, 200, 50) # Green
            
        pygame.draw.rect(self.screen, self.COLOR_GRID, (0, 0, timer_width, bar_height))
        pygame.draw.rect(self.screen, bar_color, (0, 0, int(timer_width * progress), bar_height))

        # Next Tile display
        next_tile_x = self.GRID_X + self.GRID_WIDTH + 60
        next_tile_y = self.HEIGHT // 2
        
        title_text = self.font_main.render("Next Tile", True, self.COLOR_TEXT)
        title_rect = title_text.get_rect(center=(next_tile_x, next_tile_y - 80))
        self.screen.blit(title_text, title_rect)
        
        pygame.draw.rect(self.screen, self.COLOR_GRID, (next_tile_x - 50, next_tile_y - 50, 100, 100), 2, border_radius=5)
        
        color = self.TILE_COLORS[self.next_tile_color_idx % len(self.TILE_COLORS)]
        pygame.gfxdraw.filled_circle(self.screen, next_tile_x, next_tile_y, 30, color)
        pygame.gfxdraw.aacircle(self.screen, next_tile_x, next_tile_y, 30, color)
        
        text = self.font_score.render("2", True, self.COLOR_BG)
        text_rect = text.get_rect(center=(next_tile_x, next_tile_y))
        self.screen.blit(text, text_rect)


    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_remaining": self.time_remaining,
            "win": self.win_condition_met
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # Example of how to run the environment
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Setup Pygame window for human play
    # This part requires a display. If you run this in a headless environment,
    # it might fail. The environment itself is headless.
    try:
        pygame.display.init()
        pygame.display.set_caption("Tile Merge Game")
        screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
        human_render = True
    except pygame.error:
        print("Pygame display could not be initialized. Running without visual output for human play.")
        human_render = False

    
    terminated = False
    total_reward = 0
    
    # Game loop
    running = True
    while running:
        # Default action is "do nothing"
        action = [0, 0, 0] # [movement, space, shift]
        
        if human_render:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            # Get keyboard state
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]:
                action[0] = 1
            elif keys[pygame.K_DOWN]:
                action[0] = 2
            elif keys[pygame.K_LEFT]:
                action[0] = 3
            elif keys[pygame.K_RIGHT]:
                action[0] = 4
                
            if keys[pygame.K_SPACE]:
                action[1] = 1
            
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
                action[2] = 1
        else: # If no display, just run with random actions
            action = env.action_space.sample()
            if terminated:
                running = False


        if not terminated:
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if terminated:
                print(f"Game Over! Final Score: {info['score']:.2f}, Win: {info['win']}")
        
        # Render the observation from the environment to the screen
        if human_render:
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
        
        env.clock.tick(env.FPS)

    env.close()
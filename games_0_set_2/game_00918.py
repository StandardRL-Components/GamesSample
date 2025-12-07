
# Generated: 2025-08-27T15:12:29.812404
# Source Brief: brief_00918.md
# Brief Index: 918

        
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
        "Controls: Use arrows to move the cursor. Press Space to select/deselect a crystal. "
        "When a crystal is selected, use arrows to move it."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "An isometric puzzle game. Move crystals to illuminate all cavern paths within a limited number of moves."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_WIDTH, GRID_HEIGHT = 20, 14
    TILE_W_HALF, TILE_H_HALF = 24, 12
    
    # Colors
    COLOR_BG = (25, 28, 36)
    COLOR_PATH_UNLIT = (60, 65, 80)
    COLOR_PATH_LIT = (255, 255, 255)
    COLOR_GRID_LINE = (40, 44, 55)
    CRYSTAL_COLORS = [
        (0, 200, 255),  # Cyan
        (255, 80, 120),  # Pink
        (100, 255, 100), # Green
        (255, 180, 0),   # Yellow
        (200, 100, 255)  # Purple
    ]
    COLOR_CURSOR = (255, 255, 0)
    COLOR_TEXT = (220, 220, 220)
    COLOR_TEXT_SHADOW = (10, 10, 10)

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
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 36)
        
        # Game state variables are initialized in reset()
        self.grid = None
        self.crystals = None
        self.path_illumination = None
        self.total_path_tiles = 0
        self.moves_left = 0
        self.cursor_idx = 0
        self.selected_crystal_idx = None
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_state = False
        self.last_action_feedback = {}

        # Center the grid
        self.origin_x = self.SCREEN_WIDTH // 2
        self.origin_y = self.SCREEN_HEIGHT // 2 - (self.GRID_HEIGHT * self.TILE_H_HALF) // 2 + 20

        # Initialize state
        self.reset()

        # Validate implementation
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_state = False
        self.moves_left = 50 
        self.last_action_feedback = {}

        self._generate_level()
        self._place_crystals(num_crystals=3)
        
        self.cursor_idx = 0
        self.selected_crystal_idx = None
        
        self._update_illumination()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]
        space_press = action[1] == 1
        
        reward = 0
        self.steps += 1
        self.last_action_feedback.clear()

        prev_illuminated_count = len(self.path_illumination)

        action_taken = self._handle_input(movement, space_press)

        if action_taken:
            self.moves_left -= 1
            reward -= 0.1 # Cost of moving
            # Sound: Player move sfx

            self._update_illumination()
            newly_illuminated = len(self.path_illumination) - prev_illuminated_count
            if newly_illuminated > 0:
                reward += newly_illuminated * 1.0
                # Sound: Path illuminated sfx

        self.score += reward
        terminated = self._check_termination()
        
        if terminated:
            if self.win_state:
                reward += 100
                # Sound: Win jingle
            else:
                reward -= 50
                # Sound: Lose sad tone
            self.score += reward

        return self._get_observation(), reward, terminated, False, self._get_info()
    
    def _handle_input(self, movement, space_press):
        """Processes player input and updates game state. Returns True if a move was consumed."""
        if self.selected_crystal_idx is not None:
            # --- MOVEMENT PHASE ---
            if space_press:
                self.selected_crystal_idx = None
                self.last_action_feedback['text'] = 'Crystal Deselected'
                self.last_action_feedback['pos'] = self._iso_to_screen(*self.crystals[self.cursor_idx]['pos'])
                # Sound: Deselect sfx
                return False # Does not consume a move
            
            if movement != 0:
                return self._move_crystal(self.selected_crystal_idx, movement)
        else:
            # --- SELECTION PHASE ---
            if space_press:
                self.selected_crystal_idx = self.cursor_idx
                self.last_action_feedback['text'] = 'Crystal Selected!'
                self.last_action_feedback['pos'] = self._iso_to_screen(*self.crystals[self.cursor_idx]['pos'])
                # Sound: Select sfx
                return False # Does not consume a move

            if movement != 0:
                self._move_cursor(movement)
                return False # Does not consume a move
        return False
        
    def _move_cursor(self, movement):
        num_crystals = len(self.crystals)
        if movement in [1, 4]: # Up or Right
            self.cursor_idx = (self.cursor_idx + 1) % num_crystals
        elif movement in [2, 3]: # Down or Left
            self.cursor_idx = (self.cursor_idx - 1 + num_crystals) % num_crystals
        # Sound: UI blip sfx

    def _move_crystal(self, crystal_idx, movement):
        crystal = self.crystals[crystal_idx]
        x, y = crystal['pos']
        dx, dy = 0, 0
        if movement == 1: dy = -1 # Up
        elif movement == 2: dy = 1 # Down
        elif movement == 3: dx = -1 # Left
        elif movement == 4: dx = 1 # Right
        
        nx, ny = x + dx, y + dy
        
        # Boundary check
        if not (0 <= nx < self.GRID_WIDTH and 0 <= ny < self.GRID_HEIGHT):
            return False
        # Path check
        if self.grid[ny, nx] == 0:
            return False
        # Collision with other crystals
        for i, other_crystal in enumerate(self.crystals):
            if i != crystal_idx and other_crystal['pos'] == [nx, ny]:
                return False
        
        crystal['pos'] = [nx, ny]
        return True

    def _check_termination(self):
        if self.game_over:
            return True
        
        # Win condition
        if len(self.path_illumination) == self.total_path_tiles:
            self.game_over = True
            self.win_state = True
            return True
            
        # Loss condition
        if self.moves_left <= 0:
            self.game_over = True
            self.win_state = False
            return True
        
        # Step limit
        if self.steps >= 1000:
            self.game_over = True
            self.win_state = False
            return True
            
        return False

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
            "moves_left": self.moves_left,
            "illumination_pct": (len(self.path_illumination) / self.total_path_tiles) * 100 if self.total_path_tiles > 0 else 0,
            "win": self.win_state
        }

    def _iso_to_screen(self, x, y):
        screen_x = self.origin_x + (x - y) * self.TILE_W_HALF
        screen_y = self.origin_y + (x + y) * self.TILE_H_HALF
        return int(screen_x), int(screen_y)

    def _generate_level(self):
        self.grid = np.zeros((self.GRID_HEIGHT, self.GRID_WIDTH), dtype=int)
        visited = np.zeros_like(self.grid, dtype=bool)

        # Randomized DFS for maze generation
        def is_valid(x, y):
            return 0 <= y < self.GRID_HEIGHT and 0 <= x < self.GRID_WIDTH

        stack = [(self.np_random.integers(0, self.GRID_WIDTH), self.np_random.integers(0, self.GRID_HEIGHT))]
        
        while stack:
            x, y = stack.pop()
            if visited[y, x]:
                continue
            visited[y, x] = True
            self.grid[y, x] = 1

            neighbors = [(x, y - 1), (x, y + 1), (x - 1, y), (x + 1, y)]
            self.np_random.shuffle(neighbors)

            path_created = False
            for nx, ny in neighbors:
                if is_valid(nx, ny) and not visited[ny, nx]:
                    stack.append((nx, ny))
                    if not path_created:
                        self.grid[ny, nx] = 1
                        visited[ny, nx] = True
                        stack.append((nx, ny))
                        path_created = True
        
        # Ensure a minimum number of path tiles
        self.total_path_tiles = np.sum(self.grid)
        if self.total_path_tiles < (self.GRID_WIDTH * self.GRID_HEIGHT * 0.3):
            self._generate_level() # Retry if maze is too small

    def _place_crystals(self, num_crystals):
        self.crystals = []
        path_coords = np.argwhere(self.grid == 1)
        self.np_random.shuffle(path_coords)
        
        for i in range(min(num_crystals, len(path_coords))):
            y, x = path_coords[i]
            self.crystals.append({
                'pos': [x, y],
                'color': self.CRYSTAL_COLORS[i % len(self.CRYSTAL_COLORS)]
            })

    def _update_illumination(self):
        self.path_illumination = set()
        for crystal in self.crystals:
            cx, cy = crystal['pos']
            self.path_illumination.add(tuple(crystal['pos']))
            # Illuminate adjacent path tiles
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < self.GRID_WIDTH and 0 <= ny < self.GRID_HEIGHT and self.grid[ny, nx] == 1:
                    self.path_illumination.add((nx, ny))

    def _render_game(self):
        # Draw grid and paths
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                if self.grid[y, x] == 1:
                    is_lit = (x, y) in self.path_illumination
                    self._draw_iso_tile(x, y, self.COLOR_PATH_LIT if is_lit else self.COLOR_PATH_UNLIT, self.COLOR_GRID_LINE)

        # Draw crystals and cursor
        for i, crystal in enumerate(self.crystals):
            is_selected = (self.selected_crystal_idx == i)
            is_cursor = (self.cursor_idx == i)
            self._draw_crystal(crystal['pos'], crystal['color'], is_selected, is_cursor)

    def _draw_iso_tile(self, x, y, fill_color, border_color):
        sx, sy = self._iso_to_screen(x, y)
        points = [
            (sx, sy - self.TILE_H_HALF),
            (sx + self.TILE_W_HALF, sy),
            (sx, sy + self.TILE_H_HALF),
            (sx - self.TILE_W_HALF, sy)
        ]
        pygame.gfxdraw.filled_polygon(self.screen, points, fill_color)
        pygame.gfxdraw.aapolygon(self.screen, points, border_color)

    def _draw_crystal(self, pos, color, is_selected, is_cursor):
        sx, sy = self._iso_to_screen(*pos)
        base_y = sy - self.TILE_H_HALF
        
        # Glow effect
        glow_radius = int(self.TILE_W_HALF * 1.5)
        glow_color = color
        temp_surface = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(temp_surface, (*glow_color, 40), (glow_radius, glow_radius), glow_radius)
        pygame.draw.circle(temp_surface, (*glow_color, 60), (glow_radius, glow_radius), int(glow_radius * 0.7))
        self.screen.blit(temp_surface, (sx - glow_radius, base_y - glow_radius))

        # Crystal body
        crystal_height = self.TILE_H_HALF * 2.5
        points = [
            (sx, base_y - crystal_height),
            (sx + self.TILE_W_HALF * 0.6, base_y - crystal_height * 0.4),
            (sx, base_y),
            (sx - self.TILE_W_HALF * 0.6, base_y - crystal_height * 0.4)
        ]
        pygame.gfxdraw.filled_polygon(self.screen, points, color)
        pygame.gfxdraw.aapolygon(self.screen, points, (255, 255, 255))
        
        # Cursor/Selection indicator
        if is_selected:
            pygame.gfxdraw.aacircle(self.screen, sx, base_y + 5, int(self.TILE_W_HALF * 0.8), self.COLOR_CURSOR)
            pygame.gfxdraw.aacircle(self.screen, sx, base_y + 5, int(self.TILE_W_HALF * 0.8) - 1, self.COLOR_CURSOR)
        elif is_cursor and self.selected_crystal_idx is None:
            points = [
                (sx - 12, base_y + 10), (sx + 12, base_y + 10), (sx, base_y + 18)
            ]
            pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_CURSOR)

    def _render_ui(self):
        # --- Render Text Helper ---
        def draw_text(text, font, color, pos, shadow=True):
            if shadow:
                text_surf_shadow = font.render(text, True, self.COLOR_TEXT_SHADOW)
                self.screen.blit(text_surf_shadow, (pos[0] + 2, pos[1] + 2))
            text_surf = font.render(text, True, color)
            self.screen.blit(text_surf, pos)

        # --- Moves Left ---
        moves_text = f"Moves Left: {self.moves_left}"
        draw_text(moves_text, self.font_large, self.COLOR_TEXT, (10, 10))

        # --- Illumination Percentage ---
        illum_pct = (len(self.path_illumination) / self.total_path_tiles) * 100 if self.total_path_tiles > 0 else 0
        illum_text = f"Illuminated: {illum_pct:.1f}%"
        text_width = self.font_large.size(illum_text)[0]
        draw_text(illum_text, self.font_large, self.COLOR_TEXT, (self.SCREEN_WIDTH - text_width - 10, 10))

        # --- Game Over Message ---
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            end_text = "YOU WIN!" if self.win_state else "OUT OF MOVES"
            end_color = (100, 255, 100) if self.win_state else (255, 100, 100)
            
            text_surf = self.font_large.render(end_text, True, end_color)
            text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2))
            self.screen.blit(text_surf, text_rect)
            
            reset_text = "Resetting..."
            reset_surf = self.font_small.render(reset_text, True, self.COLOR_TEXT)
            reset_rect = reset_surf.get_rect(center=(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2 + 30))
            self.screen.blit(reset_surf, reset_rect)

        # --- Action Feedback ---
        if 'text' in self.last_action_feedback and not self.game_over:
            pos = self.last_action_feedback['pos']
            draw_text(self.last_action_feedback['text'], self.font_small, (255, 255, 255), (pos[0] + 15, pos[1] - 30))
            
    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
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

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Crystal Caverns")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    # Game loop
    while running:
        movement, space, shift = 0, 0, 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1
        
        # For a turn-based game, we only step on a key press event
        action_taken = False
        for event in pygame.event.get(pygame.KEYDOWN):
            if event.key in [pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT, pygame.K_SPACE]:
                action_taken = True
                break
        
        if action_taken:
            action = [movement, space, shift]
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            print(f"Action: {action}, Reward: {reward:.2f}, Total Reward: {total_reward:.2f}, Terminated: {terminated}")
            print(f"Info: {info}")
            
            if terminated:
                print("--- Episode Finished ---")
                # Display final screen for a moment before reset
                screen.blit(pygame.transform.flip(pygame.surfarray.make_surface(obs), False, True), (0, 0))
                pygame.display.flip()
                pygame.time.wait(3000)
                
                obs, info = env.reset()
                total_reward = 0
                print("--- New Episode Started ---")

        # Update the display
        # The observation is already the rendered screen, so we just blit it
        rendered_frame = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(rendered_frame, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Limit frame rate
        
    env.close()
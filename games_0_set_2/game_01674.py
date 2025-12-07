import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import os
import os
import pygame


# Set Pygame to run in a headless mode
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys to move the crystal. Match the color of adjacent tiles to fill them."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "An isometric puzzle game. Move your crystal to a tile to activate it. "
        "Activated tiles will fill themselves and any adjacent tiles of the same color. "
        "Fill the entire board before you run out of moves!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    GRID_SIZE = 5
    MAX_MOVES = 10
    NUM_COLORS = 4
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400

    # --- Colors ---
    COLOR_BG = pygame.Color(30, 30, 40)
    COLOR_GRID_LINE = pygame.Color(60, 60, 70)
    COLOR_PLAYER_OUTLINE = pygame.Color(255, 255, 255)
    COLOR_TEXT = pygame.Color(230, 230, 230)
    
    PALETTE = [
        pygame.Color(255, 80, 80),   # Fiery Red
        pygame.Color(80, 255, 120),  # Bright Green
        pygame.Color(80, 150, 255),  # Sky Blue
        pygame.Color(255, 200, 80),  # Gold Yellow
    ]
    
    # --- Pygame Setup ---
    TILE_WIDTH_HALF = 24
    TILE_HEIGHT_HALF = 12
    ORIGIN_X = SCREEN_WIDTH // 2
    ORIGIN_Y = 140

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        try:
            self.font_main = pygame.font.SysFont("Segoe UI", 24, bold=True)
            self.font_game_over = pygame.font.SysFont("Segoe UI", 48, bold=True)
        except pygame.error:
            self.font_main = pygame.font.SysFont(None, 28, bold=True)
            self.font_game_over = pygame.font.SysFont(None, 52, bold=True)
        
        # Initialize state variables to None
        self.grid_colors = None
        self.grid_filled = None
        self.player_pos = None
        self.moves_left = 0
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.last_filled_coords = []
        
        # Note: reset() is called by the environment wrapper, so no need to call it here.

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.grid_colors = self.np_random.integers(1, self.NUM_COLORS + 1, size=(self.GRID_SIZE, self.GRID_SIZE))
        self.grid_filled = np.full((self.GRID_SIZE, self.GRID_SIZE), False, dtype=bool)
        
        self.player_pos = [self.GRID_SIZE // 2, self.GRID_SIZE // 2]
        
        self.moves_left = self.MAX_MOVES
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.last_filled_coords = []
        
        # Initial fill at the starting position to kickstart the game
        initial_fill_score = self._match_at_pos(self.player_pos)
        self.score += initial_fill_score

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        self.steps += 1
        self.last_filled_coords = []
        
        reward = 0.0
        
        dx, dy = 0, 0
        if movement == 1: dy = -1  # Up
        elif movement == 2: dy = 1   # Down
        elif movement == 3: dx = -1  # Left
        elif movement == 4: dx = 1   # Right

        # Any action other than no-op consumes a move
        if movement != 0:
            self.moves_left -= 1
            new_pos = [self.player_pos[0] + dx, self.player_pos[1] + dy]

            if 0 <= new_pos[0] < self.GRID_SIZE and 0 <= new_pos[1] < self.GRID_SIZE:
                self.player_pos = new_pos
                cells_filled = self._match_at_pos(self.player_pos)
                
                if cells_filled == 0:
                    reward += -0.2 # Penalty for a move that fills nothing
                else:
                    reward += cells_filled # +1 for each cell filled
                    if cells_filled > 2:
                        reward += 5 # Bonus for chain reactions
                self.score += max(0, int(reward)) # Score can't go down from a single move
            else:
                # Invalid move (off the board)
                reward += -0.2
        
        # FIX: Cast the result to a standard Python bool
        terminated = bool(self.moves_left <= 0 or np.all(self.grid_filled))
        truncated = False
        
        if terminated:
            self.game_over = True
            if np.all(self.grid_filled):
                reward += 100 # Victory bonus
                self.score += 100
            else:
                reward += -10 # Loss penalty
                self.score -= 10

        return self._get_observation(), float(reward), terminated, truncated, self._get_info()

    def _match_at_pos(self, pos):
        x, y = pos
        if self.grid_filled[y, x]:
            return 0 # Already filled, no new matches possible from here

        color_to_match = self.grid_colors[y, x]
        newly_filled = set()

        # Flood fill logic
        q = [(x, y)]
        visited = set([(x,y)])
        
        while q:
            cx, cy = q.pop(0)
            
            # This cell is part of the match
            if not self.grid_filled[cy, cx]:
                self.grid_filled[cy, cx] = True
                newly_filled.add((cx, cy))
            
            # Check neighbors
            for dx_n, dy_n in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = cx + dx_n, cy + dy_n
                
                if 0 <= nx < self.GRID_SIZE and 0 <= ny < self.GRID_SIZE and (nx, ny) not in visited:
                    visited.add((nx, ny))
                    if self.grid_colors[ny, nx] == color_to_match:
                        q.append((nx, ny))
        
        self.last_filled_coords = list(newly_filled)
        return len(newly_filled)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        # Pygame returns (width, height, 3), we need (height, width, 3)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_left": self.moves_left,
            "cells_filled": int(np.sum(self.grid_filled)),
        }

    def _grid_to_screen(self, x, y):
        screen_x = self.ORIGIN_X + (x - y) * self.TILE_WIDTH_HALF
        screen_y = self.ORIGIN_Y + (x + y) * self.TILE_HEIGHT_HALF
        return int(screen_x), int(screen_y)

    def _draw_iso_tile(self, surface, center_pos, color, outline_color=None, size_mod=0):
        cx, cy = center_pos
        points = [
            (cx, cy - self.TILE_HEIGHT_HALF - size_mod),
            (cx + self.TILE_WIDTH_HALF + size_mod, cy),
            (cx, cy + self.TILE_HEIGHT_HALF + size_mod),
            (cx - self.TILE_WIDTH_HALF - size_mod, cy),
        ]
        pygame.gfxdraw.filled_polygon(surface, points, color)
        if outline_color:
            pygame.gfxdraw.aapolygon(surface, points, outline_color)

    def _render_game(self):
        for y in range(self.GRID_SIZE):
            for x in range(self.GRID_SIZE):
                screen_pos = self._grid_to_screen(x, y)
                color_idx = self.grid_colors[y, x] - 1
                
                self._draw_iso_tile(self.screen, screen_pos, self.COLOR_GRID_LINE)
                
                if not self.grid_filled[y, x]:
                    target_color = self.PALETTE[color_idx].lerp(self.COLOR_BG, 0.6)
                    self._draw_iso_tile(self.screen, screen_pos, target_color, size_mod=-4)
                else:
                    if (x, y) in self.last_filled_coords:
                        glow_color = self.PALETTE[color_idx].lerp((255,255,255), 0.7)
                        self._draw_iso_tile(self.screen, screen_pos, glow_color, size_mod=4)

                    fill_color = self.PALETTE[color_idx]
                    self._draw_iso_tile(self.screen, screen_pos, fill_color, size_mod=-2)
        
        player_screen_pos = self._grid_to_screen(self.player_pos[0], self.player_pos[1])
        player_color_idx = self.grid_colors[self.player_pos[1], self.player_pos[0]] - 1
        player_color = self.PALETTE[player_color_idx]
        
        shadow_pos = (player_screen_pos[0], player_screen_pos[1] + 10)
        pygame.gfxdraw.filled_ellipse(self.screen, shadow_pos[0], shadow_pos[1], 15, 8, (0,0,0,100))

        pygame.gfxdraw.filled_circle(self.screen, player_screen_pos[0], player_screen_pos[1], 16, player_color)
        pygame.gfxdraw.aacircle(self.screen, player_screen_pos[0], player_screen_pos[1], 16, self.COLOR_PLAYER_OUTLINE)
        
        highlight_pos = (player_screen_pos[0] - 5, player_screen_pos[1] - 5)
        pygame.gfxdraw.filled_circle(self.screen, highlight_pos[0], highlight_pos[1], 5, (255,255,255,150))

    def _render_ui(self):
        ui_rect = pygame.Rect(10, 10, 200, 85)
        pygame.draw.rect(self.screen, (0,0,0,150), ui_rect, border_radius=10)
        
        moves_text = self.font_main.render(f"Moves: {self.moves_left}", True, self.COLOR_TEXT)
        self.screen.blit(moves_text, (25, 20))
        
        score_text = self.font_main.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (25, 50))

        if self.game_over:
            s = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            s.fill((0,0,0,180))
            self.screen.blit(s, (0,0))
            
            msg, color = ("Board Cleared!", self.PALETTE[3]) if np.all(self.grid_filled) else ("Out of Moves!", self.PALETTE[0])
            
            end_text = self.font_game_over.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def close(self):
        pygame.font.quit()
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to run the environment directly for human playtesting
    # It will not be run by the evaluation system.
    
    # Un-set the dummy video driver to allow a window to be created
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
        
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    pygame.display.set_caption("Crystal Grid")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    
    running = True
    while running:
        action = [0, 0, 0] # Default action: no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key in [pygame.K_q, pygame.K_ESCAPE]:
                    running = False
                elif event.key == pygame.K_r:
                    obs, info = env.reset()
                elif not env.game_over:
                    if event.key == pygame.K_UP: action[0] = 1
                    elif event.key == pygame.K_DOWN: action[0] = 2
                    elif event.key == pygame.K_LEFT: action[0] = 3
                    elif event.key == pygame.K_RIGHT: action[0] = 4
        
        # Only step if a movement key was pressed
        if action[0] != 0:
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Action: {action}, Reward: {reward}, Terminated: {terminated}, Info: {info}")
        
        # Get the current rendered frame from the environment
        frame = env._get_observation()
        # Pygame needs the axes swapped back to (width, height, 3)
        surf = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(30)

    env.close()
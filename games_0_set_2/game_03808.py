
# Generated: 2025-08-28T00:30:05.265609
# Source Brief: brief_03808.md
# Brief Index: 3808

        
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

    user_guide = (
        "Controls: Use arrow keys to move the cursor. Press space to select a tile."
    )

    game_description = (
        "A grid-based matching pairs puzzle. Match all pairs before you run out of moves. "
        "Risky matches give more points!"
    )

    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_COLS = 5
    GRID_ROWS = 4
    GRID_MARGIN = 40
    TILE_GAP = 10
    UI_HEIGHT = 60
    
    TOTAL_PAIRS = (GRID_COLS * GRID_ROWS) // 2
    RISKY_PAIRS_COUNT = TOTAL_PAIRS // 2
    SAFE_PAIRS_COUNT = TOTAL_PAIRS - RISKY_PAIRS_COUNT
    INITIAL_MOVES = 35
    MAX_STEPS = 1000

    # --- Colors ---
    COLOR_BG = (25, 35, 45)
    COLOR_UI_BG = (20, 30, 40)
    COLOR_GRID_BG = (40, 50, 60)
    COLOR_TILE_HIDDEN = (60, 75, 90)
    COLOR_TILE_BORDER = (80, 95, 110)
    COLOR_CURSOR = (255, 200, 0)
    COLOR_TEXT = (220, 220, 220)
    COLOR_TEXT_VALUE = (255, 255, 255)
    
    # --- Pair Colors and Shapes ---
    PAIR_DEFINITIONS = [
        # Safe Pairs (muted colors, simple shapes)
        {'color': (70, 130, 180), 'shape': 'circle', 'risky': False},   # Steel Blue
        {'color': (60, 179, 113), 'shape': 'square', 'risky': False},   # Medium Sea Green
        {'color': (218, 165, 32), 'shape': 'triangle_up', 'risky': False}, # Goldenrod
        {'color': (186, 85, 211), 'shape': 'diamond', 'risky': False},  # Medium Orchid
        {'color': (240, 128, 128), 'shape': 'triangle_down', 'risky': False},# Light Coral
        
        # Risky Pairs (bright colors, complex shapes)
        {'color': (255, 20, 147), 'shape': 'star', 'risky': True},      # Deep Pink
        {'color': (0, 255, 255), 'shape': 'hex', 'risky': True},       # Cyan
        {'color': (255, 165, 0), 'shape': 'plus', 'risky': True},      # Orange
        {'color': (123, 104, 238), 'shape': 'cross', 'risky': True},     # Medium Slate Blue
        {'color': (50, 205, 50), 'shape': 'heart', 'risky': True}       # Lime Green
    ]

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
        self.font_main = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 18)

        # Calculate tile dimensions based on grid layout
        grid_render_width = self.SCREEN_WIDTH - 2 * self.GRID_MARGIN
        grid_render_height = self.SCREEN_HEIGHT - self.UI_HEIGHT - self.GRID_MARGIN
        self.tile_width = (grid_render_width - (self.GRID_COLS - 1) * self.TILE_GAP) // self.GRID_COLS
        self.tile_height = (grid_render_height - (self.GRID_ROWS - 1) * self.TILE_GAP) // self.GRID_ROWS
        
        self.reset()

        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.moves_left = self.INITIAL_MOVES
        self.pairs_matched = 0
        self.game_over = False

        self.cursor_pos = [0, 0]
        self.selected_tiles = []
        self.mismatch_cooldown = 0
        self.space_was_held = False
        
        self.particles = []

        self._initialize_grid()
        
        return self._get_observation(), self._get_info()

    def _initialize_grid(self):
        tile_values = list(range(self.TOTAL_PAIRS)) * 2
        self.np_random.shuffle(tile_values)
        
        self.grid = []
        for r in range(self.GRID_ROWS):
            row = []
            for c in range(self.GRID_COLS):
                value = tile_values.pop()
                definition = self.PAIR_DEFINITIONS[value]
                tile = {
                    'value': value,
                    'state': 'hidden', # hidden, revealed, matched
                    'risky': definition['risky'],
                    'shape': definition['shape'],
                    'color': definition['color'],
                    'pos': (c, r)
                }
                row.append(tile)
            self.grid.append(row)

    def step(self, action):
        reward = 0
        
        # --- Handle Mismatch Cooldown ---
        # On the step *after* a mismatch, flip tiles back
        if self.mismatch_cooldown > 0:
            self.mismatch_cooldown -= 1
            if self.mismatch_cooldown == 0:
                for pos in self.selected_tiles:
                    self.grid[pos[1]][pos[0]]['state'] = 'hidden'
                self.selected_tiles = []

        # --- Process Actions ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # 1. Handle cursor movement
        if movement == 1 and self.cursor_pos[1] > 0: self.cursor_pos[1] -= 1  # Up
        if movement == 2 and self.cursor_pos[1] < self.GRID_ROWS - 1: self.cursor_pos[1] += 1  # Down
        if movement == 3 and self.cursor_pos[0] > 0: self.cursor_pos[0] -= 1  # Left
        if movement == 4 and self.cursor_pos[0] < self.GRID_COLS - 1: self.cursor_pos[0] += 1  # Right
        
        # 2. Handle tile selection
        is_space_press = space_held and not self.space_was_held
        if is_space_press and self.mismatch_cooldown == 0 and len(self.selected_tiles) < 2:
            tile = self.grid[self.cursor_pos[1]][self.cursor_pos[0]]
            if tile['state'] == 'hidden':
                tile['state'] = 'revealed'
                self.selected_tiles.append(self.cursor_pos.copy())
                self.moves_left -= 1
                
                # Reward for revealing a tile
                reward += 1.0 if tile['risky'] else -0.2
                # sfx: tile_reveal.wav

        self.space_was_held = space_held

        # 3. Check for a match
        if len(self.selected_tiles) == 2:
            pos1, pos2 = self.selected_tiles
            tile1 = self.grid[pos1[1]][pos1[0]]
            tile2 = self.grid[pos2[1]][pos2[0]]

            if tile1['value'] == tile2['value']:
                # --- MATCH ---
                tile1['state'] = 'matched'
                tile2['state'] = 'matched'
                self.pairs_matched += 1
                
                # Reward for successful match
                reward += 5.0 if tile1['risky'] else 2.0
                self.score += 100 if tile1['risky'] else 50
                
                self.selected_tiles = []
                self._create_particles(pos1, tile1['color'])
                self._create_particles(pos2, tile2['color'])
                # sfx: match_success.wav
            else:
                # --- MISMATCH ---
                self.mismatch_cooldown = 1 # Keep revealed for this frame
                reward += -1.0
                self.score -= 10
                # sfx: mismatch_fail.wav

        # --- Update Game State ---
        self.steps += 1
        self._update_particles()
        
        terminated = self._check_termination()
        if terminated:
            if self.pairs_matched == self.TOTAL_PAIRS:
                reward += 50.0  # Win bonus
                self.score += 500
                # sfx: game_win.wav
            else:
                reward += -50.0 # Lose penalty
                # sfx: game_lose.wav

        self.score = max(0, self.score)

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )
    
    def _check_termination(self):
        if self.game_over: return True
        
        win = self.pairs_matched == self.TOTAL_PAIRS
        lose = self.moves_left <= 0
        timeout = self.steps >= self.MAX_STEPS

        if win or lose or timeout:
            self.game_over = True
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
            "pairs_matched": self.pairs_matched,
        }

    def _render_game(self):
        # Draw grid background
        grid_rect = pygame.Rect(
            self.GRID_MARGIN, self.UI_HEIGHT, 
            self.SCREEN_WIDTH - 2 * self.GRID_MARGIN, 
            self.SCREEN_HEIGHT - self.UI_HEIGHT - self.GRID_MARGIN + self.TILE_GAP
        )
        pygame.draw.rect(self.screen, self.COLOR_GRID_BG, grid_rect, border_radius=10)

        # Draw tiles
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                tile = self.grid[r][c]
                tile_x = self.GRID_MARGIN + c * (self.tile_width + self.TILE_GAP)
                tile_y = self.UI_HEIGHT + r * (self.tile_height + self.TILE_GAP)
                rect = pygame.Rect(tile_x, tile_y, self.tile_width, self.tile_height)

                if tile['state'] == 'hidden':
                    pygame.draw.rect(self.screen, self.COLOR_TILE_HIDDEN, rect, border_radius=5)
                    pygame.draw.rect(self.screen, self.COLOR_TILE_BORDER, rect, width=2, border_radius=5)
                elif tile['state'] == 'revealed':
                    self._draw_tile_face(tile, rect)
                    if tile['risky']:
                        self._draw_glow(rect)
                elif tile['state'] == 'matched':
                    pass # Draw nothing for matched tiles

        # Draw particles
        for p in self.particles:
            p_color = p['color']
            alpha = int(255 * (p['life'] / p['max_life']))
            s = int(p['size'] * (p['life'] / p['max_life']))
            if s > 0:
                rect = pygame.Rect(int(p['pos'][0] - s/2), int(p['pos'][1] - s/2), s, s)
                shape_surf = pygame.Surface(rect.size, pygame.SRCALPHA)
                pygame.draw.rect(shape_surf, (*p_color, alpha), shape_surf.get_rect())
                self.screen.blit(shape_surf, rect)

        # Draw cursor
        cursor_x = self.GRID_MARGIN + self.cursor_pos[0] * (self.tile_width + self.TILE_GAP)
        cursor_y = self.UI_HEIGHT + self.cursor_pos[1] * (self.tile_height + self.TILE_GAP)
        cursor_rect = pygame.Rect(cursor_x - 4, cursor_y - 4, self.tile_width + 8, self.tile_height + 8)
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, width=3, border_radius=8)

    def _draw_tile_face(self, tile, rect):
        pygame.draw.rect(self.screen, self.COLOR_TILE_BORDER, rect, border_radius=5)
        inner_rect = rect.inflate(-4, -4)
        pygame.draw.rect(self.screen, self.COLOR_BG, inner_rect, border_radius=5)
        
        center = rect.center
        size = min(rect.width, rect.height) * 0.4
        color = tile['color']
        
        # Draw symbol based on shape
        if tile['shape'] == 'circle':
            pygame.gfxdraw.aacircle(self.screen, center[0], center[1], int(size), color)
            pygame.gfxdraw.filled_circle(self.screen, center[0], center[1], int(size), color)
        elif tile['shape'] == 'square':
            shape_rect = pygame.Rect(0, 0, size*1.8, size*1.8)
            shape_rect.center = center
            pygame.draw.rect(self.screen, color, shape_rect, border_radius=3)
        elif tile['shape'] == 'triangle_up':
            points = [(center[0], center[1] - size), (center[0] - size, center[1] + size*0.7), (center[0] + size, center[1] + size*0.7)]
            pygame.gfxdraw.aapolygon(self.screen, points, color)
            pygame.gfxdraw.filled_polygon(self.screen, points, color)
        elif tile['shape'] == 'triangle_down':
            points = [(center[0], center[1] + size), (center[0] - size, center[1] - size*0.7), (center[0] + size, center[1] - size*0.7)]
            pygame.gfxdraw.aapolygon(self.screen, points, color)
            pygame.gfxdraw.filled_polygon(self.screen, points, color)
        elif tile['shape'] == 'diamond':
            points = [(center[0], center[1] - size*1.2), (center[0] - size, center[1]), (center[0], center[1] + size*1.2), (center[0] + size, center[1])]
            pygame.gfxdraw.aapolygon(self.screen, points, color)
            pygame.gfxdraw.filled_polygon(self.screen, points, color)
        elif tile['shape'] == 'star':
            self._draw_star(center, 6, size, size*0.4, color)
        elif tile['shape'] == 'hex':
            self._draw_polygon(center, 6, size, color)
        elif tile['shape'] == 'plus':
            pygame.draw.rect(self.screen, color, (center[0] - size*1.2, center[1] - size*0.3, size*2.4, size*0.6), border_radius=2)
            pygame.draw.rect(self.screen, color, (center[0] - size*0.3, center[1] - size*1.2, size*0.6, size*2.4), border_radius=2)
        elif tile['shape'] == 'cross':
            pygame.draw.line(self.screen, color, (center[0] - size, center[1] - size), (center[0] + size, center[1] + size), width=int(size*0.3))
            pygame.draw.line(self.screen, color, (center[0] - size, center[1] + size), (center[0] + size, center[1] - size), width=int(size*0.3))
        elif tile['shape'] == 'heart':
            self._draw_heart(center, size, color)
            
    def _draw_polygon(self, center, n_sides, radius, color):
        points = []
        for i in range(n_sides):
            angle = i * (2 * math.pi / n_sides) - (math.pi / 2)
            x = center[0] + radius * math.cos(angle)
            y = center[1] + radius * math.sin(angle)
            points.append((int(x), int(y)))
        pygame.gfxdraw.aapolygon(self.screen, points, color)
        pygame.gfxdraw.filled_polygon(self.screen, points, color)
    
    def _draw_star(self, center, n_points, outer_r, inner_r, color):
        points = []
        for i in range(n_points * 2):
            angle = i * (math.pi / n_points) - (math.pi / 2)
            r = outer_r if i % 2 == 0 else inner_r
            x = center[0] + r * math.cos(angle)
            y = center[1] + r * math.sin(angle)
            points.append((int(x), int(y)))
        pygame.gfxdraw.aapolygon(self.screen, points, color)
        pygame.gfxdraw.filled_polygon(self.screen, points, color)

    def _draw_heart(self, center, size, color):
        points = []
        for i in range(100):
            t = (2 * math.pi * i) / 100
            x = center[0] + size * 0.8 * (16 * math.sin(t)**3)
            y = center[1] - size * 0.8 * (13 * math.cos(t) - 5 * math.cos(2*t) - 2 * math.cos(3*t) - math.cos(4*t))
            points.append((int(x), int(y)))
        if len(points) > 2:
            pygame.gfxdraw.aapolygon(self.screen, points, color)
            pygame.gfxdraw.filled_polygon(self.screen, points, color)

    def _draw_glow(self, rect):
        glow_alpha = 100 + 60 * math.sin(self.steps * 0.2)
        glow_size = 15 + 5 * math.sin(self.steps * 0.2)
        glow_surf = pygame.Surface((rect.width + glow_size, rect.height + glow_size), pygame.SRCALPHA)
        glow_color = (*self.COLOR_CURSOR, glow_alpha)
        pygame.draw.rect(glow_surf, glow_color, glow_surf.get_rect(), border_radius=12)
        self.screen.blit(glow_surf, (rect.x - glow_size/2, rect.y - glow_size/2), special_flags=pygame.BLEND_RGBA_ADD)

    def _render_ui(self):
        # Draw UI background
        pygame.draw.rect(self.screen, self.COLOR_UI_BG, (0, 0, self.SCREEN_WIDTH, self.UI_HEIGHT))
        pygame.draw.line(self.screen, self.COLOR_GRID_BG, (0, self.UI_HEIGHT-1), (self.SCREEN_WIDTH, self.UI_HEIGHT-1), 2)

        # Score
        score_text = self.font_main.render("SCORE", True, self.COLOR_TEXT)
        score_val = self.font_main.render(f"{self.score:06d}", True, self.COLOR_TEXT_VALUE)
        self.screen.blit(score_text, (self.GRID_MARGIN, 15))
        self.screen.blit(score_val, (self.GRID_MARGIN + 90, 15))

        # Moves Left
        moves_text = self.font_main.render("MOVES", True, self.COLOR_TEXT)
        moves_val = self.font_main.render(f"{self.moves_left:02d}", True, self.COLOR_TEXT_VALUE)
        self.screen.blit(moves_text, (self.SCREEN_WIDTH - self.GRID_MARGIN - 150, 15))
        self.screen.blit(moves_val, (self.SCREEN_WIDTH - self.GRID_MARGIN - 60, 15))

    def _create_particles(self, tile_pos, color):
        tile_center_x = self.GRID_MARGIN + tile_pos[0] * (self.tile_width + self.TILE_GAP) + self.tile_width / 2
        tile_center_y = self.UI_HEIGHT + tile_pos[1] * (self.tile_height + self.TILE_GAP) + self.tile_height / 2

        for _ in range(20):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(2, 5)
            self.particles.append({
                'pos': [tile_center_x, tile_center_y],
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'life': self.np_random.integers(15, 25),
                'max_life': 25,
                'color': color,
                'size': self.np_random.integers(5, 10)
            })

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]
    
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

# To run and visualize the environment (optional, for testing)
if __name__ == '__main__':
    env = GameEnv()
    obs, info = env.reset()
    
    pygame.display.set_caption("Grid Match Puzzle")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    
    done = False
    while not done:
        action = env.action_space.sample() # Start with random actions
        action[0] = 0 # Default to no-op for movement
        action[1] = 0 # Default to no space press
        
        # --- Human Controls ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP: action[0] = 1
                elif event.key == pygame.K_DOWN: action[0] = 2
                elif event.key == pygame.K_LEFT: action[0] = 3
                elif event.key == pygame.K_RIGHT: action[0] = 4
                elif event.key == pygame.K_SPACE: action[1] = 1
                elif event.key == pygame.K_r: # Reset on 'r'
                    obs, info = env.reset()
        
        # If any key is pressed, take an action. Otherwise, do nothing.
        # This simulates the turn-based nature for a human player.
        keys = pygame.key.get_pressed()
        if any(keys):
             # For human play, we want to register the key being held down for space
            if keys[pygame.K_SPACE]:
                action[1] = 1
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated:
                print(f"Game Over! Final Score: {info['score']}")
                obs, info = env.reset() # Auto-reset on game over
        
        # --- Rendering ---
        frame = env._get_observation()
        frame = np.transpose(frame, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        env.clock.tick(30)
        
    pygame.quit()
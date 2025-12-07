
# Generated: 2025-08-27T21:33:01.602542
# Source Brief: brief_02824.md
# Brief Index: 2824

        
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
        "Controls: Use arrow keys to move the cursor. Press Space to select a gem. "
        "Move the cursor to an adjacent gem and press Space again to swap. "
        "Press Shift to deselect."
    )

    game_description = (
        "A vibrant match-3 puzzle game. Swap adjacent gems to create lines of three or more. "
        "Trigger chain reactions to maximize your score and reach 1000 points before you run out of 20 moves!"
    )

    auto_advance = False

    # --- Constants ---
    # Game parameters
    GRID_WIDTH, GRID_HEIGHT = 8, 8
    NUM_GEM_TYPES = 6
    MIN_MATCH_LENGTH = 3
    STARTING_MOVES = 20
    TARGET_SCORE = 1000
    MAX_STEPS = 1000

    # Screen dimensions
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    
    # Visuals
    BOARD_X_OFFSET = (SCREEN_WIDTH - SCREEN_HEIGHT) // 2 + 20
    BOARD_Y_OFFSET = 20
    CELL_SIZE = (SCREEN_HEIGHT - 40) // GRID_HEIGHT
    GEM_SIZE = int(CELL_SIZE * 0.8)
    
    # Colors
    COLOR_BG = (20, 25, 40)
    COLOR_GRID = (40, 50, 80)
    COLOR_CURSOR = (255, 255, 255)
    COLOR_TEXT = (220, 220, 240)
    COLOR_SCORE = (100, 255, 180)
    COLOR_MOVES = (255, 180, 100)
    
    GEM_COLORS = [
        (255, 80, 80),    # Red
        (80, 255, 80),    # Green
        (80, 150, 255),   # Blue
        (255, 255, 80),   # Yellow
        (255, 80, 255),   # Magenta
        (80, 255, 255),   # Cyan
    ]

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.SysFont("Arial", 36, bold=True)
        self.font_small = pygame.font.SysFont("Arial", 24)
        
        self.grid = np.zeros((self.GRID_WIDTH, self.GRID_HEIGHT), dtype=int)
        self.cursor_pos = (0, 0)
        self.selected_gem_pos = None
        self.particles = []
        self.game_over_message = ""
        self.rng = None

        self.reset()
        
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        elif self.rng is None:
            self.rng = np.random.default_rng()

        self.steps = 0
        self.score = 0
        self.moves_left = self.STARTING_MOVES
        self.game_over = False
        self.game_over_message = ""
        
        self.cursor_pos = (self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2)
        self.selected_gem_pos = None
        self.particles = []

        self._generate_initial_board()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            obs = self._get_observation()
            return obs, 0, True, False, self._get_info()

        movement, space_pressed, shift_pressed = action[0], action[1] == 1, action[2] == 1
        self.steps += 1
        reward = 0

        # 1. Handle Cancel Action
        if shift_pressed and self.selected_gem_pos:
            self.selected_gem_pos = None

        # 2. Handle Cursor Movement
        cx, cy = self.cursor_pos
        if movement == 1: cy = (cy - 1 + self.GRID_HEIGHT) % self.GRID_HEIGHT
        elif movement == 2: cy = (cy + 1) % self.GRID_HEIGHT
        elif movement == 3: cx = (cx - 1 + self.GRID_WIDTH) % self.GRID_WIDTH
        elif movement == 4: cx = (cx + 1) % self.GRID_WIDTH
        self.cursor_pos = (cx, cy)

        # 3. Handle Primary Action (Space)
        if space_pressed:
            if not self.selected_gem_pos:
                self.selected_gem_pos = self.cursor_pos
            else:
                p1 = self.selected_gem_pos
                p2 = self.cursor_pos
                if self._are_adjacent(p1, p2):
                    self.moves_left -= 1
                    self._swap_gems(p1, p2)
                    
                    all_matches = self._find_all_matches()
                    if not all_matches:
                        self._swap_gems(p1, p2) # Swap back
                        self.moves_left += 1
                        reward = -1 # Penalty for invalid move
                    else:
                        chain = 0
                        while all_matches:
                            chain += 1
                            reward += self._process_matches(all_matches, chain)
                            all_matches = self._find_all_matches()
                        
                        if not self._has_possible_moves():
                            self._reshuffle()
                
                self.selected_gem_pos = None

        terminated = self._check_termination()
        if terminated and not self.game_over:
            self.game_over = True
            if self.score >= self.TARGET_SCORE:
                reward += 100
                self.game_over_message = "YOU WIN!"
            else:
                reward -= 10
                self.game_over_message = "GAME OVER"
        
        if self.steps >= self.MAX_STEPS:
            terminated = True

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        self._update_and_render_particles()

        if self.game_over:
            self._render_game_over_overlay()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "moves_left": self.moves_left}

    # --- Game Logic Helpers ---
    def _generate_initial_board(self):
        while True:
            for y in range(self.GRID_HEIGHT):
                for x in range(self.GRID_WIDTH):
                    self.grid[x, y] = self.rng.integers(1, self.NUM_GEM_TYPES + 1)
            
            if not self._find_all_matches() and self._has_possible_moves():
                break

    def _are_adjacent(self, p1, p2):
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1]) == 1

    def _swap_gems(self, p1, p2):
        self.grid[p1], self.grid[p2] = self.grid[p2], self.grid[p1]

    def _find_all_matches(self):
        matches = set()
        # Horizontal
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH - self.MIN_MATCH_LENGTH + 1):
                if self.grid[x, y] == 0: continue
                if len(set(self.grid[x:x+self.MIN_MATCH_LENGTH, y])) == 1:
                    for i in range(self.MIN_MATCH_LENGTH): matches.add((x+i, y))
                    # Check for longer matches
                    for i in range(self.MIN_MATCH_LENGTH, self.GRID_WIDTH - x):
                        if self.grid[x+i, y] == self.grid[x,y]: matches.add((x+i, y))
                        else: break
        # Vertical
        for x in range(self.GRID_WIDTH):
            for y in range(self.GRID_HEIGHT - self.MIN_MATCH_LENGTH + 1):
                if self.grid[x, y] == 0: continue
                if len(set(self.grid[x, y:y+self.MIN_MATCH_LENGTH])) == 1:
                    for i in range(self.MIN_MATCH_LENGTH): matches.add((x, y+i))
                    # Check for longer matches
                    for i in range(self.MIN_MATCH_LENGTH, self.GRID_HEIGHT - y):
                        if self.grid[x, y+i] == self.grid[x,y]: matches.add((x, y+i))
                        else: break
        return list(matches)

    def _process_matches(self, matches, chain_level):
        match_groups = self._group_matches(matches)
        reward = 0
        
        for group in match_groups:
            n = len(group)
            if n == 3: reward += 3 # +1 per gem
            elif n == 4: reward += 8 # +2 per gem
            else: reward += n * 3 # +3 per gem
            self.score += n * 10 * chain_level
            
            # Create particles for each gem in the group
            for x, y in group:
                self._create_gem_particles(x, y, self.grid[x, y])

        for x, y in matches:
            self.grid[x, y] = 0 # Mark for removal
        
        self._apply_gravity()
        self._refill_board()
        return reward

    def _group_matches(self, flat_matches):
        # A simple way to group connected matches for scoring
        if not flat_matches: return []
        groups = []
        
        while flat_matches:
            group = set()
            queue = [flat_matches.pop(0)]
            group.add(tuple(queue[0]))
            
            while queue:
                x1, y1 = queue.pop(0)
                
                # Find neighbors in the remaining matches
                remaining_after_iter = []
                for p2 in flat_matches:
                    x2, y2 = p2
                    if abs(x1 - x2) + abs(y1 - y2) == 1:
                        group.add(tuple(p2))
                        queue.append(p2)
                    else:
                        remaining_after_iter.append(p2)
                flat_matches = remaining_after_iter
            groups.append(list(group))
        return groups


    def _apply_gravity(self):
        for x in range(self.GRID_WIDTH):
            empty_row = self.GRID_HEIGHT - 1
            for y in range(self.GRID_HEIGHT - 1, -1, -1):
                if self.grid[x, y] != 0:
                    self.grid[x, empty_row], self.grid[x, y] = self.grid[x, y], self.grid[x, empty_row]
                    empty_row -= 1

    def _refill_board(self):
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                if self.grid[x, y] == 0:
                    self.grid[x, y] = self.rng.integers(1, self.NUM_GEM_TYPES + 1)

    def _has_possible_moves(self):
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                # Try swapping right
                if x < self.GRID_WIDTH - 1:
                    self._swap_gems((x, y), (x + 1, y))
                    if self._find_all_matches():
                        self._swap_gems((x, y), (x + 1, y))
                        return True
                    self._swap_gems((x, y), (x + 1, y))
                # Try swapping down
                if y < self.GRID_HEIGHT - 1:
                    self._swap_gems((x, y), (x, y + 1))
                    if self._find_all_matches():
                        self._swap_gems((x, y), (x, y + 1))
                        return True
                    self._swap_gems((x, y), (x, y + 1))
        return False

    def _reshuffle(self):
        flat_grid = list(self.grid.flatten())
        self.rng.shuffle(flat_grid)
        self.grid = np.array(flat_grid).reshape((self.GRID_WIDTH, self.GRID_HEIGHT))
        if not self._has_possible_moves() or self._find_all_matches():
             self._generate_initial_board() # Failsafe

    def _check_termination(self):
        return self.moves_left <= 0 or self.score >= self.TARGET_SCORE or self.steps >= self.MAX_STEPS

    # --- Rendering Helpers ---
    def _render_game(self):
        # Draw grid lines
        for i in range(self.GRID_WIDTH + 1):
            x = self.BOARD_X_OFFSET + i * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, self.BOARD_Y_OFFSET), (x, self.BOARD_Y_OFFSET + self.GRID_HEIGHT * self.CELL_SIZE))
        for i in range(self.GRID_HEIGHT + 1):
            y = self.BOARD_Y_OFFSET + i * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.BOARD_X_OFFSET, y), (self.BOARD_X_OFFSET + self.GRID_WIDTH * self.CELL_SIZE, y))

        # Draw cursor
        cx, cy = self.cursor_pos
        cursor_rect = pygame.Rect(
            self.BOARD_X_OFFSET + cx * self.CELL_SIZE,
            self.BOARD_Y_OFFSET + cy * self.CELL_SIZE,
            self.CELL_SIZE, self.CELL_SIZE
        )
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, border_radius=8, width=3)

        # Draw selected gem highlight
        if self.selected_gem_pos:
            sx, sy = self.selected_gem_pos
            pulse = (math.sin(self.steps * 0.3) + 1) / 2
            alpha = 100 + int(pulse * 100)
            
            highlight_surface = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
            pygame.draw.rect(highlight_surface, (*self.COLOR_CURSOR, alpha), (0,0, self.CELL_SIZE, self.CELL_SIZE), border_radius=8)
            self.screen.blit(highlight_surface, (self.BOARD_X_OFFSET + sx * self.CELL_SIZE, self.BOARD_Y_OFFSET + sy * self.CELL_SIZE))
        
        # Draw gems
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                gem_type = self.grid[x, y]
                if gem_type > 0:
                    self._draw_gem(x, y, gem_type)

    def _draw_gem(self, x, y, gem_type):
        center_x = self.BOARD_X_OFFSET + x * self.CELL_SIZE + self.CELL_SIZE // 2
        center_y = self.BOARD_Y_OFFSET + y * self.CELL_SIZE + self.CELL_SIZE // 2
        color = self.GEM_COLORS[gem_type - 1]
        radius = self.GEM_SIZE // 2
        
        # Draw a subtle darker version for depth
        dark_color = tuple(c * 0.6 for c in color)

        if gem_type == 1: # Circle
            pygame.gfxdraw.aacircle(self.screen, center_x, center_y, radius, dark_color)
            pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, radius, color)
        elif gem_type == 2: # Square
            rect = pygame.Rect(center_x - radius, center_y - radius, self.GEM_SIZE, self.GEM_SIZE)
            pygame.draw.rect(self.screen, dark_color, rect.move(2, 2), border_radius=4)
            pygame.draw.rect(self.screen, color, rect, border_radius=4)
        elif gem_type == 3: # Triangle
            points = [(center_x, center_y - radius), (center_x - radius, center_y + radius), (center_x + radius, center_y + radius)]
            pygame.gfxdraw.aapolygon(self.screen, points, dark_color)
            pygame.gfxdraw.filled_polygon(self.screen, points, color)
        elif gem_type == 4: # Diamond
            points = [(center_x, center_y - radius), (center_x + radius, center_y), (center_x, center_y + radius), (center_x - radius, center_y)]
            pygame.gfxdraw.aapolygon(self.screen, points, dark_color)
            pygame.gfxdraw.filled_polygon(self.screen, points, color)
        elif gem_type == 5: # Hexagon
            points = [(center_x + radius * math.cos(math.radians(a)), center_y + radius * math.sin(math.radians(a))) for a in range(30, 390, 60)]
            pygame.gfxdraw.aapolygon(self.screen, points, dark_color)
            pygame.gfxdraw.filled_polygon(self.screen, points, color)
        elif gem_type == 6: # Star
            points = []
            for i in range(10):
                r = radius if i % 2 == 0 else radius / 2
                angle = math.radians(i * 36 - 90)
                points.append((center_x + r * math.cos(angle), center_y + r * math.sin(angle)))
            pygame.gfxdraw.aapolygon(self.screen, points, dark_color)
            pygame.gfxdraw.filled_polygon(self.screen, points, color)

    def _render_ui(self):
        # Score
        score_text = self.font_large.render(f"{self.score}", True, self.COLOR_SCORE)
        score_label = self.font_small.render("SCORE", True, self.COLOR_TEXT)
        self.screen.blit(score_label, (20, 20))
        self.screen.blit(score_text, (20, 50))
        
        # Moves
        moves_text = self.font_large.render(f"{self.moves_left}", True, self.COLOR_MOVES)
        moves_label = self.font_small.render("MOVES", True, self.COLOR_TEXT)
        moves_label_rect = moves_label.get_rect(topright=(self.SCREEN_WIDTH - 20, 20))
        moves_text_rect = moves_text.get_rect(topright=(self.SCREEN_WIDTH - 20, 50))
        self.screen.blit(moves_label, moves_label_rect)
        self.screen.blit(moves_text, moves_text_rect)

    def _render_game_over_overlay(self):
        overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        
        text = self.font_large.render(self.game_over_message, True, self.COLOR_TEXT)
        text_rect = text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
        
        self.screen.blit(overlay, (0, 0))
        self.screen.blit(text, text_rect)

    # --- Particle System ---
    def _create_gem_particles(self, grid_x, grid_y, gem_type):
        # sound: gem_match.wav
        center_x = self.BOARD_X_OFFSET + grid_x * self.CELL_SIZE + self.CELL_SIZE // 2
        center_y = self.BOARD_Y_OFFSET + grid_y * self.CELL_SIZE + self.CELL_SIZE // 2
        color = self.GEM_COLORS[gem_type - 1]
        
        for _ in range(20):
            angle = self.rng.random() * 2 * math.pi
            speed = self.rng.random() * 2 + 1
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            size = self.rng.integers(3, 7)
            lifetime = self.rng.integers(20, 40)
            self.particles.append([ [center_x, center_y], vel, size, lifetime, color ])

    def _update_and_render_particles(self):
        for p in self.particles:
            p[0][0] += p[1][0] # pos.x += vel.x
            p[0][1] += p[1][1] # pos.y += vel.y
            p[3] -= 1 # lifetime
            p[2] -= 0.1 # size
            
            if p[2] > 0:
                pos = [int(p[0][0]), int(p[0][1])]
                size = int(p[2])
                
                # Fade out
                alpha = max(0, min(255, int(255 * (p[3] / 40))))
                temp_surf = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
                pygame.draw.rect(temp_surf, (*p[4], alpha), (0,0, size*2, size*2))
                self.screen.blit(temp_surf, (pos[0]-size, pos[1]-size), special_flags=pygame.BLEND_RGBA_ADD)

        self.particles = [p for p in self.particles if p[3] > 0 and p[2] > 0]

    # --- Validation ---
    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # --- Pygame Interactive Loop ---
    interactive_screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Match-3 Gym Environment")
    clock = pygame.time.Clock()
    
    action = env.action_space.sample()
    action.fill(0) # Start with no-op

    running = True
    while running:
        movement = 0
        space = 0
        shift = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                done = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1
        
        action = np.array([movement, space, shift])
        
        # Only step if an action is taken, mimicking auto_advance=False
        if not np.array_equal(action, [0,0,0]) or space == 1:
            if not done:
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                print(f"Action: {action}, Reward: {reward:.2f}, Score: {info['score']}, Moves: {info['moves_left']}, Done: {done}")

        # Draw the observation to the interactive screen
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        interactive_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Limit frame rate for human play

    pygame.quit()
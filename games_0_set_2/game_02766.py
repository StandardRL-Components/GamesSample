
# Generated: 2025-08-27T21:22:04.273691
# Source Brief: brief_02766.md
# Brief Index: 2766

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys to move the cursor. Press Space to select a fruit cluster of 3 or more."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Clear the board by matching clusters of 3 or more fruits in this fast-paced isometric puzzle game. Beat the 60-second timer to win!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    # --- Game Constants ---
    GRID_WIDTH = 8
    GRID_HEIGHT = 10
    NUM_FRUIT_TYPES = 5
    MIN_CLUSTER_SIZE = 3
    
    # --- Timing ---
    FPS = 30
    GAME_DURATION_SECONDS = 60
    MATCH_ANIMATION_FRAMES = 10
    FALL_ANIMATION_FRAMES = 8
    
    # --- Colors ---
    COLOR_BG = (20, 30, 40)
    COLOR_GRID = (40, 60, 80)
    COLOR_CURSOR = (255, 255, 0, 150)
    FRUIT_COLORS = [
        (255, 80, 80),   # Red
        (80, 255, 80),   # Green
        (80, 150, 255),  # Blue
        (255, 150, 50),  # Orange
        (200, 80, 255),  # Purple
    ]
    
    # --- Isometric Projection ---
    TILE_WIDTH_HALF = 24
    TILE_HEIGHT_HALF = 14
    ISO_OFFSET_X = 320
    ISO_OFFSET_Y = 80
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((640, 400))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.SysFont("Arial", 24, bold=True)
        self.font_small = pygame.font.SysFont("Arial", 18)
        
        # Game state variables
        self.grid = None
        self.cursor_pos = None
        self.score = None
        self.time_remaining = None
        self.game_over = None
        self.game_phase = None # 'IDLE', 'MATCH', 'FALL'
        self.animation_timer = None
        self.matched_fruits = None
        self.falling_fruits = None
        self.particles = None
        self.total_fruits_on_board = None
        self.last_space_press = False
        
        self.reset()
        
        # Run self-check
        # self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.score = 0
        self.time_remaining = self.GAME_DURATION_SECONDS * self.FPS
        self.game_over = False
        
        self._populate_grid()
        self.total_fruits_on_board = np.count_nonzero(self.grid)

        self.cursor_pos = (self.GRID_HEIGHT // 2, self.GRID_WIDTH // 2)
        
        self.game_phase = 'IDLE'
        self.animation_timer = 0
        self.matched_fruits = set()
        self.falling_fruits = []
        self.particles = []
        self.last_space_press = False
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = 0
        
        # --- Update Timers and Animations ---
        self.time_remaining -= 1
        if self.animation_timer > 0:
            self.animation_timer -= 1
        
        self._update_particles()
        
        # --- Game State Machine ---
        if self.game_phase == 'IDLE':
            reward += self._handle_player_action(action)
        elif self.game_phase == 'MATCH' and self.animation_timer == 0:
            self._process_matches()
            self.game_phase = 'FALL'
            self.animation_timer = self.FALL_ANIMATION_FRAMES
        elif self.game_phase == 'FALL' and self.animation_timer == 0:
            self._apply_gravity()
            new_matches = self._find_all_matches()
            if new_matches:
                reward += self._start_match_phase(new_matches, is_chain=True)
            else:
                self.game_phase = 'IDLE'
                # Check for win condition after board settles
                if np.count_nonzero(self.grid) == 0:
                    self.game_over = True
                    reward += 100 # Win bonus
        
        # --- Check Termination Conditions ---
        terminated = self.game_over
        if self.time_remaining <= 0 and not self.game_over:
            self.game_over = True
            terminated = True
            reward = -100 # Loss penalty
            
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_player_action(self, action):
        movement, space_val, shift_val = action
        space_pressed = space_val == 1 and not self.last_space_press
        self.last_space_press = (space_val == 1)
        
        reward = 0
        action_taken = False

        # --- Cursor Movement ---
        if movement != 0:
            action_taken = True
            r, c = self.cursor_pos
            if movement == 1: r = max(0, r - 1) # Up
            elif movement == 2: r = min(self.GRID_HEIGHT - 1, r + 1) # Down
            elif movement == 3: c = max(0, c - 1) # Left
            elif movement == 4: c = min(self.GRID_WIDTH - 1, c + 1) # Right
            self.cursor_pos = (r, c)
            
        # --- Selection ---
        if space_pressed:
            action_taken = True
            r, c = self.cursor_pos
            if self.grid[r, c] > 0:
                cluster = self._find_cluster(r, c)
                if len(cluster) >= self.MIN_CLUSTER_SIZE:
                    reward += self._start_match_phase(cluster)
                else:
                    # Invalid selection
                    reward = -0.1
            else:
                # Selected empty space
                reward = -0.1
        
        # --- No-op Penalty ---
        if not action_taken:
            reward = -0.1
            
        return reward

    def _start_match_phase(self, cluster, is_chain=False):
        reward = 0
        self.matched_fruits = cluster
        self.game_phase = 'MATCH'
        self.animation_timer = self.MATCH_ANIMATION_FRAMES
        
        num_cleared = len(cluster)
        reward += num_cleared # +1 per fruit
        if num_cleared >= 4:
            reward += 5 # Bonus for large cluster
        if is_chain:
            reward *= 1.5 # Chain reaction bonus
            
        self.score += reward
        
        # Create particles
        for r, c in cluster:
            fruit_type = self.grid[r,c]
            color = self.FRUIT_COLORS[fruit_type - 1]
            iso_x, iso_y = self._grid_to_iso(r, c)
            for _ in range(10):
                self._spawn_particle(iso_x, iso_y, color)
        
        # SFX: Play match sound
        return reward

    def _process_matches(self):
        for r, c in self.matched_fruits:
            self.grid[r, c] = 0
        self.matched_fruits.clear()
        
        # Calculate falling fruits
        self.falling_fruits = []
        for c in range(self.GRID_WIDTH):
            empty_count = 0
            for r in range(self.GRID_HEIGHT - 1, -1, -1):
                if self.grid[r, c] == 0:
                    empty_count += 1
                elif empty_count > 0:
                    fruit_type = self.grid[r, c]
                    self.falling_fruits.append({
                        "from": (r, c),
                        "to": (r + empty_count, c),
                        "type": fruit_type
                    })
        # SFX: Play whoosh sound

    def _apply_gravity(self):
        new_grid = np.zeros_like(self.grid)
        for c in range(self.GRID_WIDTH):
            write_r = self.GRID_HEIGHT - 1
            for r in range(self.GRID_HEIGHT - 1, -1, -1):
                if self.grid[r, c] != 0:
                    new_grid[write_r, c] = self.grid[r, c]
                    write_r -= 1
        self.grid = new_grid
        self.falling_fruits.clear()

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "time_remaining": self.time_remaining / self.FPS,
            "fruits_cleared": self.total_fruits_on_board - np.count_nonzero(self.grid),
        }

    def _render_game(self):
        # --- Draw Grid ---
        for r in range(self.GRID_HEIGHT + 1):
            p1 = self._grid_to_iso(r, 0, offset=True)
            p2 = self._grid_to_iso(r, self.GRID_WIDTH, offset=True)
            pygame.draw.line(self.screen, self.COLOR_GRID, p1, p2, 1)
        for c in range(self.GRID_WIDTH + 1):
            p1 = self._grid_to_iso(0, c, offset=True)
            p2 = self._grid_to_iso(self.GRID_HEIGHT, c, offset=True)
            pygame.draw.line(self.screen, self.COLOR_GRID, p1, p2, 1)

        # --- Draw Fruits ---
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                fruit_type = self.grid[r, c]
                if fruit_type == 0:
                    continue
                
                iso_x, iso_y = self._grid_to_iso(r, c)
                
                # Check if fruit is part of a fall animation
                is_falling = False
                for fruit in self.falling_fruits:
                    if fruit["from"] == (r, c):
                        progress = 1.0 - (self.animation_timer / self.FALL_ANIMATION_FRAMES)
                        from_x, from_y = self._grid_to_iso(*fruit["from"])
                        to_x, to_y = self._grid_to_iso(*fruit["to"])
                        iso_x = from_x + (to_x - from_x) * progress
                        iso_y = from_y + (to_y - from_y) * progress
                        is_falling = True
                        break
                
                if not is_falling:
                    self._draw_fruit(iso_x, iso_y, fruit_type, (r, c) in self.matched_fruits)

        # Draw falling fruits separately to ensure they are on top
        for fruit in self.falling_fruits:
             progress = 1.0 - (self.animation_timer / self.FALL_ANIMATION_FRAMES)
             from_x, from_y = self._grid_to_iso(*fruit["from"])
             to_x, to_y = self._grid_to_iso(*fruit["to"])
             iso_x = from_x + (to_x - from_x) * progress
             iso_y = from_y + (to_y - from_y) * progress
             self._draw_fruit(iso_x, iso_y, fruit["type"], False)
             
        # --- Draw Particles ---
        self._draw_particles()

        # --- Draw Cursor ---
        if self.game_phase == 'IDLE':
            r, c = self.cursor_pos
            iso_x, iso_y = self._grid_to_iso(r, c, offset=True)
            points = [
                self._grid_to_iso(r, c, offset=True),
                self._grid_to_iso(r + 1, c, offset=True),
                self._grid_to_iso(r + 1, c + 1, offset=True),
                self._grid_to_iso(r, c + 1, offset=True),
            ]
            pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_CURSOR)
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_CURSOR)

    def _draw_fruit(self, x, y, fruit_type, is_matching):
        color = self.FRUIT_COLORS[fruit_type - 1]
        radius = self.TILE_WIDTH_HALF * 0.7
        
        # Animation for matching
        if is_matching:
            progress = self.animation_timer / self.MATCH_ANIMATION_FRAMES
            scale = 1.0 + (1.0 - progress) * 0.5
            alpha = int(255 * (0.5 + 0.5 * math.sin(pygame.time.get_ticks() / 50)))
            radius *= scale
        
        # Base shape
        pygame.gfxdraw.filled_circle(self.screen, int(x), int(y), int(radius), color)
        pygame.gfxdraw.aacircle(self.screen, int(x), int(y), int(radius), color)
        
        # Highlight
        highlight_color = tuple(min(255, c + 60) for c in color)
        pygame.gfxdraw.filled_circle(self.screen, int(x - radius*0.2), int(y - radius*0.3), int(radius*0.3), highlight_color)

    def _render_ui(self):
        # Score
        score_text = self.font_large.render(f"Score: {int(self.score)}", True, (255, 255, 255))
        self.screen.blit(score_text, (10, 10))
        
        # Timer
        time_left = max(0, self.time_remaining / self.FPS)
        time_color = (255, 255, 255) if time_left > 10 else (255, 100, 100)
        time_text = self.font_large.render(f"Time: {time_left:.1f}", True, time_color)
        time_rect = time_text.get_rect(topright=(630, 10))
        self.screen.blit(time_text, time_rect)
        
        # Game Over Message
        if self.game_over:
            overlay = pygame.Surface((640, 400), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            
            cleared_all = np.count_nonzero(self.grid) == 0
            msg = "YOU WIN!" if cleared_all else "TIME'S UP!"
            
            end_text = self.font_large.render(msg, True, (255, 255, 0))
            end_rect = end_text.get_rect(center=(320, 200))
            
            overlay.blit(end_text, end_rect)
            self.screen.blit(overlay, (0, 0))

    def _populate_grid(self):
        self.grid = self.np_random.integers(1, self.NUM_FRUIT_TYPES + 1, size=(self.GRID_HEIGHT, self.GRID_WIDTH))
        # Ensure no initial matches
        while True:
            matches = self._find_all_matches()
            if not matches:
                break
            for cluster in matches:
                for r, c in cluster:
                    self.grid[r, c] = self.np_random.integers(1, self.NUM_FRUIT_TYPES + 1)

    def _find_cluster(self, start_r, start_c):
        fruit_type = self.grid[start_r, start_c]
        if fruit_type == 0:
            return set()
            
        q = deque([(start_r, start_c)])
        cluster = set([(start_r, start_c)])
        
        while q:
            r, c = q.popleft()
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.GRID_HEIGHT and 0 <= nc < self.GRID_WIDTH:
                    if (nr, nc) not in cluster and self.grid[nr, nc] == fruit_type:
                        cluster.add((nr, nc))
                        q.append((nr, nc))
        return cluster

    def _find_all_matches(self):
        all_matches = []
        checked = set()
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                if (r, c) not in checked and self.grid[r,c] > 0:
                    cluster = self._find_cluster(r, c)
                    checked.update(cluster)
                    if len(cluster) >= self.MIN_CLUSTER_SIZE:
                        all_matches.append(cluster)
        return all_matches

    def _grid_to_iso(self, r, c, offset=False):
        x = (c - r) * self.TILE_WIDTH_HALF + self.ISO_OFFSET_X
        y = (c + r) * self.TILE_HEIGHT_HALF + self.ISO_OFFSET_Y
        if not offset:
             y += self.TILE_HEIGHT_HALF # Center in cell
        return x, y
        
    def _spawn_particle(self, x, y, color):
        angle = self.np_random.uniform(0, 2 * math.pi)
        speed = self.np_random.uniform(1, 4)
        vx = math.cos(angle) * speed
        vy = math.sin(angle) * speed
        life = self.np_random.integers(15, 30)
        self.particles.append([x, y, vx, vy, life, color])
        
    def _update_particles(self):
        for p in self.particles:
            p[0] += p[2] # x += vx
            p[1] += p[3] # y += vy
            p[4] -= 1    # life -= 1
        self.particles = [p for p in self.particles if p[4] > 0]
        
    def _draw_particles(self):
        for x, y, vx, vy, life, color in self.particles:
            alpha = int(255 * (life / 30.0))
            final_color = (*color, alpha)
            pygame.gfxdraw.filled_circle(self.screen, int(x), int(y), 2, final_color)

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
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
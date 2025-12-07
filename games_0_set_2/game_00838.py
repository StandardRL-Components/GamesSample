# Generated: 2025-08-27T14:57:03.977323
# Source Brief: brief_00838.md
# Brief Index: 838

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrows to move the cursor. Hold Space and press an arrow to swap the selected crystal with an adjacent one."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "An isometric puzzle game. Align 5 or more matching crystals to trigger chain reactions and score points before you run out of moves."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    # --- Constants ---
    GRID_WIDTH = 8
    GRID_HEIGHT = 8
    NUM_COLORS = 5
    MATCH_MIN_LENGTH = 5
    
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    
    MAX_MOVES = 50
    WIN_SCORE = 5000

    TILE_WIDTH = 40
    TILE_HEIGHT = 20
    
    # Colors
    COLOR_BG = (25, 30, 45)
    CRYSTAL_COLORS = {
        1: (255, 80, 80),   # Red
        2: (80, 255, 80),   # Green
        3: (80, 120, 255),  # Blue
        4: (255, 255, 80),  # Yellow
        5: (200, 80, 255),  # Purple
    }
    COLOR_GRID = (40, 50, 70)
    COLOR_CURSOR = (255, 255, 255)
    COLOR_TEXT = (220, 220, 220)
    COLOR_UI_BG = (40, 50, 70, 180)

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
        
        self.font_main = pygame.font.Font(None, 28)
        
        self.grid_origin_x = self.SCREEN_WIDTH // 2
        self.grid_origin_y = 120
        
        # State variables are initialized in reset()
        self.grid = None
        self.cursor_pos = None
        self.score = None
        self.moves_left = None
        self.game_over = None
        self.steps = None
        self.particles = []
        
        self.DIRECTIONS = {
            1: (0, -1), # Up
            2: (0, 1),  # Down
            3: (-1, 0), # Left
            4: (1, 0),  # Right
        }

        # Final check
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.moves_left = self.MAX_MOVES
        self.game_over = False
        self.cursor_pos = (self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2)
        self.particles = []
        
        self._generate_initial_board()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        self.particles = [] # Clear particles from previous step
        
        move_dir, space_pressed, _ = action
        is_swap_action = space_pressed == 1 and move_dir != 0

        step_reward = 0
        
        if is_swap_action:
            self.moves_left -= 1
            
            x1, y1 = self.cursor_pos
            dx, dy = self.DIRECTIONS[move_dir]
            x2 = (x1 + dx) % self.GRID_WIDTH
            y2 = (y1 + dy) % self.GRID_HEIGHT
            
            # Perform swap
            self.grid[y1, x1], self.grid[y2, x2] = self.grid[y2, x2], self.grid[y1, x1]
            # sound: crystal_swap.wav
            
            # Check for matches and resolve chain
            total_chain_reward, match_found = self._resolve_chains()

            if not match_found:
                step_reward = -0.2
                # Revert swap if it was invalid
                self.grid[y1, x1], self.grid[y2, x2] = self.grid[y2, x2], self.grid[y1, x1]
                # sound: invalid_swap.wav
            else:
                step_reward = total_chain_reward
        else:
            # Cursor movement
            if move_dir != 0:
                dx, dy = self.DIRECTIONS[move_dir]
                self.cursor_pos = (
                    np.clip(self.cursor_pos[0] + dx, 0, self.GRID_WIDTH - 1),
                    np.clip(self.cursor_pos[1] + dy, 0, self.GRID_HEIGHT - 1)
                )
                # sound: cursor_move.wav

        terminated = self.moves_left <= 0 or self.score >= self.WIN_SCORE
        if terminated and not self.game_over:
            self.game_over = True
            if self.score >= self.WIN_SCORE:
                step_reward += 100 # Win bonus
                # sound: win_jingle.wav
            else:
                # sound: lose_fanfare.wav
                pass
        
        return (
            self._get_observation(),
            step_reward,
            terminated,
            False,
            self._get_info()
        )

    def _resolve_chains(self):
        total_reward = 0
        chain_multiplier = 1
        any_match_found = False

        while True:
            match_groups = self._find_all_matches()
            if not match_groups:
                break
            
            any_match_found = True
            
            all_matched_coords = set()
            for group in match_groups:
                all_matched_coords.update(group)

            # Calculate reward for this link in the chain
            num_cleared = len(all_matched_coords)
            reward_this_chain = num_cleared * chain_multiplier
            self.score += num_cleared * chain_multiplier

            for group in match_groups:
                bonus = (len(group) - (self.MATCH_MIN_LENGTH - 1)) * 10 * chain_multiplier
                reward_this_chain += bonus
                self.score += bonus
            
            total_reward += reward_this_chain
            
            # Create particles and remove crystals
            for r, c in all_matched_coords:
                self._create_particles(c, r, self.grid[r, c])
                self.grid[r, c] = 0
            # sound: match_clear.wav
            
            # Apply gravity and refill
            self._apply_gravity()
            # sound: crystals_fall.wav
            self._refill_board()
            
            chain_multiplier += 1
        
        return total_reward, any_match_found

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        
        self._render_background()
        self._render_crystals()
        self._render_cursor()
        self._render_particles()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_left": self.moves_left,
        }

    def _cart_to_iso(self, x, y):
        iso_x = self.grid_origin_x + (x - y) * (self.TILE_WIDTH / 2)
        iso_y = self.grid_origin_y + (x + y) * (self.TILE_HEIGHT / 2)
        return int(iso_x), int(iso_y)

    def _render_background(self):
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                sx, sy = self._cart_to_iso(c, r)
                points = [
                    (sx, sy),
                    (sx + self.TILE_WIDTH / 2, sy + self.TILE_HEIGHT / 2),
                    (sx, sy + self.TILE_HEIGHT),
                    (sx - self.TILE_WIDTH / 2, sy + self.TILE_HEIGHT / 2)
                ]
                pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_GRID)

    def _render_crystals(self):
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                crystal_type = self.grid[r, c]
                if crystal_type == 0:
                    continue
                
                sx, sy = self._cart_to_iso(c, r)
                base_color = self.CRYSTAL_COLORS[crystal_type]
                
                # Shimmer effect
                shimmer = (math.sin(self.steps * 0.1 + c * 0.5 + r * 0.3) + 1) / 2
                highlight_color = tuple(min(255, val + 40 * shimmer) for val in base_color)
                
                # Draw crystal body
                points = [
                    (sx, sy + self.TILE_HEIGHT * 0.2),
                    (sx + self.TILE_WIDTH / 2, sy + self.TILE_HEIGHT * 0.7),
                    (sx, sy + self.TILE_HEIGHT * 1.2),
                    (sx - self.TILE_WIDTH / 2, sy + self.TILE_HEIGHT * 0.7)
                ]
                pygame.gfxdraw.filled_polygon(self.screen, points, base_color)
                pygame.gfxdraw.aapolygon(self.screen, points, tuple(min(255, val+20) for val in base_color))
                
                # Draw highlight
                highlight_points = [
                    (sx, sy + self.TILE_HEIGHT * 0.2),
                    (sx + self.TILE_WIDTH / 2, sy + self.TILE_HEIGHT * 0.7),
                    (sx, sy + self.TILE_HEIGHT * 0.5),
                    (sx - self.TILE_WIDTH / 2, sy + self.TILE_HEIGHT * 0.7)
                ]
                pygame.gfxdraw.filled_polygon(self.screen, highlight_points, highlight_color)
                pygame.gfxdraw.aapolygon(self.screen, highlight_points, highlight_color)

    def _render_cursor(self):
        if self.game_over: return
        cx, cy = self.cursor_pos
        sx, sy = self._cart_to_iso(cx, cy)
        
        pulse = (math.sin(self.steps * 0.2) + 1) / 2
        line_width = int(2 + pulse * 2)
        
        points = [
            (sx, sy + self.TILE_HEIGHT * 0.2),
            (sx + self.TILE_WIDTH / 2, sy + self.TILE_HEIGHT * 0.7),
            (sx, sy + self.TILE_HEIGHT * 1.2),
            (sx - self.TILE_WIDTH / 2, sy + self.TILE_HEIGHT * 0.7)
        ]
        
        for i in range(len(points)):
            start = points[i]
            end = points[(i + 1) % len(points)]
            pygame.draw.line(self.screen, self.COLOR_CURSOR, start, end, line_width)
            
    def _render_particles(self):
        # Particles only exist for one frame in auto_advance=False
        for p in self.particles:
            pygame.draw.circle(self.screen, p['color'], p['pos'], int(p['radius']))

    def _render_ui(self):
        # Score
        score_surf = self.font_main.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_surf, (10, 10))
        
        # Moves
        moves_surf = self.font_main.render(f"Moves: {self.moves_left}", True, self.COLOR_TEXT)
        self.screen.blit(moves_surf, (self.SCREEN_WIDTH - moves_surf.get_width() - 10, 10))

        # Game Over
        if self.game_over:
            s = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            s.fill((0,0,0,180))
            self.screen.blit(s, (0,0))
            
            msg = "YOU WIN!" if self.score >= self.WIN_SCORE else "GAME OVER"
            end_surf = pygame.font.Font(None, 60).render(msg, True, self.COLOR_CURSOR)
            self.screen.blit(end_surf, (self.SCREEN_WIDTH/2 - end_surf.get_width()/2, self.SCREEN_HEIGHT/2 - end_surf.get_height()/2 - 20))
            
            final_score_surf = self.font_main.render(f"Final Score: {self.score}", True, self.COLOR_TEXT)
            self.screen.blit(final_score_surf, (self.SCREEN_WIDTH/2 - final_score_surf.get_width()/2, self.SCREEN_HEIGHT/2 + 30))

    def _generate_initial_board(self):
        while True:
            self.grid = self.np_random.integers(1, self.NUM_COLORS + 1, size=(self.GRID_HEIGHT, self.GRID_WIDTH))
            if not self._find_all_matches():
                break

    def _find_all_matches(self):
        all_matched_coords = set()
        
        # Horizontal matches
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                color = self.grid[r, c]
                if color == 0: continue
                match_len = 0
                while c + match_len < self.GRID_WIDTH and self.grid[r, c + match_len] == color:
                    match_len += 1
                if match_len >= self.MATCH_MIN_LENGTH:
                    for i in range(match_len):
                        all_matched_coords.add((r, c + i))
        
        # Vertical matches
        for c in range(self.GRID_WIDTH):
            for r in range(self.GRID_HEIGHT):
                color = self.grid[r, c]
                if color == 0: continue
                match_len = 0
                while r + match_len < self.GRID_HEIGHT and self.grid[r + match_len, c] == color:
                    match_len += 1
                if match_len >= self.MATCH_MIN_LENGTH:
                    for i in range(match_len):
                        all_matched_coords.add((r + i, c))

        return self._get_match_groups(all_matched_coords)

    def _get_match_groups(self, matched_coords):
        if not matched_coords:
            return []
        
        groups = []
        coords_to_visit = set(matched_coords)
        
        while coords_to_visit:
            start_node = coords_to_visit.pop()
            q = [start_node]
            current_group = {start_node}
            
            head = 0
            while head < len(q):
                r, c = q[head]
                head += 1
                
                for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    nr, nc = r + dr, c + dc
                    neighbor = (nr, nc)
                    if neighbor in coords_to_visit:
                        coords_to_visit.remove(neighbor)
                        current_group.add(neighbor)
                        q.append(neighbor)
            groups.append(current_group)
        return groups

    def _apply_gravity(self):
        for c in range(self.GRID_WIDTH):
            empty_row = self.GRID_HEIGHT - 1
            for r in range(self.GRID_HEIGHT - 1, -1, -1):
                if self.grid[r, c] != 0:
                    self.grid[empty_row, c], self.grid[r, c] = self.grid[r, c], self.grid[empty_row, c]
                    empty_row -= 1

    def _refill_board(self):
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                if self.grid[r, c] == 0:
                    self.grid[r, c] = self.np_random.integers(1, self.NUM_COLORS + 1)
    
    def _create_particles(self, c, r, crystal_type):
        sx, sy = self._cart_to_iso(c, r)
        center_y = sy + self.TILE_HEIGHT * 0.7
        base_color = self.CRYSTAL_COLORS[crystal_type]
        
        for _ in range(10): # Number of particles
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 5)
            offset_x = math.cos(angle) * speed * (self.TILE_WIDTH / 4)
            offset_y = math.sin(angle) * speed * (self.TILE_HEIGHT / 4)
            
            self.particles.append({
                'pos': [sx + offset_x, center_y + offset_y],
                'radius': self.np_random.uniform(1, 4),
                'color': base_color
            })

    def close(self):
        pygame.quit()
        super().close()

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Call reset() to initialize state and get the first observation
        obs, info = self.reset()
        
        # Validate the output of reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert obs.dtype == np.uint8
        assert isinstance(info, dict)
        
        # Test step()
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game manually
    env = GameEnv()
    obs, info = env.reset()
    
    running = True
    game_over = False
    
    # Game loop
    while running:
        action = [0, 0, 0] # Default no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and not game_over:
                move_dir = 0
                if event.key == pygame.K_UP:
                    move_dir = 1
                elif event.key == pygame.K_DOWN:
                    move_dir = 2
                elif event.key == pygame.K_LEFT:
                    move_dir = 3
                elif event.key == pygame.K_RIGHT:
                    move_dir = 4
                
                keys = pygame.key.get_pressed()
                space_held = keys[pygame.K_SPACE]
                
                action = [move_dir, 1 if space_held else 0, 0]
                
                # Only step if a key was pressed, since auto_advance is False
                if move_dir != 0:
                    obs, reward, terminated, truncated, info = env.step(action)
                    print(f"Action: {action}, Reward: {reward:.2f}, Score: {info['score']}, Moves: {info['moves_left']}")
                    if terminated:
                        game_over = True
            
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                game_over = False
                print("--- GAME RESET ---")

        # Pygame screen update
        # We need a display for the manual play mode
        if 'display' not in locals():
            display = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
            pygame.display.set_caption("Crystal Caverns")

        # Transpose the observation back to pygame's format (H, W, C) -> (W, H, C)
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(30) # Limit FPS for manual play

    env.close()
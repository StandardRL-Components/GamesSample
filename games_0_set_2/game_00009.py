
# Generated: 2025-08-27T16:21:46.901293
# Source Brief: brief_00009.md
# Brief Index: 9

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
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
        "Controls: Use arrow keys to move the cursor. Press Space to select a gem, then move to an adjacent gem and press Space again to swap. Press Shift to deselect."
    )

    game_description = (
        "A strategic isometric match-3 game. Clear the board by swapping adjacent gems to create lines of 3 or more before you run out of moves. Create cascades for bonus points!"
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_ROWS, self.GRID_COLS = 8, 8
        self.NUM_GEM_TYPES = 6
        self.INITIAL_MOVES = 20
        self.MAX_STEPS = 1000

        # --- Colors ---
        self.COLOR_BG = (15, 25, 40)
        self.COLOR_GRID = (30, 50, 80)
        self.COLOR_CURSOR = (255, 255, 255)
        self.COLOR_SELECTED = (255, 255, 0)
        self.COLOR_TEXT = (220, 220, 240)
        self.GEM_COLORS = [
            (0, 0, 0),  # 0: Empty
            (255, 80, 80),   # 1: Red
            (80, 255, 80),   # 2: Green
            (80, 150, 255),  # 3: Blue
            (255, 150, 50),  # 4: Orange
            (200, 80, 255),  # 5: Purple
            (255, 255, 100), # 6: Yellow
        ]

        # --- Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)

        # --- Isometric Projection Constants ---
        self.tile_width = 48
        self.tile_height = 24
        self.gem_radius = self.tile_height // 2 - 2
        self.grid_offset_x = self.WIDTH // 2
        self.grid_offset_y = self.HEIGHT // 2 - (self.GRID_ROWS * self.tile_height // 3)

        # --- State Variables ---
        self.grid = None
        self.cursor_pos = None
        self.selected_gem_pos = None
        self.moves_left = 0
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.particles = []
        self.last_action_was_press = {'space': False, 'shift': False}
        self.np_random = None

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random = np.random.default_rng(seed)

        self.cursor_pos = [self.GRID_ROWS // 2, self.GRID_COLS // 2]
        self.selected_gem_pos = None
        self.moves_left = self.INITIAL_MOVES
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.particles = []
        self.last_action_was_press = {'space': False, 'shift': False}
        
        self._generate_board()
        while not self._find_all_possible_moves():
            self._generate_board()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        step_reward = 0

        movement, space_pressed, shift_pressed = action[0], action[1] == 1, action[2] == 1

        # --- Handle Input ---
        # Detect rising edge for presses to avoid repeated actions
        is_space_press = space_pressed and not self.last_action_was_press['space']
        is_shift_press = shift_pressed and not self.last_action_was_press['shift']
        self.last_action_was_press['space'] = space_pressed
        self.last_action_was_press['shift'] = shift_pressed
        
        # 1. Handle Deselection (highest priority)
        if is_shift_press:
            self.selected_gem_pos = None
            # sound: deselect_sound

        # 2. Handle Cursor Movement
        if movement == 1: self.cursor_pos[0] -= 1  # Up
        if movement == 2: self.cursor_pos[0] += 1  # Down
        if movement == 3: self.cursor_pos[1] -= 1  # Left
        if movement == 4: self.cursor_pos[1] += 1  # Right
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_ROWS - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_COLS - 1)

        # 3. Handle Selection/Swap
        if is_space_press:
            # sound: select_sound
            if self.selected_gem_pos is None:
                if self.grid[self.cursor_pos[0], self.cursor_pos[1]] != 0:
                    self.selected_gem_pos = list(self.cursor_pos)
            else:
                # Attempt a swap
                r1, c1 = self.selected_gem_pos
                r2, c2 = self.cursor_pos
                
                # Check for adjacency
                if abs(r1 - r2) + abs(c1 - c2) == 1:
                    self.moves_left -= 1
                    # sound: swap_sound
                    
                    # Perform swap
                    self.grid[r1, c1], self.grid[r2, c2] = self.grid[r2, c2], self.grid[r1, c1]
                    
                    match_found, cascade_bonus = self._process_all_matches()

                    if match_found:
                        step_reward += match_found # Base reward for matches
                        step_reward += cascade_bonus # Bonus for cascades
                    else:
                        # No match, swap back
                        self.grid[r1, c1], self.grid[r2, c2] = self.grid[r2, c2], self.grid[r1, c1]
                        step_reward -= 0.1 # Penalty for invalid move
                        # sound: invalid_swap_sound
                    
                    self.selected_gem_pos = None
        
        # --- Update Game State ---
        self._update_particles()
        
        # Check for no more moves after a cascade
        if not self.game_over and not self._find_all_possible_moves():
            self._reshuffle_board()
            # sound: reshuffle_sound

        terminated = self._check_termination()
        if terminated:
            if np.all(self.grid == 0):
                step_reward += 100 # Win bonus
                # sound: win_sound
            else:
                step_reward -= 10 # Loss penalty
                # sound: lose_sound

        if self.steps >= self.MAX_STEPS:
            terminated = True

        return self._get_observation(), step_reward, terminated, False, self._get_info()

    def _generate_board(self):
        self.grid = self.np_random.integers(1, self.NUM_GEM_TYPES + 1, size=(self.GRID_ROWS, self.GRID_COLS))
        # Ensure no initial matches
        while True:
            matches = self._find_matches()
            if not matches:
                break
            for r, c in matches:
                self.grid[r, c] = self.np_random.integers(1, self.NUM_GEM_TYPES + 1)

    def _process_all_matches(self):
        total_matches = 0
        cascade_count = 0
        while True:
            matches = self._find_matches()
            if not matches:
                break
            
            # sound: match_clear_sound
            if cascade_count > 0:
                total_matches += 5 # Cascade bonus
            
            total_matches += len(matches)
            
            for r, c in matches:
                self._create_particles(r, c)
                self.grid[r, c] = 0 # Mark as empty
            
            self._drop_and_fill_gems()
            cascade_count += 1
        
        return total_matches, max(0, (cascade_count-1) * 5)

    def _find_matches(self):
        matches = set()
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                gem = self.grid[r, c]
                if gem == 0: continue
                # Horizontal
                if c < self.GRID_COLS - 2 and self.grid[r, c+1] == gem and self.grid[r, c+2] == gem:
                    matches.update([(r, c), (r, c+1), (r, c+2)])
                # Vertical
                if r < self.GRID_ROWS - 2 and self.grid[r+1, c] == gem and self.grid[r+2, c] == gem:
                    matches.update([(r, c), (r+1, c), (r+2, c)])
        return matches

    def _drop_and_fill_gems(self):
        for c in range(self.GRID_COLS):
            empty_row = self.GRID_ROWS - 1
            for r in range(self.GRID_ROWS - 1, -1, -1):
                if self.grid[r, c] != 0:
                    self.grid[empty_row, c], self.grid[r, c] = self.grid[r, c], self.grid[empty_row, c]
                    empty_row -= 1
        
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                if self.grid[r, c] == 0:
                    self.grid[r, c] = self.np_random.integers(1, self.NUM_GEM_TYPES + 1)

    def _find_all_possible_moves(self):
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                # Swap right
                if c < self.GRID_COLS - 1:
                    self.grid[r, c], self.grid[r, c+1] = self.grid[r, c+1], self.grid[r, c]
                    if self._find_matches():
                        self.grid[r, c], self.grid[r, c+1] = self.grid[r, c+1], self.grid[r, c]
                        return True
                    self.grid[r, c], self.grid[r, c+1] = self.grid[r, c+1], self.grid[r, c]
                # Swap down
                if r < self.GRID_ROWS - 1:
                    self.grid[r, c], self.grid[r+1, c] = self.grid[r+1, c], self.grid[r, c]
                    if self._find_matches():
                        self.grid[r, c], self.grid[r+1, c] = self.grid[r+1, c], self.grid[r, c]
                        return True
                    self.grid[r, c], self.grid[r+1, c] = self.grid[r+1, c], self.grid[r, c]
        return False
    
    def _reshuffle_board(self):
        gems = self.grid[self.grid > 0].flatten()
        self.np_random.shuffle(gems)
        
        new_grid = np.zeros_like(self.grid)
        k = 0
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                if self.grid[r, c] > 0:
                    new_grid[r, c] = gems[k]
                    k += 1
        self.grid = new_grid

        # Ensure reshuffled board is valid
        while self._find_matches() or not self._find_all_possible_moves():
            self._reshuffle_board()


    def _check_termination(self):
        if self.game_over: return True
        if self.moves_left <= 0 or np.all(self.grid == 0):
            self.game_over = True
            return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _iso_to_screen(self, r, c):
        x = self.grid_offset_x + (c - r) * self.tile_width / 2
        y = self.grid_offset_y + (c + r) * self.tile_height / 2
        return int(x), int(y)

    def _render_game(self):
        # Draw grid
        for r in range(self.GRID_ROWS + 1):
            for c in range(self.GRID_COLS + 1):
                p1 = self._iso_to_screen(r, c)
                if r < self.GRID_ROWS:
                    p2 = self._iso_to_screen(r + 1, c)
                    pygame.draw.aaline(self.screen, self.COLOR_GRID, p1, p2)
                if c < self.GRID_COLS:
                    p3 = self._iso_to_screen(r, c + 1)
                    pygame.draw.aaline(self.screen, self.COLOR_GRID, p1, p3)
        
        # Draw gems
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                gem_type = self.grid[r, c]
                if gem_type != 0:
                    pos = self._iso_to_screen(r, c)
                    color = self.GEM_COLORS[gem_type]
                    
                    # Shadow
                    shadow_pos = (pos[0], pos[1] + 3)
                    pygame.gfxdraw.filled_circle(self.screen, shadow_pos[0], shadow_pos[1], self.gem_radius, (0,0,0,80))
                    
                    # Main gem
                    pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.gem_radius, color)
                    pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.gem_radius, color)

                    # Highlight
                    highlight_pos = (pos[0] + 3, pos[1] - 3)
                    pygame.gfxdraw.filled_circle(self.screen, highlight_pos[0], highlight_pos[1], self.gem_radius // 3, (255,255,255,100))

        # Draw selected gem highlight
        if self.selected_gem_pos is not None:
            r, c = self.selected_gem_pos
            pos = self._iso_to_screen(r, c)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.gem_radius + 4, self.COLOR_SELECTED)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.gem_radius + 5, self.COLOR_SELECTED)

        # Draw cursor
        r, c = self.cursor_pos
        pos = self._iso_to_screen(r, c)
        points = [
            self._iso_to_screen(r, c), self._iso_to_screen(r + 1, c),
            self._iso_to_screen(r + 1, c + 1), self._iso_to_screen(r, c + 1)
        ]
        pygame.draw.aalines(self.screen, self.COLOR_CURSOR, True, points, 2)
        
        self._render_particles()

    def _render_ui(self):
        score_text = self.font_main.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        moves_text = self.font_main.render(f"Moves: {self.moves_left}", True, self.COLOR_TEXT)
        self.screen.blit(moves_text, (self.WIDTH - moves_text.get_width() - 10, 10))

        if self.game_over:
            outcome_text_str = "BOARD CLEARED!" if np.all(self.grid == 0) else "GAME OVER"
            outcome_text = self.font_main.render(outcome_text_str, True, self.COLOR_SELECTED)
            text_rect = outcome_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            pygame.draw.rect(self.screen, (0,0,0,150), text_rect.inflate(20, 20))
            self.screen.blit(outcome_text, text_rect)

    def _create_particles(self, r, c):
        pos = self._iso_to_screen(r, c)
        color = self.GEM_COLORS[self.grid[r, c]]
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            life = self.np_random.integers(20, 40)
            self.particles.append([list(pos), vel, life, color])

    def _update_particles(self):
        for p in self.particles:
            p[0][0] += p[1][0]
            p[0][1] += p[1][1]
            p[1][1] += 0.05 # gravity
            p[2] -= 1
        self.particles = [p for p in self.particles if p[2] > 0]

    def _render_particles(self):
        for p in self.particles:
            pos, vel, life, color = p
            alpha = max(0, min(255, int(255 * (life / 30.0))))
            size = int(max(1, life / 8))
            pygame.gfxdraw.filled_circle(self.screen, int(pos[0]), int(pos[1]), size, color + (alpha,))

    def _get_info(self):
        # Update score after all processing for the step is done
        self.score = max(0, self.score)
        return {"score": self.score, "steps": self.steps, "moves_left": self.moves_left}

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Gemstone Grid")
    clock = pygame.time.Clock()

    print(env.user_guide)
    print(env.game_description)

    action = [0, 0, 0] # no-op, released, released
    
    while not done:
        # --- Human Controls ---
        movement = 0
        space_pressed = 0
        shift_pressed = 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space_pressed = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_pressed = 1
        
        action = [movement, space_pressed, shift_pressed]
        
        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        if reward != 0:
            print(f"Step: {info['steps']}, Reward: {reward:.2f}, Score: {info['score']}, Moves Left: {info['moves_left']}")

        # --- Render to Display ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # Since auto_advance is False, we only step on key presses.
        # But for human play, we need a small delay to make it playable.
        clock.tick(30)

    print(f"Game Over! Final Score: {info['score']}")
    pygame.time.wait(2000)
    env.close()
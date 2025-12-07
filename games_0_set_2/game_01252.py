
# Generated: 2025-08-27T16:31:37.703789
# Source Brief: brief_01252.md
# Brief Index: 1252

        
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
        "Controls: Arrow keys to move cursor. Hold an arrow key and press Space to shift a row/column."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Isometric puzzle game. Shift rows and columns of crystals to create matches of 3 or more. Clear the board before time runs out!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    FPS = 30
    
    # Board dimensions
    GRID_WIDTH, GRID_HEIGHT = 8, 7
    
    # Visuals
    TILE_WIDTH_HALF, TILE_HEIGHT_HALF = 28, 14
    CRYSTAL_HEIGHT = 20
    
    # Colors
    COLOR_BG = (25, 20, 35)
    COLOR_GRID = (45, 40, 55)
    COLOR_CURSOR = (255, 255, 100)
    
    CRYSTAL_COLORS = {
        0: None, # Empty
        1: ((255, 50, 50), (220, 20, 20), (180, 0, 0)), # Red
        2: ((50, 255, 50), (20, 220, 20), (0, 180, 0)), # Green
        3: ((80, 80, 255), (50, 50, 220), (20, 20, 180)), # Blue
    }
    
    # Game parameters
    TIME_LIMIT_SECONDS = 60
    MAX_STEPS = 1000 # Hard cap

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        
        # Fonts
        self.font_large = pygame.font.Font(None, 36)
        self.font_medium = pygame.font.Font(None, 28)
        self.font_small = pygame.font.Font(None, 20)
        
        # Board origin for centering
        self.board_offset_x = self.SCREEN_WIDTH // 2
        self.board_offset_y = self.SCREEN_HEIGHT // 2 - (self.GRID_HEIGHT * self.TILE_HEIGHT_HALF) // 2 + 30
        
        # Initialize state variables
        self.board = None
        self.cursor_pos = None
        self.score = 0
        self.steps = 0
        self.time_remaining = 0
        self.game_over = False
        self.space_was_held = False
        self.particles = []
        self.crystals_remaining = 0
        
        self.reset()
        
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_remaining = self.TIME_LIMIT_SECONDS
        self.cursor_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        self.space_was_held = True # Prevent action on first frame
        self.particles = []
        
        self._generate_board()
        self.crystals_remaining = np.count_nonzero(self.board)
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.auto_advance:
            self.clock.tick(self.FPS)

        self.steps += 1
        reward = -0.01 # Time penalty
        
        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_held = action[1] == 1
        shift_held = action[2] == 1 # Unused in this implementation but available
        
        self._handle_input(movement, space_held)
        
        action_taken = space_held and not self.space_was_held and movement != 0
        if action_taken:
            # Sound: Crystal shift sfx
            reward += self._resolve_board()
        
        self._update_particles()
        
        self.time_remaining -= 1.0 / self.FPS
        
        terminated = self._check_termination()
        if terminated:
            if self.crystals_remaining == 0:
                reward += 100 # Win bonus
                # Sound: Win jingle
            elif self.time_remaining <= 0:
                reward -= 50 # Timeout penalty
                # Sound: Loss buzzer
            self.game_over = True
        
        self.space_was_held = space_held
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, movement, space_held):
        # Move cursor
        if movement != 0:
            dx, dy = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}[movement]
            self.cursor_pos[0] = (self.cursor_pos[0] + dx) % self.GRID_WIDTH
            self.cursor_pos[1] = (self.cursor_pos[1] + dy) % self.GRID_HEIGHT
        
        # Shift action is handled in step() to allow reward calculation
        # Here we just detect the action for the main loop
        is_shift_action = space_held and not self.space_was_held and movement != 0
        if is_shift_action:
            cx, cy = self.cursor_pos
            direction = {1: 'up', 2: 'down', 3: 'left', 4: 'right'}[movement]
            
            if direction in ['up', 'down']:
                col = self.board[:, cx].copy()
                shifted_col = np.roll(col, -1 if direction == 'up' else 1)
                self.board[:, cx] = shifted_col
            elif direction in ['left', 'right']:
                row = self.board[cy, :].copy()
                shifted_row = np.roll(row, -1 if direction == 'left' else 1)
                self.board[cy, :] = shifted_row

    def _resolve_board(self):
        total_reward = 0
        is_chain_reaction = False
        
        while True:
            matches = self._find_matches()
            if not matches:
                break
            
            # Sound: Match sfx
            num_matched = len(matches)
            total_reward += num_matched
            if is_chain_reaction:
                total_reward += 5 # Chain reaction bonus
            
            for x, y in matches:
                self._spawn_particles(x, y, self.board[y, x])
                self.board[y, x] = 0 # Mark as empty
            
            self._apply_gravity()
            is_chain_reaction = True

        self.crystals_remaining = np.count_nonzero(self.board)
        return total_reward

    def _find_matches(self):
        matches = set()
        # Horizontal matches
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH - 2):
                color = self.board[y, x]
                if color != 0 and color == self.board[y, x+1] and color == self.board[y, x+2]:
                    match_len = 3
                    while x + match_len < self.GRID_WIDTH and self.board[y, x + match_len] == color:
                        match_len += 1
                    for i in range(match_len):
                        matches.add((x + i, y))
        
        # Vertical matches
        for x in range(self.GRID_WIDTH):
            for y in range(self.GRID_HEIGHT - 2):
                color = self.board[y, x]
                if color != 0 and color == self.board[y+1, x] and color == self.board[y+2, x]:
                    match_len = 3
                    while y + match_len < self.GRID_HEIGHT and self.board[y + match_len, x] == color:
                        match_len += 1
                    for i in range(match_len):
                        matches.add((x, y + i))
        
        return list(matches)

    def _apply_gravity(self):
        for x in range(self.GRID_WIDTH):
            col = self.board[:, x]
            empty_count = np.count_nonzero(col == 0)
            if empty_count > 0:
                non_empty = col[col != 0]
                self.board[:, x] = np.concatenate([np.zeros(empty_count), non_empty])

    def _spawn_particles(self, grid_x, grid_y, color_id):
        px, py = self._iso_to_screen(grid_x, grid_y)
        color = self.CRYSTAL_COLORS[color_id][0]
        for _ in range(10):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifetime = random.randint(15, 30)
            self.particles.append({'pos': [px, py], 'vel': vel, 'lifetime': lifetime, 'color': color})

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['lifetime'] -= 1
        self.particles = [p for p in self.particles if p['lifetime'] > 0]

    def _check_termination(self):
        return (
            self.crystals_remaining == 0
            or self.time_remaining <= 0
            or self.steps >= self.MAX_STEPS
        )

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_remaining": self.time_remaining,
            "crystals_remaining": self.crystals_remaining,
        }

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid lines
        for y in range(self.GRID_HEIGHT + 1):
            start = self._iso_to_screen(0, y)
            end = self._iso_to_screen(self.GRID_WIDTH, y)
            pygame.draw.line(self.screen, self.COLOR_GRID, start, end, 1)
        for x in range(self.GRID_WIDTH + 1):
            start = self._iso_to_screen(x, 0)
            end = self._iso_to_screen(x, self.GRID_HEIGHT)
            pygame.draw.line(self.screen, self.COLOR_GRID, start, end, 1)
            
        # Draw crystals
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                color_id = self.board[y, x]
                if color_id != 0:
                    self._draw_iso_cube(self.screen, x, y, self.CRYSTAL_COLORS[color_id])
        
        # Draw cursor
        self._draw_cursor()
        
        # Draw particles
        for p in self.particles:
            alpha = max(0, 255 * (p['lifetime'] / 30))
            color = (*p['color'], alpha)
            temp_surf = pygame.Surface((4, 4), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (2, 2), 2)
            self.screen.blit(temp_surf, (int(p['pos'][0]-2), int(p['pos'][1]-2)))

    def _render_ui(self):
        # Score
        score_text = self.font_large.render(f"Score: {self.score}", True, (255, 255, 255))
        self.screen.blit(score_text, (10, 10))
        
        # Time
        time_str = f"Time: {max(0, self.time_remaining):.1f}"
        time_color = (255, 100, 100) if self.time_remaining < 10 else (255, 255, 255)
        time_text = self.font_large.render(time_str, True, time_color)
        self.screen.blit(time_text, (self.SCREEN_WIDTH - time_text.get_width() - 10, 10))
        
        # Crystals Remaining
        crystal_text = self.font_medium.render(f"Crystals Left: {self.crystals_remaining}", True, (200, 200, 200))
        self.screen.blit(crystal_text, ((self.SCREEN_WIDTH - crystal_text.get_width()) // 2, self.SCREEN_HEIGHT - 35))

        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            msg = "YOU WIN!" if self.crystals_remaining == 0 else "GAME OVER"
            end_text = self.font_large.render(msg, True, (255, 255, 100))
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def _iso_to_screen(self, x, y):
        screen_x = self.board_offset_x + (x - y) * self.TILE_WIDTH_HALF
        screen_y = self.board_offset_y + (x + y) * self.TILE_HEIGHT_HALF
        return int(screen_x), int(screen_y)

    def _draw_iso_cube(self, surface, grid_x, grid_y, colors):
        top_color, side_color1, side_color2 = colors
        px, py = self._iso_to_screen(grid_x, grid_y)
        
        # Points for the cube
        p_top_center = (px, py - self.CRYSTAL_HEIGHT)
        p_top_left = (px - self.TILE_WIDTH_HALF, py - self.CRYSTAL_HEIGHT + self.TILE_HEIGHT_HALF)
        p_top_right = (px + self.TILE_WIDTH_HALF, py - self.CRYSTAL_HEIGHT + self.TILE_HEIGHT_HALF)
        p_top_front = (px, py - self.CRYSTAL_HEIGHT + 2 * self.TILE_HEIGHT_HALF)

        p_bottom_center = (px, py)
        p_bottom_left = (px - self.TILE_WIDTH_HALF, py + self.TILE_HEIGHT_HALF)
        p_bottom_right = (px + self.TILE_WIDTH_HALF, py + self.TILE_HEIGHT_HALF)
        
        # Draw polygons
        # Top face
        pygame.draw.polygon(surface, top_color, [p_top_center, p_top_right, p_top_front, p_top_left])
        # Left face
        pygame.draw.polygon(surface, side_color1, [p_bottom_center, p_bottom_left, p_top_left, p_top_center])
        # Right face
        pygame.draw.polygon(surface, side_color2, [p_bottom_center, p_bottom_right, p_top_right, p_top_center])

        # Glow effect
        glow_surf = pygame.Surface((self.TILE_WIDTH_HALF * 4, self.TILE_WIDTH_HALF * 4), pygame.SRCALPHA)
        pygame.gfxdraw.filled_circle(glow_surf, self.TILE_WIDTH_HALF * 2, self.TILE_WIDTH_HALF * 2, self.TILE_WIDTH_HALF, (*top_color, 30))
        surface.blit(glow_surf, (p_top_center[0] - self.TILE_WIDTH_HALF * 2, p_top_center[1] - self.TILE_WIDTH_HALF * 2))

    def _draw_cursor(self):
        cx, cy = self.cursor_pos
        px, py = self._iso_to_screen(cx, cy)
        
        points = [
            (px, py - self.CRYSTAL_HEIGHT), # top
            (px + self.TILE_WIDTH_HALF, py - self.CRYSTAL_HEIGHT + self.TILE_HEIGHT_HALF), # right
            (px, py - self.CRYSTAL_HEIGHT + 2 * self.TILE_HEIGHT_HALF), # front
            (px - self.TILE_WIDTH_HALF, py - self.CRYSTAL_HEIGHT + self.TILE_HEIGHT_HALF), # left
        ]
        
        pygame.draw.polygon(self.screen, self.COLOR_CURSOR, points, 3)

    def _generate_board(self):
        while True:
            self.board = self.np_random.integers(1, len(self.CRYSTAL_COLORS), size=(self.GRID_HEIGHT, self.GRID_WIDTH))
            if not self._find_matches():
                break

    def close(self):
        pygame.quit()

    def validate_implementation(self):
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
    # It will not be executed by the evaluation system
    # but is useful for testing and debugging.
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # --- Pygame setup for human play ---
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption(GameEnv.game_description)
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement = 0 # no-op
        space_held = 0
        shift_held = 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1
            
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward}")
            # Wait a bit before resetting
            pygame.time.wait(3000)
            obs, info = env.reset()
            total_reward = 0
            
        clock.tick(GameEnv.FPS)
        
    env.close()

# Generated: 2025-08-28T02:49:09.905905
# Source Brief: brief_04576.md
# Brief Index: 4576

        
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

    user_guide = (
        "Controls: Use arrow keys to move the selector. Hold SPACE and press an arrow key to shift the selected crystal."
    )

    game_description = (
        "Navigate a procedurally generated isometric cavern, strategically shifting crystals to create color matches and clear the board within a limited number of moves."
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.GRID_WIDTH = 8
        self.GRID_HEIGHT = 8
        self.NUM_COLORS = 3
        self.MAX_MOVES = 5
        self.MAX_STEPS = 1000

        # Visual constants
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 640, 400
        self.TILE_WIDTH_ISO = 48
        self.TILE_HEIGHT_ISO = 24
        self.CRYSTAL_HEIGHT = 28
        self.ORIGIN_X = self.SCREEN_WIDTH // 2
        self.ORIGIN_Y = 100

        # Colors
        self.COLOR_BG = (15, 20, 35)
        self.COLOR_GRID = (30, 40, 65)
        self.COLOR_CURSOR = (255, 255, 100)
        self.CRYSTAL_COLORS = [
            ((255, 80, 80), (200, 30, 30)),   # Red (bright, dark)
            ((80, 255, 80), (30, 200, 30)),   # Green
            ((80, 120, 255), (30, 60, 200)),  # Blue
        ]
        self.COLOR_UI_TEXT = (230, 230, 240)
        self.COLOR_UI_SHADOW = (10, 10, 15)

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
        try:
            self.font_large = pygame.font.Font(pygame.font.get_default_font(), 24)
            self.font_small = pygame.font.Font(pygame.font.get_default_font(), 16)
        except IOError:
            self.font_large = pygame.font.SysFont("sans", 24)
            self.font_small = pygame.font.SysFont("sans", 16)
        
        # Initialize state variables
        self.grid = None
        self.cursor_pos = None
        self.moves_left = None
        self.score = None
        self.steps = None
        self.game_over = None
        self.particles = None
        self.last_action_was_shift = False
        self.rng = None

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.rng = np.random.default_rng(seed)

        self.score = 0
        self.steps = 0
        self.moves_left = self.MAX_MOVES
        self.game_over = False
        self.cursor_pos = (self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2)
        self.particles = deque()
        self.last_action_was_shift = False
        
        self._generate_board()

        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0
        self.steps += 1
        self.last_action_was_shift = False

        if not self.game_over:
            if space_held and movement != 0:
                # This is a "shift" action that consumes a move
                self.moves_left -= 1
                self.last_action_was_shift = True
                reward += self._perform_shift(movement)
            elif movement != 0:
                # This is a cursor movement action
                dx, dy = [(0, -1), (0, 1), (-1, 0), (1, 0)][movement - 1]
                self.cursor_pos = (
                    max(0, min(self.GRID_WIDTH - 1, self.cursor_pos[0] + dx)),
                    max(0, min(self.GRID_HEIGHT - 1, self.cursor_pos[1] + dy))
                )

        terminated = self._check_termination()
        if terminated and not self.game_over:
            if self._is_board_clear():
                reward += 100  # Victory bonus
                # sfx: game_win
            elif self.moves_left <= 0:
                reward += -50 # Loss penalty
                # sfx: game_lose
            self.game_over = True
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _perform_shift(self, movement):
        shift_dx, shift_dy = [(0, -1), (0, 1), (-1, 0), (1, 0)][movement - 1]
        start_x, start_y = self.cursor_pos

        if self.grid[start_y][start_x] is None:
            # sfx: error_buzz
            return -0.2 # Penalty for trying to shift an empty space

        # Trace the line of crystals to be pushed
        line = []
        cx, cy = start_x, start_y
        while 0 <= cx < self.GRID_WIDTH and 0 <= cy < self.GRID_HEIGHT:
            if self.grid[cy][cx] is not None:
                line.append(((cx, cy), self.grid[cy][cx]))
            else:
                break
            cx, cy = cx + shift_dx, cy + shift_dy
        
        # Perform the shift with wraparound
        # sfx: crystal_shift
        for i in range(len(line)):
            (x, y), color_idx = line[i]
            nx, ny = (x + shift_dx) % self.GRID_WIDTH, (y + shift_dy) % self.GRID_HEIGHT
            self.grid[ny][nx] = color_idx
        
        # The starting crystal moves into the now-empty space
        self.grid[start_y][start_x] = None
        nx, ny = (start_x + shift_dx) % self.GRID_WIDTH, (start_y + shift_dy) % self.GRID_HEIGHT
        self.grid[ny][nx] = line[0][1]


        # Process matches and cascades
        total_reward = 0
        combo_multiplier = 1
        while True:
            matches = self._find_matches()
            if not matches:
                break
            
            # sfx: match_clear
            cleared_count = len(matches)
            reward_from_match = cleared_count * combo_multiplier
            if cleared_count >= 4:
                reward_from_match += 5 # Combo bonus
                # sfx: combo_bonus

            total_reward += reward_from_match
            self.score += int(reward_from_match * 10) # Update score for UI

            for r, c in matches:
                self._spawn_particles(r, c, self.grid[r][c])
                self.grid[r][c] = None

            self._apply_gravity()
            combo_multiplier += 1
        
        if total_reward == 0:
            return -0.2 # Penalty for a move that creates no match
        
        return total_reward

    def _generate_board(self):
        while True:
            self.grid = self.rng.integers(0, self.NUM_COLORS, size=(self.GRID_HEIGHT, self.GRID_WIDTH)).tolist()
            # Clear initial accidental matches
            while True:
                matches = self._find_matches()
                if not matches:
                    break
                for r, c in matches:
                    self.grid[r][c] = self.rng.integers(0, self.NUM_COLORS)
            
            # Ensure at least one valid move exists
            if self._check_for_possible_matches():
                break

    def _check_for_possible_matches(self):
        temp_grid = [row[:] for row in self.grid]
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < self.GRID_HEIGHT and 0 <= nc < self.GRID_WIDTH:
                        # Simulate swap
                        self.grid[r][c], self.grid[nr][nc] = self.grid[nr][nc], self.grid[r][c]
                        if self._find_matches():
                            self.grid = temp_grid # Restore grid
                            return True
                        self.grid[r][c], self.grid[nr][nc] = self.grid[nr][nc], self.grid[r][c] # Swap back
        self.grid = temp_grid
        return False

    def _find_matches(self):
        to_remove = set()
        # Horizontal matches
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH - 2):
                if self.grid[r][c] is not None and self.grid[r][c] == self.grid[r][c+1] == self.grid[r][c+2]:
                    to_remove.update([(r, c), (r, c+1), (r, c+2)])
        # Vertical matches
        for c in range(self.GRID_WIDTH):
            for r in range(self.GRID_HEIGHT - 2):
                if self.grid[r][c] is not None and self.grid[r][c] == self.grid[r+1][c] == self.grid[r+2][c]:
                    to_remove.update([(r, c), (r+1, c), (r+2, c)])
        return to_remove

    def _apply_gravity(self):
        for c in range(self.GRID_WIDTH):
            empty_row = self.GRID_HEIGHT - 1
            for r in range(self.GRID_HEIGHT - 1, -1, -1):
                if self.grid[r][c] is not None:
                    if r != empty_row:
                        self.grid[empty_row][c] = self.grid[r][c]
                        self.grid[r][c] = None
                    empty_row -= 1

    def _is_board_clear(self):
        return all(all(cell is None for cell in row) for row in self.grid)

    def _check_termination(self):
        return self.moves_left <= 0 or self._is_board_clear() or self.steps >= self.MAX_STEPS

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "moves_left": self.moves_left}

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_particles()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _iso_to_screen(self, r, c):
        x = self.ORIGIN_X + (c - r) * (self.TILE_WIDTH_ISO / 2)
        y = self.ORIGIN_Y + (c + r) * (self.TILE_HEIGHT_ISO / 2)
        return int(x), int(y)

    def _render_game(self):
        # Draw grid cells
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                screen_x, screen_y = self._iso_to_screen(r, c)
                points = [
                    (screen_x, screen_y),
                    (screen_x + self.TILE_WIDTH_ISO / 2, screen_y + self.TILE_HEIGHT_ISO / 2),
                    (screen_x, screen_y + self.TILE_HEIGHT_ISO),
                    (screen_x - self.TILE_WIDTH_ISO / 2, screen_y + self.TILE_HEIGHT_ISO / 2),
                ]
                pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_GRID)

        # Draw cursor
        cursor_x, cursor_y = self.cursor_pos
        screen_x, screen_y = self._iso_to_screen(cursor_y, cursor_x)
        cursor_points = [
            (screen_x, screen_y),
            (screen_x + self.TILE_WIDTH_ISO / 2, screen_y + self.TILE_HEIGHT_ISO / 2),
            (screen_x, screen_y + self.TILE_HEIGHT_ISO),
            (screen_x - self.TILE_WIDTH_ISO / 2, screen_y + self.TILE_HEIGHT_ISO / 2),
        ]
        pygame.draw.polygon(self.screen, self.COLOR_CURSOR, cursor_points, 3)

        # Draw crystals
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                color_idx = self.grid[r][c]
                if color_idx is not None:
                    bob = math.sin(pygame.time.get_ticks() * 0.005 + r + c) * 3 if (c,r) == self.cursor_pos else 0
                    self._render_crystal(self.screen, r, c, color_idx, bob)
    
    def _render_crystal(self, surface, r, c, color_idx, bob_offset=0):
        base_x, base_y = self._iso_to_screen(r, c)
        base_y -= bob_offset
        
        main_color, dark_color = self.CRYSTAL_COLORS[color_idx]
        
        # Points for the crystal shape
        top_y = base_y - self.CRYSTAL_HEIGHT
        mid_y = base_y - self.CRYSTAL_HEIGHT / 2
        
        top_point = (base_x, top_y)
        left_point = (base_x - self.TILE_WIDTH_ISO / 2, mid_y)
        right_point = (base_x + self.TILE_WIDTH_ISO / 2, mid_y)
        bottom_point = (base_x, base_y)

        # Glow effect
        glow_color = (*main_color, 60)
        glow_surface = pygame.Surface((self.TILE_WIDTH_ISO*2, self.CRYSTAL_HEIGHT*2), pygame.SRCALPHA)
        pygame.draw.ellipse(glow_surface, glow_color, glow_surface.get_rect())
        glow_surface = pygame.transform.smoothscale(glow_surface, (int(self.TILE_WIDTH_ISO*1.5), int(self.CRYSTAL_HEIGHT*1.5)))
        surface.blit(glow_surface, (base_x - glow_surface.get_width()//2, base_y - glow_surface.get_height()//2 - self.CRYSTAL_HEIGHT/2))

        # Draw faces
        # Left face
        pygame.gfxdraw.filled_polygon(surface, [top_point, left_point, bottom_point], dark_color)
        pygame.gfxdraw.aapolygon(surface, [top_point, left_point, bottom_point], dark_color)
        # Right face
        pygame.gfxdraw.filled_polygon(surface, [top_point, right_point, bottom_point], main_color)
        pygame.gfxdraw.aapolygon(surface, [top_point, right_point, bottom_point], main_color)
        # Top face
        pygame.gfxdraw.filled_polygon(surface, [left_point, top_point, right_point], main_color)
        pygame.gfxdraw.aapolygon(surface, [left_point, top_point, right_point], main_color)

    def _spawn_particles(self, r, c, color_idx):
        screen_x, screen_y = self._iso_to_screen(r, c)
        screen_y -= self.CRYSTAL_HEIGHT / 2
        main_color, _ = self.CRYSTAL_COLORS[color_idx]
        for _ in range(20):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifespan = random.randint(20, 40)
            self.particles.append([[screen_x, screen_y], vel, main_color, lifespan])

    def _render_particles(self):
        for i in range(len(self.particles) - 1, -1, -1):
            p = self.particles[i]
            p[0][0] += p[1][0] # pos.x += vel.x
            p[0][1] += p[1][1] # pos.y += vel.y
            p[3] -= 1 # lifespan--
            if p[3] <= 0:
                self.particles.popleft()
            else:
                size = max(1, p[3] / 7)
                alpha = int(255 * (p[3] / 40))
                color = (*p[2], alpha)
                
                temp_surf = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
                pygame.draw.circle(temp_surf, color, (size, size), size)
                self.screen.blit(temp_surf, (int(p[0][0] - size), int(p[0][1] - size)), special_flags=pygame.BLEND_RGBA_ADD)

    def _render_ui(self):
        # Moves Left
        moves_text = f"Moves: {self.moves_left}"
        shadow = self.font_large.render(moves_text, True, self.COLOR_UI_SHADOW)
        text = self.font_large.render(moves_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(shadow, (11, 11))
        self.screen.blit(text, (10, 10))

        # Score
        score_text = f"Score: {self.score}"
        shadow = self.font_large.render(score_text, True, self.COLOR_UI_SHADOW)
        text = self.font_large.render(score_text, True, self.COLOR_UI_TEXT)
        text_rect = text.get_rect(topright=(self.SCREEN_WIDTH - 10, 10))
        shadow_rect = shadow.get_rect(topright=(self.SCREEN_WIDTH - 9, 11))
        self.screen.blit(shadow, shadow_rect)
        self.screen.blit(text, text_rect)
        
        # Game Over message
        if self.game_over:
            if self._is_board_clear():
                msg = "BOARD CLEARED!"
            else:
                msg = "OUT OF MOVES"
            
            shadow = self.font_large.render(msg, True, self.COLOR_UI_SHADOW)
            text = self.font_large.render(msg, True, self.COLOR_UI_TEXT)
            text_rect = text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2 + 150))
            shadow_rect = shadow.get_rect(center=(self.SCREEN_WIDTH / 2 + 1, self.SCREEN_HEIGHT / 2 + 151))
            self.screen.blit(shadow, shadow_rect)
            self.screen.blit(text, text_rect)

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
        assert self.moves_left == self.MAX_MOVES
        assert self.score == 0
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # This block allows you to play the game directly
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Pygame setup for human play
    render_screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Crystal Caverns")
    clock = pygame.time.Clock()

    action = np.array([0, 0, 0]) # No-op
    
    print("="*30)
    print("Crystal Caverns - Manual Control")
    print(env.user_guide)
    print("="*30)

    running = True
    while running:
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment.")
                obs, info = env.reset()
                done = False

        # --- Action Polling ---
        keys = pygame.key.get_pressed()
        
        movement = 0
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        current_action = np.array([movement, space_held, shift_held])
        
        # Only step if an action is taken (for turn-based game)
        if not np.array_equal(current_action, np.array([0,0,0])):
            if not done:
                obs, reward, terminated, truncated, info = env.step(current_action)
                done = terminated
                print(f"Action: {current_action}, Reward: {reward:.2f}, Score: {info['score']}, Moves: {info['moves_left']}, Done: {done}")

        # --- Rendering ---
        # The observation is already the rendered screen
        # We just need to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        render_screen.blit(surf, (0, 0))
        pygame.display.flip()

        clock.tick(30) # Limit frame rate

    pygame.quit()
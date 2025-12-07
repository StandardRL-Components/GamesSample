
# Generated: 2025-08-28T03:26:29.843767
# Source Brief: brief_02021.md
# Brief Index: 2021

        
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
        "Controls: Arrows to move cursor. Space to select a gem, then select an adjacent empty tile to move it. Shift to deselect."
    )

    game_description = (
        "An isometric puzzle game. Align 3 or more gems of the same color to collect them. Collect 10 gems within 20 moves to win."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.render_mode = render_mode
        self.screen_width, self.screen_height = 640, 400
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.screen_height, self.screen_width, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        self.rng = np.random.default_rng()

        # Game constants
        self.GRID_WIDTH, self.GRID_HEIGHT = 8, 8
        self.NUM_GEM_TYPES = 5
        self.STARTING_MOVES = 20
        self.WIN_SCORE = 10
        self.MATCH_MIN_LENGTH = 3
        self.FPS = 30
        self.ANIMATION_SPEED = 4 # frames per tile
        
        # Visuals
        self.TILE_W, self.TILE_H = 48, 24
        self.GRID_OX = self.screen_width // 2
        self.GRID_OY = self.screen_height // 2 - self.GRID_HEIGHT * self.TILE_H // 2 + 20

        # Colors
        self.COLOR_BG = (25, 30, 35)
        self.COLOR_GRID = (50, 60, 70)
        self.GEM_COLORS = [
            (255, 80, 80),   # Red
            (80, 255, 80),   # Green
            (80, 150, 255),  # Blue
            (255, 255, 80),  # Yellow
            (200, 80, 255),  # Purple
        ]
        self.COLOR_CURSOR = (255, 255, 255)
        self.COLOR_SELECTED = (255, 255, 0)
        self.COLOR_VALID_MOVE = (100, 255, 100, 100)
        self.COLOR_UI_TEXT = (220, 220, 220)
        self.COLOR_WIN = (100, 255, 100)
        self.COLOR_LOSE = (255, 100, 100)
        
        self.font_main = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_game_over = pygame.font.SysFont("Consolas", 48, bold=True)

        # Game state variables are initialized in reset()
        self.grid = None
        self.cursor_pos = None
        self.selected_gem_pos = None
        self.score = 0
        self.moves_left = 0
        self.game_over = False
        self.win = False
        self.animation = None
        self.particles = []
        self.reward_this_step = 0
        self.steps = 0
        
        # Initial call to reset to set up the first state
        self.reset()
        
        # Validate implementation
        self.validate_implementation()

    def _iso_to_screen(self, gx, gy):
        sx = self.GRID_OX + (gx - gy) * self.TILE_W / 2
        sy = self.GRID_OY + (gx + gy) * self.TILE_H / 2
        return int(sx), int(sy)

    def _generate_initial_grid(self):
        self.grid = self.rng.integers(1, self.NUM_GEM_TYPES + 1, size=(self.GRID_WIDTH, self.GRID_HEIGHT))
        
        # Ensure some empty spaces
        num_empty = self.rng.integers(4, 8)
        for _ in range(num_empty):
            while True:
                x, y = self.rng.integers(0, self.GRID_WIDTH), self.rng.integers(0, self.GRID_HEIGHT)
                if self.grid[x, y] != 0:
                    self.grid[x, y] = 0
                    break

    def _has_possible_moves(self):
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                if self.grid[x, y] != 0: # If there is a gem
                    # Check neighbors
                    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < self.GRID_WIDTH and 0 <= ny < self.GRID_HEIGHT and self.grid[nx, ny] == 0:
                            return True # Found a gem next to an empty space
        return False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed=seed)

        while True:
            self._generate_initial_grid()
            while self._check_and_remove_matches(apply_gravity=True, add_score=False) > 0:
                pass
            if self._has_possible_moves():
                break

        self.cursor_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        self.selected_gem_pos = None
        self.score = 0
        self.moves_left = self.STARTING_MOVES
        self.game_over = False
        self.win = False
        self.animation = None
        self.particles = []
        self.steps = 0

        return self._get_observation(), self._get_info()

    def step(self, action):
        self.reward_this_step = 0
        self.steps += 1

        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        self._update_animation()
        self._update_particles()
        
        if not self.animation:
            self._handle_action(action)
        
        if not self.game_over and self.moves_left <= 0 and not self.animation:
            self.game_over = True
            self.win = self.score >= self.WIN_SCORE

        return (
            self._get_observation(),
            self.reward_this_step,
            self.game_over,
            False,
            self._get_info()
        )

    def _handle_action(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # 1. Handle Deselection (Shift)
        if shift_held and self.selected_gem_pos is not None:
            self.selected_gem_pos = None
            # sfx: cancel_sound

        # 2. Handle Cursor Movement
        if movement != 0:
            cx, cy = self.cursor_pos
            if movement == 1: cy -= 1  # Up
            elif movement == 2: cy += 1  # Down
            elif movement == 3: cx -= 1  # Left
            elif movement == 4: cx += 1  # Right
            self.cursor_pos = [cx % self.GRID_WIDTH, cy % self.GRID_HEIGHT]

        # 3. Handle Selection/Move (Space)
        if space_held:
            cx, cy = self.cursor_pos
            if self.selected_gem_pos is None:
                # Try to select a gem
                if self.grid[cx, cy] > 0:
                    self.selected_gem_pos = [cx, cy]
                    # sfx: select_gem_sound
            else:
                # Try to move the selected gem
                sx, sy = self.selected_gem_pos
                dist = abs(cx - sx) + abs(cy - sy)
                if dist == 1 and self.grid[cx, cy] == 0:
                    # Valid move, start animation
                    self.animation = {
                        "start_pos": [sx, sy],
                        "end_pos": [cx, cy],
                        "gem_type": self.grid[sx, sy],
                        "progress": 0
                    }
                    self.grid[sx, sy] = 0 # Gem is now "in the air"
                    self.moves_left -= 1
                    self.selected_gem_pos = None
                    # sfx: move_gem_sound

    def _update_animation(self):
        if not self.animation:
            return

        self.animation["progress"] += 1
        if self.animation["progress"] >= self.ANIMATION_SPEED:
            # Animation finished
            end_pos = self.animation["end_pos"]
            self.grid[end_pos[0], end_pos[1]] = self.animation["gem_type"]
            self.animation = None
            
            # Now, check for matches and handle consequences
            total_collected = 0
            while True:
                collected = self._check_and_remove_matches(apply_gravity=True, add_score=True)
                if collected > 0:
                    total_collected += collected
                    # sfx: match_cascade_sound
                else:
                    break
            
            if total_collected > 0:
                self.reward_this_step += total_collected
                # sfx: match_success_sound
            
            # Check for win condition
            if self.score >= self.WIN_SCORE:
                self.game_over = True
                self.win = True
                self.reward_this_step += 50
                # sfx: win_sound

    def _check_and_remove_matches(self, apply_gravity, add_score):
        to_remove = set()
        # Check horizontal
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH - (self.MATCH_MIN_LENGTH - 1)):
                gem_type = self.grid[x, y]
                if gem_type == 0: continue
                if all(self.grid[x + i, y] == gem_type for i in range(self.MATCH_MIN_LENGTH)):
                    for i in range(self.MATCH_MIN_LENGTH): to_remove.add((x + i, y))
                    # Extend match
                    for i in range(self.MATCH_MIN_LENGTH, self.GRID_WIDTH - x):
                        if self.grid[x+i, y] == gem_type: to_remove.add((x+i, y))
                        else: break

        # Check vertical
        for x in range(self.GRID_WIDTH):
            for y in range(self.GRID_HEIGHT - (self.MATCH_MIN_LENGTH - 1)):
                gem_type = self.grid[x, y]
                if gem_type == 0: continue
                if all(self.grid[x, y + i] == gem_type for i in range(self.MATCH_MIN_LENGTH)):
                    for i in range(self.MATCH_MIN_LENGTH): to_remove.add((x, y + i))
                    # Extend match
                    for i in range(self.MATCH_MIN_LENGTH, self.GRID_HEIGHT - y):
                        if self.grid[x, y+i] == gem_type: to_remove.add((x, y+i))
                        else: break
        
        if not to_remove:
            return 0

        for x, y in to_remove:
            gem_type = self.grid[x, y]
            if gem_type > 0:
                self._spawn_particles(x, y, gem_type)
            self.grid[x, y] = 0
        
        if add_score:
            self.score += len(to_remove)

        if apply_gravity:
            self._apply_gravity()
        
        return len(to_remove)

    def _apply_gravity(self):
        for x in range(self.GRID_WIDTH):
            empty_row = self.GRID_HEIGHT - 1
            for y in range(self.GRID_HEIGHT - 1, -1, -1):
                if self.grid[x, y] != 0:
                    if y != empty_row:
                        self.grid[x, empty_row] = self.grid[x, y]
                        self.grid[x, y] = 0
                    empty_row -= 1
    
    def _spawn_particles(self, gx, gy, gem_type):
        sx, sy = self._iso_to_screen(gx, gy)
        color = self.GEM_COLORS[gem_type - 1]
        for _ in range(15):
            angle = self.rng.random() * 2 * math.pi
            speed = 2 + self.rng.random() * 3
            vx = math.cos(angle) * speed
            vy = math.sin(angle) * speed
            lifetime = 10 + self.rng.integers(0, 10)
            self.particles.append([sx, sy, vx, vy, lifetime, color])

    def _update_particles(self):
        self.particles = [
            [p[0] + p[2], p[1] + p[3], p[2]*0.95, p[3]*0.95, p[4] - 1, p[5]]
            for p in self.particles if p[4] > 0
        ]

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid cells
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                self._render_tile(x, y, self.COLOR_GRID)

        # Highlight valid moves if a gem is selected
        if self.selected_gem_pos is not None:
            sx, sy = self.selected_gem_pos
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = sx + dx, sy + dy
                if 0 <= nx < self.GRID_WIDTH and 0 <= ny < self.GRID_HEIGHT and self.grid[nx, ny] == 0:
                    self._render_tile(nx, ny, self.COLOR_VALID_MOVE, is_surface=True)
        
        # Draw gems
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                gem_type = self.grid[x, y]
                if gem_type > 0:
                    self._render_gem(x, y, gem_type)
        
        # Draw animated gem
        if self.animation:
            prog = self.animation["progress"] / self.ANIMATION_SPEED
            start_gx, start_gy = self.animation["start_pos"]
            end_gx, end_gy = self.animation["end_pos"]
            
            gx = start_gx + (end_gx - start_gx) * prog
            gy = start_gy + (end_gy - start_gy) * prog
            self._render_gem(gx, gy, self.animation["gem_type"], is_float=True)
            
        # Draw cursor and selection highlight
        self._render_tile(self.cursor_pos[0], self.cursor_pos[1], self.COLOR_CURSOR, width=2)
        if self.selected_gem_pos is not None:
            self._render_tile(self.selected_gem_pos[0], self.selected_gem_pos[1], self.COLOR_SELECTED, width=3)
        
        # Draw particles
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p[4] / 20))))
            color = p[5] + (alpha,)
            size = max(1, int(p[4] / 4))
            temp_surf = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (size, size), size)
            self.screen.blit(temp_surf, (int(p[0]) - size, int(p[1]) - size))

    def _render_tile(self, gx, gy, color, width=1, is_surface=False):
        sx, sy = self._iso_to_screen(gx, gy)
        points = [
            (sx, sy - self.TILE_H / 2),
            (sx + self.TILE_W / 2, sy),
            (sx, sy + self.TILE_H / 2),
            (sx - self.TILE_W / 2, sy)
        ]
        if is_surface:
            s = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
            pygame.gfxdraw.filled_polygon(s, points, color)
            self.screen.blit(s, (0,0))
        else:
            pygame.draw.polygon(self.screen, color, points, width)

    def _render_gem(self, gx, gy, gem_type, is_float=False):
        if not is_float:
            sx, sy = self._iso_to_screen(gx, gy)
        else: # Handle float coordinates for animation
            sxf = self.GRID_OX + (gx - gy) * self.TILE_W / 2
            syf = self.GRID_OY + (gx + gy) * self.TILE_H / 2
            sx, sy = int(sxf), int(syf)

        color = self.GEM_COLORS[gem_type - 1]
        light_color = tuple(min(255, c + 60) for c in color)
        dark_color = tuple(max(0, c - 60) for c in color)

        w, h = self.TILE_W * 0.8, self.TILE_H * 0.8
        
        # Main body
        points = [
            (sx, sy - h / 2),
            (sx + w / 2, sy),
            (sx, sy + h / 2),
            (sx - w / 2, sy)
        ]
        pygame.gfxdraw.filled_polygon(self.screen, points, color)
        pygame.gfxdraw.aapolygon(self.screen, points, color)

        # Facets for 3D effect
        pygame.draw.line(self.screen, light_color, (sx - w / 2, sy), (sx, sy - h / 2), 2)
        pygame.draw.line(self.screen, dark_color, (sx - w / 2, sy), (sx, sy + h / 2), 2)
        pygame.draw.line(self.screen, dark_color, (sx + w / 2, sy), (sx, sy + h / 2), 2)


    def _render_ui(self):
        # Score
        score_text = self.font_main.render(f"Gems: {self.score}/{self.WIN_SCORE}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Moves
        moves_text = self.font_main.render(f"Moves: {self.moves_left}", True, self.COLOR_UI_TEXT)
        self.screen.blit(moves_text, (self.screen_width - moves_text.get_width() - 10, 10))
        
        # Game Over Message
        if self.game_over:
            overlay = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            if self.win:
                msg = "YOU WIN!"
                color = self.COLOR_WIN
            else:
                msg = "GAME OVER"
                color = self.COLOR_LOSE
                
            end_text = self.font_game_over.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.screen_width / 2, self.screen_height / 2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_left": self.moves_left,
            "game_over": self.game_over,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.screen_height, self.screen_width, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.screen_height, self.screen_width, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.screen_height, self.screen_width, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.screen_width, env.screen_height))
    pygame.display.set_caption("Gem Swap")
    
    running = True
    while running:
        # Human input to action conversion
        movement = 0 # none
        space_held = 0
        shift_held = 0
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1
            
        action = [movement, space_held, shift_held]
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting game.")
                obs, info = env.reset()

        obs, reward, terminated, truncated, info = env.step(action)
        
        if reward > 0:
            print(f"Step: {info['steps']}, Reward: {reward}, Score: {info['score']}, Moves: {info['moves_left']}")

        if terminated:
            print(f"Game Over! Final Score: {info['score']}")
        
        # Display the frame
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(env.FPS)
        
    env.close()
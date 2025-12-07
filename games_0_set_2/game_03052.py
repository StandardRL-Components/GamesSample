
# Generated: 2025-08-27T22:13:29.798573
# Source Brief: brief_03052.md
# Brief Index: 3052

        
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
        "Controls: ←↑→↓ to move selected block. Space/Shift to cycle selection. Match the target pattern."
    )

    game_description = (
        "A pixel-pushing puzzle. Rearrange the colored blocks on the main grid to match the small target pattern in the top-left before you run out of moves."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_SIZE = 8
        self.CELL_SIZE = 40
        self.GRID_WIDTH = self.GRID_SIZE * self.CELL_SIZE
        self.GRID_HEIGHT = self.GRID_SIZE * self.CELL_SIZE
        self.GRID_X = (self.WIDTH - self.GRID_WIDTH) // 2
        self.GRID_Y = (self.HEIGHT - self.GRID_HEIGHT) // 2
        self.MAX_MOVES = 30
        self.MAX_STEPS = 1000
        self.INITIAL_PIXELS = 5

        # Colors
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_GRID = (40, 50, 70)
        self.COLOR_EMPTY = self.COLOR_BG
        self.COLOR_WHITE = (240, 240, 240)
        self.COLOR_GOLD = (255, 215, 0)
        self.COLOR_RED = (255, 80, 80)
        
        self.PIXEL_PALETTE = [
            (55, 148, 110), (79, 105, 198), (170, 102, 204), (219, 125, 62),
            (219, 158, 62), (153, 229, 80), (230, 230, 230), (255, 105, 180),
            (100, 200, 250), (255, 0, 0)
        ]

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 64)
        self.font_medium = pygame.font.Font(None, 32)
        self.font_small = pygame.font.Font(None, 24)
        
        # State variables
        self.grid = None
        self.target_grid = None
        self.target_positions = {}
        self.pixel_list = []
        self.selected_pixel_idx = 0
        
        self.steps = 0
        self.score = 0
        self.moves_left = 0
        self.game_over = False
        
        self.successful_episodes = 0
        
        # Animation state
        self.game_state = "IDLE"  # IDLE, MOVING, SELECTING
        self.animation_timer = 0
        self.animation_duration = 6  # 200ms at 30fps
        self.moving_pixel_info = {}
        
        # Input handling
        self.last_space_held = False
        self.last_shift_held = False
        
        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.moves_left = self.MAX_MOVES
        
        self._generate_puzzle()
        
        self.pixel_list = self._get_pixel_list()
        self.selected_pixel_idx = 0 if self.pixel_list else -1
        
        self.game_state = "IDLE"
        self.animation_timer = 0
        self.moving_pixel_info = {}
        self.last_space_held = False
        self.last_shift_held = False
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_held, shift_held = action
        reward = 0
        self.steps += 1
        
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        if self.game_state == "IDLE":
            # Handle selection first (on key press, not hold)
            if space_held and not self.last_space_held:
                self._select_next_pixel()
                self.game_state = "SELECTING"
                self.animation_timer = 3 # Short flash effect
                # sound: select_sound()
            elif shift_held and not self.last_shift_held:
                self._select_previous_pixel()
                self.game_state = "SELECTING"
                self.animation_timer = 3
                # sound: select_sound()
            elif movement > 0 and self.selected_pixel_idx != -1:
                move_info = self._prepare_move(movement)
                if move_info:
                    self.moves_left -= 1
                    move_reward = self._calculate_move_reward(move_info)
                    reward += move_reward
                    self.score += move_reward

                    self.moving_pixel_info = move_info
                    self.game_state = "MOVING"
                    self.animation_timer = self.animation_duration
                    # sound: move_start_sound()

        elif self.game_state == "MOVING":
            self.animation_timer -= 1
            if self.animation_timer <= 0:
                self._finalize_move()
                self.game_state = "IDLE"
                # sound: move_end_sound()
                
                # Check for win/loss after move completes
                if self._check_win_condition():
                    reward += 50
                    self.score += 50
                    self.game_over = True
                    self.successful_episodes += 1
                elif self.moves_left <= 0:
                    reward += -50
                    self.score += -50
                    self.game_over = True

        elif self.game_state == "SELECTING":
            self.animation_timer -= 1
            if self.animation_timer <= 0:
                self.game_state = "IDLE"
        
        self.last_space_held = bool(space_held)
        self.last_shift_held = bool(shift_held)
        
        terminated = self.game_over or self.steps >= self.MAX_STEPS
        if self.steps >= self.MAX_STEPS and not self.game_over:
            # Penalize for timeout
            reward -= 50
            self.score -= 50

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _generate_puzzle(self):
        # Determine number of pixels based on difficulty
        pixel_increase = (self.successful_episodes // 5) * (self.INITIAL_PIXELS * 0.05)
        num_pixels = min(self.INITIAL_PIXELS + int(pixel_increase), self.GRID_SIZE * self.GRID_SIZE - 1)
        
        # Create target grid
        self.target_grid = [[self.COLOR_EMPTY for _ in range(self.GRID_SIZE)] for _ in range(self.GRID_SIZE)]
        self.target_positions = {}
        
        available_colors = self.PIXEL_PALETTE[:]
        self.np_random.shuffle(available_colors)
        
        placed_pixels = 0
        while placed_pixels < num_pixels:
            x, y = self.np_random.integers(0, self.GRID_SIZE, size=2)
            if self.target_grid[y][x] == self.COLOR_EMPTY:
                color = available_colors.pop()
                self.target_grid[y][x] = color
                self.target_positions[color] = (x, y)
                placed_pixels += 1
        
        # Create shuffled grid for gameplay
        self.grid = [row[:] for row in self.target_grid]
        num_shuffles = num_pixels * 5
        for _ in range(num_shuffles):
            pixels = self._get_pixel_list(self.grid)
            if not pixels: break
            px, py = self.np_random.choice(pixels)
            
            empty_neighbors = []
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = px + dx, py + dy
                if 0 <= nx < self.GRID_SIZE and 0 <= ny < self.GRID_SIZE and self.grid[ny][nx] == self.COLOR_EMPTY:
                    empty_neighbors.append((nx, ny))
            
            if empty_neighbors:
                nx, ny = random.choice(empty_neighbors)
                self.grid[ny][nx], self.grid[py][px] = self.grid[py][px], self.grid[ny][nx]

    def _get_pixel_list(self, grid=None):
        if grid is None:
            grid = self.grid
        pixels = []
        for y in range(self.GRID_SIZE):
            for x in range(self.GRID_SIZE):
                if grid[y][x] != self.COLOR_EMPTY:
                    pixels.append((x, y))
        return pixels
    
    def _select_next_pixel(self):
        if not self.pixel_list: return
        self.selected_pixel_idx = (self.selected_pixel_idx + 1) % len(self.pixel_list)

    def _select_previous_pixel(self):
        if not self.pixel_list: return
        self.selected_pixel_idx = (self.selected_pixel_idx - 1 + len(self.pixel_list)) % len(self.pixel_list)

    def _prepare_move(self, movement):
        px, py = self.pixel_list[self.selected_pixel_idx]
        
        direction_map = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)} # up, down, left, right
        dx, dy = direction_map[movement]
        
        nx, ny = px + dx, py + dy
        
        if 0 <= nx < self.GRID_SIZE and 0 <= ny < self.GRID_SIZE and self.grid[ny][nx] == self.COLOR_EMPTY:
            return {
                "color": self.grid[py][px],
                "start_grid": (px, py),
                "end_grid": (nx, ny)
            }
        return None

    def _finalize_move(self):
        color = self.moving_pixel_info["color"]
        px, py = self.moving_pixel_info["start_grid"]
        nx, ny = self.moving_pixel_info["end_grid"]
        
        self.grid[py][px] = self.COLOR_EMPTY
        self.grid[ny][nx] = color
        
        self.pixel_list = self._get_pixel_list()
        try:
            self.selected_pixel_idx = self.pixel_list.index((nx, ny))
        except ValueError:
            self.selected_pixel_idx = 0 if self.pixel_list else -1
            
        self.moving_pixel_info = {}

    def _calculate_move_reward(self, move_info):
        color = move_info["color"]
        px, py = move_info["start_grid"]
        nx, ny = move_info["end_grid"]
        
        tx, ty = self.target_positions[color]
        
        dist_before = abs(px - tx) + abs(py - ty)
        dist_after = abs(nx - tx) + abs(ny - ty)
        
        reward = 0
        if dist_after < dist_before:
            reward += 0.1  # Moved closer
        elif dist_after > dist_before:
            reward -= 0.1  # Moved further
            
        if dist_after == 0:
            reward += 5.0 # Placed correctly
            # sound: correct_place_sound()
            
        return reward

    def _check_win_condition(self):
        for y in range(self.GRID_SIZE):
            for x in range(self.GRID_SIZE):
                if self.grid[y][x] != self.target_grid[y][x]:
                    return False
        return True

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "moves_left": self.moves_left}

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid lines
        for i in range(self.GRID_SIZE + 1):
            x = self.GRID_X + i * self.CELL_SIZE
            y = self.GRID_Y + i * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, self.GRID_Y), (x, self.GRID_Y + self.GRID_HEIGHT))
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.GRID_X, y), (self.GRID_X + self.GRID_WIDTH, y))

        # Draw static pixels
        for y in range(self.GRID_SIZE):
            for x in range(self.GRID_SIZE):
                color = self.grid[y][x]
                if color != self.COLOR_EMPTY:
                    # Don't draw the pixel that is currently being animated
                    is_moving = (self.game_state == "MOVING" and (x, y) == self.moving_pixel_info.get("start_grid"))
                    if not is_moving:
                        self._render_pixel(x, y, color)
        
        # Draw moving pixel with interpolation
        if self.game_state == "MOVING":
            progress = 1.0 - (self.animation_timer / self.animation_duration)
            progress = 1 - (1 - progress) ** 3 # Ease out cubic
            
            start_gx, start_gy = self.moving_pixel_info["start_grid"]
            end_gx, end_gy = self.moving_pixel_info["end_grid"]
            
            draw_x = start_gx + (end_gx - start_gx) * progress
            draw_y = start_gy + (end_gy - start_gy) * progress
            
            self._render_pixel(draw_x, draw_y, self.moving_pixel_info["color"])

        # Draw selector
        if self.selected_pixel_idx != -1 and self.game_state != "MOVING":
            sel_x, sel_y = self.pixel_list[self.selected_pixel_idx]
            rect = pygame.Rect(self.GRID_X + sel_x * self.CELL_SIZE, self.GRID_Y + sel_y * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
            
            # Pulsing glow effect
            pulse = (math.sin(self.steps * 0.2) + 1) / 2 # 0 to 1
            glow_size = int(4 + pulse * 4)
            glow_alpha = int(80 + pulse * 60)
            
            glow_color = self.COLOR_WHITE if self.game_state != "SELECTING" else self.COLOR_GOLD
            
            glow_rect = rect.inflate(glow_size, glow_size)
            s = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
            pygame.draw.rect(s, glow_color + (glow_alpha,), s.get_rect(), border_radius=8)
            self.screen.blit(s, glow_rect.topleft)
            
            pygame.draw.rect(self.screen, self.COLOR_WHITE, rect, 2, border_radius=4)
            
    def _render_pixel(self, grid_x, grid_y, color):
        inset = 4
        px = self.GRID_X + grid_x * self.CELL_SIZE + inset
        py = self.GRID_Y + grid_y * self.CELL_SIZE + inset
        size = self.CELL_SIZE - inset * 2
        rect = pygame.Rect(int(px), int(py), size, size)
        
        # Draw pixel with a subtle 3D effect
        highlight = tuple(min(255, c + 25) for c in color)
        shadow = tuple(max(0, c - 25) for c in color)
        
        pygame.draw.rect(self.screen, shadow, rect, border_radius=4)
        pygame.draw.rect(self.screen, color, rect.inflate(-2, -2), border_radius=4)
        
        # Top-left highlight
        pygame.draw.line(self.screen, highlight, rect.move(1, 1).topleft, rect.move(1, 1).topright, 1)
        pygame.draw.line(self.screen, highlight, rect.move(1, 1).topleft, rect.move(1, 1).bottomleft, 1)


    def _render_ui(self):
        # Render Target Preview
        target_cell_size = 10
        preview_x, preview_y = 20, 20
        preview_w = self.GRID_SIZE * target_cell_size
        
        self._draw_text("TARGET", (preview_x, preview_y - 15), self.font_small, self.COLOR_GRID)
        
        preview_bg_rect = pygame.Rect(preview_x, preview_y, preview_w, preview_w)
        pygame.draw.rect(self.screen, self.COLOR_GRID, preview_bg_rect, 0, 4)
        
        for y in range(self.GRID_SIZE):
            for x in range(self.GRID_SIZE):
                color = self.target_grid[y][x]
                if color != self.COLOR_EMPTY:
                    rect = pygame.Rect(preview_x + x * target_cell_size, preview_y + y * target_cell_size, target_cell_size, target_cell_size)
                    pygame.draw.rect(self.screen, color, rect)

        # Render Moves Left
        moves_text = f"MOVES: {self.moves_left}"
        self._draw_text(moves_text, (self.WIDTH - 20, 20), self.font_medium, self.COLOR_WHITE, align="topright")
        
        # Render Score
        score_text = f"SCORE: {int(self.score)}"
        self._draw_text(score_text, (self.WIDTH - 20, 50), self.font_medium, self.COLOR_WHITE, align="topright")

        # Render Game Over Screen
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            if self._check_win_condition():
                self._draw_text("COMPLETE!", (self.WIDTH // 2, self.HEIGHT // 2 - 20), self.font_large, self.COLOR_GOLD, align="center")
            else:
                self._draw_text("GAME OVER", (self.WIDTH // 2, self.HEIGHT // 2 - 20), self.font_large, self.COLOR_RED, align="center")
    
    def _draw_text(self, text, pos, font, color, align="topleft"):
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect()
        if align == "center":
            text_rect.center = pos
        elif align == "topright":
            text_rect.topright = pos
        else: # topleft
            text_rect.topleft = pos
        self.screen.blit(text_surface, text_rect)

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    pygame.display.set_caption("Pixel Push")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    running = True
    total_reward = 0
    
    while running:
        movement, space, shift = 0, 0, 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0
                print("--- Game Reset ---")

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1
            
        action = [movement, space, shift]
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        if reward != 0:
            total_reward += reward
            print(f"Step: {info['steps']}, Action: {action}, Reward: {reward:.2f}, Total: {total_reward:.2f}, Moves: {info['moves_left']}")

        if terminated or truncated:
            print(f"Episode Finished. Final Score: {info['score']}. Press 'R' to reset.")
        
        # Render to screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(30) # Run at 30 FPS

    pygame.quit()
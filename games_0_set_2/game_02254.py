
# Generated: 2025-08-27T19:46:27.347826
# Source Brief: brief_02254.md
# Brief Index: 2254

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import os
import pygame


# Set a dummy video driver to run pygame headless
os.environ["SDL_VIDEODRIVER"] = "dummy"

class GameEnv(gym.Env):
    """
    A puzzle game where the player pushes colored pixels around a grid
    to fill a target area within a limited number of moves.
    """
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Arrows to move selector. Arrows + Space/Shift to push."
    )

    game_description = (
        "Push colored pixels into the target zone. Fill 75% of the zone before you run out of moves!"
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_W, self.GRID_H = 32, 20
        self.CELL_SIZE = 20
        self.MAX_MOVES = 100
        self.MAX_STEPS = 1000
        self.WIN_PERCENTAGE = 0.75

        # --- Colors ---
        self.COLOR_BG = (15, 15, 25)
        self.COLOR_GRID = (30, 30, 45)
        self.COLOR_TARGET_OUTLINE = (180, 180, 180)
        self.COLOR_TARGET_AREA = (50, 50, 70)
        self.COLOR_TARGET_FILLED = (80, 80, 100)
        self.COLOR_SELECTOR = (255, 255, 0)
        self.PIXEL_COLORS = [
            (255, 80, 80),   # Red
            (80, 255, 80),   # Green
            (80, 120, 255),  # Blue
            (255, 200, 80),  # Orange
            (200, 80, 255),  # Purple
        ]

        # --- Gymnasium Spaces ---
        self.observation_space = Box(low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_msg = pygame.font.SysFont("monospace", 48, bold=True)
        
        # --- Game State (initialized in reset) ---
        self.pixels = []
        self.target_rect = None
        self.selector_pos = [0, 0]
        self.moves_left = 0
        self.fill_percentage = 0.0
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.win = False
        self.np_random = None

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.moves_left = self.MAX_MOVES
        
        self._generate_level()

        self.selector_pos = [self.GRID_W // 2, self.GRID_H // 2]
        
        self._update_fill_percentage()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        reward = 0
        self.steps += 1
        self.moves_left -= 1
        
        old_fill = self.fill_percentage

        # 1. Handle selector movement
        dx, dy = 0, 0
        if movement == 1: dy = -1  # Up
        elif movement == 2: dy = 1  # Down
        elif movement == 3: dx = -1  # Left
        elif movement == 4: dx = 1  # Right

        if dx != 0 or dy != 0:
            self.selector_pos[0] = (self.selector_pos[0] + dx) % self.GRID_W
            self.selector_pos[1] = (self.selector_pos[1] + dy) % self.GRID_H

        # 2. Handle push action (must be simultaneous with movement)
        push_triggered = (space_held or shift_held) and (dx != 0 or dy != 0)
        if push_triggered:
            pixel_idx_to_push = -1
            for i, p in enumerate(self.pixels):
                if p[0] == self.selector_pos[0] and p[1] == self.selector_pos[1]:
                    pixel_idx_to_push = i
                    break
            
            if pixel_idx_to_push != -1:
                # SFX: // sound of block sliding
                self._push_pixels(pixel_idx_to_push, dx, dy)
        
        # 3. Update state and calculate reward
        self._update_fill_percentage()
        fill_change = self.fill_percentage - old_fill
        
        if fill_change > 0:
            reward += fill_change * 10  # +0.1 per percentage point
        elif fill_change < 0:
            reward += fill_change * 2   # -0.02 per percentage point
        
        self.score += reward

        # 4. Check for termination
        terminated = False
        if self.fill_percentage >= self.WIN_PERCENTAGE:
            self.game_over = True
            self.win = True
            terminated = True
            reward += 100
            self.score += 100
        elif self.moves_left <= 0:
            self.game_over = True
            self.win = False
            terminated = True
            reward -= 10
            self.score -= 10
        
        if self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _generate_level(self):
        tw = self.np_random.integers(self.GRID_W // 4, self.GRID_W // 2)
        th = self.np_random.integers(self.GRID_H // 4, self.GRID_H // 2)
        tx = self.np_random.integers(1, self.GRID_W - tw - 1)
        ty = self.np_random.integers(1, self.GRID_H - th - 1)
        self.target_rect = pygame.Rect(tx, ty, tw, th)
        
        self.pixels = []
        num_pixels = self.np_random.integers(
            (tw * th) // 2, int(tw * th * 1.5)
        )
        occupied_positions = set()
        for _ in range(num_pixels):
            for _ in range(100): # Attempt to find a free spot
                px = self.np_random.integers(0, self.GRID_W)
                py = self.np_random.integers(0, self.GRID_H)
                if (px, py) not in occupied_positions:
                    occupied_positions.add((px, py))
                    color_idx = self.np_random.integers(0, len(self.PIXEL_COLORS))
                    self.pixels.append([px, py, color_idx])
                    break
    
    def _push_pixels(self, start_pixel_idx, dx, dy):
        chain = []
        current_pos = list(self.pixels[start_pixel_idx][:2])
        pixel_map = {(p[0], p[1]): i for i, p in enumerate(self.pixels)}

        # 1. Identify the chain of pixels to be pushed
        # We use a set to detect if the chain wraps and collides with itself
        chain_pos_set = set()
        while tuple(current_pos) in pixel_map:
            idx = pixel_map[tuple(current_pos)]
            chain.append(idx)
            chain_pos_set.add(tuple(current_pos))
            current_pos[0] = (current_pos[0] + dx) % self.GRID_W
            current_pos[1] = (current_pos[1] + dy) % self.GRID_H
        
        # 2. Check if destination is blocked by a non-chain pixel
        if tuple(current_pos) in pixel_map:
            # SFX: // sound of a dull thud
            return

        # 3. If valid, move all pixels in the chain (in reverse)
        for pixel_idx in reversed(chain):
            self.pixels[pixel_idx][0] = (self.pixels[pixel_idx][0] + dx) % self.GRID_W
            self.pixels[pixel_idx][1] = (self.pixels[pixel_idx][1] + dy) % self.GRID_H

    def _update_fill_percentage(self):
        filled_count = 0
        if not self.target_rect: return
        
        for p in self.pixels:
            if self.target_rect.collidepoint(p[0], p[1]):
                filled_count += 1
        
        total_target_cells = self.target_rect.width * self.target_rect.height
        if total_target_cells == 0:
            self.fill_percentage = 0.0
        else:
            self.fill_percentage = filled_count / total_target_cells

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid lines
        for x in range(0, self.WIDTH, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT))
        for y in range(0, self.HEIGHT, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y))

        # Draw target area
        if self.target_rect:
            r = self.target_rect.copy()
            r.x *= self.CELL_SIZE
            r.y *= self.CELL_SIZE
            r.w *= self.CELL_SIZE
            r.h *= self.CELL_SIZE
            pygame.draw.rect(self.screen, self.COLOR_TARGET_AREA, r)
        
        # Draw pixels and filled target cells
        pixel_positions = {(p[0], p[1]) for p in self.pixels}
        if self.target_rect:
            for y in range(self.target_rect.top, self.target_rect.bottom):
                for x in range(self.target_rect.left, self.target_rect.right):
                    if (x, y) in pixel_positions:
                        filled_rect = pygame.Rect(x * self.CELL_SIZE, y * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
                        pygame.draw.rect(self.screen, self.COLOR_TARGET_FILLED, filled_rect)

        for p in self.pixels:
            px_rect = pygame.Rect(p[0] * self.CELL_SIZE + 2, p[1] * self.CELL_SIZE + 2, self.CELL_SIZE - 4, self.CELL_SIZE - 4)
            pygame.draw.rect(self.screen, self.PIXEL_COLORS[p[2]], px_rect, border_radius=3)
        
        if self.target_rect:
            pygame.draw.rect(self.screen, self.COLOR_TARGET_OUTLINE, r, 2)

        # Draw selector
        if not self.game_over:
            sel_rect = pygame.Rect(self.selector_pos[0] * self.CELL_SIZE, self.selector_pos[1] * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
            alpha = int((math.sin(pygame.time.get_ticks() * 0.005) + 1) / 2 * 100 + 50)
            pygame.gfxdraw.box(self.screen, sel_rect, (*self.COLOR_SELECTOR, alpha))
            pygame.draw.rect(self.screen, self.COLOR_SELECTOR, sel_rect, 2)

    def _render_ui(self):
        # Moves left
        moves_text = self.font_ui.render(f"MOVES: {self.moves_left}", True, (200, 200, 220))
        self.screen.blit(moves_text, (10, 10))

        # Fill percentage
        fill_str = f"FILL: {self.fill_percentage:.1%}"
        fill_text = self.font_ui.render(fill_str, True, (200, 200, 220))
        self.screen.blit(fill_text, (self.WIDTH - fill_text.get_width() - 10, 10))
        
        # Game Over Message
        if self.game_over:
            msg_text = "YOU WIN!" if self.win else "GAME OVER"
            color = (100, 255, 100) if self.win else (255, 100, 100)
            rendered_text = self.font_msg.render(msg_text, True, color)
            text_rect = rendered_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            
            # Draw a semi-transparent background for the text
            bg_surface = pygame.Surface(text_rect.size, pygame.SRCALPHA)
            bg_surface.fill((0, 0, 0, 150))
            self.screen.blit(bg_surface, text_rect.topleft)
            self.screen.blit(rendered_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_left": self.moves_left,
            "fill_percentage": self.fill_percentage,
            "win": self.win,
        }

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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # Requires pygame to be installed with display support
    os.environ["SDL_VIDEODRIVER"] = "x11" # or "windows", "macOS"
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Pixel Pusher")
    
    terminated = False
    clock = pygame.time.Clock()
    
    while not terminated:
        movement = 0 # no-op
        space = 0
        shift = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1

        action = [movement, space, shift]
        
        # In manual play, we only step when a key is pressed
        if any(action):
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Action: {action}, Reward: {reward:.2f}, Score: {info['score']:.2f}, Fill: {info['fill_percentage']:.1%}, Terminated: {terminated}")
        
        # Render the observation to the screen
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if info.get('win') or (info.get('moves_left', 1) <= 0):
            pygame.time.wait(2000) # Pause on win/loss
            obs, info = env.reset()
            terminated = False

        clock.tick(30) # Limit frame rate
        
    env.close()
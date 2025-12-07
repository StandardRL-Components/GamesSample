
# Generated: 2025-08-28T06:57:02.984826
# Source Brief: brief_03091.md
# Brief Index: 3091

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
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
        "Controls: ↑↓←→ to move the cursor. Press space to select a block and clear adjacent same-colored blocks."
    )

    game_description = (
        "Clear the grid of colored blocks by matching adjacent colors before time runs out."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_SIZE = 7
        self.NUM_COLORS = 5
        self.FPS = 30
        self.MAX_TIME_SECONDS = 30
        self.MAX_STEPS = self.MAX_TIME_SECONDS * self.FPS + 5 # A little buffer

        # --- Colors ---
        self.COLOR_BG = (20, 25, 30)
        self.COLOR_GRID_LINES = (40, 45, 50)
        self.BLOCK_COLORS = [
            (255, 65, 54),   # Red
            (0, 116, 217),  # Blue
            (46, 204, 64),  # Green
            (255, 133, 27), # Orange
            (177, 13, 201), # Purple
        ]
        self.COLOR_WHITE = (240, 240, 240)
        self.COLOR_GOLD = (255, 215, 0)
        
        # --- Dimensions ---
        self.GRID_AREA_HEIGHT = self.HEIGHT - 40
        self.BLOCK_SIZE = min((self.WIDTH - 40) // self.GRID_SIZE, self.GRID_AREA_HEIGHT // self.GRID_SIZE)
        self.GRID_WIDTH_PX = self.GRID_SIZE * self.BLOCK_SIZE
        self.GRID_HEIGHT_PX = self.GRID_SIZE * self.BLOCK_SIZE
        self.GRID_OFFSET_X = (self.WIDTH - self.GRID_WIDTH_PX) // 2
        self.GRID_OFFSET_Y = (self.HEIGHT - self.GRID_HEIGHT_PX) // 2

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_game_over = pygame.font.SysFont("Consolas", 50, bold=True)

        # --- Game State ---
        self.grid = None
        self.cursor_pos = None
        self.score = None
        self.time_remaining = None
        self.steps = None
        self.game_over = None
        self.particles = []
        
        self.prev_space_held = False
        self.last_move_frame = -1
        self.MOVE_COOLDOWN = 4 # frames

        self.np_random = None
        self.current_step_reward = 0

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        else:
            self.np_random = np.random.default_rng()

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_remaining = self.MAX_TIME_SECONDS * self.FPS
        self.cursor_pos = [self.GRID_SIZE // 2, self.GRID_SIZE // 2]
        self.particles = []
        
        self.prev_space_held = False
        self.last_move_frame = -1
        self.current_step_reward = 0

        self._generate_grid()

        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        space_pressed = space_held and not self.prev_space_held
        self.prev_space_held = space_held
        
        self.current_step_reward = 0
        
        if not self.game_over:
            self.time_remaining -= 1
            self._handle_input(movement, space_pressed)

        self._update_particles()
        
        reward = self.current_step_reward
        
        terminated, terminal_reward = self._check_termination()
        reward += terminal_reward
        
        if terminated:
            self.game_over = True

        self.steps += 1
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, movement, space_pressed):
        # --- Cursor Movement with Cooldown ---
        if movement != 0 and (self.steps - self.last_move_frame) > self.MOVE_COOLDOWN:
            dx, dy = 0, 0
            if movement == 1: dy = -1  # Up
            elif movement == 2: dy = 1   # Down
            elif movement == 3: dx = -1  # Left
            elif movement == 4: dx = 1   # Right
            
            self.cursor_pos[0] = (self.cursor_pos[0] + dx) % self.GRID_SIZE
            self.cursor_pos[1] = (self.cursor_pos[1] + dy) % self.GRID_SIZE
            self.last_move_frame = self.steps

        # --- Block Selection ---
        if space_pressed:
            self._process_selection()

    def _process_selection(self):
        cx, cy = self.cursor_pos
        if self.grid[cy][cx] == -1:
            return # Selected an empty block

        matches = self._find_matches_bfs(cx, cy)
        
        if len(matches) > 1:
            # Sound effect placeholder: # sfx_match.play()
            num_cleared = len(matches)
            
            # Calculate reward
            reward = num_cleared
            if num_cleared > 3:
                reward += 5 # Bonus for larger chains
            self.current_step_reward += reward
            self.score += reward

            # Clear blocks and create particles
            for r, c in matches:
                block_color = self.BLOCK_COLORS[self.grid[r][c]]
                self.grid[r][c] = -1
                self._create_particles(c, r, block_color)
            
            self._collapse_and_refill_grid()
        else:
            # Sound effect placeholder: # sfx_no_match.play()
            pass

    def _find_matches_bfs(self, start_c, start_r):
        if self.grid[start_r][start_c] == -1:
            return []
            
        target_color = self.grid[start_r][start_c]
        q = deque([(start_r, start_c)])
        visited = set([(start_r, start_c)])
        matches = []

        while q:
            r, c = q.popleft()
            matches.append((r, c))

            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.GRID_SIZE and 0 <= nc < self.GRID_SIZE and \
                   (nr, nc) not in visited and self.grid[nr][nc] == target_color:
                    visited.add((nr, nc))
                    q.append((nr, nc))
        return matches

    def _collapse_and_refill_grid(self):
        for c in range(self.GRID_SIZE):
            write_ptr = self.GRID_SIZE - 1
            for r in range(self.GRID_SIZE - 1, -1, -1):
                if self.grid[r][c] != -1:
                    self.grid[write_ptr][c], self.grid[r][c] = self.grid[r][c], self.grid[write_ptr][c]
                    write_ptr -= 1
            
            # Refill empty top cells
            for r in range(write_ptr, -1, -1):
                self.grid[r][c] = self.np_random.integers(0, self.NUM_COLORS)

    def _generate_grid(self):
        while True:
            self.grid = self.np_random.integers(0, self.NUM_COLORS, size=(self.GRID_SIZE, self.GRID_SIZE)).tolist()
            if self._check_for_any_match():
                break

    def _check_for_any_match(self):
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                if len(self._find_matches_bfs(c, r)) > 1:
                    return True
        return False

    def _is_board_clear(self):
        return all(self.grid[r][c] == -1 for r in range(self.GRID_SIZE) for c in range(self.GRID_SIZE))

    def _check_termination(self):
        if self._is_board_clear():
            return True, 100 # Win
        if self.time_remaining <= 0:
            return True, -100 # Loss (time)
        if self.steps >= self.MAX_STEPS:
            return True, 0 # Loss (steps)
        return False, 0

    def _create_particles(self, c, r, color):
        px = self.GRID_OFFSET_X + c * self.BLOCK_SIZE + self.BLOCK_SIZE / 2
        py = self.GRID_OFFSET_Y + r * self.BLOCK_SIZE + self.BLOCK_SIZE / 2
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifetime = self.np_random.integers(15, 30)
            self.particles.append({'pos': [px, py], 'vel': vel, 'life': lifetime, 'max_life': lifetime, 'color': color})

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['life'] -= 1
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1 # Gravity

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid lines
        for i in range(self.GRID_SIZE + 1):
            # Vertical
            start_v = (self.GRID_OFFSET_X + i * self.BLOCK_SIZE, self.GRID_OFFSET_Y)
            end_v = (self.GRID_OFFSET_X + i * self.BLOCK_SIZE, self.GRID_OFFSET_Y + self.GRID_HEIGHT_PX)
            pygame.draw.line(self.screen, self.COLOR_GRID_LINES, start_v, end_v, 1)
            # Horizontal
            start_h = (self.GRID_OFFSET_X, self.GRID_OFFSET_Y + i * self.BLOCK_SIZE)
            end_h = (self.GRID_OFFSET_X + self.GRID_WIDTH_PX, self.GRID_OFFSET_Y + i * self.BLOCK_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID_LINES, start_h, end_h, 1)
            
        # Draw blocks
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                color_idx = self.grid[r][c]
                if color_idx != -1:
                    rect = pygame.Rect(self.GRID_OFFSET_X + c * self.BLOCK_SIZE,
                                       self.GRID_OFFSET_Y + r * self.BLOCK_SIZE,
                                       self.BLOCK_SIZE, self.BLOCK_SIZE)
                    
                    color = self.BLOCK_COLORS[color_idx]
                    shadow_color = tuple(max(0, val - 40) for val in color)
                    highlight_color = tuple(min(255, val + 40) for val in color)

                    pygame.draw.rect(self.screen, shadow_color, rect)
                    pygame.draw.rect(self.screen, color, rect.inflate(-4,-4))
                    pygame.draw.line(self.screen, highlight_color, rect.topleft, rect.topright, 2)
                    pygame.draw.line(self.screen, highlight_color, rect.topleft, rect.bottomleft, 2)


        # Draw cursor
        cursor_x, cursor_y = self.cursor_pos
        cursor_rect = pygame.Rect(self.GRID_OFFSET_X + cursor_x * self.BLOCK_SIZE,
                                  self.GRID_OFFSET_Y + cursor_y * self.BLOCK_SIZE,
                                  self.BLOCK_SIZE, self.BLOCK_SIZE)

        pulse = (math.sin(self.steps * 0.3) + 1) / 2 # 0 to 1
        alpha = 100 + pulse * 155
        glow_color = (*self.COLOR_WHITE[:3], alpha)
        
        # Use a separate surface for the glowing cursor for alpha blending
        glow_surf = pygame.Surface((self.BLOCK_SIZE, self.BLOCK_SIZE), pygame.SRCALPHA)
        pygame.draw.rect(glow_surf, glow_color, (0, 0, self.BLOCK_SIZE, self.BLOCK_SIZE), border_radius=4)
        pygame.draw.rect(glow_surf, (0,0,0,0), (3, 3, self.BLOCK_SIZE-6, self.BLOCK_SIZE-6), border_radius=4) # Punch a hole
        self.screen.blit(glow_surf, cursor_rect.topleft)

        # Draw particles
        for p in self.particles:
            fade_ratio = p['life'] / p['max_life']
            radius = int(fade_ratio * 4)
            if radius > 0:
                color = p['color']
                alpha_color = (*color, int(fade_ratio * 255))
                pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), radius, alpha_color)

    def _render_ui(self):
        # Score
        score_text = self.font_main.render(f"SCORE: {self.score}", True, self.COLOR_WHITE)
        self.screen.blit(score_text, (20, 10))

        # Timer
        time_sec = max(0, self.time_remaining / self.FPS)
        time_color = self.COLOR_WHITE if time_sec > 10 else (255, 80, 80)
        time_text = self.font_main.render(f"TIME: {time_sec:.1f}", True, time_color)
        time_rect = time_text.get_rect(topright=(self.WIDTH - 20, 10))
        self.screen.blit(time_text, time_rect)
        
        # Game Over message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            if self._is_board_clear():
                msg = "YOU WIN!"
                color = self.COLOR_GOLD
            else:
                msg = "TIME UP!"
                color = (200, 50, 50)
                
            end_text = self.font_game_over.render(msg, True, color)
            end_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, end_rect)


    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_remaining_seconds": round(self.time_remaining / self.FPS, 2)
        }

    def close(self):
        pygame.quit()

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

# Example usage to run and visualize the game
if __name__ == '__main__':
    env = GameEnv()
    obs, info = env.reset()
    
    # --- Pygame window for visualization ---
    pygame.display.set_caption("Block Breaker")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        # --- Human Controls ---
        movement = 0 # No-op
        space = 0
        shift = 0
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1
            
        action = [movement, space, shift]
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment.")
                obs, info = env.reset()
                total_reward = 0

        # --- Step Environment ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # --- Render ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated or truncated:
            print(f"Episode finished. Total Reward: {total_reward}, Score: {info['score']}, Steps: {info['steps']}")
            # Wait for a moment before auto-resetting, or wait for 'r' key
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0

        clock.tick(env.FPS)
        
    env.close()
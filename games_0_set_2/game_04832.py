
# Generated: 2025-08-28T03:09:33.696981
# Source Brief: brief_04832.md
# Brief Index: 4832

        
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
        "Controls: Arrows to move cursor. Space to select a gem, then move and "
        "press Space on an adjacent gem to swap."
    )

    game_description = (
        "Match-3 puzzle game. Swap adjacent gems to create lines of 3 or more. "
        "Race against the clock to make 10 matches!"
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_SIZE = 10
        self.NUM_GEM_TYPES = 5
        self.TIME_LIMIT = 180  # seconds
        self.MAX_STEPS = 1000
        self.TARGET_SETS = 10

        self.CELL_SIZE = self.HEIGHT // self.GRID_SIZE
        self.GRID_WIDTH = self.CELL_SIZE * self.GRID_SIZE
        self.GRID_OFFSET_X = (self.WIDTH - self.GRID_WIDTH) // 2
        self.GRID_OFFSET_Y = (self.HEIGHT - self.GRID_WIDTH) // 2

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
        self.font_ui = pygame.font.SysFont("sans-serif", 24, bold=True)
        self.font_big = pygame.font.SysFont("sans-serif", 48, bold=True)

        # Colors
        self.COLOR_BG = (15, 20, 35)
        self.COLOR_GRID = (30, 40, 60)
        self.COLOR_CURSOR = (255, 255, 0)
        self.COLOR_SELECTION = (255, 255, 255)
        self.COLOR_TEXT = (220, 220, 230)
        self.COLOR_TEXT_WARN = (255, 100, 100)
        self.GEM_COLORS = [
            (255, 80, 80),   # Red
            (80, 120, 255),  # Blue
            (80, 255, 80),   # Green
            (255, 255, 80),  # Yellow
            (200, 80, 255),  # Purple
        ]

        # Initialize state variables
        self.grid = None
        self.cursor_pos = None
        self.selected_gem = None
        self.steps = 0
        self.score = 0
        self.sets_collected = 0
        self.time_remaining = 0.0
        self.game_over = False
        self.game_won = False
        self.particles = []
        self.last_action_reward = 0

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.sets_collected = 0
        self.time_remaining = float(self.TIME_LIMIT)
        self.game_over = False
        self.game_won = False
        self.last_action_reward = 0

        self.cursor_pos = [self.GRID_SIZE // 2, self.GRID_SIZE // 2]
        self.selected_gem = None
        self.particles = []
        
        self._initialize_grid()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_pressed, _ = action[0], action[1] == 1, action[2] == 1
        
        reward = -0.1  # Cost of making a move
        self.steps += 1
        self.time_remaining = max(0.0, self.time_remaining - 1.0)
        
        # 1. Handle cursor movement
        if movement == 1: self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1)
        elif movement == 2: self.cursor_pos[1] = min(self.GRID_SIZE - 1, self.cursor_pos[1] + 1)
        elif movement == 3: self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1)
        elif movement == 4: self.cursor_pos[0] = min(self.GRID_SIZE - 1, self.cursor_pos[0] + 1)

        # 2. Handle selection and swapping
        if space_pressed:
            if self.selected_gem is None:
                self.selected_gem = list(self.cursor_pos)
                # sfx: select_gem
            else:
                dist = abs(self.selected_gem[0] - self.cursor_pos[0]) + abs(self.selected_gem[1] - self.cursor_pos[1])
                if dist == 1: # Adjacent swap
                    # sfx: swap_attempt
                    match_found, match_reward, sets_made = self._attempt_swap()
                    if match_found:
                        reward += match_reward
                        self.sets_collected += sets_made
                    self.selected_gem = None
                else: # New selection
                    self.selected_gem = list(self.cursor_pos)
                    # sfx: select_gem
        
        self.last_action_reward = reward
        self.score += reward

        # 3. Check for termination
        terminated = False
        if self.sets_collected >= self.TARGET_SETS:
            self.game_over = True
            self.game_won = True
            self.score += 100
            reward += 100
            terminated = True
            # sfx: game_win
        elif self.time_remaining <= 0 or self.steps >= self.MAX_STEPS:
            self.game_over = True
            self.score -= 50
            reward -= 50
            terminated = True
            # sfx: game_lose

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _initialize_grid(self):
        self.grid = self.np_random.integers(1, self.NUM_GEM_TYPES + 1, size=(self.GRID_SIZE, self.GRID_SIZE))
        while True:
            matches = self._find_matches()
            if not matches:
                break
            for r, c in matches:
                self.grid[r, c] = self.np_random.integers(1, self.NUM_GEM_TYPES + 1)

    def _attempt_swap(self):
        r1, c1 = self.selected_gem
        r2, c2 = self.cursor_pos
        
        # Perform swap
        self.grid[r1, c1], self.grid[r2, c2] = self.grid[r2, c2], self.grid[r1, c1]

        total_reward = 0
        total_sets = 0
        all_matches = set()
        
        # Chain reaction loop
        while True:
            matches = self._find_matches()
            if not matches:
                break

            all_matches.update(matches)
            
            # Calculate reward for this cascade
            for r, c in matches:
                total_reward += 1 # +1 for each gem matched
                self._create_particles(r, c)
            
            # Count sets (a "set" is a continuous line of 3+)
            # This is a simplification; a more complex system would count each line.
            # For RL purposes, counting total matched gems is more stable.
            # We add +10 per chain reaction event.
            total_reward += 10
            total_sets += 1
            # sfx: match_success

            # Remove gems and apply gravity
            for r, c in matches:
                self.grid[r, c] = 0
            self._apply_gravity()
            self._fill_top_rows()

        if not all_matches:
            # Invalid swap, swap back
            self.grid[r1, c1], self.grid[r2, c2] = self.grid[r2, c2], self.grid[r1, c1]
            # sfx: swap_fail
            return False, 0, 0
        
        return True, total_reward, total_sets

    def _find_matches(self):
        matches = set()
        # Horizontal matches
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE - 2):
                if self.grid[r, c] == self.grid[r, c+1] == self.grid[r, c+2] != 0:
                    matches.add((r, c)); matches.add((r, c+1)); matches.add((r, c+2))
        
        # Vertical matches
        for c in range(self.GRID_SIZE):
            for r in range(self.GRID_SIZE - 2):
                if self.grid[r, c] == self.grid[r+1, c] == self.grid[r+2, c] != 0:
                    matches.add((r, c)); matches.add((r+1, c)); matches.add((r+2, c))
        return matches

    def _apply_gravity(self):
        for c in range(self.GRID_SIZE):
            empty_row = self.GRID_SIZE - 1
            for r in range(self.GRID_SIZE - 1, -1, -1):
                if self.grid[r, c] != 0:
                    self.grid[empty_row, c], self.grid[r, c] = self.grid[r, c], self.grid[empty_row, c]
                    empty_row -= 1
    
    def _fill_top_rows(self):
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                if self.grid[r, c] == 0:
                    self.grid[r, c] = self.np_random.integers(1, self.NUM_GEM_TYPES + 1)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_remaining": self.time_remaining,
            "sets_collected": self.sets_collected,
        }

    def _render_game(self):
        # Draw grid lines
        for i in range(self.GRID_SIZE + 1):
            # Vertical
            start_pos = (self.GRID_OFFSET_X + i * self.CELL_SIZE, self.GRID_OFFSET_Y)
            end_pos = (self.GRID_OFFSET_X + i * self.CELL_SIZE, self.GRID_OFFSET_Y + self.GRID_WIDTH)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos, 1)
            # Horizontal
            start_pos = (self.GRID_OFFSET_X, self.GRID_OFFSET_Y + i * self.CELL_SIZE)
            end_pos = (self.GRID_OFFSET_X + self.GRID_WIDTH, self.GRID_OFFSET_Y + i * self.CELL_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos, 1)

        # Draw gems
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                gem_type = self.grid[r, c]
                if gem_type > 0:
                    self._draw_gem(c, r, gem_type)

        # Draw selection highlight
        if self.selected_gem:
            c, r = self.selected_gem
            rect = pygame.Rect(
                self.GRID_OFFSET_X + c * self.CELL_SIZE,
                self.GRID_OFFSET_Y + r * self.CELL_SIZE,
                self.CELL_SIZE, self.CELL_SIZE
            )
            pygame.draw.rect(self.screen, self.COLOR_SELECTION, rect, 3)

        # Draw cursor
        c, r = self.cursor_pos
        cursor_rect = pygame.Rect(
            self.GRID_OFFSET_X + c * self.CELL_SIZE,
            self.GRID_OFFSET_Y + r * self.CELL_SIZE,
            self.CELL_SIZE, self.CELL_SIZE
        )
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, 3)

        self._update_and_draw_particles()

    def _draw_gem(self, c, r, gem_type):
        x = self.GRID_OFFSET_X + c * self.CELL_SIZE + self.CELL_SIZE // 2
        y = self.GRID_OFFSET_Y + r * self.CELL_SIZE + self.CELL_SIZE // 2
        radius = int(self.CELL_SIZE * 0.4)
        color = self.GEM_COLORS[gem_type - 1]
        
        if gem_type == 1: # Circle
            pygame.gfxdraw.aacircle(self.screen, x, y, radius, color)
            pygame.gfxdraw.filled_circle(self.screen, x, y, radius, color)
        elif gem_type == 2: # Square
            rect = pygame.Rect(x - radius, y - radius, radius*2, radius*2)
            pygame.draw.rect(self.screen, color, rect)
        elif gem_type == 3: # Triangle
            points = [
                (x, y - radius),
                (x - radius, y + radius * 0.7),
                (x + radius, y + radius * 0.7),
            ]
            pygame.gfxdraw.aapolygon(self.screen, points, color)
            pygame.gfxdraw.filled_polygon(self.screen, points, color)
        elif gem_type == 4: # Diamond
            points = [
                (x, y - radius), (x + radius, y),
                (x, y + radius), (x - radius, y)
            ]
            pygame.gfxdraw.aapolygon(self.screen, points, color)
            pygame.gfxdraw.filled_polygon(self.screen, points, color)
        elif gem_type == 5: # Hexagon
            points = []
            for i in range(6):
                angle = math.pi / 3 * i
                px = x + int(radius * math.cos(angle))
                py = y + int(radius * math.sin(angle))
                points.append((px, py))
            pygame.gfxdraw.aapolygon(self.screen, points, color)
            pygame.gfxdraw.filled_polygon(self.screen, points, color)

    def _create_particles(self, r, c):
        gem_type = self.grid[r, c]
        if gem_type <= 0: return
        
        color = self.GEM_COLORS[gem_type - 1]
        px = self.GRID_OFFSET_X + c * self.CELL_SIZE + self.CELL_SIZE // 2
        py = self.GRID_OFFSET_Y + r * self.CELL_SIZE + self.CELL_SIZE // 2
        
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifespan = self.np_random.uniform(15, 30)
            radius = self.np_random.uniform(2, 5)
            self.particles.append({'pos': [px, py], 'vel': vel, 'color': color, 'radius': radius, 'life': lifespan})

    def _update_and_draw_particles(self):
        # In auto_advance=False, we update particles during render for visual effect
        dt = 1 # A fixed update step
        for p in self.particles:
            p['pos'][0] += p['vel'][0] * dt
            p['pos'][1] += p['vel'][1] * dt
            p['life'] -= 1
            p['radius'] -= 0.1
        
        self.particles = [p for p in self.particles if p['life'] > 0 and p['radius'] > 0]
        
        for p in self.particles:
            pos = (int(p['pos'][0]), int(p['pos'][1]))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], int(p['radius']), p['color'])

    def _render_ui(self):
        # Score
        score_surf = self.font_ui.render(f"SCORE: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_surf, (10, 10))

        # Time
        time_color = self.COLOR_TEXT if self.time_remaining > 30 else self.COLOR_TEXT_WARN
        time_surf = self.font_ui.render(f"TIME: {int(self.time_remaining)}", True, time_color)
        time_rect = time_surf.get_rect(topright=(self.WIDTH - 10, 10))
        self.screen.blit(time_surf, time_rect)

        # Sets Collected
        sets_text = f"SETS: {self.sets_collected} / {self.TARGET_SETS}"
        sets_surf = self.font_ui.render(sets_text, True, self.COLOR_TEXT)
        sets_rect = sets_surf.get_rect(midbottom=(self.WIDTH / 2, self.HEIGHT - 10))
        self.screen.blit(sets_surf, (self.GRID_OFFSET_X - sets_surf.get_width() - 20, self.HEIGHT/2 - 12))

        # Game Over message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            msg = "YOU WIN!" if self.game_won else "TIME UP!"
            msg_surf = self.font_big.render(msg, True, self.COLOR_TEXT)
            msg_rect = msg_surf.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(msg_surf, msg_rect)

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
    env = GameEnv()
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Gem Matcher")
    clock = pygame.time.Clock()
    
    running = True
    while running:
        # Action defaults to no-op
        action = [0, 0, 0] # move, space, shift
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP: action[0] = 1
                elif event.key == pygame.K_DOWN: action[0] = 2
                elif event.key == pygame.K_LEFT: action[0] = 3
                elif event.key == pygame.K_RIGHT: action[0] = 4
                elif event.key == pygame.K_SPACE: action[1] = 1
                elif event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT: action[2] = 1
                elif event.key == pygame.K_r: # Reset game
                    obs, info = env.reset()
                    continue
                
                obs, reward, terminated, truncated, info = env.step(action)
                
                print(f"Action: {action}, Reward: {reward:.2f}, Score: {info['score']:.2f}, Terminated: {terminated}")
                
                if terminated:
                    print("Game Over! Press 'R' to reset.")

        # Draw the current observation to the screen
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(30)
        
    pygame.quit()
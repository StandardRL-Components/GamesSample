
# Generated: 2025-08-27T17:24:28.545117
# Source Brief: brief_01522.md
# Brief Index: 1522

        
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
        "Controls: Use arrow keys to move the cursor. Press space to place the current card."
    )

    game_description = (
        "Place cards on the grid. Placing a card next to one or more cards of the same suit "
        "creates a match, clearing all connected cards of that suit. Clear the board before time runs out!"
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_COLS, self.GRID_ROWS = 8, 6
        self.MAX_STEPS = 1800  # 60 seconds * 30 FPS
        self.TIME_LIMIT = self.MAX_STEPS

        # --- Colors ---
        self.COLOR_BG = (20, 30, 40)
        self.COLOR_GRID = (40, 50, 60)
        self.COLOR_TEXT = (230, 230, 230)
        self.COLOR_CURSOR = (255, 220, 0)
        self.SUIT_COLORS = [
            (255, 80, 80),   # Red
            (80, 255, 80),   # Green
            (80, 120, 255),  # Blue
        ]
        self.EMPTY_CELL = -1

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
        self.font_main = pygame.font.SysFont("monospace", 24, bold=True)
        self.font_small = pygame.font.SysFont("monospace", 16)

        # --- Grid Layout ---
        self.GRID_AREA_WIDTH = 480
        self.GRID_AREA_HEIGHT = 360
        self.GRID_OFFSET_X = (self.WIDTH - self.GRID_AREA_WIDTH) // 2
        self.GRID_OFFSET_Y = (self.HEIGHT - self.GRID_AREA_HEIGHT) + 10
        self.CELL_WIDTH = self.GRID_AREA_WIDTH // self.GRID_COLS
        self.CELL_HEIGHT = self.GRID_AREA_HEIGHT // self.GRID_ROWS

        # --- Game State (initialized in reset) ---
        self.np_random = None
        self.grid = None
        self.cursor_pos = None
        self.current_card_suit = None
        self.score = 0
        self.steps = 0
        self.time_remaining = 0
        self.game_over = False
        self.last_space_held = False
        self.particles = []
        self.score_popup = 0 # for animation

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)
        else:
            # Fallback if no seed is provided
            if self.np_random is None:
                self.np_random, _ = gym.utils.seeding.np_random(random.randint(0, 1e9))


        self.grid = np.full((self.GRID_ROWS, self.GRID_COLS), self.EMPTY_CELL, dtype=int)
        self.cursor_pos = [self.GRID_COLS // 2, self.GRID_ROWS // 2]
        self.current_card_suit = self.np_random.integers(0, len(self.SUIT_COLORS))
        
        # Pre-populate board for interesting starts
        num_initial_cards = self.np_random.integers(3, 6)
        for _ in range(num_initial_cards):
            x, y = self.np_random.integers(0, self.GRID_COLS), self.np_random.integers(0, self.GRID_ROWS)
            suit = self.np_random.integers(0, len(self.SUIT_COLORS))
            if self.grid[y, x] == self.EMPTY_CELL:
                self.grid[y, x] = suit

        self.score = 0
        self.steps = 0
        self.time_remaining = self.TIME_LIMIT
        self.game_over = False
        self.last_space_held = False
        self.particles = []
        self.score_popup = 0

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.auto_advance:
            self.clock.tick(30)

        reward = 0
        terminated = self.game_over

        if not terminated:
            movement = action[0]
            space_held = action[1] == 1

            self._handle_input(movement, space_held)
            reward += self._update_game_state(space_held)
            self._update_particles()
            
            self.steps += 1
            self.time_remaining -= 1

            # Check termination conditions
            if self._is_board_clear():
                reward += 100
                self.game_over = True
            elif self.time_remaining <= 0:
                reward -= 50
                self.game_over = True
            elif self._no_valid_moves():
                reward -= 50
                self.game_over = True
            elif self.steps >= self.MAX_STEPS:
                self.game_over = True
            
            terminated = self.game_over
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, movement, space_held):
        if movement == 1:  # Up
            self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1)
        elif movement == 2:  # Down
            self.cursor_pos[1] = min(self.GRID_ROWS - 1, self.cursor_pos[1] + 1)
        elif movement == 3:  # Left
            self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1)
        elif movement == 4:  # Right
            self.cursor_pos[0] = min(self.GRID_COLS - 1, self.cursor_pos[0] + 1)

    def _update_game_state(self, space_held):
        reward = 0
        space_press = space_held and not self.last_space_held
        self.last_space_held = space_held

        if space_press:
            cx, cy = self.cursor_pos
            if self.grid[cy, cx] == self.EMPTY_CELL:
                # SFX placeholder: // Sound: CardPlace.wav
                
                # Placement reward
                has_potential_match = False
                for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < self.GRID_COLS and 0 <= ny < self.GRID_ROWS:
                        if self.grid[ny, nx] == self.current_card_suit:
                            has_potential_match = True
                            break
                reward += 1.0 if has_potential_match else -0.2

                self.grid[cy, cx] = self.current_card_suit
                
                # Check for matches
                matches = self._find_matches(cx, cy)
                if len(matches) > 1:
                    # SFX placeholder: // Sound: MatchClear.wav
                    match_reward = 5 * len(matches)
                    reward += match_reward
                    self.score += match_reward
                    self.score_popup = 15 # frames for popup animation

                    for x, y in matches:
                        self.grid[y, x] = self.EMPTY_CELL
                        self._create_particles(x, y, self.current_card_suit)
                
                self.current_card_suit = self.np_random.integers(0, len(self.SUIT_COLORS))
        
        return reward

    def _find_matches(self, start_x, start_y):
        target_suit = self.grid[start_y, start_x]
        if target_suit == self.EMPTY_CELL:
            return []

        q = [(start_x, start_y)]
        visited = set([(start_x, start_y)])
        matches = []

        while q:
            x, y = q.pop(0)
            matches.append((x, y))

            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.GRID_COLS and 0 <= ny < self.GRID_ROWS:
                    if (nx, ny) not in visited and self.grid[ny, nx] == target_suit:
                        visited.add((nx, ny))
                        q.append((nx, ny))
        return matches

    def _is_board_clear(self):
        return np.all(self.grid == self.EMPTY_CELL)

    def _no_valid_moves(self):
        return not np.any(self.grid == self.EMPTY_CELL)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Render grid and cards
        for y in range(self.GRID_ROWS):
            for x in range(self.GRID_COLS):
                rect = pygame.Rect(
                    self.GRID_OFFSET_X + x * self.CELL_WIDTH,
                    self.GRID_OFFSET_Y + y * self.CELL_HEIGHT,
                    self.CELL_WIDTH,
                    self.CELL_HEIGHT,
                )
                
                # Draw grid cell background
                pygame.draw.rect(self.screen, self.COLOR_GRID, rect.inflate(-4, -4), border_radius=4)

                suit = self.grid[y, x]
                if suit != self.EMPTY_CELL:
                    self._draw_card(rect.center, self.SUIT_COLORS[suit])

        # Render particles
        for p in self.particles:
            p['alpha'] -= 10
            if p['alpha'] > 0:
                p['pos'][0] += p['vel'][0]
                p['pos'][1] += p['vel'][1]
                p['size'] += 0.2
                
                color = list(p['color'])
                color_with_alpha = (*color, int(p['alpha']))
                
                radius = int(p['size'] / 2)
                if radius > 0:
                    pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), radius, color_with_alpha)


        # Render cursor
        cx, cy = self.cursor_pos
        cursor_rect = pygame.Rect(
            self.GRID_OFFSET_X + cx * self.CELL_WIDTH,
            self.GRID_OFFSET_Y + cy * self.CELL_HEIGHT,
            self.CELL_WIDTH,
            self.CELL_HEIGHT,
        )
        pulse = (math.sin(self.steps * 0.3) + 1) / 2
        line_width = int(2 + pulse * 2)
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, line_width, border_radius=6)

    def _draw_card(self, center_pos, color, scale=1.0):
        card_size = min(self.CELL_WIDTH, self.CELL_HEIGHT) * 0.7 * scale
        card_rect = pygame.Rect(0, 0, card_size, card_size)
        card_rect.center = center_pos

        pygame.gfxdraw.box(self.screen, card_rect, (*color, 150))
        pygame.gfxdraw.rectangle(self.screen, card_rect, color)

    def _render_ui(self):
        # Score display
        if self.score_popup > 0:
            self.score_popup -= 1
        
        score_scale = 1.0 + 0.3 * (self.score_popup / 15)
        score_font_size = int(24 * score_scale)
        score_font = pygame.font.SysFont("monospace", score_font_size, bold=True)
        score_text = score_font.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        score_rect = score_text.get_rect(topleft=(20, 15))
        self.screen.blit(score_text, score_rect)

        # Timer display
        time_percent = max(0, self.time_remaining / self.TIME_LIMIT)
        time_text = self.font_main.render(f"TIME", True, self.COLOR_TEXT)
        time_rect = time_text.get_rect(topright=(self.WIDTH - 20, 15))
        self.screen.blit(time_text, time_rect)

        bar_width = 150
        bar_height = 10
        bar_x = self.WIDTH - 20 - bar_width
        bar_y = time_rect.bottom + 5
        
        # Lerp color from green to red
        time_color = (
            min(255, 255 * (1 - time_percent) * 2),
            min(255, 255 * time_percent * 2),
            50
        )

        pygame.draw.rect(self.screen, self.COLOR_GRID, (bar_x, bar_y, bar_width, bar_height), border_radius=3)
        pygame.draw.rect(self.screen, time_color, (bar_x, bar_y, bar_width * time_percent, bar_height), border_radius=3)

        # Next card display
        next_text = self.font_small.render("NEXT:", True, self.COLOR_TEXT)
        self.screen.blit(next_text, (self.GRID_OFFSET_X, self.GRID_OFFSET_Y - 30))
        self._draw_card(
            (self.GRID_OFFSET_X + 60, self.GRID_OFFSET_Y - 22),
            self.SUIT_COLORS[self.current_card_suit],
            scale=0.6
        )

    def _create_particles(self, grid_x, grid_y, suit_index):
        center_x = self.GRID_OFFSET_X + grid_x * self.CELL_WIDTH + self.CELL_WIDTH / 2
        center_y = self.GRID_OFFSET_Y + grid_y * self.CELL_HEIGHT + self.CELL_HEIGHT / 2
        color = self.SUIT_COLORS[suit_index]

        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            self.particles.append({
                'pos': [center_x, center_y],
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'alpha': 255,
                'color': color,
                'size': self.np_random.uniform(3, 8)
            })

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['alpha'] > 0]

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_remaining_steps": self.time_remaining,
            "board_clear": self._is_board_clear(),
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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")


if __name__ == "__main__":
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # --- Pygame setup for human play ---
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Card Match Grid")
    clock = pygame.time.Clock()
    
    action = env.action_space.sample()
    action.fill(0)

    print("\n" + "="*30)
    print(env.game_description)
    print(env.user_guide)
    print("="*30 + "\n")

    while not done:
        # --- Human input mapping ---
        movement, space, shift = 0, 0, 0
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1

        action = np.array([movement, space, shift])

        # --- Gym step ---
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        if reward != 0:
            print(f"Step: {info['steps']}, Score: {info['score']}, Reward: {reward:.2f}")

        # --- Render to screen ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Event handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("--- RESETTING ---")
                obs, info = env.reset()
        
        clock.tick(30)

    print(f"Game Over! Final Score: {info['score']}")
    env.close()
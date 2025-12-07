import gymnasium as gym
import os
import pygame
import numpy as np
import pygame.gfxdraw
import math
import random
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from gymnasium.spaces import MultiDiscrete

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Collect coins on a grid that periodically shifts. "
        "Anticipate the directional shift warnings to move correctly and avoid penalties."
    )
    user_guide = (
        "Controls: Use the arrow keys (↑↓←→) to move and collect coins. "
        "Match the direction of the grid shift during warnings to avoid a penalty."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    UI_HEIGHT = 40

    GRID_ROWS = 8
    GRID_COLS = 12
    CELL_SIZE = 40

    GRID_WIDTH = GRID_COLS * CELL_SIZE
    GRID_HEIGHT = GRID_ROWS * CELL_SIZE
    GRID_OFFSET_X = (SCREEN_WIDTH - GRID_WIDTH) // 2
    GRID_OFFSET_Y = (SCREEN_HEIGHT - GRID_HEIGHT) // 2 + UI_HEIGHT // 2

    # Colors
    COLOR_BG = (15, 20, 30)
    COLOR_GRID_LINES = (40, 50, 60)
    COLOR_UI_BG = (10, 15, 25)
    COLOR_TEXT = (220, 220, 240)
    COLOR_TEXT_WARN = (255, 100, 100)

    COLOR_PLAYER = (50, 150, 255)
    COLOR_PLAYER_GLOW = (50, 150, 255, 50)
    COLOR_COIN = (255, 200, 0)
    COLOR_COIN_GLOW = (255, 200, 0, 60)
    COLOR_SHIFT_WARN = (0, 255, 100, 30)
    COLOR_SHIFT_ARROW = (0, 255, 100)
    COLOR_PENALTY_FLASH = (255, 50, 50, 150)

    # Game Rules
    WIN_SCORE = 100
    MAX_STEPS = 1800  # 60 seconds at 30 FPS
    
    SHIFT_INTERVAL = 300 # Every 10 seconds
    SHIFT_WARN_TIME = 90 # Warning for last 3 seconds
    
    COIN_COUNT = 15
    REWARD_COIN = 5
    REWARD_PENALTY = -2
    REWARD_WIN = 100
    
    INTERPOLATION_FACTOR = 0.4

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
        self.font_main = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 16)
        
        # Initialize state variables to prevent AttributeError
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_grid_pos = [0, 0]
        self.player_visual_pos = [0, 0]
        self.coins = []
        self.particles = []
        self.shift_timer = 0
        self.shift_direction = 0  # 0=None, 1=Up, 2=Down, 3=Left, 4=Right
        self.penalty_flash_timer = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False

        self.player_grid_pos = [self.GRID_COLS // 2, self.GRID_ROWS // 2]
        self.player_visual_pos = self._grid_to_pixel(self.player_grid_pos)
        
        self.coins = []
        self._spawn_coins(self.COIN_COUNT)

        self.particles = []
        self.shift_timer = self.SHIFT_INTERVAL
        self.shift_direction = 0
        self.penalty_flash_timer = 0

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0
        
        self.steps += 1
        self._update_timers()
        
        # --- Grid Shift Logic ---
        if self.shift_timer <= self.SHIFT_WARN_TIME and self.shift_direction == 0:
            self.shift_direction = self.np_random.integers(1, 5)

        if self.shift_timer <= 0:
            self._execute_shift()
            # sfx: grid_shift_sound
            self.shift_timer = self.SHIFT_INTERVAL
            self.shift_direction = 0

        # --- Player Movement ---
        if movement != 0:
            # Penalty for unsynchronized movement during warning phase
            if self.shift_direction != 0 and movement != self.shift_direction:
                reward += self.REWARD_PENALTY
                self.penalty_flash_timer = 10
                # sfx: penalty_sound
            
            dx, dy = 0, 0
            if movement == 1: dy = -1  # Up
            elif movement == 2: dy = 1  # Down
            elif movement == 3: dx = -1  # Left
            elif movement == 4: dx = 1  # Right
            
            self.player_grid_pos[0] = np.clip(self.player_grid_pos[0] + dx, 0, self.GRID_COLS - 1)
            self.player_grid_pos[1] = np.clip(self.player_grid_pos[1] + dy, 0, self.GRID_ROWS - 1)

        # --- Coin Collection ---
        collected_indices = []
        for i, coin_pos in enumerate(self.coins):
            if coin_pos == self.player_grid_pos:
                collected_indices.append(i)
                self.score += 1
                reward += self.REWARD_COIN
                # sfx: coin_collect_sound
                self._create_particles(self._grid_to_pixel(coin_pos), self.COLOR_COIN)

        if collected_indices:
            self.coins = [c for i, c in enumerate(self.coins) if i not in collected_indices]
            self._spawn_coins(len(collected_indices))

        # --- Termination ---
        terminated = self.steps >= self.MAX_STEPS or self.score >= self.WIN_SCORE
        truncated = False
        if terminated and not self.game_over:
            self.game_over = True
            if self.score >= self.WIN_SCORE:
                reward += self.REWARD_WIN
                # sfx: win_sound
            else:
                # sfx: lose_sound
                pass

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _update_timers(self):
        self.shift_timer -= 1
        if self.penalty_flash_timer > 0:
            self.penalty_flash_timer -= 1
        
        # Update particles
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1 # Gravity
            p['life'] -= 1

        # Update player visual interpolation
        target_pixel_pos = self._grid_to_pixel(self.player_grid_pos)
        self.player_visual_pos[0] += (target_pixel_pos[0] - self.player_visual_pos[0]) * self.INTERPOLATION_FACTOR
        self.player_visual_pos[1] += (target_pixel_pos[1] - self.player_visual_pos[1]) * self.INTERPOLATION_FACTOR

    def _execute_shift(self):
        if self.shift_direction == 0: return
        dx, dy = 0, 0
        if self.shift_direction == 1: dy = -1
        elif self.shift_direction == 2: dy = 1
        elif self.shift_direction == 3: dx = -1
        elif self.shift_direction == 4: dx = 1

        for coin in self.coins:
            coin[0] = (coin[0] + dx) % self.GRID_COLS
            coin[1] = (coin[1] + dy) % self.GRID_ROWS

    def _spawn_coins(self, num_to_spawn):
        occupied_cells = set(tuple(c) for c in self.coins)
        occupied_cells.add(tuple(self.player_grid_pos))
        
        for _ in range(num_to_spawn):
            if len(occupied_cells) >= self.GRID_COLS * self.GRID_ROWS:
                break
            
            pos = [self.np_random.integers(0, self.GRID_COLS), self.np_random.integers(0, self.GRID_ROWS)]
            while tuple(pos) in occupied_cells:
                pos = [self.np_random.integers(0, self.GRID_COLS), self.np_random.integers(0, self.GRID_ROWS)]
            
            self.coins.append(pos)
            occupied_cells.add(tuple(pos))

    def _grid_to_pixel(self, grid_pos):
        x = self.GRID_OFFSET_X + grid_pos[0] * self.CELL_SIZE + self.CELL_SIZE // 2
        y = self.GRID_OFFSET_Y + grid_pos[1] * self.CELL_SIZE + self.CELL_SIZE // 2
        return [x, y]

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid lines
        for i in range(self.GRID_COLS + 1):
            x = self.GRID_OFFSET_X + i * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID_LINES, (x, self.GRID_OFFSET_Y), (x, self.GRID_OFFSET_Y + self.GRID_HEIGHT))
        for i in range(self.GRID_ROWS + 1):
            y = self.GRID_OFFSET_Y + i * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID_LINES, (self.GRID_OFFSET_X, y), (self.GRID_OFFSET_X + self.GRID_WIDTH, y))

        # Shift warning
        if self.shift_direction != 0:
            self._render_shift_warning()

        # Draw coins
        for coin_pos in self.coins:
            px, py = self._grid_to_pixel(coin_pos)
            radius = int(self.CELL_SIZE * 0.3)
            pygame.gfxdraw.filled_circle(self.screen, int(px), int(py), radius + 4, self.COLOR_COIN_GLOW)
            pygame.gfxdraw.aacircle(self.screen, int(px), int(py), radius, self.COLOR_COIN)
            pygame.gfxdraw.filled_circle(self.screen, int(px), int(py), radius, self.COLOR_COIN)

        # Draw particles
        for p in self.particles:
            alpha = max(0, 255 * (p['life'] / p['max_life']))
            color = (*p['color'], alpha)
            pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), int(p['size']), color)
            
        # Draw player
        px, py = self.player_visual_pos
        size = int(self.CELL_SIZE * 0.7)
        player_rect = pygame.Rect(px - size // 2, py - size // 2, size, size)
        
        glow_size = int(size * 1.8)
        glow_surf = pygame.Surface((glow_size, glow_size), pygame.SRCALPHA)
        pygame.draw.rect(glow_surf, self.COLOR_PLAYER_GLOW, (0, 0, glow_size, glow_size), border_radius=int(glow_size*0.3))
        self.screen.blit(glow_surf, (px - glow_size // 2, py - glow_size // 2))
        
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=int(size*0.2))

        # Penalty flash
        if self.penalty_flash_timer > 0:
            flash_alpha = 150 * (self.penalty_flash_timer / 10)
            flash_surface = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            flash_surface.fill((*self.COLOR_PENALTY_FLASH[:3], flash_alpha))
            self.screen.blit(flash_surface, (0, 0))

    def _render_shift_warning(self):
        # Semi-transparent overlay
        warn_surface = pygame.Surface((self.GRID_WIDTH, self.GRID_HEIGHT), pygame.SRCALPHA)
        alpha = 30 + 30 * math.sin(self.steps * 0.2) # Pulsing effect
        warn_surface.fill((*self.COLOR_SHIFT_WARN[:3], alpha))
        self.screen.blit(warn_surface, (self.GRID_OFFSET_X, self.GRID_OFFSET_Y))
        
        # Draw arrows
        arrow_len = 20
        arrow_thick = 4
        
        if self.shift_direction == 1: # Up
            for i in range(self.GRID_COLS):
                cx = self.GRID_OFFSET_X + i * self.CELL_SIZE + self.CELL_SIZE // 2
                cy = self.GRID_OFFSET_Y + self.CELL_SIZE // 4
                pygame.draw.line(self.screen, self.COLOR_SHIFT_ARROW, (cx, cy), (cx-arrow_len//3, cy+arrow_len//3), arrow_thick)
                pygame.draw.line(self.screen, self.COLOR_SHIFT_ARROW, (cx, cy), (cx+arrow_len//3, cy+arrow_len//3), arrow_thick)
        elif self.shift_direction == 2: # Down
            for i in range(self.GRID_COLS):
                cx = self.GRID_OFFSET_X + i * self.CELL_SIZE + self.CELL_SIZE // 2
                cy = self.GRID_OFFSET_Y + self.GRID_HEIGHT - self.CELL_SIZE // 4
                pygame.draw.line(self.screen, self.COLOR_SHIFT_ARROW, (cx, cy), (cx-arrow_len//3, cy-arrow_len//3), arrow_thick)
                pygame.draw.line(self.screen, self.COLOR_SHIFT_ARROW, (cx, cy), (cx+arrow_len//3, cy-arrow_len//3), arrow_thick)
        elif self.shift_direction == 3: # Left
            for i in range(self.GRID_ROWS):
                cx = self.GRID_OFFSET_X + self.CELL_SIZE // 4
                cy = self.GRID_OFFSET_Y + i * self.CELL_SIZE + self.CELL_SIZE // 2
                pygame.draw.line(self.screen, self.COLOR_SHIFT_ARROW, (cx, cy), (cx+arrow_len//3, cy-arrow_len//3), arrow_thick)
                pygame.draw.line(self.screen, self.COLOR_SHIFT_ARROW, (cx, cy), (cx+arrow_len//3, cy+arrow_len//3), arrow_thick)
        elif self.shift_direction == 4: # Right
            for i in range(self.GRID_ROWS):
                cx = self.GRID_OFFSET_X + self.GRID_WIDTH - self.CELL_SIZE // 4
                cy = self.GRID_OFFSET_Y + i * self.CELL_SIZE + self.CELL_SIZE // 2
                pygame.draw.line(self.screen, self.COLOR_SHIFT_ARROW, (cx, cy), (cx-arrow_len//3, cy-arrow_len//3), arrow_thick)
                pygame.draw.line(self.screen, self.COLOR_SHIFT_ARROW, (cx, cy), (cx-arrow_len//3, cy+arrow_len//3), arrow_thick)

    def _render_ui(self):
        pygame.draw.rect(self.screen, self.COLOR_UI_BG, (0, 0, self.SCREEN_WIDTH, self.UI_HEIGHT))
        pygame.draw.line(self.screen, self.COLOR_GRID_LINES, (0, self.UI_HEIGHT-1), (self.SCREEN_WIDTH, self.UI_HEIGHT-1))

        # Score
        score_text = self.font_main.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (15, self.UI_HEIGHT // 2 - score_text.get_height() // 2))

        # Timer
        time_left = (self.MAX_STEPS - self.steps) / 30.0
        time_color = self.COLOR_TEXT if time_left > 10 else self.COLOR_TEXT_WARN
        timer_text = self.font_main.render(f"TIME: {max(0, time_left):.1f}", True, time_color)
        self.screen.blit(timer_text, (self.SCREEN_WIDTH - timer_text.get_width() - 15, self.UI_HEIGHT // 2 - timer_text.get_height() // 2))

        # Game Over Text
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            result_text_str = "YOU WIN!" if self.score >= self.WIN_SCORE else "TIME UP!"
            result_text = self.font_main.render(result_text_str, True, self.COLOR_TEXT)
            text_rect = result_text.get_rect(center=(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2))
            self.screen.blit(result_text, text_rect)

    def _create_particles(self, pos, color, count=20):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            life = random.randint(15, 30)
            self.particles.append({
                'pos': list(pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'life': life,
                'max_life': life,
                'size': random.uniform(2, 5),
                'color': color,
            })

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
        }

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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block is for manual testing and visualization.
    # It is not part of the required Gymnasium interface.
    # To run, you might need to unset the dummy video driver:
    # if "SDL_VIDEODRIVER" in os.environ:
    #     del os.environ["SDL_VIDEODRIVER"]
    
    env = GameEnv()
    env.reset()
    env.validate_implementation()
    obs, info = env.reset()
    done = False
    
    # --- Manual Play Controls ---
    # Arrow keys for movement
    # Game is turn-based, press a key to advance one step
    
    action = [0, 0, 0] # [movement, space, shift]
    
    # Setup a display for rendering
    pygame.display.init()
    display_surface = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("GameEnv")
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                # Reset action on new key press
                action = [0, 0, 0]
                
                if event.key == pygame.K_UP: action[0] = 1
                elif event.key == pygame.K_DOWN: action[0] = 2
                elif event.key == pygame.K_LEFT: action[0] = 3
                elif event.key == pygame.K_RIGHT: action[0] = 4
                elif event.key == pygame.K_SPACE: action[1] = 1
                elif event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT: action[2] = 1
                elif event.key == pygame.K_r: # Reset env
                    obs, info = env.reset()
                    done = False
                    continue
                
                if done: # If game is over, R is the only active key
                    continue

                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                print(f"Action: {action}, Reward: {reward:.2f}, Score: {info['score']}, Done: {done}")

        # Render the environment to the display
        render_surface = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_surface.blit(render_surface, (0, 0))
        
        pygame.display.flip()
        env.clock.tick(30) # Limit FPS for manual play

    env.close()
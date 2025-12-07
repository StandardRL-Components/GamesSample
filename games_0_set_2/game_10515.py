import gymnasium as gym
import os
import pygame
import numpy as np
from gymnasium.spaces import MultiDiscrete, Box
import pygame.gfxdraw
import math
import random
from collections import deque
import copy
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class Memory:
    """Helper class to store data for each tradable memory."""
    def __init__(self, name, initial_price):
        self.name = name
        self.price = initial_price
        self.price_history = deque([initial_price] * 20, maxlen=20)
        self.owned_units = 0
        self.avg_cost_basis = 0.0

    def update_price(self, volatility, trend):
        """Updates the memory's price based on volatility and a trend."""
        change_multiplier = 1.0 + random.uniform(-volatility, volatility) + trend
        self.price = max(1.0, self.price * change_multiplier)
        self.price_history.append(self.price)

    def get_price_change(self):
        """Returns the difference between the current and previous price."""
        if len(self.price_history) < 2:
            return 0
        return self.price - self.price_history[-2]

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Trade memories as commodities in a fast-paced, cyberpunk stock market. "
        "Buy low, sell high, and use your reputation to rewind time to secure your fortune."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to select a memory. "
        "Press space to buy, shift to sell, and space+shift to rewind time."
    )
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_COLS, GRID_ROWS = 4, 3
    GRID_CELL_WIDTH, GRID_CELL_HEIGHT = 120, 80
    GRID_MARGIN_X = (SCREEN_WIDTH - GRID_COLS * GRID_CELL_WIDTH) // 2
    GRID_MARGIN_Y = 60

    COLOR_BG = (10, 5, 25)
    COLOR_GRID = (30, 20, 60)
    COLOR_TEXT = (220, 220, 255)
    COLOR_NEON_BLUE = (100, 150, 255)
    COLOR_NEON_YELLOW = (255, 255, 100)
    COLOR_NEON_GREEN = (100, 255, 150)
    COLOR_NEON_RED = (255, 100, 100)

    # --- Initialization ---
    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = Box(low=0, high=255, shape=(400, 640, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("monospace", 16, bold=True)
        self.font_small = pygame.font.SysFont("monospace", 12, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 24, bold=True)

        self.render_mode = render_mode
        self._initialize_state_variables()
        self.np_random = None # Will be seeded in reset

    def _initialize_state_variables(self):
        # Game state
        self.steps = 0
        self.game_over = False
        self.capital = 0
        self.reputation = 0
        self.price_volatility = 0.0
        self.cursor_pos = [0, 0]
        self.render_cursor_pos = [0.0, 0.0]
        self.memories = []
        self.state_history = None
        self.last_action_feedback = []
        self.particles = []
        self.rewind_effect_timer = 0
        self.win_condition = 1_000_000
        self.max_steps = 5000
        
        # Reward tracking
        self.score = 0
        self.last_capital = 0

    # --- Gymnasium API ---
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._initialize_state_variables()

        self.capital = 10000
        self.last_capital = self.capital
        self.reputation = 100
        self.price_volatility = 0.001 # 0.1%
        
        MEMORY_NAMES = [
            "First Kiss", "Lost Dog", "Graduation", "Childhood Joy", "Regret", "Vengeance",
            "Serenity", "Eureka Moment", "Lucid Dream", "Forbidden Lore", "Hero's Triumph", "False Hope"
        ]
        # Use a copy to shuffle in place without affecting the original list
        shuffled_names = MEMORY_NAMES[:]
        random.shuffle(shuffled_names)
        
        self.memories = []
        for i in range(self.GRID_COLS * self.GRID_ROWS):
            initial_price = self.np_random.uniform(50, 250)
            self.memories.append(Memory(shuffled_names[i % len(shuffled_names)], initial_price))

        self.cursor_pos = [0, 0]
        cursor_screen_pos = self._get_screen_pos_for_grid(self.cursor_pos[0], self.cursor_pos[1])
        self.render_cursor_pos = [float(cursor_screen_pos[0]), float(cursor_screen_pos[1])]
        
        self.state_history = deque(maxlen=10)
        self._save_state_for_rewind()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        reward = 0
        self.last_action_feedback.clear()
        old_reputation = self.reputation

        # 1. Handle actions (Buy/Sell/Rewind)
        if space_held and shift_held:
            reward += self._perform_rewind()
        elif space_held:
            reward += self._perform_buy()
        elif shift_held:
            reward += self._perform_sell()
        else:
            self._perform_wait()

        # 2. Handle movement
        if movement != 0:
            if movement == 1: self.cursor_pos[1] -= 1 # Up
            elif movement == 2: self.cursor_pos[1] += 1 # Down
            elif movement == 3: self.cursor_pos[0] -= 1 # Left
            elif movement == 4: self.cursor_pos[0] += 1 # Right
            self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_COLS - 1)
            self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_ROWS - 1)

        # 3. Update game state (if not a rewind action)
        if not (space_held and shift_held):
            self._update_market()
            self.steps += 1
            self._save_state_for_rewind()

        # 4. Calculate rewards
        capital_change = self.capital - self.last_capital
        reward += (capital_change / 1000.0) * 0.1
        self.last_capital = self.capital

        rep_loss = old_reputation - self.reputation
        if rep_loss > 0:
            reward -= rep_loss * 0.1
        self.score += reward

        # 5. Check termination
        terminated = self._check_termination()
        truncated = False # No truncation condition specified other than steps, which is termination
        if terminated:
            if self.capital >= self.win_condition:
                reward += 100
            elif self.reputation <= 0 or self.steps >= self.max_steps:
                reward -= 100
            self.score += reward
        
        return self._get_observation(), reward, terminated, truncated, self._get_info()
    
    def _get_observation(self):
        self._update_and_draw_frame()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "capital": self.capital, "reputation": self.reputation}
    
    # --- Action Handlers ---
    def _perform_rewind(self):
        if len(self.state_history) > 1 and self.reputation >= 10:
            self.state_history.pop() # Remove current state
            last_state = self.state_history[-1] # Peek at new latest state
            
            self.memories = copy.deepcopy(last_state['memories'])
            self.capital = last_state['capital']
            self.reputation = last_state['reputation'] - 10 # Apply penalty after restoring
            self.steps = last_state['steps']
            
            self._add_feedback(f"-10 REP", self.COLOR_NEON_RED, duration=90)
            self.rewind_effect_timer = 15 # frames
            return 0
        else:
            return 0

    def _perform_buy(self):
        mem = self.memories[self.cursor_pos[1] * self.GRID_COLS + self.cursor_pos[0]]
        if self.capital >= mem.price:
            self.capital -= mem.price
            
            total_value = mem.avg_cost_basis * mem.owned_units
            mem.owned_units += 1
            mem.avg_cost_basis = (total_value + mem.price) / mem.owned_units

            mem.price *= 1.02 # Buying pressure increases price
            self._add_feedback(f"+1 {mem.name[:4]}", self.COLOR_NEON_GREEN)
            self._create_particles(self._get_cursor_screen_pos(), 10, self.COLOR_NEON_GREEN)
        else:
            self._add_feedback(f"INSUFFICIENT", self.COLOR_NEON_RED)
        return 0

    def _perform_sell(self):
        mem = self.memories[self.cursor_pos[1] * self.GRID_COLS + self.cursor_pos[0]]
        if mem.owned_units > 0:
            profit = mem.price - mem.avg_cost_basis
            reward = 1.0 if profit > 0 else 0
            
            self.capital += mem.price
            mem.owned_units -= 1
            if mem.owned_units == 0:
                mem.avg_cost_basis = 0

            mem.price *= 0.98 # Selling pressure decreases price
            self._add_feedback(f"-1 {mem.name[:4]}", self.COLOR_NEON_RED)
            self._create_particles(self._get_cursor_screen_pos(), 10, self.COLOR_NEON_RED)
            return reward
        else:
            self._add_feedback(f"NO UNITS", self.COLOR_NEON_YELLOW)
            return 0
    
    def _perform_wait(self):
        # No action, just advance time
        pass

    # --- Game Logic ---
    def _update_market(self):
        if self.steps > 0 and self.steps % 50 == 0:
            self.price_volatility = min(0.05, self.price_volatility + 0.0001)

        market_trend = self.np_random.uniform(-0.0005, 0.0005)
        for mem in self.memories:
            local_trend = self.np_random.uniform(-0.0001, 0.0001)
            mem.update_price(self.price_volatility, market_trend + local_trend)

    def _save_state_for_rewind(self):
        state = {
            'memories': copy.deepcopy(self.memories),
            'capital': self.capital,
            'reputation': self.reputation,
            'steps': self.steps,
        }
        self.state_history.append(state)

    def _check_termination(self):
        if self.capital >= self.win_condition or self.reputation <= 0 or self.steps >= self.max_steps:
            self.game_over = True
            return True
        return False
        
    # --- Rendering ---
    def _update_and_draw_frame(self):
        self.screen.fill(self.COLOR_BG)
        self._draw_background_grid()
        self._update_and_draw_particles()
        self._draw_memories()
        self._draw_cursor()
        self._draw_ui()
        self._draw_action_feedback()
        if self.rewind_effect_timer > 0:
            self._draw_rewind_effect()
            self.rewind_effect_timer -= 1
        if self.game_over:
            self._draw_game_over()
    
    def _draw_background_grid(self):
        for x in range(0, self.SCREEN_WIDTH, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT), 1)
        for y in range(0, self.SCREEN_HEIGHT, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y), 1)

    def _draw_memories(self):
        for i, mem in enumerate(self.memories):
            col = i % self.GRID_COLS
            row = i // self.GRID_COLS
            x, y = self._get_screen_pos_for_grid(col, row)
            
            rect = pygame.Rect(x, y, self.GRID_CELL_WIDTH, self.GRID_CELL_HEIGHT)
            pygame.draw.rect(self.screen, (20, 15, 45), rect, 0, 5)
            pygame.draw.rect(self.screen, self.COLOR_GRID, rect, 1, 5)

            self._render_text_with_glow(mem.name, self.font_main, self.COLOR_TEXT, (x + 10, y + 10))

            price_change = mem.get_price_change()
            price_color = self.COLOR_NEON_BLUE
            if price_change > 0.01: price_color = self.COLOR_NEON_GREEN
            elif price_change < -0.01: price_color = self.COLOR_NEON_RED
            self._render_text_with_glow(f"${mem.price:.2f}", self.font_main, price_color, (x + 10, y + 30))

            if mem.owned_units > 0:
                self._render_text_with_glow(f"OWN: {mem.owned_units}", self.font_small, self.COLOR_NEON_YELLOW, (x + 10, y + 55))
    
    def _draw_cursor(self):
        target_x, target_y = self._get_screen_pos_for_grid(self.cursor_pos[0], self.cursor_pos[1])
        
        self.render_cursor_pos[0] += (target_x - self.render_cursor_pos[0]) * 0.5
        self.render_cursor_pos[1] += (target_y - self.render_cursor_pos[1]) * 0.5
        
        rx, ry = self.render_cursor_pos
        rect = pygame.Rect(rx, ry, self.GRID_CELL_WIDTH, self.GRID_CELL_HEIGHT)
        
        for i in range(5):
            alpha = 150 - i * 30
            color = (*self.COLOR_NEON_YELLOW[:3], alpha)
            s = pygame.Surface((self.GRID_CELL_WIDTH + i*2, self.GRID_CELL_HEIGHT + i*2), pygame.SRCALPHA)
            pygame.draw.rect(s, color, s.get_rect(), 2, 6)
            self.screen.blit(s, (rx - i, ry - i))
        
        pygame.draw.rect(self.screen, self.COLOR_NEON_YELLOW, rect, 2, 6)

    def _draw_ui(self):
        pygame.draw.rect(self.screen, self.COLOR_BG, (0, 0, self.SCREEN_WIDTH, 45))
        pygame.draw.line(self.screen, self.COLOR_NEON_BLUE, (0, 45), (self.SCREEN_WIDTH, 45), 2)

        capital_str = f"CAPITAL: ${self.capital:,.2f}"
        self._render_text_with_glow(capital_str, self.font_large, self.COLOR_NEON_GREEN, (10, 10))

        rep_str = f"REPUTATION: {self.reputation}"
        rep_color = self.COLOR_NEON_BLUE if self.reputation > 25 else self.COLOR_NEON_RED
        self._render_text_with_glow(rep_str, self.font_main, rep_color, (350, 18))
        
        day_str = f"DAY: {self.steps}"
        self._render_text_with_glow(day_str, self.font_main, self.COLOR_TEXT, (540, 18))

    def _draw_action_feedback(self):
        for i, item in enumerate(self.last_action_feedback):
            text, color, pos, life = item
            if life > 0:
                alpha = min(255, life * 4)
                self._render_text_with_glow(text, self.font_main, (*color[:3], alpha), (pos[0], pos[1] - (60 - life)), use_alpha=True)
                item[3] -= 1
        self.last_action_feedback = [item for item in self.last_action_feedback if item[3] > 0]

    def _draw_rewind_effect(self):
        s = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        progress = self.rewind_effect_timer / 15.0
        s.fill((255, 255, 255, int(100 * progress)))
        for offset, color_mask in zip([-5, 5], [(255,0,0,0), (0,0,255,0)]):
            offset_val = int(offset * (1-progress))
            shifted_surf = self.screen.copy()
            shifted_surf.fill(color_mask, special_flags=pygame.BLEND_RGBA_MULT)
            s.blit(shifted_surf, (offset_val, 0), special_flags=pygame.BLEND_RGBA_ADD)
        self.screen.blit(s, (0, 0))

    def _draw_game_over(self):
        overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((self.COLOR_BG[0], self.COLOR_BG[1], self.COLOR_BG[2], 200))
        self.screen.blit(overlay, (0, 0))
        
        msg = "MEMORY MOGUL" if self.capital >= self.win_condition else "MARKET CRASH"
        color = self.COLOR_NEON_GREEN if self.capital >= self.win_condition else self.COLOR_NEON_RED
            
        text_surf = self.font_large.render(msg, True, color)
        text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
        self._render_text_with_glow(msg, self.font_large, color, text_rect.topleft)

    # --- Particle System ---
    def _create_particles(self, pos, count, color):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            life = self.np_random.integers(20, 41)
            self.particles.append({'pos': list(pos), 'vel': vel, 'life': life, 'color': color})

    def _update_and_draw_particles(self):
        active_particles = []
        for p in self.particles:
            p['life'] -= 1
            if p['life'] > 0:
                p['pos'][0] += p['vel'][0]
                p['pos'][1] += p['vel'][1]
                p['vel'][1] += 0.1 # Gravity
                
                alpha = int(255 * (p['life'] / 40.0))
                color = (*p['color'][:3], alpha)
                size = max(1, 3 * (p['life'] / 40.0))
                
                pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), int(size), color)
                active_particles.append(p)
        self.particles = active_particles

    # --- Helpers ---
    def _get_screen_pos_for_grid(self, col, row):
        return (self.GRID_MARGIN_X + col * self.GRID_CELL_WIDTH, 
                self.GRID_MARGIN_Y + row * self.GRID_CELL_HEIGHT)

    def _get_cursor_screen_pos(self):
        x, y = self._get_screen_pos_for_grid(self.cursor_pos[0], self.cursor_pos[1])
        return x + self.GRID_CELL_WIDTH / 2, y + self.GRID_CELL_HEIGHT / 2

    def _render_text_with_glow(self, text, font, color, pos, use_alpha=False):
        text_surf = font.render(text, True, color)
        glow_surf = font.render(text, True, color)
        if use_alpha:
            text_surf.set_alpha(color[3])
            glow_surf.set_alpha(color[3] // 4)
        else:
            glow_surf.set_alpha(100)
        
        for dx, dy in [(d,0) for d in range(-2,3)] + [(0,d) for d in range(-2,3)]:
             self.screen.blit(glow_surf, (pos[0] + dx, pos[1] + dy))
        self.screen.blit(text_surf, pos)
        
    def _add_feedback(self, text, color, duration=60):
        pos = self._get_cursor_screen_pos()
        self.last_action_feedback.append([text, color, (pos[0] - 30, pos[1]), duration])


if __name__ == '__main__':
    # This block is for human play and visualization.
    # It is not used by the evaluation system.
    os.environ["SDL_VIDEODRIVER"] = "x11" # Or "windows", "macOS"
    env = GameEnv()
    obs, info = env.reset(seed=42)
    
    pygame.display.set_caption("Memory Market")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    running = True
    
    while running:
        movement = 0 # None
        space_held = 0
        shift_held = 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1
        
        action = [movement, space_held, shift_held]
        obs, reward, terminated, truncated, info = env.step(action)
        
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Capital: {info['capital']}")
            pygame.time.wait(3000)
            obs, info = env.reset()

        clock.tick(30)

    pygame.quit()
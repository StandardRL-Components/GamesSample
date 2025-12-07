import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T10:55:02.569983
# Source Brief: brief_00753.md
# Brief Index: 753
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    Dream Merchant: A Gymnasium environment where the player trades dream fragments
    in a time-bending market to maximize their net worth.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Trade dream fragments in a time-bending market to maximize your net worth before time runs out. "
        "Buy low and sell high by navigating a circular board of fluctuating prices."
    )
    user_guide = (
        "Press [SPACE] to buy the currently selected dream fragment and [SHIFT] to sell. "
        "Your movement is determined automatically by a dice roll each turn."
    )
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    MAX_STEPS = 1000
    STARTING_CASH = 1000

    # Colors
    COLOR_BG = (10, 5, 25)
    COLOR_TEXT = (220, 220, 255)
    COLOR_TEXT_SHADOW = (5, 2, 12)
    COLOR_PANEL_BG = (20, 10, 50, 200)
    COLOR_PANEL_BORDER = (100, 80, 150)
    COLOR_PLAYER = (255, 255, 100)
    COLOR_GREEN_GLOW = (100, 255, 100)
    COLOR_RED_GLOW = (255, 100, 100)
    
    FRAGMENTS = [
        {'name': 'Joy', 'color': (255, 200, 50), 'avg_price': 100, 'amplitude': 50, 'period': 100},
        {'name': 'Fear', 'color': (150, 50, 200), 'avg_price': 50, 'amplitude': 40, 'period': 75},
        {'name': 'Calm', 'color': (60, 180, 255), 'avg_price': 80, 'amplitude': 30, 'period': 120},
        {'name': 'Curiosity', 'color': (100, 255, 150), 'avg_price': 120, 'amplitude': 60, 'period': 150},
        {'name': 'Grief', 'color': (120, 120, 140), 'avg_price': 30, 'amplitude': 25, 'period': 60},
        {'name': 'Hope', 'color': (255, 150, 200), 'avg_price': 150, 'amplitude': 70, 'period': 200},
    ]
    NUM_TILES = len(FRAGMENTS)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        
        self.font_small = pygame.font.Font(None, 18)
        self.font_medium = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 36)
        
        # Game state variables initialized in reset()
        self.steps = 0
        self.cash = 0
        self.inventory = []
        self.market_prices = []
        self.player_pos = 0
        self.game_over = False
        self.dice_roll = 0
        self.particles = []
        self.background_stars = []
        self.action_feedback = [] # To store feedback text like "+1 Joy"

        self._generate_background_stars()
        # self.reset() is called by the wrapper or user, no need to call it here.

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.cash = self.STARTING_CASH
        self.inventory = [0] * self.NUM_TILES
        self.player_pos = 0
        self.game_over = False
        self.dice_roll = 0
        self.particles = []
        self.action_feedback = []
        
        self._update_market()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # --- Pre-action state ---
        old_net_worth = self._calculate_net_worth()
        reward = 0

        # --- 1. Player Action (Buy/Sell) ---
        _movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        fragment_idx = self.player_pos
        price = self.market_prices[fragment_idx]

        # Prioritize Buy over Sell if both are pressed
        if space_held:
            if self.cash >= price:
                self.cash -= price
                self.inventory[fragment_idx] += 1
                # Sound: buy_sfx.wav
                self._create_particles(self.FRAGMENTS[fragment_idx]['color'], 20, is_buy=True)
                self.action_feedback.append({'text': f'BOUGHT {self.FRAGMENTS[fragment_idx]["name"]}', 'color': self.COLOR_GREEN_GLOW, 'life': 30})
                
                # Reward for good trades
                if price < self.FRAGMENTS[fragment_idx]['avg_price']:
                    reward += 0.1
                else:
                    reward -= 0.1
        elif shift_held:
            if self.inventory[fragment_idx] > 0:
                self.inventory[fragment_idx] -= 1
                self.cash += price
                # Sound: sell_sfx.wav
                self._create_particles(self.FRAGMENTS[fragment_idx]['color'], 20, is_buy=False)
                self.action_feedback.append({'text': f'SOLD {self.FRAGMENTS[fragment_idx]["name"]}', 'color': self.COLOR_RED_GLOW, 'life': 30})
                
                # Reward for good trades
                if price > self.FRAGMENTS[fragment_idx]['avg_price']:
                    reward += 0.1
                else:
                    reward -= 0.1

        # --- 2. Dice Roll & Movement ---
        self.dice_roll = self.np_random.integers(1, 7)
        self.player_pos = (self.player_pos + self.dice_roll) % self.NUM_TILES
        # Sound: dice_roll.wav

        # --- 3. Market Update & Termination ---
        self.steps += 1
        self._update_market()
        
        terminated = self.steps >= self.MAX_STEPS
        if terminated:
            self.game_over = True

        # --- 4. Calculate Reward ---
        new_net_worth = self._calculate_net_worth()
        # Reward is the change in net worth, scaled
        reward += (new_net_worth - old_net_worth) / 100.0 
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _update_market(self):
        self.market_prices = []
        for frag in self.FRAGMENTS:
            price = frag['avg_price'] + frag['amplitude'] * math.sin(2 * math.pi * self.steps / frag['period'])
            self.market_prices.append(int(price))

    def _calculate_net_worth(self):
        inventory_value = sum(self.inventory[i] * self.market_prices[i] for i in range(self.NUM_TILES))
        return self.cash + inventory_value

    def _get_observation(self):
        self._render_all()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self._calculate_net_worth(),
            "steps": self.steps,
            "cash": self.cash,
            "inventory": self.inventory,
        }

    def _render_all(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_market()
        self._render_player()
        self._render_particles()
        self._render_panels()
        self._render_ui()
        self._update_action_feedback()

    def _generate_background_stars(self):
        self.background_stars = []
        for _ in range(150):
            self.background_stars.append({
                'pos': [random.uniform(0, self.SCREEN_WIDTH), random.uniform(0, self.SCREEN_HEIGHT)],
                'size': random.uniform(0.5, 2),
                'speed': random.uniform(0.05, 0.2)
            })

    def _render_background(self):
        for star in self.background_stars:
            star['pos'][0] = (star['pos'][0] - star['speed']) % self.SCREEN_WIDTH
            size = int(star['size'])
            color_val = int(star['size'] * 40) + 20
            color = (color_val, color_val, color_val + 20)
            if size > 1:
                pygame.gfxdraw.filled_circle(self.screen, int(star['pos'][0]), int(star['pos'][1]), size, color)
            else:
                self.screen.set_at((int(star['pos'][0]), int(star['pos'][1])), color)

    def _render_market(self):
        center_x, center_y = 220, self.SCREEN_HEIGHT // 2
        radius = 160
        
        for i in range(self.NUM_TILES):
            angle = (i / self.NUM_TILES) * 2 * math.pi - math.pi / 2
            x = center_x + radius * math.cos(angle)
            y = center_y + radius * math.sin(angle)
            
            frag = self.FRAGMENTS[i]
            price = self.market_prices[i]
            price_ratio = (price - (frag['avg_price'] - frag['amplitude'])) / (frag['amplitude'] * 2)
            price_ratio = max(0, min(1, price_ratio))
            
            # Interpolate between cool and warm colors
            tile_color = (
                int(60 + 195 * price_ratio),
                int(180 - 180 * price_ratio),
                int(255 - 205 * price_ratio)
            )
            
            self._draw_glow_circle(self.screen, tile_color, (int(x), int(y)), 25, 40)
            self._draw_text(frag['name'], (x, y - 12), self.font_small, self.COLOR_TEXT, self.COLOR_TEXT_SHADOW, center=True)
            self._draw_text(f"${price}", (x, y + 12), self.font_medium, self.COLOR_TEXT, self.COLOR_TEXT_SHADOW, center=True)

    def _render_player(self):
        center_x, center_y = 220, self.SCREEN_HEIGHT // 2
        radius = 160
        angle = (self.player_pos / self.NUM_TILES) * 2 * math.pi - math.pi / 2
        x = center_x + radius * math.cos(angle)
        y = center_y + radius * math.sin(angle)
        
        self._draw_glow_circle(self.screen, self.COLOR_PLAYER, (int(x), int(y)), 12, 100)
        pygame.gfxdraw.filled_circle(self.screen, int(x), int(y), 8, self.COLOR_PLAYER)

    def _render_panels(self):
        panel_x = 440
        panel_width = 190
        
        # --- Inventory Panel ---
        inv_panel_rect = pygame.Rect(panel_x, 10, panel_width, 210)
        pygame.gfxdraw.box(self.screen, inv_panel_rect, self.COLOR_PANEL_BG)
        pygame.draw.rect(self.screen, self.COLOR_PANEL_BORDER, inv_panel_rect, 1, 5)
        self._draw_text("INVENTORY", (panel_x + panel_width/2, 25), self.font_medium, self.COLOR_TEXT, self.COLOR_TEXT_SHADOW, center=True)
        
        y_offset = 50
        for i, frag in enumerate(self.FRAGMENTS):
            if self.inventory[i] > 0:
                pygame.draw.circle(self.screen, frag['color'], (panel_x + 20, y_offset), 5)
                self._draw_text(f"{frag['name']}: {self.inventory[i]}", (panel_x + 35, y_offset), self.font_small, self.COLOR_TEXT, self.COLOR_TEXT_SHADOW)
                y_offset += 20

        # --- Current Tile Panel ---
        current_panel_rect = pygame.Rect(panel_x, 230, panel_width, 160)
        pygame.gfxdraw.box(self.screen, current_panel_rect, self.COLOR_PANEL_BG)
        pygame.draw.rect(self.screen, self.COLOR_PANEL_BORDER, current_panel_rect, 1, 5)

        frag_idx = self.player_pos
        frag = self.FRAGMENTS[frag_idx]
        price = self.market_prices[frag_idx]
        
        self._draw_text("CURRENT TILE", (panel_x + panel_width/2, 245), self.font_medium, self.COLOR_TEXT, self.COLOR_TEXT_SHADOW, center=True)
        self._draw_text(frag['name'], (panel_x + panel_width/2, 275), self.font_large, frag['color'], self.COLOR_TEXT_SHADOW, center=True)
        self._draw_text(f"Price: ${price}", (panel_x + panel_width/2, 310), self.font_medium, self.COLOR_TEXT, self.COLOR_TEXT_SHADOW, center=True)
        self._draw_text(f"You Have: {self.inventory[frag_idx]}", (panel_x + panel_width/2, 335), self.font_medium, self.COLOR_TEXT, self.COLOR_TEXT_SHADOW, center=True)

        # Action prompts
        self._draw_text("[SPACE] to Buy", (panel_x + panel_width/2, 360), self.font_small, self.COLOR_GREEN_GLOW, self.COLOR_TEXT_SHADOW, center=True)
        self._draw_text("[SHIFT] to Sell", (panel_x + panel_width/2, 375), self.font_small, self.COLOR_RED_GLOW, self.COLOR_TEXT_SHADOW, center=True)


    def _render_ui(self):
        # Top UI Bar
        self._draw_text(f"Net Worth: ${int(self._calculate_net_worth())}", (10, 15), self.font_large, self.COLOR_TEXT, self.COLOR_TEXT_SHADOW)
        self._draw_text(f"Cash: ${self.cash}", (10, 50), self.font_medium, self.COLOR_TEXT, self.COLOR_TEXT_SHADOW)
        
        turn_text = f"Turn: {self.steps} / {self.MAX_STEPS}"
        self._draw_text(turn_text, (self.SCREEN_WIDTH - 10, 15), self.font_large, self.COLOR_TEXT, self.COLOR_TEXT_SHADOW, align_right=True)
        
        dice_text = f"Dice Roll: {self.dice_roll}"
        self._draw_text(dice_text, (self.SCREEN_WIDTH - 10, 50), self.font_medium, self.COLOR_TEXT, self.COLOR_TEXT_SHADOW, align_right=True)

        # Action feedback
        for i, feedback in enumerate(self.action_feedback):
            self._draw_text(feedback['text'], (220, 100 + i*20), self.font_medium, feedback['color'], self.COLOR_TEXT_SHADOW, center=True, alpha=feedback['life']*8)

    def _create_particles(self, color, count, is_buy):
        center_x, center_y = 220, self.SCREEN_HEIGHT // 2
        radius = 160
        angle = (self.player_pos / self.NUM_TILES) * 2 * math.pi - math.pi / 2
        x = center_x + radius * math.cos(angle)
        y = center_y + radius * math.sin(angle)
        
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 3)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            if is_buy: # Particles fly towards player
                vel = [-v for v in vel]
            self.particles.append({
                'pos': [x, y], 'vel': vel, 'life': random.randint(20, 40), 'color': color
            })

    def _render_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][0] *= 0.95 # friction
            p['vel'][1] *= 0.95
            p['life'] -= 1
            
            if p['life'] > 0:
                alpha = int(255 * (p['life'] / 40))
                color = p['color'] + (alpha,)
                size = int(p['life'] / 10)
                if size > 0:
                    pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), size, color)

        self.particles = [p for p in self.particles if p['life'] > 0]
    
    def _update_action_feedback(self):
        for feedback in self.action_feedback:
            feedback['life'] -= 1
        self.action_feedback = [f for f in self.action_feedback if f['life'] > 0]

    def _draw_text(self, text, pos, font, color, shadow_color=None, center=False, align_right=False, alpha=255):
        text_surf = font.render(text, True, color)
        text_surf.set_alpha(alpha)
        text_rect = text_surf.get_rect()

        if center:
            text_rect.center = pos
        elif align_right:
            text_rect.topright = pos
        else:
            text_rect.topleft = pos
        
        if shadow_color:
            shadow_surf = font.render(text, True, shadow_color)
            shadow_surf.set_alpha(alpha)
            shadow_rect = shadow_surf.get_rect()
            shadow_rect.topleft = (text_rect.left + 1, text_rect.top + 1)
            self.screen.blit(shadow_surf, shadow_rect)
            
        self.screen.blit(text_surf, text_rect)

    def _draw_glow_circle(self, surface, color, center, radius, max_alpha):
        for i in range(radius, 0, -2):
            alpha = int(max_alpha * (1 - (i / radius))**2)
            glow_color = color + (alpha,)
            pygame.gfxdraw.filled_circle(surface, center[0], center[1], i, glow_color)

    def close(self):
        pygame.font.quit()
        pygame.quit()

# Example of how to run the environment
if __name__ == '__main__':
    # The main loop is for demonstration and interactive testing.
    # It requires a display, so we unset the dummy video driver.
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv(render_mode="rgb_array")
    
    # --- Manual Play ---
    obs, info = env.reset()
    terminated = False
    
    # Setup Pygame window for human play
    pygame.display.set_caption("Dream Merchant")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    running = True

    while running:
        # Default action is to do nothing
        action = [0, 0, 0] # [movement, space, shift]

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_SPACE]:
            action[1] = 1 # Buy
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            action[2] = 1 # Sell

        # In manual play, we step every frame to have a responsive demo.
        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}")
            obs, info = env.reset()
        
        # Render the observation from the environment
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(30) # Run at 30 FPS

    env.close()
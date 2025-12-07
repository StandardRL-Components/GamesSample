import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T15:49:40.391668
# Source Brief: brief_01215.md
# Brief Index: 1215
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}
    game_description = (
        "Manage three factory production lines to assemble and fulfill component orders as quickly as possible before time runs out."
    )
    user_guide = (
        "Use ↑, ↓, and ← to cycle the item produced by the top, middle, and bottom production lines."
    )
    auto_advance = True

    # --- CONSTANTS ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    FACTORY_WIDTH = 440
    UI_WIDTH = SCREEN_WIDTH - FACTORY_WIDTH

    # Game parameters
    MAX_ORDERS_TO_WIN = 15
    GAME_DURATION_SECONDS = 120
    LOGIC_FPS = 30  # How many logic steps per second
    MAX_STEPS = GAME_DURATION_SECONDS * LOGIC_FPS
    
    # Colors
    COLOR_BG = (25, 28, 36)
    COLOR_UI_BG = (35, 38, 48)
    COLOR_LINE_BG = (45, 48, 58)
    COLOR_TEXT = (220, 220, 230)
    COLOR_TEXT_DIM = (150, 150, 160)
    COLOR_ACCENT = (76, 175, 80)
    COLOR_WARNING = (255, 193, 7)
    COLOR_ERROR = (244, 67, 54)

    ITEM_COLORS = [
        (239, 83, 80),   # Red (Square)
        (66, 165, 245),  # Blue (Circle)
        (126, 204, 130), # Green (Triangle)
    ]
    
    CHAIN_BONUS_COLOR = (255, 234, 0)
    
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
        
        # Fonts
        try:
            self.font_main = pygame.font.SysFont("Roboto Condensed", 24, bold=True)
            self.font_small = pygame.font.SysFont("Roboto Condensed", 18)
            self.font_tiny = pygame.font.SysFont("Roboto Condensed", 14)
        except pygame.error:
            self.font_main = pygame.font.SysFont("sans", 24, bold=True)
            self.font_small = pygame.font.SysFont("sans", 18)
            self.font_tiny = pygame.font.SysFont("sans", 14)
            
        # State variables are initialized in reset()
        # self.reset() is called here to ensure np_random is initialized for validate_implementation
        self.reset()

        # Critical self-check
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.win_condition_met = False
        
        self.orders_fulfilled = 0
        self.inventory = np.array([0.0, 0.0, 0.0])
        self.production_focus = [0, 1, 2] # Line i produces item production_focus[i]
        self.last_production_focus = list(self.production_focus)
        self.chain_bonus_active = [False, False, False]
        
        # Production rates: items per second
        self.base_rates = self.np_random.uniform(2.0/60.0, 8.0/60.0, size=(3, 3)) # items/sec for [line][item]
        
        self.order_queue = []
        self.next_order_countdown = self.np_random.integers(5, 15) * self.LOGIC_FPS

        # Visuals
        self.particles = []
        self.moving_items = []
        self.production_progress = [0.0, 0.0, 0.0]

        # Generate initial orders
        for _ in range(3):
            self._generate_order()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0.0
        self.steps += 1

        # 1. Handle player actions
        self._handle_action(action)
        
        # 2. Update game logic
        reward += self._update_production()
        self._update_orders()
        reward += self._fulfill_orders()
        
        # 3. Check for termination
        terminated = False
        if self.orders_fulfilled >= self.MAX_ORDERS_TO_WIN:
            if not self.win_condition_met: # Grant reward only once
                reward += 100.0
                self.win_condition_met = True
            terminated = True
            self.game_over = True
            # sound: game_win
        
        if self.steps >= self.MAX_STEPS:
            if not self.win_condition_met:
                reward -= 100.0
            terminated = True
            self.game_over = True
            # sound: game_lose

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_action(self, action):
        movement = action[0]

        line_to_change = -1
        if movement == 1: line_to_change = 0 # Up for Line 1
        elif movement == 2: line_to_change = 1 # Down for Line 2
        elif movement == 3: line_to_change = 2 # Left for Line 3

        if line_to_change != -1:
            # sound: ui_click
            self.production_focus[line_to_change] = (self.production_focus[line_to_change] + 1) % 3

    def _update_production(self):
        production_reward = 0.0
        
        for i in range(3):
            # Update chain bonus status
            if self.production_focus[i] == self.last_production_focus[i]:
                if not self.chain_bonus_active[i]:
                    self.chain_bonus_active[i] = True
                    # sound: chain_bonus_start
            else:
                if self.chain_bonus_active[i]:
                    self.chain_bonus_active[i] = False
                    # sound: chain_bonus_break

            self.last_production_focus[i] = self.production_focus[i]

            # Calculate production for this step
            item_type = self.production_focus[i]
            base_rate_per_step = self.base_rates[i][item_type] / self.LOGIC_FPS
            rate_modifier = 1.20 if self.chain_bonus_active[i] else 1.0
            
            production_this_step = base_rate_per_step * rate_modifier
            
            self.inventory[item_type] += production_this_step
            production_reward += 0.1 * production_this_step

            # Visuals: spawn moving items
            self.production_progress[i] += production_this_step * 2.0 # speed up visual representation
            if self.production_progress[i] >= 1.0:
                self.production_progress[i] -= 1.0
                self.moving_items.append({
                    "type": item_type,
                    "line": i,
                    "progress": 0.0,
                    "y_offset": self.np_random.uniform(-5, 5)
                })
                # sound: item_produced
        
        return production_reward

    def _update_orders(self):
        self.next_order_countdown -= 1
        if self.next_order_countdown <= 0 and len(self.order_queue) < 6:
            self._generate_order()
            self.next_order_countdown = self.np_random.integers(8, 20) * self.LOGIC_FPS
            # sound: new_order

    def _generate_order(self):
        item_type = self.np_random.integers(0, 3)
        quantity = self.np_random.integers(5, 15 + self.orders_fulfilled) # Orders get slightly larger over time
        self.order_queue.append({"type": item_type, "quantity": quantity})

    def _fulfill_orders(self):
        fulfilled_reward = 0.0
        fulfilled_indices = []
        for i, order in enumerate(self.order_queue):
            if self.inventory[order["type"]] >= order["quantity"]:
                self.inventory[order["type"]] -= order["quantity"]
                self.orders_fulfilled += 1
                fulfilled_reward += 1.0
                if self.chain_bonus_active[self.np_random.integers(0,3)]: # Random small bonus
                    fulfilled_reward += 0.5
                fulfilled_indices.append(i)
                # sound: order_complete
                
                # Visual effect for fulfillment
                order_pos_y = 60 + 100 + i * 45
                order_pos_x = self.FACTORY_WIDTH + self.UI_WIDTH / 2
                for _ in range(20):
                    self.particles.append(Particle(
                        (order_pos_x, order_pos_y),
                        self.ITEM_COLORS[order["type"]],
                        self.np_random
                    ))
                break # Fulfill one order per step to avoid cascading clears

        if fulfilled_indices:
            self.order_queue = [o for i, o in enumerate(self.order_queue) if i not in fulfilled_indices]
        
        return fulfilled_reward

    def _get_observation(self):
        # Main render call
        self.screen.fill(self.COLOR_BG)
        self._render_factory()
        self._render_ui_panel()
        self._update_and_render_visuals()
        self._render_top_bar()
        
        if self.game_over:
            self._render_game_over()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "orders_fulfilled": self.orders_fulfilled,
            "inventory": self.inventory.tolist(),
        }

    # --- RENDER METHODS ---

    def _draw_text(self, text, font, color, position, centered=False):
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect()
        if centered:
            text_rect.center = position
        else:
            text_rect.topleft = position
        self.screen.blit(text_surface, text_rect)

    def _render_item(self, surface, item_type, center_pos, size):
        rect = pygame.Rect(0, 0, size, size)
        rect.center = center_pos
        color = self.ITEM_COLORS[item_type]
        
        if item_type == 0: # Square
            pygame.draw.rect(surface, color, rect, border_radius=int(size*0.1))
        elif item_type == 1: # Circle
            pygame.gfxdraw.aacircle(surface, int(center_pos[0]), int(center_pos[1]), int(size/2), color)
            pygame.gfxdraw.filled_circle(surface, int(center_pos[0]), int(center_pos[1]), int(size/2), color)
        elif item_type == 2: # Triangle
            points = [
                (center_pos[0], center_pos[1] - size/2),
                (center_pos[0] - size/2, center_pos[1] + size/2),
                (center_pos[0] + size/2, center_pos[1] + size/2),
            ]
            pygame.gfxdraw.aapolygon(surface, points, color)
            pygame.gfxdraw.filled_polygon(surface, points, color)

    def _render_factory(self):
        line_height = 110
        y_start = 60
        for i in range(3):
            line_y = y_start + i * line_height
            # Line background
            line_rect = pygame.Rect(10, line_y, self.FACTORY_WIDTH - 20, line_height - 10)
            pygame.draw.rect(self.screen, self.COLOR_LINE_BG, line_rect, border_radius=8)

            # Chain bonus glow
            if self.chain_bonus_active[i]:
                glow_rect = line_rect.inflate(6, 6)
                pygame.draw.rect(self.screen, self.CHAIN_BONUS_COLOR, glow_rect, width=2, border_radius=10)

            # Machine part
            machine_rect = pygame.Rect(20, line_y + 10, 80, 80)
            pygame.draw.rect(self.screen, self.COLOR_BG, machine_rect, border_radius=5)
            self._render_item(self.screen, self.production_focus[i], machine_rect.center, 40)
            
            # Conveyor belt
            belt_y = line_y + line_height / 2 - 5
            pygame.draw.line(self.screen, self.COLOR_BG, (110, belt_y), (self.FACTORY_WIDTH - 40, belt_y), 20)

    def _render_ui_panel(self):
        ui_rect = pygame.Rect(self.FACTORY_WIDTH, 40, self.UI_WIDTH, self.SCREEN_HEIGHT - 40)
        pygame.draw.rect(self.screen, self.COLOR_UI_BG, ui_rect)
        pygame.draw.line(self.screen, self.COLOR_BG, (self.FACTORY_WIDTH, 40), (self.FACTORY_WIDTH, self.SCREEN_HEIGHT), 2)

        # Inventory
        self._draw_text("INVENTORY", self.font_small, self.COLOR_TEXT, (self.FACTORY_WIDTH + 15, 55))
        for i in range(3):
            item_y = 90 + i * 35
            self._render_item(self.screen, i, (self.FACTORY_WIDTH + 30, item_y), 20)
            inv_text = f"{int(self.inventory[i])}"
            self._draw_text(inv_text, self.font_small, self.COLOR_TEXT, (self.FACTORY_WIDTH + 55, item_y - 12))

        # Orders
        self._draw_text("ORDER QUEUE", self.font_small, self.COLOR_TEXT, (self.FACTORY_WIDTH + 15, 200))
        for i, order in enumerate(self.order_queue[:4]):
            order_y = 235 + i * 45
            can_fulfill = self.inventory[order["type"]] >= order["quantity"]
            color = self.COLOR_ACCENT if can_fulfill else self.COLOR_TEXT_DIM
            
            self._render_item(self.screen, order["type"], (self.FACTORY_WIDTH + 30, order_y), 25)
            order_text = f"{order['quantity']}"
            self._draw_text(order_text, self.font_main, color, (self.FACTORY_WIDTH + 70, order_y - 15))

    def _update_and_render_visuals(self):
        # Moving items
        items_to_remove = []
        for i, item in enumerate(self.moving_items):
            item['progress'] += 1.0 / self.LOGIC_FPS * 0.75 # Adjust speed
            if item['progress'] >= 1.0:
                items_to_remove.append(i)
            else:
                line_y = 60 + item['line'] * 110 + 110 / 2 - 5 + item['y_offset']
                line_start_x, line_end_x = 110, self.FACTORY_WIDTH - 40
                item_x = line_start_x + (line_end_x - line_start_x) * item['progress']
                self._render_item(self.screen, item['type'], (item_x, line_y), 12)
        
        for i in sorted(items_to_remove, reverse=True):
            del self.moving_items[i]

        # Particles
        particles_to_remove = []
        for i, p in enumerate(self.particles):
            p.update()
            if p.lifespan <= 0:
                particles_to_remove.append(i)
            else:
                p.draw(self.screen)
        
        for i in sorted(particles_to_remove, reverse=True):
            del self.particles[i]

    def _render_top_bar(self):
        top_bar_rect = pygame.Rect(0, 0, self.SCREEN_WIDTH, 40)
        pygame.draw.rect(self.screen, self.COLOR_UI_BG, top_bar_rect)
        
        # Timer
        time_left = (self.MAX_STEPS - self.steps) / self.LOGIC_FPS
        time_text = f"TIME: {max(0, time_left):.1f}s"
        time_color = self.COLOR_TEXT if time_left > 10 else self.COLOR_ERROR
        self._draw_text(time_text, self.font_main, time_color, (20, 7))

        # Score
        score_text = f"SCORE: {int(self.score)}"
        self._draw_text(score_text, self.font_main, self.COLOR_TEXT, (self.SCREEN_WIDTH/2, 20), centered=True)

        # Orders
        orders_text = f"ORDERS: {self.orders_fulfilled}/{self.MAX_ORDERS_TO_WIN}"
        orders_color = self.COLOR_ACCENT if self.orders_fulfilled > 0 else self.COLOR_TEXT
        text_surf = self.font_main.render(orders_text, True, orders_color)
        self.screen.blit(text_surf, (self.SCREEN_WIDTH - text_surf.get_width() - 20, 7))

    def _render_game_over(self):
        overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((self.COLOR_BG[0], self.COLOR_BG[1], self.COLOR_BG[2], 200))
        self.screen.blit(overlay, (0, 0))

        if self.win_condition_met:
            msg = "MISSION COMPLETE"
            color = self.COLOR_ACCENT
        else:
            msg = "TIME UP"
            color = self.COLOR_ERROR
            
        self._draw_text(msg, pygame.font.SysFont("Roboto Condensed", 60, bold=True), color, (self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2 - 20), centered=True)
        final_score_text = f"Final Score: {int(self.score)}"
        self._draw_text(final_score_text, self.font_main, self.COLOR_TEXT, (self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2 + 40), centered=True)

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.nvec.shape == (3,)
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
        
        # print("✓ Implementation validated successfully")


class Particle:
    def __init__(self, pos, color, np_random):
        self.pos = list(pos)
        self.np_random = np_random
        angle = self.np_random.uniform(0, 2 * math.pi)
        speed = self.np_random.uniform(1, 4)
        self.vel = [math.cos(angle) * speed, math.sin(angle) * speed]
        self.lifespan = self.np_random.integers(20, 40)
        self.color = color
        self.radius = self.np_random.uniform(2, 5)

    def update(self):
        self.pos[0] += self.vel[0]
        self.pos[1] += self.vel[1]
        self.lifespan -= 1
        self.radius -= 0.1

    def draw(self, surface):
        if self.radius > 0:
            alpha = max(0, min(255, int(255 * (self.lifespan / 30))))
            color_with_alpha = self.color + (alpha,)
            
            # Use a temporary surface for alpha blending
            temp_surf = pygame.Surface((self.radius*2, self.radius*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color_with_alpha, (self.radius, self.radius), self.radius)
            surface.blit(temp_surf, (self.pos[0] - self.radius, self.pos[1] - self.radius))

# Example usage:
if __name__ == '__main__':
    # To run with display, comment out the os.environ line at the top
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv(render_mode="rgb_array")
    
    # --- Manual Play ---
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Factory Production Chain")
    clock = pygame.time.Clock()

    obs, info = env.reset()
    done = False
    
    # Mapping keys to actions
    # Arrow keys for lines 1/2/3
    key_to_movement = {
        pygame.K_UP: 1,    # Line 1 (top)
        pygame.K_DOWN: 2,  # Line 2 (middle)
        pygame.K_LEFT: 3,  # Line 3 (bottom)
    }

    running = True
    while running:
        # Default action is no-op
        movement_action = 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key in key_to_movement:
                    movement_action = key_to_movement[event.key]
                if event.key == pygame.K_r: # Reset
                    obs, info = env.reset()
                if event.key == pygame.K_ESCAPE:
                    running = False

        # Construct the full action
        action = [movement_action, 0, 0] # space and shift are not used
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            # Game over, wait for reset
            pass

        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(env.LOGIC_FPS)

    env.close()
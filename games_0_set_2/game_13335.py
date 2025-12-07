import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T20:48:10.778659
# Source Brief: brief_03335.md
# Brief Index: 3335
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

# --- Helper Classes for Game Objects ---

class LeptonCard:
    """Represents a single falling card with an element."""
    def __init__(self, element_type, x, y, card_size):
        self.element_type = element_type
        self.x = x
        self.y = y
        self.vy = 0
        self.width, self.height = card_size
        self.target_y = y
        self.is_matched = False

class Particle:
    """Represents a single particle for visual effects."""
    def __init__(self, x, y, color, lifespan):
        self.x = x
        self.y = y
        self.vx = random.uniform(-2, 2)
        self.vy = random.uniform(-4, 4)
        self.color = color
        self.lifespan = lifespan
        self.life = lifespan
        self.size = random.uniform(2, 5)

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.vy += 0.1 # Gravity on particles
        self.life -= 1
        return self.life > 0

    def draw(self, surface):
        alpha = int(255 * (self.life / self.lifespan))
        if alpha > 0:
            radius = int(self.size * (self.life / self.lifespan))
            if radius > 0:
                pygame.gfxdraw.filled_circle(surface, int(self.x), int(self.y), radius, (*self.color, alpha))

# --- Main Environment Class ---

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "A physics-based puzzle game where you arrange 'lepton' cards to match a target sequence by flipping gravity and cloning cards."
    )
    user_guide = (
        "Controls: ←→ to select a column. Press space to flip gravity and shift to clone the top card. Arrange cards to match the target."
    )
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30
    NUM_COLUMNS = 4
    MAX_CARDS_PER_COLUMN = 8
    CARD_SPEED = 8.0

    # Colors
    COLOR_BG = (15, 20, 35)
    COLOR_GRID = (40, 50, 70)
    COLOR_SELECTOR = (255, 255, 100)
    COLOR_TEXT = (220, 220, 240)
    COLOR_SCORE = (100, 255, 150)
    COLOR_TIMER_GREEN = (0, 200, 80)
    COLOR_TIMER_YELLOW = (255, 200, 0)
    COLOR_TIMER_RED = (220, 50, 50)
    
    ELEMENTS = {
        0: {"name": "Electron", "color": (80, 150, 255), "symbol": "circle"},
        1: {"name": "Muon", "color": (255, 80, 80), "symbol": "triangle"},
        2: {"name": "Tau", "color": (80, 255, 150), "symbol": "square"},
        3: {"name": "Quark", "color": (255, 220, 80), "symbol": "star"},
    }
    NUM_ELEMENTS = len(ELEMENTS)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium Spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame Setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("monospace", 16, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 24, bold=True)
        
        # Game State (persistent across resets for difficulty scaling)
        self.level = 0
        self.initial_pattern_length = 3
        self.initial_time_limit = 60.0

        # Game State (reset every episode)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.timer = 0.0
        self.current_pattern_length = 0
        self.current_time_limit = 0.0
        self.target_pattern = []
        self.columns = []
        self.column_gravity = []
        self.selected_column = 0
        self.particles = []
        self.last_space_held = False
        self.last_shift_held = False
        self.last_match_count = 0

        self.card_width = (self.WIDTH // self.NUM_COLUMNS) * 0.8
        self.card_height = (self.HEIGHT - 50) / self.MAX_CARDS_PER_COLUMN * 0.9
        
        # self.reset() is called by the wrapper, no need to call it here.

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Reset episode-specific state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.particles = []
        self.selected_column = 0
        self.last_space_held = False
        self.last_shift_held = False
        self.last_match_count = 0

        # Difficulty Scaling
        self.current_pattern_length = self.initial_pattern_length + (self.level // 5)
        self.current_time_limit = max(30.0, self.initial_time_limit - (self.level // 10) * 5)
        self.timer = self.current_time_limit

        # Generate new puzzle
        self._generate_target_pattern()
        self._spawn_initial_cards()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            # If the game is over, do nothing until reset
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        self.steps += 1
        
        reward = 0
        
        # 1. Handle Input and Action Costs
        action_cost = self._handle_input(movement, space_held, shift_held)
        reward += action_cost

        # 2. Update Game Physics and State
        self._update_physics()
        self.timer = max(0, self.timer - 1.0 / self.FPS)

        # 3. Calculate Rewards
        state_reward, current_match_count = self._calculate_state_reward()
        reward += state_reward
        
        # Event-based reward for making progress
        if current_match_count > self.last_match_count:
            reward += 10.0 # +10 for completing a part of the target pattern
        self.last_match_count = current_match_count
        
        # 4. Check for Termination
        terminated = self._check_termination()
        if terminated:
            self.game_over = True
            if self._check_win_condition():
                reward += 100.0  # +100 for winning
                self.level += 1 # Increase difficulty for next game
                # SFX: Win Jingle
            elif self.timer <= 0:
                reward -= 50.0  # -50 for timeout
                # SFX: Loss Buzzer
        
        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, movement, space_held, shift_held):
        action_cost = 0
        
        # Selector Movement (Left/Right)
        if movement == 3: # Left
            self.selected_column = max(0, self.selected_column - 1)
        elif movement == 4: # Right
            self.selected_column = min(self.NUM_COLUMNS - 1, self.selected_column + 1)
            
        # Gravity Flip (Spacebar press)
        if space_held and not self.last_space_held:
            self.column_gravity[self.selected_column] *= -1
            action_cost -= 0.1 # -0.1 for flip
            # SFX: Gravity Shift Whoosh
            self._spawn_particles(
                (self.selected_column + 0.5) * (self.WIDTH / self.NUM_COLUMNS),
                self.HEIGHT / 2, 30, (200, 200, 255)
            )
        
        # Clone Card (Shift press)
        if shift_held and not self.last_shift_held:
            col = self.columns[self.selected_column]
            if len(col) > 0 and len(col) < self.MAX_CARDS_PER_COLUMN:
                gravity = self.column_gravity[self.selected_column]
                # Get the card at the "top" relative to gravity
                top_card = min(col, key=lambda c: c.y) if gravity == 1 else max(col, key=lambda c: c.y)
                
                new_card_y = -self.card_height if gravity == 1 else self.HEIGHT + self.card_height
                
                new_card = LeptonCard(top_card.element_type, top_card.x, new_card_y, (self.card_width, self.card_height))
                self.columns[self.selected_column].append(new_card)
                action_cost -= 0.01 # -0.01 for clone
                # SFX: Clone Spawn sound
        
        self.last_space_held = space_held
        self.last_shift_held = shift_held
        return action_cost

    def _update_physics(self):
        # Update particles
        self.particles = [p for p in self.particles if p.update()]
        
        # Update cards
        for i, col in enumerate(self.columns):
            gravity = self.column_gravity[i]
            # Sort cards by position to handle stacking correctly
            col.sort(key=lambda c: c.y, reverse=(gravity == -1))
            
            base_y = self.HEIGHT if gravity == 1 else 0
            
            for card in col:
                # Determine target y based on stacking
                target_y = base_y - (self.card_height if gravity == 1 else 0)
                
                # Smooth movement towards target_y
                dy = target_y - card.y
                card.y += dy * 0.5 # Interpolation for smooth stacking
                
                # Update base for next card in stack
                base_y = card.y if gravity == 1 else card.y + self.card_height

                # Clamp to screen bounds
                card.y = max(0, min(self.HEIGHT - self.card_height, card.y))


    def _calculate_state_reward(self):
        reward = 0
        max_match_count = 0
        
        # Reset match status for all cards
        for col in self.columns:
            for card in col:
                card.is_matched = False

        for i, col in enumerate(self.columns):
            gravity = self.column_gravity[i]
            # Get cards sorted from bottom to top, regardless of gravity
            sorted_cards = sorted(col, key=lambda c: c.y, reverse=True)
            
            if not sorted_cards:
                continue

            current_match_count = 0
            for j in range(min(len(sorted_cards), len(self.target_pattern))):
                card = sorted_cards[j]
                if card.element_type == self.target_pattern[j]:
                    reward += 1.0 # +1 for each correctly placed card
                    card.is_matched = True
                    current_match_count += 1
                else:
                    break # Chain is broken
            
            if current_match_count > max_match_count:
                max_match_count = current_match_count
        
        return reward, max_match_count

    def _check_win_condition(self):
        for i, col in enumerate(self.columns):
            sorted_cards = sorted(col, key=lambda c: c.y, reverse=True)
            if len(sorted_cards) < len(self.target_pattern):
                continue
            
            match = True
            for j in range(len(self.target_pattern)):
                if sorted_cards[j].element_type != self.target_pattern[j]:
                    match = False
                    break
            if match:
                return True
        return False

    def _check_termination(self):
        return self.timer <= 0 or self.steps >= 1000 or self._check_win_condition()

    def _generate_target_pattern(self):
        self.target_pattern = [self.np_random.integers(0, self.NUM_ELEMENTS) for _ in range(self.current_pattern_length)]

    def _spawn_initial_cards(self):
        self.columns = [[] for _ in range(self.NUM_COLUMNS)]
        self.column_gravity = [1] * self.NUM_COLUMNS
        
        col_width = self.WIDTH / self.NUM_COLUMNS
        
        for i in range(self.NUM_COLUMNS):
            num_cards = self.np_random.integers(3, self.MAX_CARDS_PER_COLUMN)
            card_x = i * col_width + (col_width - self.card_width) / 2
            
            for j in range(num_cards):
                element = self.np_random.integers(0, self.NUM_ELEMENTS)
                card_y = self.HEIGHT - (j + 1) * self.card_height
                card = LeptonCard(element, card_x, card_y, (self.card_width, self.card_height))
                self.columns[i].append(card)

    def _spawn_particles(self, x, y, count, color):
        for _ in range(count):
            self.particles.append(Particle(x, y, color, lifespan=random.randint(20, 40)))

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid lines
        col_width = self.WIDTH / self.NUM_COLUMNS
        for i in range(1, self.NUM_COLUMNS):
            pygame.draw.line(self.screen, self.COLOR_GRID, (i * col_width, 50), (i * col_width, self.HEIGHT), 2)
            
        # Draw selector
        selector_x = self.selected_column * col_width
        selector_rect = pygame.Rect(selector_x, 50, col_width, self.HEIGHT - 50)
        s = pygame.Surface((col_width, self.HEIGHT - 50), pygame.SRCALPHA)
        s.fill((*self.COLOR_SELECTOR, 30))
        self.screen.blit(s, (selector_x, 50))
        pygame.draw.rect(self.screen, self.COLOR_SELECTOR, selector_rect, 2, 4)

        # Draw particles
        for p in self.particles:
            p.draw(self.screen)

        # Draw cards
        for col in self.columns:
            for card in col:
                self._draw_card(card)

    def _draw_card(self, card):
        rect = pygame.Rect(card.x, card.y, card.width, card.height)
        element_info = self.ELEMENTS[card.element_type]
        
        # Draw glowing effect for matched cards
        if card.is_matched:
            glow_radius = int(max(card.width, card.height) * 0.7)
            pygame.gfxdraw.filled_circle(self.screen, int(rect.centerx), int(rect.centery), glow_radius, (*element_info["color"], 40))
            pygame.gfxdraw.filled_circle(self.screen, int(rect.centerx), int(rect.centery), glow_radius-5, (*element_info["color"], 60))

        # Draw card body
        pygame.draw.rect(self.screen, element_info["color"], rect, 0, 5)
        pygame.draw.rect(self.screen, tuple(min(255, c + 40) for c in element_info["color"]), rect, 2, 5)

        # Draw symbol
        self._draw_element_symbol(self.screen, card.element_type, rect.centerx, rect.centery, min(rect.width, rect.height) * 0.4)

    def _draw_element_symbol(self, surface, element_type, cx, cy, size):
        symbol = self.ELEMENTS[element_type]["symbol"]
        color = (255, 255, 255)
        cx, cy, size = int(cx), int(cy), int(size)

        if symbol == "circle":
            pygame.gfxdraw.aacircle(surface, cx, cy, size, color)
            pygame.gfxdraw.filled_circle(surface, cx, cy, size, color)
        elif symbol == "triangle":
            points = [
                (cx, cy - size),
                (cx - size, cy + size * 0.7),
                (cx + size, cy + size * 0.7),
            ]
            pygame.gfxdraw.aapolygon(surface, points, color)
            pygame.gfxdraw.filled_polygon(surface, points, color)
        elif symbol == "square":
            rect = pygame.Rect(cx - size, cy - size, size * 2, size * 2)
            pygame.draw.rect(surface, color, rect)
        elif symbol == "star":
            points = []
            for i in range(10):
                angle = math.radians(i * 36)
                r = size if i % 2 == 0 else size * 0.4
                points.append((cx + r * math.sin(angle), cy - r * math.cos(angle)))
            pygame.gfxdraw.aapolygon(surface, points, color)
            pygame.gfxdraw.filled_polygon(surface, points, color)

    def _render_ui(self):
        # Draw UI background
        pygame.draw.rect(self.screen, (30, 40, 60), (0, 0, self.WIDTH, 50))
        pygame.draw.line(self.screen, self.COLOR_GRID, (0, 50), (self.WIDTH, 50), 2)

        # Draw Score
        score_text = self.font_large.render(f"SCORE: {int(self.score)}", True, self.COLOR_SCORE)
        self.screen.blit(score_text, (10, 12))

        # Draw Timer Bar
        timer_ratio = self.timer / self.current_time_limit
        timer_color = self.COLOR_TIMER_RED
        if timer_ratio > 0.5:
            timer_color = self.COLOR_TIMER_GREEN
        elif timer_ratio > 0.2:
            timer_color = self.COLOR_TIMER_YELLOW
        
        timer_bar_width = 200
        timer_bar_x = self.WIDTH - timer_bar_width - 10
        pygame.draw.rect(self.screen, self.COLOR_GRID, (timer_bar_x, 15, timer_bar_width, 20))
        pygame.draw.rect(self.screen, timer_color, (timer_bar_x, 15, timer_bar_width * timer_ratio, 20))
        
        # Draw Target Pattern
        target_label = self.font_small.render("TARGET:", True, self.COLOR_TEXT)
        self.screen.blit(target_label, (self.WIDTH / 2 - 150, 18))
        
        card_size = 30
        for i, element_type in enumerate(self.target_pattern):
            x = self.WIDTH / 2 - 50 + i * (card_size + 5)
            y = 10
            rect = pygame.Rect(x, y, card_size, card_size)
            info = self.ELEMENTS[element_type]
            pygame.draw.rect(self.screen, info["color"], rect, 0, 4)
            self._draw_element_symbol(self.screen, element_type, rect.centerx, rect.centery, card_size * 0.3)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "level": self.level}

    def close(self):
        pygame.font.quit()
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # It requires the SDL_VIDEODRIVER to be set to a real driver, e.g., by unsetting it.
    if "SDL_VIDEODRIVER" in os.environ and os.environ["SDL_VIDEODRIVER"] == "dummy":
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv()
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Lepton Chain")
    
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
                print(f"--- RESET --- Level: {info['level']}")

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        if keys[pygame.K_DOWN]: movement = 2
        if keys[pygame.K_LEFT]: movement = 3
        if keys[pygame.K_RIGHT]: movement = 4
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1
        
        action = [movement, space, shift]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if terminated:
            print(f"Episode Finished. Final Score: {info['score']:.2f}, Steps: {info['steps']}")
        
        # Draw the observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(env.FPS)
        
    env.close()
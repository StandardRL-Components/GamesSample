import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T10:41:05.168096
# Source Brief: brief_00032.md
# Brief Index: 32
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

# --- Helper Classes ---

class Vesicle:
    """Represents a vesicle carrying a protein."""
    def __init__(self, pos, protein_type, target_mods, grid_pos):
        self.pos = pygame.Vector2(pos)
        self.target_pos = pygame.Vector2(pos)
        self.protein_type = protein_type
        self.modifications = set()
        self.target_modifications = tuple(sorted(target_mods))
        self.grid_pos = grid_pos # (row, col)
        self.radius = 15
        self.lerp_speed = 0.15

    def update(self):
        """Smoothly moves the vesicle towards its target position."""
        self.pos.x += (self.target_pos.x - self.pos.x) * self.lerp_speed
        self.pos.y += (self.target_pos.y - self.pos.y) * self.lerp_speed

    def draw(self, surface, font, is_selected, colors):
        """Draws the vesicle and its information."""
        # Glow effect for selected vesicle
        if is_selected:
            for i in range(5):
                alpha = 80 - i * 15
                pygame.gfxdraw.aacircle(
                    surface, int(self.pos.x), int(self.pos.y),
                    self.radius + i + 2, (*colors["SELECT_GLOW"], alpha)
                )

        # Main body
        pygame.gfxdraw.filled_circle(surface, int(self.pos.x), int(self.pos.y), self.radius, colors["VESICLE"])
        pygame.gfxdraw.aacircle(surface, int(self.pos.x), int(self.pos.y), self.radius, colors["VESICLE_BORDER"])

        # Protein Type Label
        type_text = font.render(self.protein_type, True, colors["UI_TEXT_DARK"])
        text_rect = type_text.get_rect(center=self.pos)
        surface.blit(type_text, text_rect)

class Particle:
    """A simple particle for visual effects."""
    def __init__(self, pos, vel, radius, color, lifespan):
        self.pos = pygame.Vector2(pos)
        self.vel = pygame.Vector2(vel)
        self.radius = radius
        self.color = color
        self.lifespan = lifespan
        self.max_lifespan = lifespan

    def update(self):
        self.pos += self.vel
        self.lifespan -= 1
        self.radius -= 0.1
        self.vel *= 0.98 # Damping

    def draw(self, surface):
        if self.lifespan > 0 and self.radius > 0:
            alpha = int(255 * (self.lifespan / self.max_lifespan))
            current_color = (*self.color, alpha)
            pygame.gfxdraw.filled_circle(surface, int(self.pos.x), int(self.pos.y), int(self.radius), current_color)

class FloatingText:
    """Displays temporary text for feedback (e.g., +10)."""
    def __init__(self, pos, text, color, lifespan):
        self.pos = pygame.Vector2(pos)
        self.text = text
        self.color = color
        self.lifespan = lifespan
        self.max_lifespan = lifespan
        self.vel = pygame.Vector2(0, -0.5)

    def update(self):
        self.pos += self.vel
        self.lifespan -= 1

    def draw(self, surface, font):
        if self.lifespan > 0:
            alpha = int(255 * (self.lifespan / self.max_lifespan))
            text_surface = font.render(self.text, True, self.color)
            text_surface.set_alpha(alpha)
            surface.blit(text_surface, self.pos)

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Manage protein processing in the Golgi apparatus. Guide vesicles through cisternae to "
        "apply the correct modifications before shipping them out of the cell."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move the selected vesicle. Press space to cycle "
        "between vesicles and shift to apply a modification at the current location."
    )
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    FPS = 30

    # Colors
    COLOR_BG = (15, 25, 40)
    COLOR_CISTERNA = (30, 45, 65)
    COLOR_CISTERNA_BORDER = (50, 70, 95)
    COLOR_VESICLE = (255, 200, 100)
    COLOR_VESICLE_BORDER = (255, 255, 255)
    COLOR_UI_TEXT = (220, 220, 240)
    COLOR_UI_TEXT_DARK = (10, 10, 20)
    COLOR_SUCCESS = (100, 255, 150)
    COLOR_ERROR = (255, 100, 100)
    COLOR_TRANSPORT = (100, 150, 255)
    COLOR_RESOURCE = (255, 220, 0)
    SELECT_GLOW = (255, 255, 255)

    # Game Parameters
    MAX_STEPS = 1000
    WIN_CONDITION_SHIPPED = 50
    INITIAL_CARBOHYDRATES = 100
    CARBOHYDRATE_COST = 5
    NUM_VESICLES = 4
    MODIFICATION_TYPES = ['G', 'M', 'S', 'P', 'A', 'U'] # Glycosylation, Mannose rem, Sialylation, Phosphorylation, Acetylation, Ubiquitination

    # Layout
    CISTERNA_GRID = (2, 3) # 2 rows, 3 columns
    CISTERNA_WIDTH, CISTERNA_HEIGHT = 120, 60
    CISTERNA_H_SPACE, CISTERNA_V_SPACE = 40, 30

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = Box(low=0, high=255, shape=(400, 640, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("sans-serif", 14, bold=True)
        self.font_medium = pygame.font.SysFont("sans-serif", 18, bold=True)
        self.font_large = pygame.font.SysFont("sans-serif", 24, bold=True)

        self.cisternae = []
        self.vesicles = []
        self.particles = []
        self.floating_texts = []
        self.selected_vesicle_idx = 0
        self.last_space_held = False
        self.last_shift_held = False

        self._create_cisternae_layout()

        # self.reset() is called by the wrapper or user
        # self.validate_implementation() # Removed for production

    def _create_cisternae_layout(self):
        grid_w = self.CISTERNA_GRID[1] * self.CISTERNA_WIDTH + (self.CISTERNA_GRID[1] - 1) * self.CISTERNA_H_SPACE
        grid_h = self.CISTERNA_GRID[0] * self.CISTERNA_HEIGHT + (self.CISTERNA_GRID[0] - 1) * self.CISTERNA_V_SPACE
        start_x = (self.SCREEN_WIDTH - grid_w) / 2
        start_y = (self.SCREEN_HEIGHT - grid_h) / 2 + 20

        self.cisternae_layout = []
        mod_idx = 0
        for r in range(self.CISTERNA_GRID[0]):
            row_layout = []
            for c in range(self.CISTERNA_GRID[1]):
                x = start_x + c * (self.CISTERNA_WIDTH + self.CISTERNA_H_SPACE)
                y = start_y + r * (self.CISTERNA_HEIGHT + self.CISTERNA_V_SPACE)
                rect = pygame.Rect(x, y, self.CISTERNA_WIDTH, self.CISTERNA_HEIGHT)
                mod_type = self.MODIFICATION_TYPES[mod_idx % len(self.MODIFICATION_TYPES)]
                row_layout.append({"rect": rect, "mod": mod_type, "grid_pos": (r, c)})
                mod_idx += 1
            self.cisternae_layout.append(row_layout)

        self.entry_pos = pygame.Vector2(start_x - 50, start_y + self.CISTERNA_HEIGHT / 2)
        self.shipping_bay_y = start_y + grid_h + 20

    def _create_vesicle(self):
        protein_type = random.choice(['α', 'β', 'γ', 'δ'])
        num_mods = max(1, self.difficulty_level)
        target_mods = random.sample(self.MODIFICATION_TYPES, k=min(num_mods, len(self.MODIFICATION_TYPES)))
        vesicle = Vesicle(self.entry_pos, protein_type, target_mods, grid_pos=None)
        # Move to first available cisterna
        vesicle.grid_pos = (0,0)
        vesicle.target_pos = pygame.Vector2(self.cisternae_layout[0][0]["rect"].center)
        return vesicle

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False

        self.carbohydrates = self.INITIAL_CARBOHYDRATES
        self.proteins_shipped = 0
        self.proteins_mishipped = 0
        self.difficulty_level = 1

        self.vesicles = [self._create_vesicle() for _ in range(self.NUM_VESICLES)]
        self.selected_vesicle_idx = 0

        self.particles.clear()
        self.floating_texts.clear()

        self.last_space_held = False
        self.last_shift_held = False

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        space_pressed = space_held and not self.last_space_held
        shift_pressed = shift_held and not self.last_shift_held

        # --- Handle Actions ---
        selected_vesicle = self.vesicles[self.selected_vesicle_idx]

        if space_pressed:
            # Select next vesicle
            self.selected_vesicle_idx = (self.selected_vesicle_idx + 1) % len(self.vesicles)
            # sfx: UI_switch

        if shift_pressed:
            # Initiate modification
            if selected_vesicle.grid_pos is not None:
                r, c = selected_vesicle.grid_pos
                cisterna = self.cisternae_layout[r][c]
                mod_type = cisterna["mod"]

                if self.carbohydrates >= self.CARBOHYDRATE_COST:
                    if mod_type in selected_vesicle.target_modifications and mod_type not in selected_vesicle.modifications:
                        selected_vesicle.modifications.add(mod_type)
                        self.carbohydrates -= self.CARBOHYDRATE_COST
                        reward += 1.0
                        self._spawn_particles(selected_vesicle.pos, 20, self.COLOR_SUCCESS)
                        self._add_floating_text(selected_vesicle.pos, "+1", self.COLOR_SUCCESS)
                        # sfx: success_mod
                    else:
                        reward -= 0.5 # Wasted action
                        self._spawn_particles(selected_vesicle.pos, 10, self.COLOR_ERROR)
                        # sfx: error_mod
                else:
                    reward -= 0.5 # Not enough resources
                    self._add_floating_text(selected_vesicle.pos, "NO ATP", self.COLOR_ERROR)
                    # sfx: error_resource

        # Handle Movement & Shipping
        if movement != 0:
            if selected_vesicle.grid_pos is not None:
                r, c = selected_vesicle.grid_pos

                # Shipping action
                if movement == 2 and r == self.CISTERNA_GRID[0] - 1:
                    is_correct = set(selected_vesicle.target_modifications) == selected_vesicle.modifications
                    if is_correct:
                        reward += 10.0
                        self.score += 10
                        self.proteins_shipped += 1
                        self._spawn_particles(selected_vesicle.pos, 50, self.COLOR_RESOURCE)
                        self._add_floating_text(selected_vesicle.pos, f"+10", self.COLOR_SUCCESS)
                        # sfx: success_ship
                        if self.proteins_shipped > 0 and self.proteins_shipped % 20 == 0:
                            self.difficulty_level += 1
                    else:
                        reward -= 5.0
                        self.score -= 5
                        self.proteins_mishipped += 1
                        self._spawn_particles(selected_vesicle.pos, 30, self.COLOR_ERROR)
                        self._add_floating_text(selected_vesicle.pos, "-5", self.COLOR_ERROR)
                        # sfx: fail_ship

                    # Replace vesicle
                    self.vesicles[self.selected_vesicle_idx] = self._create_vesicle()

                else: # Regular movement
                    nr, nc = r, c
                    if movement == 1: nr -= 1 # Up
                    elif movement == 2: nr += 1 # Down
                    elif movement == 3: nc -= 1 # Left
                    elif movement == 4: nc += 1 # Right

                    if 0 <= nr < self.CISTERNA_GRID[0] and 0 <= nc < self.CISTERNA_GRID[1]:
                        if (nr, nc) != (r, c):
                            selected_vesicle.grid_pos = (nr, nc)
                            selected_vesicle.target_pos = pygame.Vector2(self.cisternae_layout[nr][nc]["rect"].center)
                            reward -= 0.1 # Movement cost
                            self._spawn_particles(selected_vesicle.pos, 5, self.COLOR_TRANSPORT, trail=True)
                            # sfx: transport_whoosh
                    else:
                        reward -= 0.2 # Invalid move
                        # sfx: invalid_move_buzz

        # --- Update Game State ---
        self.steps += 1

        for v in self.vesicles: v.update()
        for p in self.particles[:]:
            p.update()
            if p.lifespan <= 0: self.particles.remove(p)
        for t in self.floating_texts[:]:
            t.update()
            if t.lifespan <= 0: self.floating_texts.remove(t)

        self.last_space_held = space_held
        self.last_shift_held = shift_held

        # --- Check Termination ---
        terminated = False
        if self.carbohydrates <= 0:
            terminated = True
            reward -= 10.0
        elif self.proteins_shipped >= self.WIN_CONDITION_SHIPPED:
            terminated = True
            reward += 50.0
        
        truncated = self.steps >= self.MAX_STEPS
        if truncated:
            terminated = True

        self.game_over = terminated

        return self._get_observation(), reward, terminated, truncated, self._get_info()

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
            "carbohydrates": self.carbohydrates,
            "proteins_shipped": self.proteins_shipped,
            "difficulty": self.difficulty_level,
        }

    def _render_game(self):
        # Draw Cisternae
        for r_idx, row in enumerate(self.cisternae_layout):
            for c_idx, cisterna in enumerate(row):
                pygame.draw.rect(self.screen, self.COLOR_CISTERNA, cisterna["rect"], border_radius=10)
                pygame.draw.rect(self.screen, self.COLOR_CISTERNA_BORDER, cisterna["rect"], width=2, border_radius=10)
                mod_text = self.font_large.render(cisterna["mod"], True, self.COLOR_CISTERNA_BORDER)
                self.screen.blit(mod_text, mod_text.get_rect(center=cisterna["rect"].center))

        # Draw Entry/Exit areas
        entry_text = self.font_small.render("ER ENTRY", True, self.COLOR_CISTERNA_BORDER)
        self.screen.blit(entry_text, entry_text.get_rect(centerx=self.entry_pos.x, y=self.entry_pos.y - 30))
        pygame.draw.line(self.screen, self.COLOR_CISTERNA_BORDER, (self.entry_pos.x + 40, self.entry_pos.y), (self.cisternae_layout[0][0]["rect"].left, self.entry_pos.y), 2)

        shipping_text = self.font_small.render("CELL MEMBRANE EXPORT", True, self.COLOR_CISTERNA_BORDER)
        self.screen.blit(shipping_text, shipping_text.get_rect(centerx=self.SCREEN_WIDTH/2, y=self.shipping_bay_y))

        # Draw Particles
        for p in self.particles:
            p.draw(self.screen)

        # Draw Vesicles
        for i, v in enumerate(self.vesicles):
            v.draw(self.screen, self.font_medium, i == self.selected_vesicle_idx, self.colors)

        # Draw Floating Text
        for t in self.floating_texts:
            t.draw(self.screen, self.font_medium)

    def _render_ui(self):
        # Score
        score_text = self.font_medium.render(f"Score: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Resources
        carb_text = self.font_medium.render(f"ATP: {self.carbohydrates}", True, self.COLOR_RESOURCE)
        self.screen.blit(carb_text, (self.SCREEN_WIDTH - carb_text.get_width() - 10, 10))

        # Shipped
        shipped_text = self.font_medium.render(f"Shipped: {self.proteins_shipped}/{self.WIN_CONDITION_SHIPPED}", True, self.COLOR_UI_TEXT)
        self.screen.blit(shipped_text, (self.SCREEN_WIDTH - shipped_text.get_width() - 10, 35))

        # Selected Vesicle Info
        if self.vesicles:
            selected_v = self.vesicles[self.selected_vesicle_idx]

            # Target modifications
            target_str = " ".join(selected_v.target_modifications)
            target_text = self.font_medium.render(f"Target: [ {target_str} ]", True, self.COLOR_UI_TEXT)
            self.screen.blit(target_text, (10, self.SCREEN_HEIGHT - 60))

            # Current modifications
            current_mods = sorted(list(selected_v.modifications))
            current_str = " ".join(current_mods)
            current_text = self.font_medium.render(f"Current: [ {current_str} ]", True, self.COLOR_UI_TEXT)
            self.screen.blit(current_text, (10, self.SCREEN_HEIGHT - 35))

        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0,0,0,180))
            self.screen.blit(overlay, (0,0))
            if self.proteins_shipped >= self.WIN_CONDITION_SHIPPED:
                end_text = "Shipment Complete!"
                end_color = self.COLOR_SUCCESS
            else:
                end_text = "Process Failed"
                end_color = self.COLOR_ERROR

            text_surf = self.font_large.render(end_text, True, end_color)
            self.screen.blit(text_surf, text_surf.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2)))


    def _spawn_particles(self, pos, count, color, trail=False):
        for _ in range(count):
            if trail:
                angle = random.uniform(0, 2 * math.pi)
                speed = random.uniform(0.1, 0.5)
                vel = (math.cos(angle) * speed, math.sin(angle) * speed)
            else:
                angle = random.uniform(0, 2 * math.pi)
                speed = random.uniform(1, 4)
                vel = (math.cos(angle) * speed, math.sin(angle) * speed)

            radius = random.uniform(2, 5)
            lifespan = random.randint(20, 40)
            self.particles.append(Particle(pos, vel, radius, color, lifespan))

    def _add_floating_text(self, pos, text, color):
        lifespan = 60
        self.floating_texts.append(FloatingText(pos, text, color, lifespan))

    @property
    def colors(self):
        return {
            "VESICLE": self.COLOR_VESICLE,
            "VESICLE_BORDER": self.COLOR_VESICLE_BORDER,
            "SELECT_GLOW": self.SELECT_GLOW,
            "UI_TEXT_DARK": self.COLOR_UI_TEXT_DARK
        }

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game directly
    # It will not be run by the evaluation server
    # Un-comment the following line to run in a window instead of headless
    os.environ.pop("SDL_VIDEODRIVER", None)

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()

    # Setup for human play
    pygame.display.set_caption("Golgi Apparatus")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))

    running = True
    total_reward = 0

    while running:
        movement, space, shift = 0, 0, 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4

        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1

        action = [movement, space, shift]

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        env.clock.tick(env.FPS)

        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}, Steps: {info['steps']}")
            pygame.time.wait(3000)
            obs, info = env.reset()
            total_reward = 0

    env.close()
import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T10:08:20.104180
# Source Brief: brief_00285.md
# Brief Index: 285
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Defend your space station from alien attacks. Use resources to repair damage "
        "to the four quadrants of your base and survive for the designated number of turns."
    )
    user_guide = (
        "Use the arrow keys to select a quadrant to repair: ↑ for top-left, → for top-right, "
        "↓ for bottom-left, and ← for bottom-right. Taking no action passes the turn."
    )
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30
    MAX_TURNS = 100
    MAX_RESOURCES = 100
    RESOURCE_REGEN_PER_TURN = 5
    BASE_MAX_INTEGRITY_PER_QUAD = 100
    INITIAL_RESOURCES = 100
    
    # --- Colors ---
    COLOR_BG = pygame.Color("#1a1c2c")
    COLOR_BASE_HEALTHY = pygame.Color("#686f99")
    COLOR_BASE_CRITICAL = pygame.Color("#692f40")
    COLOR_BASE_OUTLINE = pygame.Color("#d0d0d8")
    COLOR_REPAIR = pygame.Color("#57e07f")
    COLOR_DAMAGE = pygame.Color("#e0576b")
    COLOR_RESOURCE = pygame.Color("#57a7e0")
    COLOR_UI_TEXT = pygame.Color("#f0f0f0")
    COLOR_UI_PANEL = pygame.Color(30, 32, 52, 180) # Semi-transparent
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)
        self.font_title = pygame.font.Font(None, 72)
        
        # --- Game Geometry ---
        base_size = 256
        base_center_x, base_center_y = self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2
        quad_size = base_size // 2
        self.quadrant_rects = [
            pygame.Rect(base_center_x - quad_size, base_center_y - quad_size, quad_size, quad_size), # 0: Top-Left
            pygame.Rect(base_center_x, base_center_y - quad_size, quad_size, quad_size),             # 1: Top-Right
            pygame.Rect(base_center_x - quad_size, base_center_y, quad_size, quad_size),             # 2: Bottom-Left
            pygame.Rect(base_center_x, base_center_y, quad_size, quad_size),                         # 3: Bottom-Right
        ]
        
        # --- State Variables (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_condition = False
        self.base_integrity = []
        self.resources = 0
        self.visual_effects = []

        # --- Action Mapping ---
        # Maps movement action (1-4) to quadrant index (0-3)
        # 1:Up -> TL, 2:Down -> BR, 3:Left -> BL, 4:Right -> TR
        # Action space: 0: No-op, 1: Up, 2: Down, 3: Left, 4: Right
        # The mapping is a bit unusual but we stick to it.
        # 1 (Up) -> 0 (TL)
        # 4 (Right) -> 1 (TR)
        # 2 (Down) -> 2 (BL)
        # 3 (Left) -> 3 (BR)
        self.movement_to_quad = {1: 0, 4: 1, 2: 2, 3: 3}
        
        # self.reset() is called by the environment wrapper, no need to call it here.

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_condition = False
        self.base_integrity = [self.BASE_MAX_INTEGRITY_PER_QUAD] * 4
        self.resources = self.INITIAL_RESOURCES
        self.visual_effects = []
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # --- Unpack Action ---
        movement = action[0]
        
        reward = 0
        
        # --- 1. Player Action Phase ---
        quad_to_repair = self.movement_to_quad.get(movement)
        
        if quad_to_repair is not None:
            integrity_to_restore = self.BASE_MAX_INTEGRITY_PER_QUAD - self.base_integrity[quad_to_repair]
            can_afford_to_repair = min(integrity_to_restore, self.resources)
            
            if can_afford_to_repair > 0:
                # SFX: repair_start.wav
                self.base_integrity[quad_to_repair] += can_afford_to_repair
                self.resources -= can_afford_to_repair
                reward += can_afford_to_repair  # +1 per point repaired
                self._add_visual_effect('repair', quad_to_repair, duration=0.7)
            
            # Add selector feedback even if no repair happened
            self._add_visual_effect('select', quad_to_repair, duration=0.3)

        # --- 2. Alien Attack Phase ---
        attack_quad_idx = self.np_random.integers(0, 4)
        base_damage = self.np_random.integers(10, 31) # 10-30 damage
        bonus_damage = self.steps // 10
        total_damage = base_damage + bonus_damage
        
        self.base_integrity[attack_quad_idx] = max(0, self.base_integrity[attack_quad_idx] - total_damage)
        self._add_visual_effect('damage', attack_quad_idx, duration=0.5, magnitude=total_damage)
        # SFX: explosion.wav, metal_hit.wav
        
        # --- 3. Update & Termination ---
        self.steps += 1
        self.resources = min(self.MAX_RESOURCES, self.resources + self.RESOURCE_REGEN_PER_TURN)
        reward += 5 # Survival reward
        
        terminated = False
        truncated = False
        total_integrity = sum(self.base_integrity)
        
        if total_integrity <= 0:
            # SFX: base_destroyed.wav
            terminated = True
            self.game_over = True
            self.win_condition = False
            reward -= 100
        elif self.steps >= self.MAX_TURNS:
            # SFX: victory.wav
            terminated = True # Could be truncated, but terminated is fine for turn-limit
            self.game_over = True
            self.win_condition = True
            reward += 100
            
        self.score += reward
        
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _add_visual_effect(self, effect_type, quad_idx, duration, magnitude=0):
        self.visual_effects.append({
            "type": effect_type,
            "quad_idx": quad_idx,
            "timer": duration * self.FPS,
            "max_timer": duration * self.FPS,
            "magnitude": magnitude,
            "particles": self._create_particles(effect_type, quad_idx) if effect_type == 'damage' else []
        })

    def _create_particles(self, effect_type, quad_idx):
        particles = []
        if effect_type == 'damage':
            rect = self.quadrant_rects[quad_idx]
            center_x, center_y = rect.center
            for _ in range(15):
                angle = random.uniform(0, 2 * math.pi)
                speed = random.uniform(1, 4)
                particles.append({
                    "pos": [center_x, center_y],
                    "vel": [math.cos(angle) * speed, math.sin(angle) * speed],
                    "life": random.randint(10, 20),
                    "size": random.uniform(2, 5)
                })
        return particles

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
            "resources": self.resources,
            "total_integrity": sum(self.base_integrity) / (4.0 * self.BASE_MAX_INTEGRITY_PER_QUAD)
        }

    def _render_game(self):
        self._render_base()
        self._render_effects()

    def _render_base(self):
        for i, rect in enumerate(self.quadrant_rects):
            integrity_percent = self.base_integrity[i] / self.BASE_MAX_INTEGRITY_PER_QUAD
            color = self.COLOR_BASE_CRITICAL.lerp(self.COLOR_BASE_HEALTHY, integrity_percent)
            
            pygame.draw.rect(self.screen, color, rect)
            pygame.draw.rect(self.screen, self.COLOR_BASE_OUTLINE, rect, 2)

    def _render_effects(self):
        # Update and draw effects
        for effect in self.visual_effects[:]:
            effect["timer"] -= 1
            if effect["timer"] <= 0:
                self.visual_effects.remove(effect)
                continue

            progress = effect["timer"] / effect["max_timer"]
            rect = self.quadrant_rects[effect["quad_idx"]]
            
            if effect["type"] == 'damage':
                alpha = int(255 * progress)
                s = pygame.Surface(rect.size, pygame.SRCALPHA)
                s.fill((self.COLOR_DAMAGE.r, self.COLOR_DAMAGE.g, self.COLOR_DAMAGE.b, alpha))
                self.screen.blit(s, rect.topleft)
                # Update and draw particles
                for p in effect["particles"][:]:
                    p["pos"][0] += p["vel"][0]
                    p["pos"][1] += p["vel"][1]
                    p["life"] -= 1
                    if p["life"] <= 0:
                        effect["particles"].remove(p)
                    else:
                        pygame.draw.circle(self.screen, self.COLOR_DAMAGE, p["pos"], p["size"])

            elif effect["type"] == 'repair':
                alpha = int(255 * (1 - progress**2)) # Fade in fast, fade out slow
                s = pygame.Surface(rect.size, pygame.SRCALPHA)
                pygame.draw.rect(s, (self.COLOR_REPAIR.r, self.COLOR_REPAIR.g, self.COLOR_REPAIR.b, alpha), s.get_rect(), border_radius=5)
                pygame.draw.rect(s, (255, 255, 255, alpha), s.get_rect(), width=3, border_radius=5)
                self.screen.blit(s, rect.topleft)

            elif effect["type"] == 'select':
                alpha = int(200 * progress)
                pygame.draw.rect(self.screen, (255, 255, 255, alpha), rect, 4, border_radius=5)

    def _render_ui(self):
        # --- UI Panel ---
        panel_rect = pygame.Rect(0, 0, self.SCREEN_WIDTH, 50)
        s = pygame.Surface(panel_rect.size, pygame.SRCALPHA)
        s.fill(self.COLOR_UI_PANEL)
        self.screen.blit(s, (0,0))
        
        # --- Base Integrity ---
        total_integrity = sum(self.base_integrity)
        total_max_integrity = self.BASE_MAX_INTEGRITY_PER_QUAD * 4
        integrity_percent = (total_integrity / total_max_integrity) * 100
        integrity_text = f"Base Integrity: {integrity_percent:.0f}%"
        text_surf = self.font_main.render(integrity_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(text_surf, (15, 12))

        # --- Turn Counter ---
        turn_text = f"Turn: {self.steps} / {self.MAX_TURNS}"
        text_surf = self.font_main.render(turn_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(text_surf, (self.SCREEN_WIDTH - text_surf.get_width() - 15, 12))
        
        # --- Resource Bar ---
        res_bar_width = 400
        res_bar_height = 25
        res_bar_x = (self.SCREEN_WIDTH - res_bar_width) / 2
        res_bar_y = self.SCREEN_HEIGHT - res_bar_height - 15
        
        # Background
        pygame.draw.rect(self.screen, self.COLOR_UI_PANEL, (res_bar_x - 5, res_bar_y - 5, res_bar_width + 10, res_bar_height + 10), border_radius=8)
        # Fill
        fill_width = (self.resources / self.MAX_RESOURCES) * res_bar_width
        pygame.draw.rect(self.screen, self.COLOR_RESOURCE, (res_bar_x, res_bar_y, fill_width, res_bar_height), border_radius=5)
        # Outline
        pygame.draw.rect(self.screen, self.COLOR_UI_TEXT, (res_bar_x, res_bar_y, res_bar_width, res_bar_height), 2, border_radius=5)
        # Text
        res_text = f"Resources: {self.resources}"
        text_surf = self.font_small.render(res_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(text_surf, (res_bar_x + res_bar_width / 2 - text_surf.get_width() / 2, res_bar_y + res_bar_height / 2 - text_surf.get_height() / 2))

        # --- Game Over/Win Screen ---
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            if self.win_condition:
                end_text = "VICTORY"
                end_color = self.COLOR_REPAIR
            else:
                end_text = "BASE DESTROYED"
                end_color = self.COLOR_DAMAGE
                
            text_surf = self.font_title.render(end_text, True, end_color)
            text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(text_surf, text_rect)
            
    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game manually
    # We need to unset the dummy video driver to see the window
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
        
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Action space: [5, 2, 2]
    # 0: No-op
    # 1: Up -> Repair Top-Left
    # 2: Down -> Repair Bottom-Left
    # 3: Left -> Repair Bottom-Right
    # 4: Right -> Repair Top-Right
    # The key mapping below makes this more intuitive for a human player.
    human_key_to_action_map = {
        pygame.K_w: 1, pygame.K_UP: 1,       # Top-Left
        pygame.K_d: 4, pygame.K_RIGHT: 4,    # Top-Right
        pygame.K_s: 2, pygame.K_DOWN: 2,     # Bottom-Left
        pygame.K_a: 3, pygame.K_LEFT: 3,     # Bottom-Right
        pygame.K_SPACE: 0                    # No-op / End Turn
    }

    # --- Pygame window for human play ---
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Base Defense")
    clock = pygame.time.Clock()
    running = True

    print("--- Manual Control ---")
    print(GameEnv.user_guide)
    print("Space: Pass turn")
    print("Q: Quit")

    while running:
        action_taken = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                
                movement = human_key_to_action_map.get(event.key)
                if movement is not None and not done:
                    # Action is [movement, 0, 0]
                    action = [movement, 0, 0]
                    obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    action_taken = True
        
        # Render the environment to the screen
        frame = np.transpose(env._get_observation(), (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(GameEnv.FPS)

    env.close()
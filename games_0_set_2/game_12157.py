import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T13:10:43.553410
# Source Brief: brief_02157.md
# Brief Index: 2157
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
        "Maintain a decaying space station by teleporting resources for repairs, expansion, and upgrades. "
        "Manage your energy and resources carefully to survive and complete the mission."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move the reticle. Press shift to cycle resources and space to "
        "teleport them to the target."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30

    # Colors
    COLOR_BG = (10, 0, 20)
    COLOR_STARS = (100, 100, 120)
    COLOR_GRID = (30, 20, 40)
    COLOR_TEXT = (220, 220, 255)
    COLOR_HUD_BG = (20, 10, 30, 200)

    COLOR_STATION_HEALTHY = (0, 255, 150)
    COLOR_STATION_WARN = (255, 255, 0)
    COLOR_STATION_CRIT = (255, 50, 50)
    
    COLOR_ENERGY = (0, 150, 255)
    COLOR_ENERGY_BG = (20, 40, 80)

    COLOR_REPAIR = (0, 255, 100)
    COLOR_EXPAND = (255, 150, 0)
    COLOR_UPGRADE = (200, 100, 255)
    
    COLOR_RETICLE = (255, 255, 255)
    COLOR_TELEPORTER_BEAM = (100, 200, 255)
    
    # Game Parameters
    RETICLE_SPEED = 15
    MAX_STEPS = 5000
    
    INITIAL_INTEGRITY = 100.0
    INITIAL_ENERGY = 100.0
    INITIAL_RESOURCES = {'repair': 10, 'module': 5, 'upgrade': 3}
    
    MAX_STATION_SIZE = 10
    MAX_TELEPORTER_LEVEL = 5
    
    BASE_DEGRADATION_RATE = 0.02  # per step
    DEGRADATION_PER_MODULE = 0.005 # per step
    
    ENERGY_RECHARGE_RATE = 0.1 # per step
    TELEPORT_BASE_COST = 25.0
    TELEPORT_COST_REDUCTION_PER_LEVEL = 4.0
    
    REPAIR_AMOUNT = 20.0
    
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
        
        self.font_small = pygame.font.Font(None, 20)
        self.font_medium = pygame.font.Font(None, 28)
        self.font_large = pygame.font.Font(None, 64)
        
        self.render_mode = render_mode
        self.stars = []
        self.particles = []

        # Initialize state variables to be defined in reset
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        
        self.station_integrity = 0
        self.station_size = 0
        self.station_core = None
        self.station_modules = []
        self.station_rects = []
        
        self.teleporter_level = 0
        self.teleporter_base_rect = None
        self.energy_level = 0
        
        self.resources = {}
        self.resource_types = ['repair', 'module', 'upgrade']
        self.selected_resource_idx = 0
        
        self.reticle_pos = [0, 0]
        
        self.last_space_held = False
        self.last_shift_held = False

        self.teleporter_animation = None # {'end_pos', 'timer'}
        
        # The original code called reset() and validate_implementation() here.
        # It's better practice to let the user call reset() after __init__().
        # self.reset()
        # self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        
        self.station_integrity = self.INITIAL_INTEGRITY
        self.station_size = 1
        
        core_size = 60
        self.station_core = pygame.Rect(
            (self.SCREEN_WIDTH - core_size) // 2,
            (self.SCREEN_HEIGHT - core_size) // 2,
            core_size, core_size
        )
        self.station_modules = []
        self._update_station_rects()

        self.teleporter_level = 1
        teleporter_base_size = (80, 20)
        self.teleporter_base_rect = pygame.Rect(
            (self.SCREEN_WIDTH - teleporter_base_size[0]) // 2,
            self.SCREEN_HEIGHT - teleporter_base_size[1] - 30,
            teleporter_base_size[0], teleporter_base_size[1]
        )
        self.energy_level = self.INITIAL_ENERGY
        
        self.resources = self.INITIAL_RESOURCES.copy()
        self.selected_resource_idx = 0
        
        self.reticle_pos = [self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2]
        
        self.last_space_held = False
        self.last_shift_held = False
        
        self.particles = []
        self.teleporter_animation = None
        
        if not self.stars:
            self.stars = [
                (random.randint(0, self.SCREEN_WIDTH), random.randint(0, self.SCREEN_HEIGHT), random.randint(1, 2))
                for _ in range(200)
            ]
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = 0.1  # Survival reward
        
        # --- 1. Handle Actions ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        self._move_reticle(movement)
        
        if shift_held and not self.last_shift_held:
            self.selected_resource_idx = (self.selected_resource_idx + 1) % len(self.resource_types)
            # SFX: UI_SWITCH
        
        if space_held and not self.last_space_held:
            teleport_reward = self._fire_teleporter()
            reward += teleport_reward

        self.last_space_held = space_held
        self.last_shift_held = shift_held
        
        # --- 2. Update Game State ---
        self.steps += 1
        
        # Update animations and effects
        self._update_particles()
        if self.teleporter_animation:
            self.teleporter_animation['timer'] -= 1
            if self.teleporter_animation['timer'] <= 0:
                self.teleporter_animation = None
        
        # Station degradation
        degradation = self.BASE_DEGRADATION_RATE + (self.station_size - 1) * self.DEGRADATION_PER_MODULE
        self.station_integrity = max(0, self.station_integrity - degradation)
        
        # Energy recharge
        self.energy_level = min(self.INITIAL_ENERGY, self.energy_level + self.ENERGY_RECHARGE_RATE)
        
        # --- 3. Check Termination ---
        terminated = False
        truncated = False
        if self.station_integrity <= 0:
            terminated = True
            reward -= 100
            self.game_over = True
            # SFX: STATION_DESTROYED
        
        if self.station_size >= self.MAX_STATION_SIZE and self.teleporter_level >= self.MAX_TELEPORTER_LEVEL:
            terminated = True
            reward += 100
            self.game_over = True
            self.win = True
            # SFX: GAME_WIN
            
        if self.steps >= self.MAX_STEPS:
            truncated = True # Use truncated for time limit
            self.game_over = True

        self.score += reward
        
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _fire_teleporter(self):
        resource_type = self.resource_types[self.selected_resource_idx]
        cost = self.TELEPORT_BASE_COST - (self.teleporter_level - 1) * self.TELEPORT_COST_REDUCTION_PER_LEVEL
        
        if self.resources[resource_type] > 0 and self.energy_level >= cost:
            # SFX: TELEPORT_FIRE
            self.energy_level -= cost
            self.resources[resource_type] -= 1
            
            self.teleporter_animation = {
                'end_pos': self.reticle_pos.copy(),
                'timer': self.FPS // 3 # Animation lasts 1/3 second
            }
            
            # Check for hit
            hit_station = any(rect.collidepoint(self.reticle_pos) for rect in self.station_rects)
            hit_teleporter = self.teleporter_base_rect.collidepoint(self.reticle_pos)

            if resource_type == 'repair' and hit_station:
                self.station_integrity = min(self.INITIAL_INTEGRITY, self.station_integrity + self.REPAIR_AMOUNT)
                self._add_particles(30, self.reticle_pos, self.COLOR_REPAIR, 3)
                # SFX: REPAIR_SUCCESS
                return 1.0
            
            elif resource_type == 'module' and hit_station and self.station_size < self.MAX_STATION_SIZE:
                if self._add_station_module():
                    self._add_particles(50, self.reticle_pos, self.COLOR_EXPAND, 4)
                    # SFX: EXPAND_SUCCESS
                    return 5.0
            
            elif resource_type == 'upgrade' and hit_teleporter and self.teleporter_level < self.MAX_TELEPORTER_LEVEL:
                self.teleporter_level += 1
                self._add_particles(50, self.reticle_pos, self.COLOR_UPGRADE, 4)
                # SFX: UPGRADE_SUCCESS
                return 10.0
            
            else:
                # Missed target or invalid action
                self._add_particles(20, self.reticle_pos, (100, 100, 100), 2)
                # SFX: TELEPORT_FAIL
                return -0.5

        return 0 # Not enough resources or energy

    def _move_reticle(self, movement):
        if movement == 1: self.reticle_pos[1] -= self.RETICLE_SPEED
        elif movement == 2: self.reticle_pos[1] += self.RETICLE_SPEED
        elif movement == 3: self.reticle_pos[0] -= self.RETICLE_SPEED
        elif movement == 4: self.reticle_pos[0] += self.RETICLE_SPEED
        
        self.reticle_pos[0] = np.clip(self.reticle_pos[0], 0, self.SCREEN_WIDTH)
        self.reticle_pos[1] = np.clip(self.reticle_pos[1], 0, self.SCREEN_HEIGHT)

    def _add_station_module(self):
        module_size = 40
        attachment_points = []
        all_rects = self.station_rects
        
        for rect in all_rects:
            attachment_points.extend([
                (rect.left - module_size, rect.centery - module_size//2),
                (rect.right, rect.centery - module_size//2),
                (rect.centerx - module_size//2, rect.top - module_size),
                (rect.centerx - module_size//2, rect.bottom),
            ])
        
        random.shuffle(attachment_points)
        
        for p in attachment_points:
            new_module_rect = pygame.Rect(p[0], p[1], module_size, module_size)
            if not any(new_module_rect.colliderect(r) for r in all_rects):
                self.station_modules.append(new_module_rect)
                self.station_size += 1
                self._update_station_rects()
                return True
        return False
    
    def _update_station_rects(self):
        self.station_rects = [self.station_core] + self.station_modules

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_game()
        self._render_effects()
        self._render_ui()
        
        if self.game_over:
            self._render_game_over()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "station_integrity": self.station_integrity,
            "station_size": self.station_size,
            "teleporter_level": self.teleporter_level,
            "energy": self.energy_level
        }

    def _render_background(self):
        for x, y, size in self.stars:
            self.screen.set_at((x, y), self.COLOR_STARS)
        for i in range(0, self.SCREEN_WIDTH, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (i, 0), (i, self.SCREEN_HEIGHT))
        for i in range(0, self.SCREEN_HEIGHT, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, i), (self.SCREEN_WIDTH, i))

    def _render_game(self):
        # Station integrity color
        if self.station_integrity > 70: station_color = self.COLOR_STATION_HEALTHY
        elif self.station_integrity > 30: station_color = self.COLOR_STATION_WARN
        else: station_color = self.COLOR_STATION_CRIT
        
        # Station damage flicker
        if self.station_integrity < 30 and self.steps % 10 < 5:
            station_color = (max(0, station_color[0]-100), max(0, station_color[1]-100), max(0, station_color[2]-100))

        # Draw station modules and core
        for rect in self.station_rects:
            self._draw_wireframe_rect(self.screen, rect, station_color, 2)
            # Blinking lights
            if self.steps % 40 > 20:
                pygame.gfxdraw.filled_circle(self.screen, rect.centerx, rect.centery, 3, station_color)
            else:
                pygame.gfxdraw.filled_circle(self.screen, rect.centerx, rect.centery, 2, self.COLOR_BG)

        # Draw teleporter base
        self._draw_wireframe_rect(self.screen, self.teleporter_base_rect, self.COLOR_ENERGY, 2)
        pygame.gfxdraw.filled_circle(self.screen, self.teleporter_base_rect.centerx, self.teleporter_base_rect.centery, 5, self.COLOR_ENERGY if self.steps % 20 > 10 else self.COLOR_BG)

    def _render_effects(self):
        # Particles
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            radius = int(p['life'] / p['max_life'] * p['size'])
            if radius > 0:
                pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), radius, p['color'])

        # Teleporter Beam
        if self.teleporter_animation:
            start_pos = self.teleporter_base_rect.midtop
            end_pos = self.teleporter_animation['end_pos']
            alpha = int(255 * (self.teleporter_animation['timer'] / (self.FPS // 3)))
            glow_color = (*self.COLOR_TELEPORTER_BEAM, alpha // 4)
            main_color = (*self.COLOR_TELEPORTER_BEAM, alpha)
            
            pygame.draw.line(self.screen, glow_color, start_pos, end_pos, 15)
            pygame.draw.line(self.screen, main_color, start_pos, end_pos, 5)
            pygame.gfxdraw.filled_circle(self.screen, int(end_pos[0]), int(end_pos[1]), 10, glow_color)
            pygame.gfxdraw.filled_circle(self.screen, int(end_pos[0]), int(end_pos[1]), 5, main_color)
        
        # Reticle
        rx, ry = int(self.reticle_pos[0]), int(self.reticle_pos[1])
        size = 12
        pygame.draw.line(self.screen, self.COLOR_RETICLE, (rx - size, ry), (rx + size, ry), 2)
        pygame.draw.line(self.screen, self.COLOR_RETICLE, (rx, ry - size), (rx, ry + size), 2)
        pygame.gfxdraw.aacircle(self.screen, rx, ry, size // 2, self.COLOR_RETICLE)

        # Scanlines
        scanline_surf = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        for y in range(0, self.SCREEN_HEIGHT, 3):
            pygame.draw.line(scanline_surf, (0, 0, 0, 60), (0, y), (self.SCREEN_WIDTH, y), 1)
        self.screen.blit(scanline_surf, (0, 0))

    def _render_ui(self):
        # HUD Background
        hud_surf = pygame.Surface((self.SCREEN_WIDTH, 60), pygame.SRCALPHA)
        hud_surf.fill(self.COLOR_HUD_BG)
        self.screen.blit(hud_surf, (0, 0))
        pygame.draw.line(self.screen, self.COLOR_TEXT, (0, 60), (self.SCREEN_WIDTH, 60), 1)

        # Integrity
        integrity_color = self.COLOR_STATION_HEALTHY if self.station_integrity > 70 else self.COLOR_STATION_WARN if self.station_integrity > 30 else self.COLOR_STATION_CRIT
        self._draw_text(f"INTEGRITY: {self.station_integrity:.1f}%", (10, 10), integrity_color, self.font_medium)
        
        # Energy
        self._draw_text("ENERGY", (10, 35), self.COLOR_ENERGY, self.font_small)
        pygame.draw.rect(self.screen, self.COLOR_ENERGY_BG, (70, 35, 150, 15))
        energy_width = int(150 * (self.energy_level / self.INITIAL_ENERGY))
        pygame.draw.rect(self.screen, self.COLOR_ENERGY, (70, 35, energy_width, 15))

        # Resources
        start_x = 280
        for i, res_type in enumerate(self.resource_types):
            color = self.COLOR_REPAIR if res_type == 'repair' else self.COLOR_EXPAND if res_type == 'module' else self.COLOR_UPGRADE
            is_selected = i == self.selected_resource_idx
            
            box_rect = pygame.Rect(start_x + i * 110, 5, 100, 50)
            bg_color = (*color, 80) if is_selected else (*color, 20)
            pygame.draw.rect(self.screen, bg_color, box_rect)
            pygame.draw.rect(self.screen, color, box_rect, 2 if is_selected else 1)
            
            label = res_type.upper()
            count = self.resources[res_type]
            self._draw_text(f"{label}: {count}", (box_rect.x + 10, box_rect.y + 15), color, self.font_small)
        
        # Score and Level
        self._draw_text(f"SCORE: {int(self.score)}", (self.SCREEN_WIDTH - 150, 10), self.COLOR_TEXT, self.font_small)
        self._draw_text(f"STATION: {self.station_size}/{self.MAX_STATION_SIZE}", (self.SCREEN_WIDTH - 150, 25), self.COLOR_TEXT, self.font_small)
        self._draw_text(f"TELEPORTER: {self.teleporter_level}/{self.MAX_TELEPORTER_LEVEL}", (self.SCREEN_WIDTH - 150, 40), self.COLOR_TEXT, self.font_small)

    def _render_game_over(self):
        overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))
        
        message = "MISSION ACCOMPLISHED" if self.win else "STATION LOST"
        color = self.COLOR_STATION_HEALTHY if self.win else self.COLOR_STATION_CRIT
        
        self._draw_text(message, (self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2 - 30), color, self.font_large, center=True)
        self._draw_text(f"Final Score: {int(self.score)}", (self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2 + 30), self.COLOR_TEXT, self.font_medium, center=True)

    def _draw_text(self, text, pos, color, font, center=False):
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect()
        if center: text_rect.center = pos
        else: text_rect.topleft = pos
        self.screen.blit(text_surface, text_rect)
        
    def _draw_wireframe_rect(self, surface, rect, color, width):
        pygame.draw.rect(surface, color, rect, width)
        # Glow effect
        glow_color = (*color, 50)
        glow_rect = rect.inflate(width*2, width*2)
        pygame.draw.rect(surface, glow_color, glow_rect, width)

    def _add_particles(self, count, pos, color, speed_mult):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 3) * speed_mult
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifetime = random.randint(10, 20)
            self.particles.append({
                'pos': list(pos), 'vel': vel, 'color': color, 
                'life': lifetime, 'max_life': lifetime, 'size': random.randint(2, 5)
            })

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]

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
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block is for manual play and is not part of the Gymnasium environment
    # It will not be run by the evaluation server.
    # We need to switch the video driver to a real one.
    os.environ["SDL_VIDEODRIVER"] = "x11" 
    
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    pygame.display.set_caption("Station Keeper")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    
    movement = 0
    space = 0
    shift = 0
    
    terminated = False
    truncated = False
    
    while not (terminated or truncated):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE: space = 1
                if event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT: shift = 1
            if event.type == pygame.KEYUP:
                if event.key == pygame.K_SPACE: space = 0
                if event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT: shift = 0

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        else: movement = 0

        action = [movement, space, shift]
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Steps: {info['steps']}")
            # Render one last time to show the game over screen
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            pygame.time.wait(3000) # Show final screen
            # obs, info = env.reset() # This would start a new game
            break

        # Render to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(GameEnv.FPS)
        
    env.close()
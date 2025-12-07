import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T15:34:51.534569
# Source Brief: brief_01071.md
# Brief Index: 1071
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    GameEnv: Defend your base from enemy waves by strategically managing your mech.

    Core Gameplay Loop:
    - Enemies spawn in waves from the right and advance towards your base on the left.
    - Your mech automatically fires at the nearest enemy.
    - You gather resources over time.
    - Use actions to move your mech, spend resources to upgrade it (SHIFT),
      and use a time-pausing 'reposition' ability to teleport (SPACE).
    - Survive all waves to win. If your base is destroyed, you lose.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Defend your base from waves of enemies by moving your mech, upgrading its abilities, "
        "and using a tactical repositioning ability."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move. Hold SPACE to pause and select a teleport location. "
        "Hold SHIFT to purchase upgrades."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_SIZE = 40
    GRID_W = SCREEN_WIDTH // GRID_SIZE
    GRID_H = SCREEN_HEIGHT // GRID_SIZE
    FPS = 30
    MAX_STEPS = 4500 # Approx 2.5 minutes at 30 FPS
    TOTAL_WAVES = 20

    # Colors (Player=Blue, Enemy=Red, Resource=Green, Effect=Yellow, UI=White)
    COLOR_BG = (15, 18, 32)
    COLOR_GRID = (30, 35, 60)
    COLOR_PLAYER = (0, 150, 255)
    COLOR_PLAYER_GLOW = (0, 150, 255, 50)
    COLOR_ENEMY = (255, 50, 50)
    COLOR_ENEMY_GLOW = (255, 50, 50, 50)
    COLOR_BASE = (100, 120, 150)
    COLOR_P_BULLET = (120, 200, 255)
    COLOR_E_BULLET = (255, 150, 50)
    COLOR_RESOURCE = (50, 255, 50)
    COLOR_YELLOW = (255, 220, 0)
    COLOR_WHITE = (220, 220, 220)
    COLOR_BLACK = (0, 0, 0)
    COLOR_GREEN = (50, 255, 50) # Added for victory message

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = Box(low=0, high=255, shape=(400, 640, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 20)
        self.font_medium = pygame.font.Font(None, 28)
        self.font_large = pygame.font.Font(None, 48)

        # Game state variables are initialized in reset()
        self._initialize_attributes()
        
        self.reset()
        # self.validate_implementation() # Commented out for production

    def _initialize_attributes(self):
        """Initialize all attributes to prevent potential AttributeError."""
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.reward_this_step = 0.0

        self.base_health = 0
        self.max_base_health = 1000

        self.resources = 0
        self.resource_timer = 0

        self.wave_number = 0
        self.wave_timer = 0
        self.wave_in_progress = False
        self.wave_difficulty_mod = 1.0

        self.mech = {}
        self.enemies = []
        self.player_projectiles = []
        self.enemy_projectiles = []
        self.particles = []

        self.tech_level = 0
        self.upgrades = self._get_upgrade_definitions()

        # Action state
        self.is_repositioning = False
        self.reposition_target = [0, 0]
        self.reposition_cooldown = 0
        self.max_reposition_cooldown = 90  # 3 seconds

    def _get_upgrade_definitions(self):
        return [
            {"name": "Fire Rate I", "cost": 50, "effect": ("fire_rate", 0.8)},
            {"name": "Damage I", "cost": 75, "effect": ("damage", 1.5)},
            {"name": "Mech HP I", "cost": 100, "effect": ("max_health", 1.5)},
            {"name": "Fire Rate II", "cost": 125, "effect": ("fire_rate", 0.8)},
            {"name": "Base Repair", "cost": 150, "effect": ("base_repair", 250)},
            {"name": "Damage II", "cost": 200, "effect": ("damage", 1.5)},
            {"name": "Mech HP II", "cost": 250, "effect": ("max_health", 1.5)},
            {"name": "Fire Rate III", "cost": 300, "effect": ("fire_rate", 0.8)},
        ]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._initialize_attributes()
        
        self.base_health = self.max_base_health
        self.mech = self._create_mech()
        
        self._start_next_wave()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        self.reward_this_step = 0.0
        self.game_over = self._check_termination()
        
        if not self.game_over:
            self._handle_input(action)
            self._update_game_state()
            self.steps += 1
        
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS
        
        if terminated and not self.game_over: # Handle terminal state reward
            if self.wave_number > self.TOTAL_WAVES:
                self.reward_this_step += 100 # Victory
            else:
                self.reward_this_step -= 100 # Defeat
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, self.reward_this_step, terminated, truncated, info

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # --- Repositioning (Space Bar) ---
        if self.reposition_cooldown == 0 and space_held and not self.is_repositioning:
            self.is_repositioning = True
            self.reposition_target = list(self.mech['pos'])
            # sfx: time_pause_activate
        
        if self.is_repositioning:
            if movement == 1: self.reposition_target[1] -= 1
            if movement == 2: self.reposition_target[1] += 1
            if movement == 3: self.reposition_target[0] -= 1
            if movement == 4: self.reposition_target[0] += 1
            self.reposition_target[0] = np.clip(self.reposition_target[0], 0, self.GRID_W - 1)
            self.reposition_target[1] = np.clip(self.reposition_target[1], 0, self.GRID_H - 1)

            if not space_held:
                self.mech['pos'] = list(self.reposition_target)
                self.is_repositioning = False
                self.reposition_cooldown = self.max_reposition_cooldown
                self._create_particle_burst(self.mech['pixel_pos'], self.COLOR_YELLOW, 20)
                # sfx: teleport_whoosh
        
        # --- Normal Movement & Crafting (Not repositioning) ---
        else:
            # Movement
            target_pos = list(self.mech['pos'])
            if movement == 1: target_pos[1] -= 1
            if movement == 2: target_pos[1] += 1
            if movement == 3: target_pos[0] -= 1
            if movement == 4: target_pos[0] += 1
            
            target_pos[0] = np.clip(target_pos[0], 0, self.GRID_W - 1)
            target_pos[1] = np.clip(target_pos[1], 0, self.GRID_H - 1)
            self.mech['pos'] = target_pos

            # Crafting
            if shift_held:
                self._attempt_craft()

    def _update_game_state(self):
        if self.is_repositioning:
            # Game is paused, only update cooldowns and particles
            self.reposition_cooldown = max(0, self.reposition_cooldown - 1)
            self._update_particles()
            return
        
        # Update timers
        self.resource_timer += 1
        self.wave_timer -= 1
        self.reposition_cooldown = max(0, self.reposition_cooldown - 1)

        # Resource generation
        if self.resource_timer >= self.FPS * 2: # 1 resource every 2 seconds
            self.resource_timer = 0
            self.resources += 1
            self.reward_this_step += 0.01

        # Wave management
        if self.wave_timer <= 0 and not self.wave_in_progress:
            self._start_next_wave()
        
        if self.wave_in_progress and not self.enemies:
            self.wave_in_progress = False
            self.wave_timer = self.FPS * 5 # 5 seconds between waves
            self.reward_this_step += 5 # Wave survived reward
            # sfx: wave_complete

        # Update mech
        self._update_mech()
        
        # Update enemies
        for enemy in self.enemies[:]:
            self._update_enemy(enemy)
        
        # Update projectiles
        self._update_projectiles(self.player_projectiles, self.enemies, True)
        self._update_projectiles(self.enemy_projectiles, [self.mech], False)

        # Update particles for visual effects
        self._update_particles()

    def _create_mech(self):
        start_pos = [2, self.GRID_H // 2]
        return {
            "pos": start_pos,
            "pixel_pos": [start_pos[0] * self.GRID_SIZE + self.GRID_SIZE/2, start_pos[1] * self.GRID_SIZE + self.GRID_SIZE/2],
            "max_health": 100,
            "health": 100,
            "fire_rate": 20, # frames per shot
            "fire_cooldown": 0,
            "damage": 10,
            "range": 250,
            "size": 15,
        }

    def _update_mech(self):
        # Smooth movement
        target_px = (self.mech['pos'][0] * self.GRID_SIZE + self.GRID_SIZE/2, self.mech['pos'][1] * self.GRID_SIZE + self.GRID_SIZE/2)
        self.mech['pixel_pos'][0] += (target_px[0] - self.mech['pixel_pos'][0]) * 0.25
        self.mech['pixel_pos'][1] += (target_px[1] - self.mech['pixel_pos'][1]) * 0.25

        # Firing logic
        self.mech['fire_cooldown'] = max(0, self.mech['fire_cooldown'] - 1)
        if self.mech['fire_cooldown'] == 0 and self.enemies:
            # Find closest enemy
            closest_enemy = min(self.enemies, key=lambda e: math.dist(self.mech['pixel_pos'], e['pixel_pos']))
            if math.dist(self.mech['pixel_pos'], closest_enemy['pixel_pos']) <= self.mech['range']:
                self._fire_projectile(self.mech, closest_enemy, self.player_projectiles)
                self.mech['fire_cooldown'] = self.mech['fire_rate']
                # sfx: player_shoot
        
        if self.mech['health'] <= 0:
            self.game_over = True # Mech destruction is not a loss condition, but could be added

    def _create_enemy(self):
        start_y = self.np_random.integers(0, self.GRID_H)
        start_pos = [self.GRID_W - 1, start_y]
        base_health = 20
        base_damage = 5
        return {
            "pos": start_pos,
            "pixel_pos": [start_pos[0] * self.GRID_SIZE + self.GRID_SIZE/2, start_pos[1] * self.GRID_SIZE + self.GRID_SIZE/2],
            "max_health": base_health * self.wave_difficulty_mod,
            "health": base_health * self.wave_difficulty_mod,
            "damage": base_damage * self.wave_difficulty_mod,
            "speed": self.np_random.uniform(0.5, 1.0),
            "size": 12,
            "attack_range": self.GRID_SIZE * 1.5,
            "fire_rate": 90, # Slower than player
            "fire_cooldown": self.np_random.integers(0, 90),
        }

    def _update_enemy(self, enemy):
        # Movement logic
        base_pos_px = (self.GRID_SIZE/2, self.SCREEN_HEIGHT/2)
        dist_to_base = math.dist(enemy['pixel_pos'], base_pos_px)

        if dist_to_base > enemy['attack_range']:
            angle = math.atan2(base_pos_px[1] - enemy['pixel_pos'][1], base_pos_px[0] - enemy['pixel_pos'][0])
            enemy['pixel_pos'][0] += math.cos(angle) * enemy['speed']
            enemy['pixel_pos'][1] += math.sin(angle) * enemy['speed']
            enemy['pos'] = [enemy['pixel_pos'][0] // self.GRID_SIZE, enemy['pixel_pos'][1] // self.GRID_SIZE]
        else: # In range, attack base
            enemy['fire_cooldown'] = max(0, enemy['fire_cooldown'] - 1)
            if enemy['fire_cooldown'] == 0:
                self.base_health -= enemy['damage']
                self._create_particle_burst(base_pos_px, self.COLOR_ENEMY, 10, speed_mult=0.5)
                enemy['fire_cooldown'] = enemy['fire_rate']
                # sfx: base_hit
        
        # Check if enemy reached base
        if enemy['pixel_pos'][0] < self.GRID_SIZE:
            self.base_health -= enemy['health'] # Kamikaze damage
            if enemy in self.enemies: self.enemies.remove(enemy)
            self._create_particle_burst(enemy['pixel_pos'], self.COLOR_ENEMY, 30)
            # sfx: explosion

    def _fire_projectile(self, source, target, projectile_list):
        angle = math.atan2(target['pixel_pos'][1] - source['pixel_pos'][1], target['pixel_pos'][0] - source['pixel_pos'][0])
        projectile = {
            "pos": list(source['pixel_pos']),
            "vel": [math.cos(angle) * 8, math.sin(angle) * 8],
            "damage": source['damage'],
            "size": 4,
            "owner": "player" if projectile_list is self.player_projectiles else "enemy"
        }
        projectile_list.append(projectile)

    def _update_projectiles(self, projectiles, targets, is_player):
        for p in projectiles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]

            # Out of bounds check
            if not (0 < p['pos'][0] < self.SCREEN_WIDTH and 0 < p['pos'][1] < self.SCREEN_HEIGHT):
                projectiles.remove(p)
                continue
            
            # Collision check
            for target in targets[:]:
                dist = math.dist(p['pos'], target['pixel_pos'])
                if dist < target['size'] + p['size']:
                    target['health'] -= p['damage']
                    self._create_particle_burst(p['pos'], self.COLOR_P_BULLET if is_player else self.COLOR_E_BULLET, 5)
                    if p in projectiles: projectiles.remove(p)
                    
                    if is_player:
                        self.reward_this_step += 0.1 # Hit reward
                        # sfx: enemy_hit
                    
                    if target['health'] <= 0:
                        if is_player: # Enemy destroyed
                            self.reward_this_step += 1.0
                            self.score += 1
                            self._create_particle_burst(target['pixel_pos'], self.COLOR_ENEMY, 40)
                            # sfx: enemy_explode
                            if target in self.enemies: self.enemies.remove(target)
                        else: # Player mech destroyed
                             self._create_particle_burst(target['pixel_pos'], self.COLOR_PLAYER, 50)
                             # In this design, mech destruction is not a loss. It could respawn.
                             # For simplicity, we just let it be destroyed. It can still be controlled.
                    break

    def _start_next_wave(self):
        if self.wave_number >= self.TOTAL_WAVES:
            self.wave_number += 1 # To trigger victory condition
            return
            
        self.wave_number += 1
        self.wave_difficulty_mod = 1 + (0.05 * (self.wave_number - 1))
        num_enemies = 3 + self.wave_number // 2
        
        for _ in range(num_enemies):
            self.enemies.append(self._create_enemy())
        
        self.wave_in_progress = True
        self.wave_timer = 9999 # Active until all enemies are gone

    def _attempt_craft(self):
        if self.tech_level >= len(self.upgrades):
            return # Max tech
        
        upgrade = self.upgrades[self.tech_level]
        if self.resources >= upgrade['cost']:
            self.resources -= upgrade['cost']
            self.tech_level += 1
            
            # Apply effect
            effect_type, value = upgrade['effect']
            if effect_type == "fire_rate": self.mech['fire_rate'] = int(self.mech['fire_rate'] * value)
            if effect_type == "damage": self.mech['damage'] *= value
            if effect_type == "max_health":
                self.mech['max_health'] *= value
                self.mech['health'] = self.mech['max_health'] # Full heal on upgrade
            if effect_type == "base_repair":
                self.base_health = min(self.max_base_health, self.base_health + value)

            self._create_particle_burst(self.mech['pixel_pos'], self.COLOR_RESOURCE, 30)
            # sfx: upgrade_success
    
    def _create_particle_burst(self, pos, color, count, speed_mult=1.0):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3) * speed_mult
            self.particles.append({
                "pos": list(pos),
                "vel": [math.cos(angle) * speed, math.sin(angle) * speed],
                "life": self.np_random.integers(15, 30),
                "color": color
            })
    
    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _check_termination(self):
        return self.base_health <= 0 or self.wave_number > self.TOTAL_WAVES

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.wave_number,
            "resources": self.resources,
            "base_health": self.base_health,
            "mech_health": self.mech['health'],
            "tech_level": self.tech_level,
        }
        
    def _render_background(self):
        for x in range(0, self.SCREEN_WIDTH, self.GRID_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, self.GRID_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))

    def _render_game(self):
        # Base
        base_rect = pygame.Rect(0, 0, self.GRID_SIZE, self.SCREEN_HEIGHT)
        pygame.draw.rect(self.screen, self.COLOR_BASE, base_rect)
        
        # Particles (rendered first to be in background)
        for p in self.particles:
            alpha = int(255 * (p['life'] / 30))
            color = (*p['color'], alpha)
            temp_surf = pygame.Surface((4, 4), pygame.SRCALPHA)
            pygame.draw.rect(temp_surf, color, temp_surf.get_rect())
            self.screen.blit(temp_surf, (int(p['pos'][0]-2), int(p['pos'][1]-2)), special_flags=pygame.BLEND_RGBA_ADD)

        # Repositioning Ghost
        if self.is_repositioning:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 50, 100, 100))
            self.screen.blit(overlay, (0, 0))
            
            r_pos = (self.reposition_target[0] * self.GRID_SIZE + self.GRID_SIZE/2, self.reposition_target[1] * self.GRID_SIZE + self.GRID_SIZE/2)
            pygame.gfxdraw.filled_circle(self.screen, int(r_pos[0]), int(r_pos[1]), self.mech['size'], (*self.COLOR_YELLOW, 100))
            pygame.gfxdraw.aacircle(self.screen, int(r_pos[0]), int(r_pos[1]), self.mech['size'], self.COLOR_YELLOW)

        # Player Projectiles
        for p in self.player_projectiles:
            pos = (int(p['pos'][0]), int(p['pos'][1]))
            pygame.draw.line(self.screen, self.COLOR_P_BULLET, pos, (pos[0]-p['vel'][0], pos[1]-p['vel'][1]), 3)

        # Mech
        m_pos = (int(self.mech['pixel_pos'][0]), int(self.mech['pixel_pos'][1]))
        pygame.gfxdraw.filled_circle(self.screen, m_pos[0], m_pos[1], self.mech['size'] + 5, self.COLOR_PLAYER_GLOW)
        pygame.gfxdraw.filled_circle(self.screen, m_pos[0], m_pos[1], self.mech['size'], self.COLOR_PLAYER)
        pygame.gfxdraw.aacircle(self.screen, m_pos[0], m_pos[1], self.mech['size'], self.COLOR_WHITE)
        self._render_health_bar(self.mech['pixel_pos'], self.mech['health'], self.mech['max_health'], 30)

        # Enemies
        for e in self.enemies:
            e_pos = (int(e['pixel_pos'][0]), int(e['pixel_pos'][1]))
            pygame.gfxdraw.filled_circle(self.screen, e_pos[0], e_pos[1], e['size'] + 4, self.COLOR_ENEMY_GLOW)
            pygame.gfxdraw.filled_circle(self.screen, e_pos[0], e_pos[1], e['size'], self.COLOR_ENEMY)
            pygame.gfxdraw.aacircle(self.screen, e_pos[0], e_pos[1], e['size'], self.COLOR_WHITE)
            self._render_health_bar(e['pixel_pos'], e['health'], e['max_health'], 24)

    def _render_health_bar(self, pos, current_hp, max_hp, width):
        if current_hp < max_hp:
            bar_w = width
            bar_h = 5
            bar_x = pos[0] - bar_w / 2
            bar_y = pos[1] - 30
            
            fill_ratio = max(0, current_hp / max_hp)
            
            pygame.draw.rect(self.screen, self.COLOR_BLACK, (bar_x - 1, bar_y - 1, bar_w + 2, bar_h + 2))
            pygame.draw.rect(self.screen, self.COLOR_ENEMY, (bar_x, bar_y, bar_w, bar_h))
            pygame.draw.rect(self.screen, self.COLOR_RESOURCE, (bar_x, bar_y, bar_w * fill_ratio, bar_h))

    def _render_ui(self):
        # Top Bar Background
        pygame.draw.rect(self.screen, self.COLOR_BLACK, (0, 0, self.SCREEN_WIDTH, 30))
        
        # Base Health
        base_hp_text = self.font_medium.render(f"Base HP: {int(self.base_health)} / {self.max_base_health}", True, self.COLOR_WHITE)
        self.screen.blit(base_hp_text, (10, 5))
        
        # Resources
        res_text = self.font_medium.render(f"Resources: {self.resources}", True, self.COLOR_RESOURCE)
        self.screen.blit(res_text, (230, 5))

        # Wave
        wave_text = self.font_medium.render(f"Wave: {self.wave_number} / {self.TOTAL_WAVES}", True, self.COLOR_WHITE)
        self.screen.blit(wave_text, (400, 5))

        # Score
        score_text = self.font_medium.render(f"Score: {int(self.score)}", True, self.COLOR_WHITE)
        self.screen.blit(score_text, (540, 5))
        
        # Next Upgrade
        if self.tech_level < len(self.upgrades):
            upgrade = self.upgrades[self.tech_level]
            cost_color = self.COLOR_RESOURCE if self.resources >= upgrade['cost'] else self.COLOR_ENEMY
            upgrade_str = f"Next Upgrade (SHIFT): {upgrade['name']}"
            upgrade_text = self.font_small.render(upgrade_str, True, self.COLOR_WHITE)
            cost_str = f"[{upgrade['cost']}]"
            cost_surf = self.font_small.render(cost_str, True, cost_color)
            
            full_surf = pygame.Surface((upgrade_text.get_width() + cost_surf.get_width() + 5, upgrade_text.get_height()), pygame.SRCALPHA)
            full_surf.blit(upgrade_text, (0,0))
            full_surf.blit(cost_surf, (upgrade_text.get_width() + 5, 0))
            
            self.screen.blit(full_surf, (10, self.SCREEN_HEIGHT - 22))

        # Reposition Cooldown
        if self.reposition_cooldown > 0:
            cooldown_ratio = self.reposition_cooldown / self.max_reposition_cooldown
            pygame.draw.rect(self.screen, self.COLOR_YELLOW, (self.SCREEN_WIDTH - 110, self.SCREEN_HEIGHT - 22, 100 * cooldown_ratio, 15))
            pygame.draw.rect(self.screen, self.COLOR_WHITE, (self.SCREEN_WIDTH - 110, self.SCREEN_HEIGHT - 22, 100, 15), 1)
            text = self.font_small.render("Reposition", True, self.COLOR_WHITE)
            self.screen.blit(text, (self.SCREEN_WIDTH - 110 + (100 - text.get_width())//2, self.SCREEN_HEIGHT - 21))


        # Game Over Message
        if self._check_termination():
            is_victory = self.wave_number > self.TOTAL_WAVES
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            message = "VICTORY" if is_victory else "BASE DESTROYED"
            color = self.COLOR_GREEN if is_victory else self.COLOR_ENEMY
            
            end_text = self.font_large.render(message, True, color)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def close(self):
        pygame.quit()

# Example of how to run the environment
if __name__ == '__main__':
    # To run with display, unset the dummy video driver
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
        
    env = GameEnv(render_mode="rgb_array")
    
    # --- Manual Play ---
    obs, info = env.reset()
    done = False
    
    # Pygame setup for display
    pygame.display.set_caption("Mech Defender")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    while running:
        # Action mapping for human player
        keys = pygame.key.get_pressed()
        movement = 0 # None
        if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
        if keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
        if keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if reward != 0:
            print(f"Step: {info['steps']}, Reward: {reward:.2f}, Score: {info['score']}, Terminated: {terminated}")
            
        if terminated or truncated:
            print("Game Over!")
            # Wait a bit before resetting
            pygame.time.wait(3000)
            obs, info = env.reset()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        clock.tick(GameEnv.FPS)
        
    env.close()
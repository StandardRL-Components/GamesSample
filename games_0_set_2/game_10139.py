import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T13:59:24.265402
# Source Brief: brief_00139.md
# Brief Index: 139
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque

class GameEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    game_description = (
        "Defend the central nucleus from waves of invading pathogens by strategically placing defensive organelles. "
        "Manage resources and unlock new organelles to survive."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move the cursor. Press space to place the selected organelle "
        "and use shift to cycle between available types."
    )
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Critical Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.MAX_STEPS = 1500  # Increased for 5 waves
        self.NUM_WAVES = 5
        self.render_mode = render_mode

        # --- Colors ---
        self.COLOR_BG = (15, 5, 25)
        self.COLOR_NUCLEUS = (255, 220, 50)
        self.COLOR_CURSOR = (255, 255, 255)
        self.COLOR_UI_BG = (30, 10, 50, 200)
        self.COLOR_UI_TEXT = (240, 240, 255)
        self.COLOR_UI_BAR = (40, 20, 70)
        self.COLOR_UI_BAR_FILL = (100, 80, 200)
        
        # --- Entity Definitions ---
        self.ORGANELLE_DEFS = {
            0: {"name": "Lysosome", "color": (50, 255, 100), "radius": 12, "range": 120, "cooldown": 45, "cost": 100, "projectile_speed": 5, "damage": 25, "weakness": 0},
            1: {"name": "Ribosome", "color": (50, 180, 255), "radius": 10, "range": 150, "cooldown": 30, "cost": 125, "projectile_speed": 7, "damage": 15, "weakness": 1},
            2: {"name": "Mitochondrion", "color": (255, 150, 50), "radius": 15, "range": 100, "cooldown": 60, "cost": 150, "projectile_speed": 4, "damage": 40, "weakness": 2}
        }
        self.PATHOGEN_DEFS = {
            0: {"name": "Pathogen A", "color": (255, 80, 80), "radius": 8, "speed": 0.5, "health": 100},
            1: {"name": "Pathogen B", "color": (200, 100, 255), "radius": 7, "speed": 0.7, "health": 75},
            2: {"name": "Pathogen C", "color": (80, 120, 255), "radius": 10, "speed": 0.4, "health": 150}
        }
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 16, bold=True)
        self.font_wave = pygame.font.SysFont("monospace", 24, bold=True)
        self.human_screen = None
        if self.render_mode == "human":
            self.human_screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
            pygame.display.set_caption("Cell Defense")

        # --- Game State (initialized in reset) ---
        self.steps = None
        self.score = None
        self.game_over = None
        self.win = None
        self.wave_number = None
        self.wave_spawning_complete = None
        self.wave_spawn_queue = None
        
        self.cursor_pos = None
        self.cursor_speed = 8
        
        self.nucleus_pos = pygame.Vector2(self.WIDTH // 2, self.HEIGHT // 2)
        self.nucleus_radius = 35

        self.organelles = None
        self.pathogens = None
        self.projectiles = None
        self.particles = None
        
        self.cloning_cooldown = None
        self.cloning_cooldown_max = 90  # 3s at 30fps
        self.selected_organelle_idx = None
        self.unlocked_organelle_types = None
        self.last_shift_held = None
        self.last_space_held = None
        self.resources = None
        
        self.wave_definitions = self._define_waves()
        self.bg_stars = [(random.randint(0, self.WIDTH), random.randint(0, self.HEIGHT), random.randint(1, 2)) for _ in range(100)]
        
        # This will set initial dummy values to pass validation
        self._initialize_state()
    
    def _initialize_state(self):
        """Initializes all state variables for a new episode."""
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.wave_number = 1
        
        self.cursor_pos = pygame.Vector2(self.WIDTH // 2, self.HEIGHT // 2 + 100)
        
        self.organelles = []
        self.pathogens = []
        self.projectiles = []
        self.particles = deque(maxlen=200)
        
        self.cloning_cooldown = 0
        self.selected_organelle_idx = 0
        self.unlocked_organelle_types = [0, 1]
        self.last_shift_held = False
        self.last_space_held = False
        self.resources = 250
        
        self._start_wave(self.wave_number)

    def _define_waves(self):
        waves = {}
        base_count = 8
        for i in range(1, self.NUM_WAVES + 1):
            wave_list = []
            enemy_types = [0, 1] if i < 3 else [0, 1, 2]
            for j in range(base_count + i * 2):
                spawn_time = j * (60 // i) + 60 # Spawn faster in later waves
                enemy_type = random.choice(enemy_types)
                wave_list.append((spawn_time, enemy_type))
            waves[i] = sorted(wave_list)
        return waves
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._initialize_state()
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0.0
        self.steps += 1
        
        # --- Handle Input ---
        reward += self._handle_input(action)
        
        # --- Update Game Logic ---
        self._update_spawning()
        self._update_organelles()
        self._update_pathogens()
        reward += self._update_projectiles()
        self._update_particles()

        # --- Check Game State ---
        if not self.pathogens and self.wave_spawning_complete:
            reward += 5.0
            self.score += 5
            self.wave_number += 1
            if self.wave_number > self.NUM_WAVES:
                self.win = True
            else:
                self._start_wave(self.wave_number)
                self.resources += 150 # Bonus resources between waves
                # Unlock Mitochondrion after wave 2 is cleared
                if self.wave_number == 3 and 2 not in self.unlocked_organelle_types:
                    self.unlocked_organelle_types.append(2)

        terminated = self.game_over or self.win or self.steps >= self.MAX_STEPS
        
        if terminated:
            if self.win:
                reward = 100.0
                self.score += 100
            elif self.game_over: # Nucleus breached
                reward = -100.0
                self.score -= 100

        truncated = self.steps >= self.MAX_STEPS
        
        if self.render_mode == "human":
            self._render_human()
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_val, shift_val = action
        space_held = space_val == 1
        shift_held = shift_val == 1
        
        # Movement
        if movement == 1: self.cursor_pos.y -= self.cursor_speed
        elif movement == 2: self.cursor_pos.y += self.cursor_speed
        elif movement == 3: self.cursor_pos.x -= self.cursor_speed
        elif movement == 4: self.cursor_pos.x += self.cursor_speed
        
        self.cursor_pos.x = np.clip(self.cursor_pos.x, 0, self.WIDTH)
        self.cursor_pos.y = np.clip(self.cursor_pos.y, 0, self.HEIGHT - 60) # Keep out of UI

        # Cycle Selection (on press)
        if shift_held and not self.last_shift_held:
            self.selected_organelle_idx = (self.selected_organelle_idx + 1) % len(self.unlocked_organelle_types)
        self.last_shift_held = shift_held

        # Clone Organelle (on press)
        reward = 0
        if space_held and not self.last_space_held:
            selected_type_id = self.unlocked_organelle_types[self.selected_organelle_idx]
            cost = self.ORGANELLE_DEFS[selected_type_id]["cost"]
            if self.cloning_cooldown <= 0 and self.resources >= cost:
                # Check for placement validity (not on another organelle or nucleus)
                can_place = True
                new_radius = self.ORGANELLE_DEFS[selected_type_id]['radius']
                if self.cursor_pos.distance_to(self.nucleus_pos) < self.nucleus_radius + new_radius + 5:
                    can_place = False
                for org in self.organelles:
                    if self.cursor_pos.distance_to(org['pos']) < org['radius'] + new_radius:
                        can_place = False
                        break
                
                if can_place:
                    # SFX: Place Organelle
                    new_organelle = self.ORGANELLE_DEFS[selected_type_id].copy()
                    new_organelle.update({
                        "id": selected_type_id,
                        "pos": self.cursor_pos.copy(),
                        "shoot_cooldown": random.randint(0, new_organelle["cooldown"]) # Stagger initial fire
                    })
                    self.organelles.append(new_organelle)
                    self.cloning_cooldown = self.cloning_cooldown_max
                    self.resources -= cost
                    self._create_particles(self.cursor_pos, new_organelle['color'], 15, 2.0)
                    reward += 0.5 # Small reward for a successful action
        self.last_space_held = space_held
        
        if self.cloning_cooldown > 0:
            self.cloning_cooldown -= 1
            
        return reward

    def _start_wave(self, wave_num):
        self.wave_spawning_complete = False
        self.wave_spawn_queue = deque(self.wave_definitions[wave_num])

    def _update_spawning(self):
        if not self.wave_spawn_queue:
            self.wave_spawning_complete = True
            return
        
        if self.steps >= self.wave_spawn_queue[0][0]:
            _, pathogen_type = self.wave_spawn_queue.popleft()
            
            # Spawn pathogen at a random edge
            side = random.randint(0, 3)
            if side == 0: pos = pygame.Vector2(random.randint(0, self.WIDTH), -20)
            elif side == 1: pos = pygame.Vector2(random.randint(0, self.WIDTH), self.HEIGHT + 20)
            elif side == 2: pos = pygame.Vector2(-20, random.randint(0, self.HEIGHT))
            else: pos = pygame.Vector2(self.WIDTH + 20, random.randint(0, self.HEIGHT))
            
            path_def = self.PATHOGEN_DEFS[pathogen_type].copy()
            # Increase health and speed based on wave number
            wave_multiplier = 1 + (self.wave_number - 1) * 0.1
            path_def['health'] *= wave_multiplier
            path_def['speed'] *= wave_multiplier
            
            new_pathogen = {
                "id": pathogen_type,
                "pos": pos,
                "max_health": path_def['health'],
                **path_def
            }
            self.pathogens.append(new_pathogen)

    def _update_organelles(self):
        for org in self.organelles:
            if org['shoot_cooldown'] > 0:
                org['shoot_cooldown'] -= 1
            else:
                # Find a target in range
                target = None
                min_dist = org['range']
                for p in self.pathogens:
                    dist = org['pos'].distance_to(p['pos'])
                    if dist < min_dist:
                        min_dist = dist
                        target = p
                
                if target:
                    # SFX: Shoot Projectile
                    org['shoot_cooldown'] = org['cooldown']
                    new_projectile = {
                        "pos": org['pos'].copy(),
                        "color": org['color'],
                        "target_pos": target['pos'].copy(),
                        "speed": org['projectile_speed'],
                        "damage": org['damage'],
                        "is_weakness": target['id'] == org['weakness']
                    }
                    self.projectiles.append(new_projectile)

    def _update_pathogens(self):
        for p in self.pathogens:
            direction = (self.nucleus_pos - p['pos']).normalize()
            p['pos'] += direction * p['speed']
            
            if p['pos'].distance_to(self.nucleus_pos) < self.nucleus_radius + p['radius']:
                self.game_over = True
                self._create_particles(self.nucleus_pos, self.COLOR_NUCLEUS, 50, 5.0, 120)
                # SFX: Nucleus Breach
                break

    def _update_projectiles(self):
        reward = 0
        projectiles_to_remove = []
        pathogens_to_remove = []
        
        for i, proj in enumerate(self.projectiles):
            direction = (proj['target_pos'] - proj['pos'])
            if direction.length() < proj['speed']:
                proj['pos'] = proj['target_pos']
            else:
                proj['pos'] += direction.normalize() * proj['speed']
            
            hit = False
            for j, path in enumerate(self.pathogens):
                if proj['pos'].distance_to(path['pos']) < path['radius']:
                    # SFX: Hit
                    damage = proj['damage'] * 1.5 if proj['is_weakness'] else proj['damage'] * 0.5
                    path['health'] -= damage
                    reward += 0.1 # Reward for any damage
                    self.score += 0.1
                    self._create_particles(proj['pos'], proj['color'], 5, 1.0)
                    
                    if path['health'] <= 0 and j not in pathogens_to_remove:
                        # SFX: Enemy Destroyed
                        self._create_particles(path['pos'], path['color'], 20, 3.0)
                        pathogens_to_remove.append(j)
                        reward += 1.0
                        self.score += 1
                        self.resources += 20 # Gain resources from kills
                    
                    hit = True
                    break
            
            if hit or proj['pos'] == proj['target_pos']:
                projectiles_to_remove.append(i)

        # Remove entities in reverse order to avoid index errors
        for i in sorted(projectiles_to_remove, reverse=True):
            del self.projectiles[i]
        for i in sorted(pathogens_to_remove, reverse=True):
            del self.pathogens[i]
            
        return reward
        
    def _update_particles(self):
        for p in list(self.particles):
            p['pos'] += p['vel']
            p['lifetime'] -= 1
            if p['lifetime'] <= 0:
                self.particles.remove(p)

    def _create_particles(self, pos, color, count, power, lifetime=30):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(0.5, power)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            self.particles.append({
                "pos": pos.copy(),
                "vel": vel,
                "lifetime": random.randint(lifetime // 2, lifetime),
                "color": color,
                "radius": random.uniform(1, 3)
            })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Background stars
        for x, y, r in self.bg_stars:
            pygame.gfxdraw.filled_circle(self.screen, x, y, r, (50, 40, 70))
            
        # Nucleus
        self._draw_glowing_circle(self.screen, self.COLOR_NUCLEUS, self.nucleus_pos, self.nucleus_radius, 3)

        # Organelles
        for org in self.organelles:
            self._draw_glowing_circle(self.screen, org['color'], org['pos'], org['radius'], 2)
            # Range indicator when placing
            if self.cloning_cooldown > self.cloning_cooldown_max - 10:
                pygame.gfxdraw.aacircle(self.screen, int(org['pos'].x), int(org['pos'].y), org['range'], (*org['color'], 100))

        # Pathogens
        for p in self.pathogens:
            pygame.gfxdraw.filled_circle(self.screen, int(p['pos'].x), int(p['pos'].y), p['radius'], p['color'])
            pygame.gfxdraw.aacircle(self.screen, int(p['pos'].x), int(p['pos'].y), p['radius'], (255,255,255))
            # Health bar
            if p['health'] < p['max_health']:
                bar_len = p['radius'] * 2
                health_pct = max(0, p['health'] / p['max_health'])
                pygame.draw.rect(self.screen, (255,0,0), (p['pos'].x - bar_len/2, p['pos'].y - p['radius'] - 8, bar_len, 4))
                pygame.draw.rect(self.screen, (0,255,0), (p['pos'].x - bar_len/2, p['pos'].y - p['radius'] - 8, bar_len * health_pct, 4))

        # Projectiles
        for proj in self.projectiles:
            end_pos = proj['pos'] + (proj['pos'] - proj['target_pos']).normalize() * 5
            pygame.draw.line(self.screen, proj['color'], proj['pos'], end_pos, 3)

        # Particles
        for p in self.particles:
            alpha = int(255 * (p['lifetime'] / 30))
            if alpha > 0:
                color = (*p['color'], alpha)
                surf = pygame.Surface((p['radius']*2, p['radius']*2), pygame.SRCALPHA)
                pygame.draw.circle(surf, color, (p['radius'], p['radius']), p['radius'])
                self.screen.blit(surf, (p['pos'].x - p['radius'], p['pos'].y - p['radius']), special_flags=pygame.BLEND_RGBA_ADD)

        # Cursor
        self._draw_glowing_circle(self.screen, self.COLOR_CURSOR, self.cursor_pos, 8, 2, 150)
        selected_type = self.unlocked_organelle_types[self.selected_organelle_idx]
        org_def = self.ORGANELLE_DEFS[selected_type]
        pygame.gfxdraw.aacircle(self.screen, int(self.cursor_pos.x), int(self.cursor_pos.y), org_def['range'], (*org_def['color'], 50))
        
    def _render_ui(self):
        # UI Background Panel
        ui_panel = pygame.Surface((self.WIDTH, 60), pygame.SRCALPHA)
        ui_panel.fill(self.COLOR_UI_BG)
        self.screen.blit(ui_panel, (0, self.HEIGHT - 60))

        # Wave Info
        wave_text = self.font_wave.render(f"WAVE: {self.wave_number}/{self.NUM_WAVES}", True, self.COLOR_UI_TEXT)
        self.screen.blit(wave_text, (10, 10))

        # Resources Info
        resource_text = self.font_ui.render(f"RESOURCES: {self.resources}", True, self.COLOR_UI_TEXT)
        self.screen.blit(resource_text, (10, self.HEIGHT - 50))

        # Selected Organelle Info
        y_pos = self.HEIGHT - 50
        x_pos = 200
        
        # Cloning Cooldown Bar
        bar_width = 150
        bar_height = 15
        cooldown_pct = 1 - (self.cloning_cooldown / self.cloning_cooldown_max)
        pygame.draw.rect(self.screen, self.COLOR_UI_BAR, (x_pos, y_pos + 20, bar_width, bar_height))
        pygame.draw.rect(self.screen, self.COLOR_UI_BAR_FILL, (x_pos, y_pos + 20, bar_width * cooldown_pct, bar_height))

        # Organelle Selection Display
        for i, org_id in enumerate(self.unlocked_organelle_types):
            org_def = self.ORGANELLE_DEFS[org_id]
            is_selected = i == self.selected_organelle_idx
            
            box_x = x_pos + i * 80 + 200
            
            box_rect = pygame.Rect(box_x, y_pos - 5, 70, 40)
            border_color = self.COLOR_UI_TEXT if is_selected else self.COLOR_UI_BAR
            pygame.draw.rect(self.screen, self.COLOR_UI_BG, box_rect)
            pygame.draw.rect(self.screen, border_color, box_rect, 2)
            
            org_text = self.font_ui.render(f"{org_def['name']}", True, self.COLOR_UI_TEXT)
            self.screen.blit(org_text, (box_x + 35 - org_text.get_width()//2, y_pos))
            cost_text = self.font_ui.render(f"Cost: {org_def['cost']}", True, self.COLOR_UI_TEXT)
            self.screen.blit(cost_text, (box_x + 35 - cost_text.get_width()//2, y_pos + 15))
            
            if is_selected:
                pygame.draw.polygon(self.screen, self.COLOR_UI_TEXT, [(box_x + 35, y_pos + 38), (box_x + 30, y_pos + 33), (box_x + 40, y_pos + 33)])
    
    def _draw_glowing_circle(self, surface, color, center, radius, tiers, alpha=255):
        for i in range(tiers, 0, -1):
            glow_radius = int(radius + i * 2)
            glow_alpha = int(alpha / (tiers * 2) * (1 - (i / (tiers * 2))))
            if glow_alpha > 0:
                pygame.gfxdraw.filled_circle(surface, int(center.x), int(center.y), glow_radius, (*color, glow_alpha))
        pygame.gfxdraw.filled_circle(surface, int(center.x), int(center.y), radius, color)
        pygame.gfxdraw.aacircle(surface, int(center.x), int(center.y), radius, color)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.wave_number,
            "resources": self.resources,
            "organelles": len(self.organelles),
            "pathogens": len(self.pathogens),
        }
    
    def render(self):
        if self.render_mode == "human":
            self._render_human()

    def _render_human(self):
        if self.human_screen is None: return
        # The observation is already the screen content
        obs = self._get_observation()
        # The observation is (H, W, C) but pygame wants (W, H) surface.
        # So we need to transpose it back.
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        self.human_screen.blit(surf, (0, 0))
        pygame.event.pump()
        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.human_screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.human_screen = None
    
    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # To run and play the game manually
    env = GameEnv(render_mode="human")
    obs, info = env.reset()
    terminated = False
    
    print("\n--- Cell Defense ---")
    print(GameEnv.game_description)
    print(GameEnv.user_guide)
    print("--------------------\n")

    while not terminated:
        # Map keyboard keys to the MultiDiscrete action space
        keys = pygame.key.get_pressed()
        
        movement = 0 # no-op
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Survived {info['wave']-1} waves.")
            if env.win:
                print("VICTORY! The cell is safe.")
            else:
                print("DEFEAT! The nucleus has been breached.")
            terminated = True

    env.close()
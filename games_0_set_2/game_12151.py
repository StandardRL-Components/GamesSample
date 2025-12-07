import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T17:25:01.027487
# Source Brief: brief_02151.md
# Brief Index: 2151
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment for "Space Cattle Rancher".

    The player manages a ranch in an asteroid field, using a teleporter
    to collect resources and cattle while defending against bandit raids.
    The goal is to grow the ranch to 100 cattle.
    """
    game_description = "Manage a space ranch, using a teleporter to collect resources and cattle while defending against bandit raids."
    user_guide = "Controls: Use arrow keys (↑↓←→) to move the reticle. Press space to activate the teleporter on the targeted object."
    auto_advance = True
    metadata = {"render_modes": ["rgb_array"]}

    # --- Constants ---
    # Screen
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    # Colors
    COLOR_BG = (15, 10, 30)
    COLOR_TEXT = (220, 220, 220)
    COLOR_RANCH_BORDER = (139, 69, 19)
    COLOR_RANCH_FILL = (80, 60, 40, 50)
    COLOR_RETICLE = (0, 255, 255)
    COLOR_RETICLE_GLOW = (0, 150, 150)
    COLOR_CATTLE = (200, 255, 200)
    COLOR_ENERGY_RESOURCE = (255, 255, 0)
    COLOR_FOOD_RESOURCE = (50, 200, 50)
    COLOR_BANDIT = (255, 50, 50)
    COLOR_BANDIT_GLOW = (150, 0, 0)
    COLOR_BEAM = (100, 150, 255)
    COLOR_ENERGY_BAR = (255, 220, 0)
    COLOR_FOOD_BAR = (80, 220, 80)
    COLOR_UI_BG = (50, 50, 70, 150)

    # Game Parameters
    MAX_STEPS = 2000
    WIN_CATTLE_COUNT = 100
    RETICLE_SPEED = 15
    INITIAL_ENERGY = 100
    MAX_ENERGY = 100
    INITIAL_FOOD = 100
    MAX_FOOD = 100
    INITIAL_CATTLE = 5
    ENERGY_DECAY_RATE = 0.05
    ENERGY_TELEPORT_COST = 5
    FOOD_CONSUMPTION_RATE = 0.005
    TELEPORT_COOLDOWN_STEPS = 5
    TELEPORT_DURATION = 10
    RETICLE_TARGET_RADIUS = 20
    
    # Spawning
    MAX_ASTEROIDS = 15
    MAX_RESOURCES = 10
    MAX_CATTLE_IN_FIELD = 8
    RESOURCE_SPAWN_CHANCE = 0.02
    CATTLE_SPAWN_CHANCE = 0.01

    # Raids
    INITIAL_RAID_FREQUENCY = 200 # steps
    RAID_FREQUENCY_INCREASE_INTERVAL = 200 # steps
    RAID_FREQUENCY_INCREASE_AMOUNT = 0.99 # multiplier
    RAID_SIZE_INCREASE_INTERVAL = 5 # successful raids
    INITIAL_RAID_SIZE = 1

    # Rewards
    REWARD_COLLECT_RESOURCE = 1.0
    REWARD_COLLECT_CATTLE = 2.0
    REWARD_DESTROY_BANDIT = 5.0
    PENALTY_LOSE_CATTLE = -10.0
    PENALTY_LOW_ENERGY = -0.1
    ENERGY_THRESHOLD = 20
    REWARD_WIN = 100.0
    PENALTY_LOSE = -100.0

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
        self.font_small = pygame.font.SysFont("monospace", 16)
        self.font_large = pygame.font.SysFont("monospace", 24, bold=True)
        
        self.ranch_rect = pygame.Rect(
            self.SCREEN_WIDTH // 4, self.SCREEN_HEIGHT * 0.75,
            self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT * 0.25
        )

        # State variables are initialized in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.reticle_pos = np.array([0.0, 0.0])
        self.asteroids = []
        self.field_cattle = []
        self.resources = []
        self.bandits = []
        self.particles = []
        self.teleport_effects = []
        self.energy = 0
        self.food = 0
        self.ranch_cattle_count = 0
        self.total_cattle_count = 0
        self.teleport_cooldown = 0
        self.last_space_held = False
        self.raid_timer = 0
        self.raid_frequency = 0
        self.raid_size = 0
        self.successful_raids = 0
        self.stars = []

        self.reset()
        # self.validate_implementation() # Removed for final submission

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False

        self.reticle_pos = np.array([self.SCREEN_WIDTH / 2.0, self.SCREEN_HEIGHT / 2.0])
        
        self.energy = self.INITIAL_ENERGY
        self.food = self.INITIAL_FOOD
        self.ranch_cattle_count = self.INITIAL_CATTLE
        self.total_cattle_count = self.INITIAL_CATTLE

        self.teleport_cooldown = 0
        self.last_space_held = False

        self.raid_timer = self.INITIAL_RAID_FREQUENCY
        self.raid_frequency = self.INITIAL_RAID_FREQUENCY
        self.raid_size = self.INITIAL_RAID_SIZE
        self.successful_raids = 0
        
        self.asteroids = [self._create_asteroid() for _ in range(self.MAX_ASTEROIDS)]
        self.field_cattle = [self._create_field_cattle() for _ in range(3)]
        self.resources = [self._create_resource() for _ in range(5)]
        self.bandits = []
        self.particles = []
        self.teleport_effects = []
        
        if not self.stars:
            self.stars = [
                (self.np_random.integers(0, self.SCREEN_WIDTH + 1), self.np_random.integers(0, self.SCREEN_HEIGHT + 1), self.np_random.integers(1, 3))
                for _ in range(150)
            ]

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            obs = self._get_observation() if hasattr(self, 'screen') else np.zeros(self.observation_space.shape, dtype=np.uint8)
            return obs, 0, True, False, self._get_info()

        self.steps += 1
        reward = 0

        # --- Handle Actions ---
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        self._handle_movement(movement)
        
        space_pressed = space_held and not self.last_space_held
        if space_pressed and self.teleport_cooldown <= 0 and self.energy >= self.ENERGY_TELEPORT_COST:
            self._activate_teleporter()
        self.last_space_held = space_held

        # --- Update Game State ---
        self._update_timers()
        self._update_entities()
        reward = self._resolve_effects(reward)
        self._spawn_entities()
        
        # --- Resource Consumption & Decay ---
        self.energy -= self.ENERGY_DECAY_RATE
        if self.energy < self.ENERGY_THRESHOLD:
            reward += self.PENALTY_LOW_ENERGY

        self.food -= self.ranch_cattle_count * self.FOOD_CONSUMPTION_RATE
        self.food = max(0, self.food)
        
        # --- Check Termination ---
        terminated, term_reward = self._check_termination()
        reward += term_reward
        self.score += reward
        
        self.game_over = terminated

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_movement(self, movement):
        move_vec = np.array([0.0, 0.0])
        if movement == 1: move_vec[1] -= 1 # Up
        elif movement == 2: move_vec[1] += 1 # Down
        elif movement == 3: move_vec[0] -= 1 # Left
        elif movement == 4: move_vec[0] += 1 # Right
        
        self.reticle_pos += move_vec * self.RETICLE_SPEED
        self.reticle_pos[0] = np.clip(self.reticle_pos[0], 0, self.SCREEN_WIDTH)
        self.reticle_pos[1] = np.clip(self.reticle_pos[1], 0, self.SCREEN_HEIGHT)

    def _activate_teleporter(self):
        target_pos = (self.reticle_pos[0], self.reticle_pos[1])

        # Target Bandits
        for bandit in self.bandits:
            if math.hypot(bandit['pos'][0] - target_pos[0], bandit['pos'][1] - target_pos[1]) < self.RETICLE_TARGET_RADIUS:
                # sfx: teleport_attack
                self.teleport_effects.append({'type': 'attack', 'target': bandit, 'timer': self.TELEPORT_DURATION})
                self.teleport_cooldown = self.TELEPORT_COOLDOWN_STEPS
                self.energy -= self.ENERGY_TELEPORT_COST
                return
        
        # Target Cattle
        for cattle in self.field_cattle:
            if math.hypot(cattle['pos'][0] - target_pos[0], cattle['pos'][1] - target_pos[1]) < self.RETICLE_TARGET_RADIUS:
                # sfx: teleport_collect_good
                self.teleport_effects.append({'type': 'collect_cattle', 'target': cattle, 'timer': self.TELEPORT_DURATION})
                self.teleport_cooldown = self.TELEPORT_COOLDOWN_STEPS
                self.energy -= self.ENERGY_TELEPORT_COST
                return

        # Target Resources
        for resource in self.resources:
            if math.hypot(resource['pos'][0] - target_pos[0], resource['pos'][1] - target_pos[1]) < self.RETICLE_TARGET_RADIUS:
                # sfx: teleport_collect_neutral
                self.teleport_effects.append({'type': 'collect_resource', 'target': resource, 'timer': self.TELEPORT_DURATION})
                self.teleport_cooldown = self.TELEPORT_COOLDOWN_STEPS
                self.energy -= self.ENERGY_TELEPORT_COST
                return

    def _update_timers(self):
        if self.teleport_cooldown > 0:
            self.teleport_cooldown -= 1
        
        self.raid_timer -= 1
        if self.raid_timer <= 0:
            self._start_raid()
            self.raid_timer = self.raid_frequency

        if self.steps > 0 and self.steps % self.RAID_FREQUENCY_INCREASE_INTERVAL == 0:
            self.raid_frequency = max(50, int(self.raid_frequency * self.RAID_FREQUENCY_INCREASE_AMOUNT))

    def _update_entities(self):
        for ast in self.asteroids:
            ast['angle'] += ast['rot_speed']

        for cattle in self.field_cattle:
            cattle['bob_angle'] = (cattle['bob_angle'] + 0.1) % (2 * math.pi)
            cattle['pos'][1] += math.sin(cattle['bob_angle']) * 0.2
            
        for bandit in self.bandits:
            if not bandit['has_stolen']:
                target_pos = self.ranch_rect.center
                direction = np.array(target_pos) - bandit['pos']
                dist = np.linalg.norm(direction)
                if dist > bandit['speed']:
                    direction = direction / dist
                    bandit['pos'] += direction * bandit['speed']
                else: # Reached ranch
                    if self.ranch_cattle_count > 0:
                        # sfx: cattle_stolen
                        bandit['has_stolen'] = True
                        bandit['target_pos'] = self._get_offscreen_pos() # Flee
                    else: # No cattle to steal, just fly away
                        bandit['has_stolen'] = True # Mark to be removed
                        bandit['target_pos'] = self._get_offscreen_pos()
            else: # Fleeing
                direction = bandit['target_pos'] - bandit['pos']
                dist = np.linalg.norm(direction)
                if dist > bandit['speed']:
                    direction /= dist
                    bandit['pos'] += direction * bandit['speed']

        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]
        
        for effect in self.teleport_effects:
            effect['timer'] -= 1

    def _resolve_effects(self, reward):
        completed_effects = [e for e in self.teleport_effects if e['timer'] <= 0]
        self.teleport_effects = [e for e in self.teleport_effects if e['timer'] > 0]

        for effect in completed_effects:
            target = effect['target']
            if effect['type'] == 'attack':
                if any(b is target for b in self.bandits):
                    self._create_particles(target['pos'], 30, self.COLOR_BANDIT)
                    self.bandits = [b for b in self.bandits if b is not target]
                    reward += self.REWARD_DESTROY_BANDIT
                    self.successful_raids += 1
                    if self.successful_raids > 0 and self.successful_raids % self.RAID_SIZE_INCREASE_INTERVAL == 0:
                        self.raid_size += 1
            
            elif effect['type'] == 'collect_cattle':
                if any(c is target for c in self.field_cattle):
                    self.field_cattle = [c for c in self.field_cattle if c is not target]
                    self.ranch_cattle_count += 1
                    self.total_cattle_count += 1
                    reward += self.REWARD_COLLECT_CATTLE
                    self._create_particles(self.ranch_rect.center, 15, self.COLOR_CATTLE)

            elif effect['type'] == 'collect_resource':
                if any(r is target for r in self.resources):
                    if target['type'] == 'energy':
                        self.energy = min(self.MAX_ENERGY, self.energy + 25)
                    elif target['type'] == 'food':
                        self.food = min(self.MAX_FOOD, self.food + 25)
                    self.resources = [r for r in self.resources if r is not target]
                    reward += self.REWARD_COLLECT_RESOURCE
                    self._create_particles(self.ranch_rect.center, 15, target['color'])
        
        # Resolve bandit stealing
        bandits_to_remove_after_stealing = []
        for bandit in self.bandits:
            if bandit['has_stolen']:
                if self.ranch_rect.collidepoint(bandit['pos']):
                    if self.ranch_cattle_count > 0:
                        self.ranch_cattle_count -= 1
                        self.total_cattle_count -= 1
                        reward += self.PENALTY_LOSE_CATTLE
                        self._create_particles(bandit['pos'], 10, (255, 100, 100))
                    bandit['pos'] -= np.array([0,1])
                
                offscreen = not self.screen.get_rect().inflate(40,40).collidepoint(bandit['pos'])
                if offscreen:
                    bandits_to_remove_after_stealing.append(bandit)

        if bandits_to_remove_after_stealing:
            ids_to_remove = {id(b) for b in bandits_to_remove_after_stealing}
            self.bandits = [b for b in self.bandits if id(b) not in ids_to_remove]

        return reward

    def _spawn_entities(self):
        if len(self.resources) < self.MAX_RESOURCES and self.np_random.random() < self.RESOURCE_SPAWN_CHANCE:
            self.resources.append(self._create_resource())
        
        if len(self.field_cattle) < self.MAX_CATTLE_IN_FIELD and self.np_random.random() < self.CATTLE_SPAWN_CHANCE:
            self.field_cattle.append(self._create_field_cattle())

    def _start_raid(self):
        # sfx: raid_alert
        if self.total_cattle_count > 0: # No raids if no cattle
            for _ in range(self.raid_size):
                self.bandits.append(self._create_bandit())

    def _check_termination(self):
        if self.energy <= 0:
            return True, self.PENALTY_LOSE
        if self.total_cattle_count <= 0 and self.INITIAL_CATTLE > 0:
            return True, self.PENALTY_LOSE
        if self.total_cattle_count >= self.WIN_CATTLE_COUNT:
            return True, self.REWARD_WIN
        if self.steps >= self.MAX_STEPS:
            return True, 0
        return False, 0

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_game_objects()
        self._render_effects()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        for x, y, size in self.stars:
            self.screen.set_at((x, y), (size * 80, size * 80, size * 80))

        ranch_surface = pygame.Surface(self.ranch_rect.size, pygame.SRCALPHA)
        ranch_surface.fill(self.COLOR_RANCH_FILL)
        self.screen.blit(ranch_surface, self.ranch_rect.topleft)
        pygame.draw.rect(self.screen, self.COLOR_RANCH_BORDER, self.ranch_rect, 3, border_radius=5)

    def _render_game_objects(self):
        for ast in self.asteroids:
            self._draw_rotated_poly(self.screen, ast['points'], ast['pos'], ast['angle'], (100, 100, 110))

        for res in self.resources:
            self._draw_glow_circle(res['pos'], res['color'], 8)

        for cattle in self.field_cattle:
            y_offset = math.sin(cattle['bob_angle']) * 2
            pos = (int(cattle['pos'][0]), int(cattle['pos'][1] + y_offset))
            self._draw_glow_circle(pos, self.COLOR_CATTLE, 10)
            pygame.draw.circle(self.screen, (255,255,255), pos, 5)

        for bandit in self.bandits:
            self._draw_glow_circle(bandit['pos'], self.COLOR_BANDIT_GLOW, 12)
            p1 = (bandit['pos'][0] - 5, bandit['pos'][1] + 5)
            p2 = (bandit['pos'][0] + 5, bandit['pos'][1] + 5)
            p3 = (bandit['pos'][0], bandit['pos'][1] - 5)
            pygame.gfxdraw.aapolygon(self.screen, [p1,p2,p3], self.COLOR_BANDIT)
            pygame.gfxdraw.filled_polygon(self.screen, [p1,p2,p3], self.COLOR_BANDIT)

    def _render_effects(self):
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['life'] / p['max_life']))))
            color = (*p['color'], alpha)
            size = int(p['size'] * (p['life'] / p['max_life']))
            if size > 0:
                self._draw_glow_circle(p['pos'], color, size, glow_strength=0.5)

        for effect in self.teleport_effects:
            progress = 1.0 - (effect['timer'] / self.TELEPORT_DURATION)
            start_pos = effect['target']['pos']
            end_pos = self.ranch_rect.center
            if effect['type'] == 'attack':
                end_pos = start_pos
                start_pos = (self.reticle_pos[0], self.reticle_pos[1])
            
            pygame.draw.aaline(self.screen, self.COLOR_BEAM, start_pos, end_pos, 2)
            
            interp_pos = (
                start_pos[0] + (end_pos[0] - start_pos[0]) * progress,
                start_pos[1] + (end_pos[1] - start_pos[1]) * progress
            )
            self._draw_glow_circle(interp_pos, (200, 220, 255), 8)
            
        self._draw_glow_circle(self.reticle_pos, self.COLOR_RETICLE_GLOW, self.RETICLE_TARGET_RADIUS)
        pygame.gfxdraw.aacircle(self.screen, int(self.reticle_pos[0]), int(self.reticle_pos[1]), self.RETICLE_TARGET_RADIUS, self.COLOR_RETICLE)
        pygame.draw.line(self.screen, self.COLOR_RETICLE, (self.reticle_pos[0] - 5, self.reticle_pos[1]), (self.reticle_pos[0] + 5, self.reticle_pos[1]))
        pygame.draw.line(self.screen, self.COLOR_RETICLE, (self.reticle_pos[0], self.reticle_pos[1] - 5), (self.reticle_pos[0], self.reticle_pos[1] + 5))

    def _render_ui(self):
        ui_panel = pygame.Surface((self.SCREEN_WIDTH, 40), pygame.SRCALPHA)
        ui_panel.fill(self.COLOR_UI_BG)
        self.screen.blit(ui_panel, (0, 0))

        self._draw_text("ENERGY", (10, 10), self.font_small)
        self._draw_bar((80, 12), 150, 15, self.energy / self.MAX_ENERGY, self.COLOR_ENERGY_BAR)

        self._draw_text("FOOD", (250, 10), self.font_small)
        self._draw_bar((300, 12), 150, 15, self.food / self.MAX_FOOD, self.COLOR_FOOD_BAR)

        cattle_text = f"CATTLE: {self.total_cattle_count}/{self.WIN_CATTLE_COUNT}"
        self._draw_text(cattle_text, (470, 10), self.font_small)

        score_text = f"SCORE: {int(self.score)}"
        text_surf = self.font_large.render(score_text, True, self.COLOR_TEXT)
        self.screen.blit(text_surf, (self.SCREEN_WIDTH - text_surf.get_width() - 10, self.SCREEN_HEIGHT - 35))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "energy": self.energy,
            "food": self.food,
            "cattle": self.total_cattle_count
        }
        
    def close(self):
        if hasattr(self, 'screen') and self.screen:
            pygame.quit()
            self.screen = None

    # --- Helper Creation Functions ---
    def _create_asteroid(self):
        size = self.np_random.integers(15, 30)
        angle = self.np_random.random() * 2 * math.pi
        num_points = self.np_random.integers(5, 9)
        points = []
        for i in range(num_points):
            r = size + self.np_random.random() * size * 0.4 - size * 0.2
            a = (i / num_points) * 2 * math.pi
            points.append((r * math.cos(a), r * math.sin(a)))
        return {
            'pos': np.array([self.np_random.random() * self.SCREEN_WIDTH, self.np_random.random() * self.SCREEN_HEIGHT * 0.7]),
            'points': points,
            'angle': angle,
            'rot_speed': (self.np_random.random() - 0.5) * 0.02
        }

    def _create_field_cattle(self):
        return {
            'pos': np.array([self.np_random.random() * self.SCREEN_WIDTH, self.np_random.random() * self.SCREEN_HEIGHT * 0.7]),
            'bob_angle': self.np_random.random() * 2 * math.pi
        }

    def _create_resource(self):
        res_type = 'energy' if self.np_random.random() > 0.5 else 'food'
        return {
            'pos': np.array([self.np_random.random() * self.SCREEN_WIDTH, self.np_random.random() * self.SCREEN_HEIGHT * 0.7]),
            'type': res_type,
            'color': self.COLOR_ENERGY_RESOURCE if res_type == 'energy' else self.COLOR_FOOD_RESOURCE
        }
        
    def _create_bandit(self):
        pos = self._get_offscreen_pos()
        return {
            'pos': pos.copy(),
            'speed': self.np_random.uniform(1.0, 2.0),
            'target_pos': pos.copy(), # Initially has no target
            'has_stolen': False
        }

    def _get_offscreen_pos(self):
        edge = self.np_random.integers(0, 4)
        if edge == 0: # Top
            return np.array([self.np_random.random() * self.SCREEN_WIDTH, -20.0])
        elif edge == 1: # Bottom
            return np.array([self.np_random.random() * self.SCREEN_WIDTH, self.SCREEN_HEIGHT + 20.0])
        elif edge == 2: # Left
            return np.array([-20.0, self.np_random.random() * self.SCREEN_HEIGHT])
        else: # Right
            return np.array([self.SCREEN_WIDTH + 20.0, self.np_random.random() * self.SCREEN_HEIGHT])

    def _create_particles(self, pos, count, color):
        for _ in range(count):
            angle = self.np_random.random() * 2 * math.pi
            speed = self.np_random.random() * 2 + 1
            self.particles.append({
                'pos': np.array(pos, dtype=float),
                'vel': np.array([math.cos(angle) * speed, math.sin(angle) * speed]),
                'life': self.np_random.integers(15, 30),
                'max_life': 30,
                'color': color,
                'size': self.np_random.random() * 3 + 1
            })

    # --- Helper Drawing Functions ---
    def _draw_text(self, text, pos, font, color=COLOR_TEXT):
        text_surface = font.render(text, True, color)
        self.screen.blit(text_surface, pos)

    def _draw_bar(self, pos, width, height, progress, color):
        progress = np.clip(progress, 0, 1)
        pygame.draw.rect(self.screen, (50, 50, 50), (*pos, width, height))
        pygame.draw.rect(self.screen, color, (*pos, int(width * progress), height))
        pygame.draw.rect(self.screen, (200, 200, 200), (*pos, width, height), 1)

    def _draw_rotated_poly(self, surface, points, pos, angle, color):
        rotated_points = []
        for x, y in points:
            x_rot = x * math.cos(angle) - y * math.sin(angle)
            y_rot = x * math.sin(angle) + y * math.cos(angle)
            rotated_points.append((int(pos[0] + x_rot), int(pos[1] + y_rot)))
        
        if len(rotated_points) > 2:
            pygame.gfxdraw.aapolygon(surface, rotated_points, color)

    def _draw_glow_circle(self, pos, color, radius, glow_strength=1.0):
        pos_int = (int(pos[0]), int(pos[1]))
        if len(color) == 4: base_alpha = color[3]
        else: base_alpha = 255

        for i in range(int(radius * 0.8), 0, -2):
            alpha = int(base_alpha * (1 - i / (radius * 0.8)) * 0.3 * glow_strength)
            if alpha > 0:
                pygame.gfxdraw.filled_circle(
                    self.screen, pos_int[0], pos_int[1],
                    int(radius + i),
                    (color[0], color[1], color[2], alpha)
                )

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc is False
        assert isinstance(info, dict)

        assert self.energy <= self.MAX_ENERGY
        assert self.food <= self.MAX_FOOD
        assert self.total_cattle_count >= 0
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to run the file directly to play the game
    # It will not run in a headless environment.
    try:
        os.environ["SDL_VIDEODRIVER"] = "x11"
        import pygame
        env = GameEnv(render_mode="rgb_array")
        obs, info = env.reset()
        
        screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
        pygame.display.set_caption("Space Cattle Rancher")
        clock = pygame.time.Clock()
        
        running = True
        total_reward = 0
        
        while running:
            movement = 0 # No-op
            space = 0
            shift = 0
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            
            keys = pygame.key.get_pressed()
            if keys[pygame.K_w] or keys[pygame.K_UP]: movement = 1
            elif keys[pygame.K_s] or keys[pygame.K_DOWN]: movement = 2
            elif keys[pygame.K_a] or keys[pygame.K_LEFT]: movement = 3
            elif keys[pygame.K_d] or keys[pygame.K_RIGHT]: movement = 4
            
            if keys[pygame.K_SPACE]: space = 1
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1
            
            action = [movement, space, shift]
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            
            if terminated:
                print(f"Game Over! Final Score: {info['score']}, Total Cattle: {info['cattle']}, Steps: {info['steps']}")
                pygame.time.wait(2000)
                obs, info = env.reset()
                total_reward = 0
                
            clock.tick(30) # Run at 30 FPS
            
        env.close()
    except pygame.error as e:
        print(f"Could not initialize Pygame display. Running in headless mode. Error: {e}")
        # You can add a simple loop here to test the environment logic without rendering
        env = GameEnv()
        obs, info = env.reset()
        for _ in range(1000):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated:
                print("Headless episode finished.")
                obs, info = env.reset()
        env.close()
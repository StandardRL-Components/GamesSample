import gymnasium as gym
import os
import pygame
import numpy as np
import math
from gymnasium.spaces import MultiDiscrete, Box
import pygame.gfxdraw
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    A Gymnasium environment where a gravity-flipping robot navigates an alien base.
    The goal is to disable all energy sources while avoiding detection by patrolling enemies
    and preventing them from destroying your base.

    Action Space: MultiDiscrete([5, 2, 2])
    - actions[0]: Movement (0=none, 1=up, 2=down, 3=left, 4=right)
    - actions[1]: Space button (0=released, 1=held) -> Flips gravity on press
    - actions[2]: Shift button (0=released, 1=held) -> Deactivates adjacent energy source

    Observation Space: Box(0, 255, (400, 640, 3), uint8)
    - An RGB image of the game screen.

    Reward Structure:
    - +100 for winning (disabling all sources).
    - +10 for disabling one energy source.
    - +0.1 for moving closer to the nearest active energy source.
    - -0.5 for moving closer to the nearest enemy.
    - -25 when an enemy damages the base.
    - -50 for being detected by an enemy (terminates).
    - -100 if the base is destroyed (terminates).
    """
    metadata = {"render_modes": ["rgb_array"]}
    
    game_description = "Navigate a robot through an alien base, flipping gravity to avoid enemies and disable all energy sources."
    user_guide = "Use arrow keys to move. Press space to flip gravity. Hold shift near an energy source to deactivate it."
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_STEPS = 1000

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        try:
            self.font_ui = pygame.font.Font(None, 24)
            self.font_game_over = pygame.font.Font(None, 64)
        except IOError:
            self.font_ui = pygame.font.SysFont("sans", 24)
            self.font_game_over = pygame.font.SysFont("sans", 64)

        # --- Colors ---
        self.COLOR_BG = (15, 20, 35)
        self.COLOR_GRID = (25, 30, 45)
        self.COLOR_WALL = (80, 90, 110)
        self.COLOR_PLAYER = (50, 150, 255)
        self.COLOR_PLAYER_GLOW = (50, 150, 255, 50)
        self.COLOR_ENEMY = (255, 60, 60)
        self.COLOR_ENEMY_GLOW = (255, 60, 60, 100)
        self.COLOR_LOS = (255, 100, 100, 40)
        self.COLOR_BASE = (0, 200, 100)
        self.COLOR_BASE_HEALTH = (0, 255, 120)
        self.COLOR_BASE_HEALTH_BG = (255, 0, 0)
        self.COLOR_ENERGY_ACTIVE = (255, 220, 0)
        self.COLOR_ENERGY_INACTIVE = (60, 60, 60)
        self.COLOR_TEXT = (230, 230, 230)
        self.COLOR_WIN = (100, 255, 150)
        self.COLOR_LOSE = (255, 100, 100)

        # --- State Variables ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_condition = False
        self.player_pos = None
        self.player_vel = None
        self.gravity_down = None
        self.base_rect = None
        self.base_health = None
        self.walls = []
        self.energy_sources = []
        self.enemies = []
        self.particles = []
        self.prev_space_held = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_condition = False

        self.player_pos = pygame.math.Vector2(self.WIDTH // 2, self.HEIGHT - 50)
        self.player_vel = pygame.math.Vector2(0, 0)
        self.gravity_down = True

        self.base_rect = pygame.Rect(self.WIDTH // 2 - 40, self.HEIGHT - 20, 80, 20)
        self.base_health = 100

        self.prev_space_held = False
        self.particles.clear()

        self._generate_level()

        return self._get_observation(), self._get_info()

    def _generate_level(self):
        self.walls = [
            # Borders
            pygame.Rect(0, 0, self.WIDTH, 10),
            pygame.Rect(0, self.HEIGHT - 10, self.WIDTH, 10),
            pygame.Rect(0, 0, 10, self.HEIGHT),
            pygame.Rect(self.WIDTH - 10, 0, 10, self.HEIGHT),
            # Internal platforms
            pygame.Rect(100, 120, 440, 10),
            pygame.Rect(100, 280, 440, 10),
            pygame.Rect(200, 200, 240, 10),
        ]
        self.energy_sources = [
            {'pos': pygame.math.Vector2(80, 95), 'active': True, 'radius': 8},
            {'pos': pygame.math.Vector2(self.WIDTH - 80, 95), 'active': True, 'radius': 8},
            {'pos': pygame.math.Vector2(self.WIDTH / 2, 175), 'active': True, 'radius': 8},
        ]
        self.enemies = [
            {
                'pos': pygame.math.Vector2(110, 110),
                'path': [pygame.math.Vector2(110, 110), pygame.math.Vector2(530, 110)],
                'waypoint_idx': 1, 'speed': 1.5, 'radius': 7
            },
            {
                'pos': pygame.math.Vector2(530, 270),
                'path': [pygame.math.Vector2(110, 270), pygame.math.Vector2(530, 270)],
                'waypoint_idx': 0, 'speed': 1.5, 'radius': 7
            }
        ]

    def step(self, action):
        reward = 0.0
        terminated = self.game_over

        if not terminated:
            # --- Store pre-update state for reward calculation ---
            min_dist_source_before = self._get_min_dist_to_source()
            min_dist_enemy_before = self._get_min_dist_to_enemy()

            # --- Process Actions ---
            movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
            deactivate_reward, deactivated_source = self._handle_deactivation(shift_held)
            reward += deactivate_reward
            if deactivated_source:
                self.score += 10

            self._handle_gravity_flip(space_held)
            self._handle_movement(movement)

            # --- Update Game World ---
            self._update_player_physics()
            base_damage_reward, base_destroyed = self._update_enemies()
            reward += base_damage_reward

            self._update_particles()

            # --- Check for Terminal Conditions ---
            detected = self._check_detection()
            all_sources_off = all(not s['active'] for s in self.energy_sources)

            if detected:
                reward -= 50
                terminated = True
            if base_destroyed:
                reward -= 100
                terminated = True
            if all_sources_off:
                reward += 100
                terminated = True
                self.win_condition = True

            # --- Calculate Continuous Rewards ---
            min_dist_source_after = self._get_min_dist_to_source()
            if min_dist_source_after is not None and min_dist_source_before is not None:
                if min_dist_source_after < min_dist_source_before:
                    reward += 0.1

            min_dist_enemy_after = self._get_min_dist_to_enemy()
            if min_dist_enemy_after < min_dist_enemy_before:
                reward -= 0.5
        
        self.steps += 1
        if self.steps >= self.MAX_STEPS:
            terminated = True
            
        self.game_over = terminated

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info(),
        )

    def _handle_deactivation(self, shift_held):
        reward = 0
        deactivated = False
        if shift_held:
            for source in self.energy_sources:
                if source['active']:
                    dist = self.player_pos.distance_to(source['pos'])
                    if dist < 25: # Interaction radius
                        source['active'] = False
                        reward += 10
                        deactivated = True
                        self._create_particles(source['pos'], self.COLOR_ENERGY_ACTIVE, 30)
                        break
        return reward, deactivated

    def _handle_gravity_flip(self, space_held):
        if space_held and not self.prev_space_held:
            self.gravity_down = not self.gravity_down
            self.player_vel.y = 0
            self._create_particles(self.player_pos, self.COLOR_PLAYER, 50, is_gravity_effect=True)
        self.prev_space_held = space_held

    def _handle_movement(self, movement_action):
        move_speed = 3
        if movement_action == 1: # Up
            self.player_vel.y = -move_speed
        elif movement_action == 2: # Down
            self.player_vel.y = move_speed
        elif movement_action == 3: # Left
            self.player_vel.x = -move_speed
        elif movement_action == 4: # Right
            self.player_vel.x = move_speed

    def _update_player_physics(self):
        gravity = 0.4
        max_fall_speed = 6

        # Apply gravity
        if self.gravity_down:
            self.player_vel.y += gravity
            self.player_vel.y = min(self.player_vel.y, max_fall_speed)
        else:
            self.player_vel.y -= gravity
            self.player_vel.y = max(self.player_vel.y, -max_fall_speed)

        # Move and collide
        self.player_pos.x += self.player_vel.x
        player_rect = pygame.Rect(self.player_pos.x - 10, self.player_pos.y - 10, 20, 20)
        for wall in self.walls:
            if player_rect.colliderect(wall):
                if self.player_vel.x > 0: player_rect.right = wall.left
                if self.player_vel.x < 0: player_rect.left = wall.right
                self.player_pos.x = player_rect.centerx

        self.player_pos.y += self.player_vel.y
        player_rect.center = self.player_pos
        for wall in self.walls:
            if player_rect.colliderect(wall):
                if self.player_vel.y > 0: player_rect.bottom = wall.top
                if self.player_vel.y < 0: player_rect.top = wall.bottom
                self.player_vel.y = 0
                self.player_pos.y = player_rect.centery

        # Dampen horizontal movement
        self.player_vel.x *= 0.8
        if abs(self.player_vel.x) < 0.1: self.player_vel.x = 0

    def _update_enemies(self):
        reward = 0
        base_destroyed = False
        for enemy in self.enemies:
            target_pos = enemy['path'][enemy['waypoint_idx']]
            direction = (target_pos - enemy['pos'])
            if direction.length() < enemy['speed']:
                enemy['pos'] = target_pos
                enemy['waypoint_idx'] = 1 - enemy['waypoint_idx'] # Flip between 0 and 1
            else:
                enemy['pos'] += direction.normalize() * enemy['speed']

            if self.base_rect.collidepoint(enemy['pos']):
                self.base_health -= 5
                reward -= 25 # Heavy penalty for base damage
                self._create_particles(enemy['pos'], self.COLOR_ENEMY, 20)
                if self.base_health <= 0:
                    self.base_health = 0
                    base_destroyed = True
        return reward, base_destroyed

    def _check_detection(self):
        for enemy in self.enemies:
            if self.player_pos.distance_to(enemy['pos']) > 200: continue # Max vision range
            
            has_los = True
            p1 = (int(enemy['pos'].x), int(enemy['pos'].y))
            p2 = (int(self.player_pos.x), int(self.player_pos.y))
            
            for wall in self.walls:
                if wall.clipline(p1, p2):
                    has_los = False
                    break
            if has_los:
                return True
        return False

    def _get_min_dist_to_source(self):
        active_sources = [s['pos'] for s in self.energy_sources if s['active']]
        if not active_sources: return None
        return min(self.player_pos.distance_to(pos) for pos in active_sources)

    def _get_min_dist_to_enemy(self):
        if not self.enemies: return float('inf')
        return min(self.player_pos.distance_to(e['pos']) for e in self.enemies)

    def _create_particles(self, pos, color, count, is_gravity_effect=False):
        for _ in range(count):
            if is_gravity_effect:
                angle = self.np_random.uniform(0, 2 * math.pi)
                speed = self.np_random.uniform(1, 4)
                vel = pygame.math.Vector2(math.cos(angle), math.sin(angle)) * speed
                p_pos = self.player_pos + vel * 5
                life = self.np_random.integers(15, 26)
            else:
                angle = self.np_random.uniform(0, 2 * math.pi)
                speed = self.np_random.uniform(1, 5)
                vel = pygame.math.Vector2(math.cos(angle), math.sin(angle)) * speed
                p_pos = pos.copy()
                life = self.np_random.integers(20, 41)
            self.particles.append({'pos': p_pos, 'vel': vel, 'life': life, 'color': color})

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['vel'] *= 0.95
            p['life'] -= 1

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_walls()
        self._render_base()
        self._render_energy_sources()
        self._render_enemies()
        self._render_player()
        self._render_particles()
        self._render_ui()

        if self.game_over:
            self._render_game_over()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        for x in range(0, self.WIDTH, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT))
        for y in range(0, self.HEIGHT, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y))

    def _render_walls(self):
        for wall in self.walls:
            pygame.draw.rect(self.screen, self.COLOR_WALL, wall)

    def _render_base(self):
        pygame.draw.rect(self.screen, self.COLOR_BASE, self.base_rect)
        # Health Bar
        if self.base_health > 0:
            health_w = int((self.base_health / 100) * self.base_rect.width)
            health_rect = pygame.Rect(self.base_rect.left, self.base_rect.top - 10, health_w, 5)
            bg_rect = pygame.Rect(self.base_rect.left, self.base_rect.top - 10, self.base_rect.width, 5)
            pygame.draw.rect(self.screen, self.COLOR_BASE_HEALTH_BG, bg_rect)
            pygame.draw.rect(self.screen, self.COLOR_BASE_HEALTH, health_rect)

    def _render_energy_sources(self):
        for source in self.energy_sources:
            pos = (int(source['pos'].x), int(source['pos'].y))
            radius = source['radius']
            color = self.COLOR_ENERGY_ACTIVE if source['active'] else self.COLOR_ENERGY_INACTIVE
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, color)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, color)
            if source['active']: # Glow
                glow_color = (*color, 60)
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius + 5, glow_color)

    def _render_enemies(self):
        for enemy in self.enemies:
            pos = (int(enemy['pos'].x), int(enemy['pos'].y))
            radius = enemy['radius']
            # Vision cone
            if self.player_pos.distance_to(enemy['pos']) < 200:
                p1 = enemy['pos']
                p2 = self.player_pos
                has_los = all(not w.clipline(p1, p2) for w in self.walls)
                if has_los:
                    pygame.draw.line(self.screen, self.COLOR_LOS, p1, p2, 3)

            # Body
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius + 4, self.COLOR_ENEMY_GLOW)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, self.COLOR_ENEMY)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, self.COLOR_ENEMY)

    def _render_player(self):
        pos = (int(self.player_pos.x), int(self.player_pos.y))
        size = 10
        player_rect = pygame.Rect(pos[0] - size, pos[1] - size, size * 2, size * 2)

        # Glow
        glow_surf = pygame.Surface((size * 4, size * 4), pygame.SRCALPHA)
        pygame.gfxdraw.filled_circle(glow_surf, size * 2, size * 2, size * 2, self.COLOR_PLAYER_GLOW)
        self.screen.blit(glow_surf, (player_rect.centerx - size * 2, player_rect.centery - size * 2))

        # Body
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=3)
        
        # Gravity indicator
        if self.gravity_down:
            p1 = (pos[0], pos[1] + size)
            p2 = (pos[0] - 4, pos[1] + size + 6)
            p3 = (pos[0] + 4, pos[1] + size + 6)
        else:
            p1 = (pos[0], pos[1] - size)
            p2 = (pos[0] - 4, pos[1] - size - 6)
            p3 = (pos[0] + 4, pos[1] - size - 6)
        pygame.draw.polygon(self.screen, self.COLOR_TEXT, [p1, p2, p3])

    def _render_particles(self):
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['life'] / 30.0))))
            color = (*p['color'], alpha)
            pos = (int(p['pos'].x), int(p['pos'].y))
            radius = int(max(1, p['life'] * 0.2))
            if radius > 0:
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, color)

    def _render_ui(self):
        remaining_sources = sum(1 for s in self.energy_sources if s['active'])
        ui_text = f"SCORE: {self.score}  |  ENERGY SOURCES: {remaining_sources}  |  STEPS: {self.steps}/{self.MAX_STEPS}"
        text_surf = self.font_ui.render(ui_text, True, self.COLOR_TEXT)
        self.screen.blit(text_surf, (20, 20))

    def _render_game_over(self):
        overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))

        if self.win_condition:
            text = "MISSION COMPLETE"
            color = self.COLOR_WIN
        else:
            text = "MISSION FAILED"
            color = self.COLOR_LOSE
        
        text_surf = self.font_game_over.render(text, True, color)
        text_rect = text_surf.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
        self.screen.blit(text_surf, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "base_health": self.base_health,
            "remaining_sources": sum(1 for s in self.energy_sources if s['active']),
        }

    def close(self):
        pygame.quit()
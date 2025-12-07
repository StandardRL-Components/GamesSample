
# Generated: 2025-08-28T05:59:24.004945
# Source Brief: brief_05756.md
# Brief Index: 5756

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ↑↓←→ to select tower spots. Space to place a tower. Shift to cycle tower types."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Defend your base from waves of enemies by strategically placing towers in a minimalist, top-down tower defense game."
    )

    # Frames auto-advance for real-time gameplay.
    auto_advance = True

    # --- Constants ---
    # Colors
    COLOR_BG = (20, 25, 40)
    COLOR_PATH = (40, 50, 70)
    COLOR_PATH_OUTLINE = (60, 70, 90)
    COLOR_BASE = (0, 200, 100)
    COLOR_ENEMY = (255, 50, 50)
    COLOR_TEXT = (220, 220, 240)
    COLOR_UI_ACCENT = (255, 180, 0)

    # Game parameters
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    MAX_STEPS = 30 * 120 # 2 minutes at 30fps
    INITIAL_BASE_HEALTH = 20
    INITIAL_RESOURCES = 150
    NUM_WAVES = 10

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont('Consolas', 16, bold=True)
        self.font_large = pygame.font.SysFont('Consolas', 24, bold=True)
        
        # Path and tower placement definition
        self._define_layout()
        
        # Tower definitions
        self.tower_types = [
            {"name": "Cannon", "cost": 50, "range": 80, "cooldown": 45, "damage": 10, "color": (80, 150, 255)},
            {"name": "Sniper", "cost": 100, "range": 150, "cooldown": 90, "damage": 40, "color": (255, 100, 255)},
        ]
        
        # Initialize state variables
        self.np_random = None
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.base_health = 0
        self.resources = 0
        self.wave_number = 0
        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.particles = []
        self.wave_state = "intermission"
        self.wave_timer = 0
        self.enemies_to_spawn = []
        self.selected_spot_index = 0
        self.selected_tower_type_index = 0
        self.last_movement_action = 0
        self.last_space_held = False
        self.last_shift_held = False

        self.reset()
        self.validate_implementation()

    def _define_layout(self):
        self.path_waypoints = [
            pygame.math.Vector2(-20, 100),
            pygame.math.Vector2(100, 100),
            pygame.math.Vector2(100, 300),
            pygame.math.Vector2(300, 300),
            pygame.math.Vector2(300, 100),
            pygame.math.Vector2(500, 100),
            pygame.math.Vector2(500, 300),
            pygame.math.Vector2(self.SCREEN_WIDTH + 20, 300)
        ]
        self.base_pos = pygame.math.Vector2(self.SCREEN_WIDTH - 40, 300)
        self.tower_spots = [
            (160, 200), (240, 200), (360, 200), (440, 200),
            (60, 200), (540, 200)
        ]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.base_health = self.INITIAL_BASE_HEALTH
        self.resources = self.INITIAL_RESOURCES
        self.wave_number = 0
        
        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.particles = []
        
        self.wave_state = "intermission"
        self.wave_timer = 150 # Time before first wave
        self.enemies_to_spawn = []
        
        self.selected_spot_index = 0
        self.selected_tower_type_index = 0
        
        self.last_movement_action = 0
        self.last_space_held = False
        self.last_shift_held = False

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = -0.001 # Small penalty for surviving to encourage efficiency
        self.steps += 1
        
        self._handle_input(action)
        
        self._update_wave_manager()
        
        new_projectiles, projectile_reward = self._update_towers()
        self.projectiles.extend(new_projectiles)
        reward += projectile_reward
        
        damage_reward, kill_reward, base_damage_penalty = self._update_projectiles()
        reward += damage_reward + kill_reward + base_damage_penalty
        
        base_damage = self._update_enemies()
        if base_damage > 0:
            self.base_health -= base_damage
            reward -= 10 * base_damage # Heavy penalty for losing health
            # sfx: base_damage_sound
        
        self._update_particles()
        
        self.score += reward

        terminated = self._check_termination()
        if terminated:
            if self.base_health <= 0:
                reward -= 100 # Loss penalty
            elif self.wave_number > self.NUM_WAVES:
                reward += 100 # Win bonus
            self.score += reward
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )
        
    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # --- Movement: Cycle through tower spots ---
        # Only register on new press
        if movement != 0 and movement != self.last_movement_action:
            if movement in [1, 3]: # Up or Left
                self.selected_spot_index = (self.selected_spot_index - 1) % len(self.tower_spots)
            elif movement in [2, 4]: # Down or Right
                self.selected_spot_index = (self.selected_spot_index + 1) % len(self.tower_spots)
        self.last_movement_action = movement

        # --- Shift: Cycle through tower types ---
        if shift_held and not self.last_shift_held:
            self.selected_tower_type_index = (self.selected_tower_type_index + 1) % len(self.tower_types)
            # sfx: ui_cycle_sound
        self.last_shift_held = shift_held

        # --- Space: Place tower ---
        if space_held and not self.last_space_held:
            self._place_tower()
        self.last_space_held = space_held
        
    def _place_tower(self):
        spot_pos = self.tower_spots[self.selected_spot_index]
        tower_def = self.tower_types[self.selected_tower_type_index]
        
        is_occupied = any(t['pos'] == spot_pos for t in self.towers)
        
        if not is_occupied and self.resources >= tower_def['cost']:
            self.resources -= tower_def['cost']
            self.towers.append({
                "pos": spot_pos,
                "type": self.selected_tower_type_index,
                "cooldown": 0,
                "target": None
            })
            # sfx: place_tower_sound
            self._create_particles(spot_pos, 20, tower_def['color'])

    def _update_wave_manager(self):
        if self.wave_state == "intermission":
            self.wave_timer -= 1
            if self.wave_timer <= 0:
                self.wave_number += 1
                if self.wave_number > self.NUM_WAVES:
                    return # All waves cleared
                self.wave_state = "spawning"
                self._generate_wave()
                
        elif self.wave_state == "spawning":
            self.wave_timer -= 1
            if self.wave_timer <= 0 and self.enemies_to_spawn:
                self.enemies.append(self.enemies_to_spawn.pop(0))
                self.wave_timer = 30 # Spawn interval
            elif not self.enemies_to_spawn:
                self.wave_state = "active"
                
        elif self.wave_state == "active":
            if not self.enemies:
                self.wave_state = "intermission"
                self.wave_timer = 300 # 10 seconds between waves

    def _generate_wave(self):
        num_enemies = 5 + self.wave_number * 2
        base_health = 20 * (1.05 ** (self.wave_number - 1))
        speed = 1.0 + self.wave_number * 0.05
        
        self.enemies_to_spawn = []
        for _ in range(num_enemies):
            self.enemies_to_spawn.append({
                "pos": self.path_waypoints[0].copy(),
                "health": base_health,
                "max_health": base_health,
                "speed": speed,
                "path_index": 1,
                "distance_on_path": 0.0,
            })
    
    def _update_enemies(self):
        base_damage = 0
        for enemy in reversed(self.enemies):
            if enemy['path_index'] >= len(self.path_waypoints):
                self.enemies.remove(enemy)
                base_damage += 1
                continue

            start_point = self.path_waypoints[enemy['path_index'] - 1]
            end_point = self.path_waypoints[enemy['path_index']]
            
            direction = (end_point - start_point)
            segment_length = direction.length()
            
            if segment_length > 0:
                direction.normalize_ip()
                enemy['pos'] += direction * enemy['speed']
                enemy['distance_on_path'] += enemy['speed']

            if (enemy['pos'] - start_point).length() >= segment_length:
                enemy['pos'] = end_point.copy()
                enemy['path_index'] += 1
        return base_damage

    def _update_towers(self):
        new_projectiles = []
        reward = 0
        for tower in self.towers:
            tower_def = self.tower_types[tower['type']]
            if tower['cooldown'] > 0:
                tower['cooldown'] -= 1
                continue
            
            # Find a target: furthest enemy in range
            best_target = None
            max_dist = -1
            
            tower_pos = pygame.math.Vector2(tower['pos'])
            for enemy in self.enemies:
                dist_to_enemy = tower_pos.distance_to(enemy['pos'])
                if dist_to_enemy <= tower_def['range']:
                    if enemy['distance_on_path'] > max_dist:
                        max_dist = enemy['distance_on_path']
                        best_target = enemy
            
            if best_target:
                tower['target'] = best_target
                tower['cooldown'] = tower_def['cooldown']
                new_projectiles.append({
                    "pos": tower_pos.copy(),
                    "target": best_target,
                    "speed": 8,
                    "damage": tower_def['damage'],
                    "color": (255, 255, 0)
                })
                # sfx: fire_sound
        return new_projectiles, reward

    def _update_projectiles(self):
        damage_reward = 0
        kill_reward = 0
        base_damage_penalty = 0

        for proj in reversed(self.projectiles):
            if proj['target'] not in self.enemies:
                self.projectiles.remove(proj)
                continue
            
            target_pos = proj['target']['pos']
            direction = (target_pos - proj['pos']).normalize()
            proj['pos'] += direction * proj['speed']
            
            if proj['pos'].distance_to(target_pos) < 5:
                proj['target']['health'] -= proj['damage']
                damage_reward += 0.01 # Small reward for hitting
                self._create_particles(proj['pos'], 5, proj['color'])
                # sfx: hit_sound
                
                if proj['target']['health'] <= 0:
                    self.resources += 10
                    kill_reward += 1.0 # Big reward for kill
                    self._create_particles(proj['target']['pos'], 30, self.COLOR_ENEMY)
                    # sfx: enemy_destroyed_sound
                    self.enemies.remove(proj['target'])
                
                self.projectiles.remove(proj)
        
        return damage_reward, kill_reward, base_damage_penalty
        
    def _update_particles(self):
        for p in reversed(self.particles):
            p['pos'] += p['vel']
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _create_particles(self, pos, count, color):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = pygame.math.Vector2(math.cos(angle), math.sin(angle)) * speed
            self.particles.append({
                'pos': pygame.math.Vector2(pos),
                'vel': vel,
                'life': self.np_random.integers(10, 20),
                'color': color
            })
            
    def _check_termination(self):
        win = self.wave_number > self.NUM_WAVES and not self.enemies
        loss = self.base_health <= 0
        timeout = self.steps >= self.MAX_STEPS
        return win or loss or timeout

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.wave_number,
            "resources": self.resources,
            "base_health": self.base_health,
        }

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        self._render_path()
        self._render_tower_spots()
        self._render_base()
        self._render_towers()
        self._render_enemies()
        self._render_projectiles()
        self._render_particles()

    def _render_path(self):
        if len(self.path_waypoints) > 1:
            pygame.draw.lines(self.screen, self.COLOR_PATH, False, [tuple(p) for p in self.path_waypoints], 30)
            pygame.draw.lines(self.screen, self.COLOR_PATH_OUTLINE, False, [tuple(p) for p in self.path_waypoints], 34)

    def _render_base(self):
        base_rect = pygame.Rect(self.base_pos.x - 15, self.base_pos.y - 15, 30, 30)
        pygame.draw.rect(self.screen, self.COLOR_BASE, base_rect)
        pygame.draw.rect(self.screen, tuple(min(255, c+50) for c in self.COLOR_BASE), base_rect, 2)

    def _render_tower_spots(self):
        for i, pos in enumerate(self.tower_spots):
            is_occupied = any(t['pos'] == pos for t in self.towers)
            if is_occupied:
                continue

            if i == self.selected_spot_index:
                # Pulsing glow for selected spot
                pulse = (math.sin(self.steps * 0.2) + 1) / 2
                radius = int(18 + pulse * 4)
                alpha = int(100 + pulse * 100)
                color = self.tower_types[self.selected_tower_type_index]['color']
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, (*color, alpha))
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, (*color, alpha))
            else:
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 15, (100, 100, 120, 100))

    def _render_towers(self):
        for tower in self.towers:
            pos = tower['pos']
            tower_def = self.tower_types[tower['type']]
            color = tower_def['color']
            
            points = [
                (pos[0], pos[1] - 12),
                (pos[0] - 10, pos[1] + 8),
                (pos[0] + 10, pos[1] + 8)
            ]
            pygame.gfxdraw.aapolygon(self.screen, points, color)
            pygame.gfxdraw.filled_polygon(self.screen, points, color)
            
            # Range indicator
            if tower.get('target'):
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], tower_def['range'], (*color, 50))
                
    def _render_enemies(self):
        for enemy in self.enemies:
            pos = (int(enemy['pos'].x), int(enemy['pos'].y))
            radius = 8
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, self.COLOR_ENEMY)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, tuple(min(255, c+50) for c in self.COLOR_ENEMY))
            
            # Health bar
            health_ratio = max(0, enemy['health'] / enemy['max_health'])
            bar_width = 16
            bar_x = pos[0] - bar_width / 2
            bar_y = pos[1] - radius - 6
            pygame.draw.rect(self.screen, (80, 0, 0), (bar_x, bar_y, bar_width, 3))
            pygame.draw.rect(self.screen, self.COLOR_BASE, (bar_x, bar_y, bar_width * health_ratio, 3))

    def _render_projectiles(self):
        for proj in self.projectiles:
            pos = (int(proj['pos'].x), int(proj['pos'].y))
            pygame.draw.rect(self.screen, proj['color'], (pos[0]-2, pos[1]-2, 4, 4))

    def _render_particles(self):
        for p in self.particles:
            pos = (int(p['pos'].x), int(p['pos'].y))
            life_ratio = p['life'] / 20.0
            size = int(3 * life_ratio)
            color = tuple(c * life_ratio for c in p['color'])
            if size > 0:
                pygame.draw.rect(self.screen, color, (pos[0]-size//2, pos[1]-size//2, size, size))

    def _render_ui(self):
        # Wave info
        wave_text = f"Wave: {self.wave_number}/{self.NUM_WAVES}"
        if self.wave_state == 'intermission' and self.wave_number < self.NUM_WAVES:
            wave_text += f" (Next in {self.wave_timer // 30 + 1}s)"
        text_surface = self.font_large.render(wave_text, True, self.COLOR_TEXT)
        self.screen.blit(text_surface, (10, 10))
        
        # Base Health
        health_text = f"Base Health: {self.base_health}"
        text_surface = self.font_large.render(health_text, True, self.COLOR_TEXT)
        self.screen.blit(text_surface, (self.SCREEN_WIDTH - text_surface.get_width() - 10, 10))

        # Resources
        res_text = f"Resources: {self.resources}"
        text_surface = self.font_large.render(res_text, True, self.COLOR_UI_ACCENT)
        text_rect = text_surface.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT - 40))
        self.screen.blit(text_surface, text_rect)
        
        # Selected Tower
        tower_def = self.tower_types[self.selected_tower_type_index]
        tower_info = f"Build: {tower_def['name']} (Cost: {tower_def['cost']})"
        text_surface = self.font_small.render(tower_info, True, self.COLOR_TEXT)
        text_rect = text_surface.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT - 20))
        self.screen.blit(text_surface, text_rect)

        # Game Over / Win message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            if self.base_health <= 0:
                msg = "BASE DESTROYED"
                color = self.COLOR_ENEMY
            else:
                msg = "VICTORY"
                color = self.COLOR_BASE
            
            text_surface = self.font_large.render(msg, True, color)
            text_rect = text_surface.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(text_surface, text_rect)

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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # This block allows you to play the game directly
    # Set SDL_VIDEODRIVER to a dummy value for headless execution
    import os
    # os.environ["SDL_VIDEODRIVER"] = "dummy"
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # --- Pygame setup for human play ---
    pygame.display.set_caption("Tower Defense")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    running = True

    while running:
        # --- Event handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # --- Get human input ---
        keys = pygame.key.get_pressed()
        movement = 0
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]

        # --- Step the environment ---
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Steps: {info['steps']}")
            # Wait a bit before resetting
            pygame.time.wait(3000)
            obs, info = env.reset()

        # --- Render to screen ---
        # The observation is already a rendered frame
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Cap the frame rate ---
        clock.tick(30)

    pygame.quit()
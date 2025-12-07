
# Generated: 2025-08-28T05:51:18.573217
# Source Brief: brief_05708.md
# Brief Index: 5708

        
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

    user_guide = (
        "Controls: ↑/↓ to select placement zone. Space to cycle tower type. Shift to place tower."
    )

    game_description = (
        "Minimalist tower defense. Survive 10 waves of enemies by strategically placing towers on the designated zones."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.width, self.height = 640, 400
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.height, self.width, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.width, self.height))
        self.clock = pygame.time.Clock()
        self.game_font = pygame.font.SysFont("monospace", 18, bold=True)
        self.title_font = pygame.font.SysFont("monospace", 40, bold=True)

        self._define_constants()
        self.reset()
        
        # self.validate_implementation()

    def _define_constants(self):
        # Colors
        self.COLOR_BG = (25, 25, 40)
        self.COLOR_PATH = (45, 45, 60)
        self.COLOR_PATH_BORDER = (65, 65, 80)
        self.COLOR_BASE = (0, 150, 0)
        self.COLOR_BASE_DAMAGED = (150, 150, 0)
        self.COLOR_ZONE = (255, 200, 0, 100)
        self.COLOR_ZONE_SELECTED = (255, 255, 0)
        self.COLOR_ENEMY = (210, 50, 50)
        self.COLOR_TEXT = (230, 230, 230)
        self.COLOR_TOWER_GUN = (0, 180, 255)
        self.COLOR_TOWER_CANNON = (255, 165, 0)
        self.COLOR_TOWER_PULSE = (180, 0, 255)
        self.TOWER_COLORS = [self.COLOR_TOWER_GUN, self.COLOR_TOWER_CANNON, self.COLOR_TOWER_PULSE]

        # Game parameters
        self.MAX_STEPS = 1500
        self.WIN_WAVE = 10
        self.INITIAL_RESOURCES = 250
        self.INITIAL_BASE_HEALTH = 100
        self.ENEMY_KILL_REWARD = 10
        self.WAVE_CLEAR_REWARD = 50

        # Path and zones
        self.path = [pygame.math.Vector2(-20, 200), pygame.math.Vector2(100, 200), pygame.math.Vector2(100, 100), pygame.math.Vector2(540, 100), pygame.math.Vector2(540, 300), pygame.math.Vector2(self.width + 20, 300)]
        self.base_pos = pygame.math.Vector2(self.width - 40, 300)
        self.tower_zones = [
            (50, 150), (150, 150), (320, 150), (490, 150), (235, 250), (400, 250)
        ]

        # Tower specifications [cost, range, damage, fire_rate_steps, projectile_speed, aoe_radius]
        self.TOWER_SPECS = {
            0: {"name": "Gatling", "cost": 50, "range": 80, "damage": 8, "fire_rate": 10, "projectile_speed": 10},
            1: {"name": "Cannon", "cost": 100, "range": 120, "damage": 50, "fire_rate": 60, "projectile_speed": 7},
            2: {"name": "Pulse", "cost": 125, "range": 60, "damage": 15, "fire_rate": 45, "projectile_speed": 0, "aoe_radius": 40},
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        self.reward_this_step = 0

        self.base_health = self.INITIAL_BASE_HEALTH
        self.resources = self.INITIAL_RESOURCES
        
        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.particles = []

        self.wave_number = 0
        self.wave_in_progress = False
        self.next_wave_step = 60 # Start first wave after 2 seconds
        self.enemies_to_spawn = []

        self.selected_zone_idx = 0
        self.tower_preview_type = 0
        self.prev_space_held = False
        self.prev_shift_held = False

        return self._get_observation(), self._get_info()

    def step(self, action):
        self.reward_this_step = -0.01 # Small penalty for existing

        self._handle_actions(action)
        
        if not self.game_over:
            self._update_waves()
            self._update_towers()
            self._update_enemies()
            self._update_projectiles()
        
        self._update_particles()
        
        self.steps += 1
        
        terminated = self._check_termination()

        return (
            self._get_observation(),
            self.reward_this_step,
            terminated,
            False,
            self._get_info()
        )

    def _handle_actions(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        if movement == 1: # Up
            self.selected_zone_idx = (self.selected_zone_idx - 1 + len(self.tower_zones)) % len(self.tower_zones)
        elif movement == 2: # Down
            self.selected_zone_idx = (self.selected_zone_idx + 1) % len(self.tower_zones)

        if space_held and not self.prev_space_held:
            self.tower_preview_type = (self.tower_preview_type + 1) % len(self.TOWER_SPECS)
            # sfx: menu_bleep.wav

        if shift_held and not self.prev_shift_held:
            zone_pos = self.tower_zones[self.selected_zone_idx]
            spec = self.TOWER_SPECS[self.tower_preview_type]
            
            is_occupied = any(t['pos'] == zone_pos for t in self.towers)
            
            if not is_occupied and self.resources >= spec['cost']:
                self.resources -= spec['cost']
                self.towers.append({
                    "pos": zone_pos,
                    "type": self.tower_preview_type,
                    "spec": spec,
                    "cooldown": 0,
                    "target": None,
                })
                # sfx: place_tower.wav

        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

    def _update_waves(self):
        if not self.wave_in_progress and self.steps >= self.next_wave_step and self.wave_number < self.WIN_WAVE:
            self.wave_in_progress = True
            self.wave_number += 1
            
            num_enemies = 8 + self.wave_number * 2
            base_health = 50 * (1 + (self.wave_number - 1) * 0.2)
            base_speed = 1.0 * (1 + (self.wave_number - 1) * 0.05)
            
            self.enemies_to_spawn = [
                {
                    "pos": pygame.math.Vector2(self.path[0]),
                    "health": base_health,
                    "max_health": base_health,
                    "speed": base_speed,
                    "path_idx": 1,
                    "id": self.np_random.random()
                } for _ in range(num_enemies)
            ]
            self.spawn_timer = 0

        if self.wave_in_progress:
            self.spawn_timer -= 1
            if self.spawn_timer <= 0 and self.enemies_to_spawn:
                self.enemies.append(self.enemies_to_spawn.pop(0))
                self.spawn_timer = 15
            
            if not self.enemies and not self.enemies_to_spawn:
                self.wave_in_progress = False
                self.next_wave_step = self.steps + 150
                self.reward_this_step += self.WAVE_CLEAR_REWARD
                self.score += self.WAVE_CLEAR_REWARD
                if self.wave_number >= self.WIN_WAVE:
                    self.game_won = True
                    self.game_over = True

    def _update_towers(self):
        for tower in self.towers:
            tower['cooldown'] = max(0, tower['cooldown'] - 1)
            
            if tower['target']:
                target_enemy = next((e for e in self.enemies if e['id'] == tower['target']), None)
                if not target_enemy or pygame.math.Vector2(tower['pos']).distance_to(target_enemy['pos']) > tower['spec']['range']:
                    tower['target'] = None
            
            if not tower['target']:
                in_range_enemies = [e for e in self.enemies if pygame.math.Vector2(tower['pos']).distance_to(e['pos']) <= tower['spec']['range']]
                if in_range_enemies:
                    tower['target'] = max(in_range_enemies, key=lambda e: e['path_idx'] + e['pos'].distance_to(self.path[e['path_idx']]) / self.path[e['path_idx']-1].distance_to(self.path[e['path_idx']]))['id']

            if tower['cooldown'] == 0 and tower['target']:
                target_enemy = next((e for e in self.enemies if e['id'] == tower['target']), None)
                if target_enemy:
                    tower['cooldown'] = tower['spec']['fire_rate']
                    # sfx: fire_weapon.wav
                    
                    if tower['type'] == 2:
                        self.projectiles.append({"pos": pygame.math.Vector2(tower['pos']), "type": tower['type'], "spec": tower['spec'], "target_pos": pygame.math.Vector2(target_enemy['pos'])})
                    else:
                        self.projectiles.append({"pos": pygame.math.Vector2(tower['pos']), "type": tower['type'], "spec": tower['spec'], "target_id": tower['target']})

    def _update_enemies(self):
        for enemy in self.enemies[:]:
            if enemy['path_idx'] >= len(self.path):
                self.enemies.remove(enemy)
                self.base_health -= 10
                self.reward_this_step -= 20
                # sfx: base_damage.wav
                self._create_particles(self.base_pos, self.COLOR_ENEMY, 20)
                continue

            target_pos = self.path[enemy['path_idx']]
            direction = (target_pos - enemy['pos']).normalize() if (target_pos - enemy['pos']).length() > 0 else pygame.math.Vector2(0,0)
            enemy['pos'] += direction * enemy['speed']

            if enemy['pos'].distance_to(target_pos) < enemy['speed']:
                enemy['path_idx'] += 1

    def _update_projectiles(self):
        for proj in self.projectiles[:]:
            if proj['type'] == 2:
                if proj['pos'].distance_to(proj['target_pos']) < 2:
                    # sfx: explosion.wav
                    self._create_particles(proj['pos'], self.TOWER_COLORS[proj['type']], 30, aoe_radius=proj['spec']['aoe_radius'])
                    for enemy in self.enemies:
                        if enemy['pos'].distance_to(proj['pos']) <= proj['spec']['aoe_radius']:
                            enemy['health'] -= proj['spec']['damage']
                            self.reward_this_step += 0.1
                    self.projectiles.remove(proj)
                else:
                    direction = (proj['target_pos'] - proj['pos']).normalize()
                    proj['pos'] += direction * 4
            else:
                target = next((e for e in self.enemies if e['id'] == proj['target_id']), None)
                if not target:
                    self.projectiles.remove(proj)
                    continue
                
                if proj['pos'].distance_to(target['pos']) < 5:
                    target['health'] -= proj['spec']['damage']
                    self.reward_this_step += 0.1
                    self._create_particles(proj['pos'], self.TOWER_COLORS[proj['type']], 5)
                    self.projectiles.remove(proj)
                    # sfx: hit.wav
                else:
                    direction = (target['pos'] - proj['pos']).normalize()
                    proj['pos'] += direction * proj['spec']['projectile_speed']

        for enemy in self.enemies[:]:
            if enemy['health'] <= 0:
                self.enemies.remove(enemy)
                self.reward_this_step += self.ENEMY_KILL_REWARD
                self.score += self.ENEMY_KILL_REWARD
                self.resources += 5 + self.wave_number
                self._create_particles(enemy['pos'], self.COLOR_ENEMY, 15)
                # sfx: enemy_die.wav

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['lifespan'] -= 1
            if p['lifespan'] <= 0:
                self.particles.remove(p)

    def _create_particles(self, pos, color, count, aoe_radius=0):
        if aoe_radius > 0:
            for _ in range(count):
                angle = self.np_random.uniform(0, 2 * math.pi)
                radius = self.np_random.uniform(0, aoe_radius)
                p_pos = pos + pygame.math.Vector2(math.cos(angle), math.sin(angle)) * radius
                self.particles.append({'pos': p_pos, 'vel': pygame.math.Vector2(0,0), 'lifespan': 10, 'color': color, 'size': self.np_random.integers(2,4)})
        else:
            for _ in range(count):
                vel = pygame.math.Vector2(self.np_random.uniform(-2, 2), self.np_random.uniform(-2, 2))
                lifespan = self.np_random.integers(10, 25)
                self.particles.append({'pos': pygame.math.Vector2(pos), 'vel': vel, 'lifespan': lifespan, 'color': color, 'size': self.np_random.integers(2,5)})

    def _check_termination(self):
        if self.game_over:
            if self.game_won:
                self.reward_this_step += 100
                self.score += 100
            else:
                self.reward_this_step -= 100
                self.score -= 100
            return True
        
        if self.base_health <= 0:
            self.game_over = True
            self.game_won = False
            return True
        
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            self.game_won = False
            return True
            
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        for i in range(len(self.path) - 1):
            pygame.draw.line(self.screen, self.COLOR_PATH_BORDER, self.path[i], self.path[i+1], 44)
            pygame.draw.line(self.screen, self.COLOR_PATH, self.path[i], self.path[i+1], 40)
        
        base_color = self.COLOR_BASE if self.base_health > 30 else self.COLOR_BASE_DAMAGED
        pygame.draw.rect(self.screen, base_color, (self.base_pos.x-10, self.base_pos.y-10, 20, 20))
        pygame.gfxdraw.rectangle(self.screen, (int(self.base_pos.x-10), int(self.base_pos.y-10), 20, 20), (*base_color, 150))
        
        selected_zone_pos = self.tower_zones[self.selected_zone_idx]
        for i, pos in enumerate(self.tower_zones):
            is_occupied = any(t['pos'] == pos for t in self.towers)
            if i == self.selected_zone_idx:
                pulse = (math.sin(self.steps * 0.2) + 1) / 2
                size = int(25 + pulse * 5)
                pygame.gfxdraw.aacircle(self.screen, int(pos[0]), int(pos[1]), size, self.COLOR_ZONE_SELECTED)
                if not is_occupied:
                    spec = self.TOWER_SPECS[self.tower_preview_type]
                    color = self.TOWER_COLORS[self.tower_preview_type]
                    pygame.gfxdraw.filled_circle(self.screen, int(pos[0]), int(pos[1]), 10, (*color, 100))
                    pygame.gfxdraw.aacircle(self.screen, int(pos[0]), int(pos[1]), spec['range'], (*color, 50))
            else:
                color_with_alpha = self.COLOR_ZONE if not is_occupied else (100, 100, 100, 50)
                pygame.gfxdraw.filled_circle(self.screen, int(pos[0]), int(pos[1]), 25, color_with_alpha)

        for tower in self.towers:
            pos, spec, type = tower['pos'], tower['spec'], tower['type']
            color = self.TOWER_COLORS[type]
            pygame.gfxdraw.filled_circle(self.screen, int(pos[0]), int(pos[1]), 12, color)
            pygame.gfxdraw.aacircle(self.screen, int(pos[0]), int(pos[1]), 12, (255,255,255))
            if tower['cooldown'] > 0:
                angle = (tower['cooldown'] / spec['fire_rate']) * 2 * math.pi
                pygame.draw.arc(self.screen, (255,255,255), (pos[0]-15, pos[1]-15, 30, 30), 0, angle, 2)

        for enemy in self.enemies:
            pos = (int(enemy['pos'].x), int(enemy['pos'].y))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 8, self.COLOR_ENEMY)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 8, (255, 150, 150))
            health_pct = max(0, enemy['health'] / enemy['max_health'])
            pygame.draw.rect(self.screen, (50,50,50), (pos[0]-10, pos[1]-15, 20, 3))
            pygame.draw.rect(self.screen, self.COLOR_ENEMY, (pos[0]-10, pos[1]-15, int(20*health_pct), 3))

        for proj in self.projectiles:
            pos = (int(proj['pos'].x), int(proj['pos'].y))
            color = self.TOWER_COLORS[proj['type']]
            size = 5 if proj['type'] == 1 else 3
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], size, color)

        for p in self.particles:
            size = int(p['size'] * (p['lifespan'] / 25))
            if size > 0:
                pygame.draw.rect(self.screen, p['color'], (int(p['pos'].x - size/2), int(p['pos'].y - size/2), size, size))

    def _render_ui(self):
        pygame.draw.rect(self.screen, (0,0,0,150), (0, 0, self.width, 40))

        res_text = self.game_font.render(f"$: {self.resources}", True, self.COLOR_TEXT)
        self.screen.blit(res_text, (10, 10))

        health_text = self.game_font.render(f"Base: {max(0, self.base_health)}%", True, self.COLOR_TEXT)
        self.screen.blit(health_text, (130, 10))
        pygame.draw.rect(self.screen, (50,0,0), (230, 12, 100, 15))
        pygame.draw.rect(self.screen, self.COLOR_BASE, (230, 12, max(0, int(self.base_health)), 15))

        wave_str = f"Wave: {self.wave_number}/{self.WIN_WAVE}" if self.wave_in_progress or self.game_won else f"Next wave in {max(0, (self.next_wave_step - self.steps)//30)}s"
        wave_text = self.game_font.render(wave_str, True, self.COLOR_TEXT)
        self.screen.blit(wave_text, (350, 10))

        score_text = self.game_font.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (520, 10))
        
        spec = self.TOWER_SPECS[self.tower_preview_type]
        color = self.TOWER_COLORS[self.tower_preview_type]
        info_str = f"Selected: {spec['name']} (Cost: {spec['cost']})"
        info_text = self.game_font.render(info_str, True, color)
        self.screen.blit(info_text, (10, self.height - 30))
        
        if self.game_over:
            s = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
            s.fill((0,0,0,180))
            self.screen.blit(s, (0,0))
            msg = "VICTORY!" if self.game_won else "GAME OVER"
            color = (0, 255, 0) if self.game_won else (255, 0, 0)
            end_text = self.title_font.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.width/2, self.height/2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.wave_number,
            "resources": self.resources,
            "base_health": self.base_health,
        }
        
    def close(self):
        pygame.quit()

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        test_obs = self._get_observation()
        assert test_obs.shape == (self.height, self.width, 3)
        assert test_obs.dtype == np.uint8
        
        obs, info = self.reset()
        assert obs.shape == (self.height, self.width, 3)
        assert isinstance(info, dict)
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.height, self.width, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    import os
    os.environ['SDL_VIDEODRIVER'] = 'dummy' # Run headless for automated test

    print("Running automated test...")
    env = GameEnv(render_mode="rgb_array")
    env.validate_implementation()
    episodes = 3
    for ep in range(episodes):
        obs, info = env.reset()
        terminated = False
        total_reward = 0
        step_count = 0
        while not terminated:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step_count += 1
            if terminated or truncated:
                break
        print(f"Episode {ep+1}: Steps={step_count}, Score={info['score']}, Total Reward={total_reward:.2f}, Reason={'Win' if env.game_won else 'Loss'}")
    env.close()
    print("Automated test finished.")
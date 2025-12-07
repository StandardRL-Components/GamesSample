import gymnasium as gym
import os
import pygame
import numpy as np
import math
import random
from gymnasium.spaces import MultiDiscrete
import pygame.gfxdraw
from collections import deque
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    Gymnasium environment for 'Mech-Combine', a real-time strategy game.
    The player commands a squad of four transforming robots to defeat waves of enemies.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Command a squad of transforming robots to defeat enemy waves. "
        "Combine mechs into powerful squads to unleash devastating attacks."
    )
    user_guide = (
        "Use arrow keys (↑↓←→) to move the selected robot. Press Shift to cycle selection and Space to transform a robot."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_SIZE = 40
    GRID_W = SCREEN_WIDTH // GRID_SIZE
    GRID_H = SCREEN_HEIGHT // GRID_SIZE

    FPS = 30
    EPISODE_SECONDS = 60
    MAX_STEPS = FPS * EPISODE_SECONDS
    WIN_SCORE = 20

    # Colors
    COLOR_BG = (15, 20, 30)
    COLOR_GRID = (25, 30, 45)
    COLOR_ENEMY = (255, 50, 50)
    COLOR_ENERGY = (255, 220, 0)
    COLOR_TEXT = (220, 220, 240)
    ROBOT_COLORS = [(0, 150, 255), (0, 255, 150), (255, 255, 0), (255, 0, 255)]
    ROBOT_TRANSFORMED_COLORS = [(100, 200, 255), (100, 255, 200), (255, 255, 120), (255, 120, 255)]

    # Game Parameters
    ROBOT_SPEED = 0.2  # Lerp factor
    ROBOT_SIZE = GRID_SIZE * 0.8
    ROBOT_ENERGY_TO_TRANSFORM = 100
    ROBOT_TRANSFORM_COOLDOWN = 1 * FPS
    ROBOT_HIT_STUN_DURATION = 0.5 * FPS

    ENEMY_SPEED = 0.03
    ENEMY_SIZE = GRID_SIZE * 0.4
    ENEMY_HEALTH = 100
    ENEMY_DAMAGE = 50
    ENEMY_ATTACK_RADIUS = GRID_SIZE * 1.1
    INITIAL_ENEMY_SPAWN_RATE = 1.0 # per second
    ENEMY_SPAWN_RAMP_UP = 0.1 # per 10 seconds

    ENERGY_SPAWN_RATE = 0.5 # per second
    ENERGY_VALUE = 50

    COMBINED_SQUAD_DURATION = 10 * FPS
    COMBINED_SQUAD_ATTACK_RADIUS = GRID_SIZE * 3.5
    COMBINED_SQUAD_DAMAGE = 25

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.render_mode = render_mode
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_big = pygame.font.SysFont("Consolas", 48, bold=True)

        # State variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_remaining = 0
        self.robots = []
        self.enemies = []
        self.energy_pickups = []
        self.combined_squads = []
        self.particles = []
        self.attack_visuals = []
        self.selected_robot_idx = 0
        self.last_action = np.array([0, 0, 0])
        self.enemy_spawn_timer = 0
        self.energy_spawn_timer = 0
        self.current_enemy_spawn_rate = 0.0
        self.step_reward = 0.0
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_remaining = self.MAX_STEPS
        self.step_reward = 0.0
        
        self.selected_robot_idx = 0
        self.last_action = np.array([0, 0, 0])
        self.current_enemy_spawn_rate = self.INITIAL_ENEMY_SPAWN_RATE

        self.robots = self._initialize_robots()
        self.enemies = []
        self.energy_pickups = []
        self.combined_squads = []
        self.particles = []
        self.attack_visuals = []

        self.enemy_spawn_timer = 1 / self.current_enemy_spawn_rate * self.FPS
        self.energy_spawn_timer = 1 / self.ENERGY_SPAWN_RATE * self.FPS

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        self.time_remaining -= 1
        self.step_reward = 0.0 # Small penalty for each step to encourage efficiency
        
        self._handle_input(action)
        self._update_game_state()
        self._handle_interactions()
        self._check_combinations()
        self._update_squads()
        
        self.last_action = action
        
        terminated = self._check_termination()
        reward = self.step_reward

        if terminated:
            if self.score >= self.WIN_SCORE:
                reward += 100.0 # Win bonus
            else:
                reward -= 100.0 # Lose penalty
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_pressed, shift_pressed = action[0], action[1] == 1, action[2] == 1
        last_space, last_shift = self.last_action[1] == 1, self.last_action[2] == 1

        # Cycle selection on Shift press (rising edge)
        if shift_pressed and not last_shift:
            self.selected_robot_idx = (self.selected_robot_idx + 1) % len(self.robots)

        selected_robot = self.robots[self.selected_robot_idx]
        
        # Don't allow actions if robot is stunned or combined
        if selected_robot['stun_timer'] > 0 or selected_robot['state'] == 'combined':
            return
            
        # Transform on Space press (rising edge)
        if space_pressed and not last_space:
            if selected_robot['state'] == 'normal' and selected_robot['energy'] >= self.ROBOT_ENERGY_TO_TRANSFORM and selected_robot['transform_cooldown'] == 0:
                selected_robot['state'] = 'transformed'
                selected_robot['energy'] -= self.ROBOT_ENERGY_TO_TRANSFORM
                selected_robot['transform_cooldown'] = self.ROBOT_TRANSFORM_COOLDOWN
                self._add_particles(selected_robot['pos'], 20, self.ROBOT_TRANSFORMED_COLORS[selected_robot['id']], 2, 15)

        # Movement
        if movement > 0:
            current_grid_pos = (selected_robot['target_pos'] / self.GRID_SIZE).astype(int)
            dx, dy = 0, 0
            if movement == 1: dy = -1 # Up
            elif movement == 2: dy = 1 # Down
            elif movement == 3: dx = -1 # Left
            elif movement == 4: dx = 1 # Right

            new_grid_x = np.clip(current_grid_pos[0] + dx, 0, self.GRID_W - 1)
            new_grid_y = np.clip(current_grid_pos[1] + dy, 0, self.GRID_H - 1)
            
            is_occupied = False
            for r in self.robots:
                if r['id'] != selected_robot['id']:
                    other_grid_pos = (r['target_pos'] / self.GRID_SIZE).astype(int)
                    if other_grid_pos[0] == new_grid_x and other_grid_pos[1] == new_grid_y:
                        is_occupied = True
                        break
            
            if not is_occupied:
                selected_robot['target_pos'] = np.array([
                    new_grid_x * self.GRID_SIZE + self.GRID_SIZE / 2,
                    new_grid_y * self.GRID_SIZE + self.GRID_SIZE / 2
                ])

    def _update_game_state(self):
        # Update robots
        for r in self.robots:
            r['pos'] += (r['target_pos'] - r['pos']) * self.ROBOT_SPEED
            if r['transform_cooldown'] > 0: r['transform_cooldown'] -= 1
            if r['stun_timer'] > 0: r['stun_timer'] -= 1

        # Update enemies
        for e in self.enemies:
            if not self.robots: break
            non_combined_robots = [r for r in self.robots if r['state'] != 'combined']
            if not non_combined_robots: break
            
            target_robot = min(non_combined_robots, key=lambda r: np.linalg.norm(r['pos'] - e['pos']))
            direction = target_robot['pos'] - e['pos']
            norm = np.linalg.norm(direction)
            if norm > 0:
                e['pos'] += direction / norm * self.ENEMY_SPEED * self.GRID_SIZE

        # Update particles
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1
            p['radius'] *= 0.95

        # Update attack visuals
        self.attack_visuals = [v for v in self.attack_visuals if v['life'] > 0]
        for v in self.attack_visuals:
            v['life'] -= 1

        self._spawn_entities()

    def _handle_interactions(self):
        # Robot attacks
        for r in self.robots:
            if r['state'] == 'normal' or r['state'] == 'transformed':
                attack_damage = 75 if r['state'] == 'transformed' else 25
                attack_radius = self.GRID_SIZE * 1.1
                for e in self.enemies:
                    if np.linalg.norm(r['pos'] - e['pos']) < attack_radius:
                        self._damage_enemy(e, attack_damage)
                        self.attack_visuals.append({'start': r['pos'], 'end': e['pos'], 'color': r['color'], 'life': 3})
                        break 

        # Enemy attacks
        for e in self.enemies:
            for r in self.robots:
                if r['state'] != 'combined' and r['stun_timer'] == 0:
                    if np.linalg.norm(e['pos'] - r['pos']) < self.ENEMY_ATTACK_RADIUS:
                        r['stun_timer'] = self.ROBOT_HIT_STUN_DURATION
                        self.step_reward -= 0.5
                        self._add_particles(r['pos'], 15, self.COLOR_ENEMY, 3, 10)

        # Energy pickups
        for r in self.robots:
            for pickup in self.energy_pickups[:]:
                if np.linalg.norm(r['pos'] - pickup['pos']) < self.GRID_SIZE * 0.6:
                    r['energy'] = min(r['energy'] + self.ENERGY_VALUE, self.ROBOT_ENERGY_TO_TRANSFORM)
                    self.energy_pickups.remove(pickup)
                    self._add_particles(r['pos'], 10, self.COLOR_ENERGY, 2, 20)
                    
    def _damage_enemy(self, enemy, damage):
        enemy['health'] -= damage
        self.step_reward += 0.1
        if enemy['health'] <= 0:
            if enemy in self.enemies:
                self.enemies.remove(enemy)
                self.score += 1
                self.step_reward += 1.0
                self._add_particles(enemy['pos'], 30, (255, 255, 255), 5, 25)

    def _check_combinations(self):
        transformed_robots = [r for r in self.robots if r['state'] == 'transformed']
        if len(transformed_robots) < 2:
            return

        adj = {r['id']: [] for r in transformed_robots}
        robot_map = {r['id']: r for r in transformed_robots}

        for i in range(len(transformed_robots)):
            for j in range(i + 1, len(transformed_robots)):
                r1 = transformed_robots[i]
                r2 = transformed_robots[j]
                pos1 = (r1['target_pos'] / self.GRID_SIZE).astype(int)
                pos2 = (r2['target_pos'] / self.GRID_SIZE).astype(int)
                dist = np.sum(np.abs(pos1 - pos2))
                if dist == 1:
                    adj[r1['id']].append(r2['id'])
                    adj[r2['id']].append(r1['id'])

        visited = set()
        for r_id in adj:
            if r_id not in visited:
                component = []
                stack = [r_id]
                visited.add(r_id)
                while stack:
                    curr_id = stack.pop()
                    component.append(robot_map[curr_id])
                    for neighbor_id in adj[curr_id]:
                        if neighbor_id not in visited:
                            visited.add(neighbor_id)
                            stack.append(neighbor_id)
                
                if len(component) >= 2:
                    self._form_squad(component)

    def _form_squad(self, component_robots):
        for r in component_robots:
            r['state'] = 'combined'
        
        squad_pos = np.mean([r['pos'] for r in component_robots], axis=0)
        squad_colors = [r['color'] for r in component_robots]

        self.combined_squads.append({
            'pos': squad_pos,
            'robots': component_robots,
            'colors': squad_colors,
            'timer': self.COMBINED_SQUAD_DURATION,
            'attack_cooldown': 0
        })
        self.step_reward += 5.0
        self._add_particles(squad_pos, 50, (200, 200, 255), 4, 30)

    def _update_squads(self):
        for squad in self.combined_squads[:]:
            squad['timer'] -= 1
            if squad['timer'] <= 0:
                for r in squad['robots']:
                    r['state'] = 'transformed' 
                    r['transform_cooldown'] = self.ROBOT_TRANSFORM_COOLDOWN
                self.combined_squads.remove(squad)
                self._add_particles(squad['pos'], 30, (150, 150, 150), 3, 20)
                continue
            
            squad['pos'] = np.mean([r['pos'] for r in squad['robots']], axis=0)
            if squad['attack_cooldown'] > 0:
                squad['attack_cooldown'] -= 1
            else:
                squad['attack_cooldown'] = 0.5 * self.FPS
                self._add_particles(squad['pos'], 1, (255,255,255), self.COMBINED_SQUAD_ATTACK_RADIUS, 5, is_ring=True)
                for e in self.enemies:
                    if np.linalg.norm(squad['pos'] - e['pos']) < self.COMBINED_SQUAD_ATTACK_RADIUS:
                        self._damage_enemy(e, self.COMBINED_SQUAD_DAMAGE)
                        self.attack_visuals.append({'start': squad['pos'], 'end': e['pos'], 'color': self.np_random.choice(squad['colors']), 'life': 5})

    def _spawn_entities(self):
        self.enemy_spawn_timer -= 1
        if self.enemy_spawn_timer <= 0:
            self._spawn_enemy()
            self.current_enemy_spawn_rate = self.INITIAL_ENEMY_SPAWN_RATE + (self.steps / (10 * self.FPS)) * self.ENEMY_SPAWN_RAMP_UP
            self.enemy_spawn_timer = (1 / self.current_enemy_spawn_rate) * self.FPS

        self.energy_spawn_timer -= 1
        if self.energy_spawn_timer <= 0 and len(self.energy_pickups) < 5:
            self._spawn_energy_pickup()
            self.energy_spawn_timer = (1 / self.ENERGY_SPAWN_RATE) * self.FPS

    def _initialize_robots(self):
        robots = []
        start_positions = [
            (self.GRID_W // 4, self.GRID_H // 2),
            (self.GRID_W // 4, self.GRID_H // 2 - 1),
            (self.GRID_W // 4, self.GRID_H // 2 + 1),
            (self.GRID_W // 4 - 1, self.GRID_H // 2)
        ]
        for i in range(4):
            grid_pos = start_positions[i]
            pos = np.array([grid_pos[0] * self.GRID_SIZE + self.GRID_SIZE / 2, grid_pos[1] * self.GRID_SIZE + self.GRID_SIZE / 2], dtype=float)
            robots.append({
                'id': i,
                'pos': pos.copy(),
                'target_pos': pos.copy(),
                'color': self.ROBOT_COLORS[i],
                'transformed_color': self.ROBOT_TRANSFORMED_COLORS[i],
                'state': 'normal', 
                'energy': 0,
                'transform_cooldown': 0,
                'stun_timer': 0,
            })
        return robots

    def _spawn_enemy(self):
        edge = self.np_random.integers(4)
        if edge == 0: # top
            pos = np.array([self.np_random.uniform(0, self.SCREEN_WIDTH), -self.ENEMY_SIZE])
        elif edge == 1: # bottom
            pos = np.array([self.np_random.uniform(0, self.SCREEN_WIDTH), self.SCREEN_HEIGHT + self.ENEMY_SIZE])
        elif edge == 2: # left
            pos = np.array([-self.ENEMY_SIZE, self.np_random.uniform(0, self.SCREEN_HEIGHT)])
        else: # right
            pos = np.array([self.SCREEN_WIDTH + self.ENEMY_SIZE, self.np_random.uniform(0, self.SCREEN_HEIGHT)])
        
        self.enemies.append({'pos': pos, 'health': self.ENEMY_HEALTH})

    def _spawn_energy_pickup(self):
        pos = self.np_random.uniform(low=[self.GRID_SIZE, self.GRID_SIZE], high=[self.SCREEN_WIDTH - self.GRID_SIZE, self.SCREEN_HEIGHT - self.GRID_SIZE], size=2)
        self.energy_pickups.append({'pos': pos})
    
    def _add_particles(self, pos, count, color, speed_scale, life, is_ring=False):
        for _ in range(count):
            if is_ring:
                angle = self.np_random.uniform(0, 2 * math.pi)
                vel = np.array([math.cos(angle), math.sin(angle)]) * speed_scale
            else:
                vel = self.np_random.uniform(-1, 1, size=2)
                vel = vel / np.linalg.norm(vel) * self.np_random.uniform(0.5, 1.5) * speed_scale
            
            self.particles.append({
                'pos': pos.copy(),
                'vel': vel,
                'life': self.np_random.integers(life // 2, life),
                'radius': self.np_random.uniform(2, 5),
                'color': color,
                'is_ring': is_ring,
            })

    def _check_termination(self):
        return self.time_remaining <= 0 or self.score >= self.WIN_SCORE

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
            "time_remaining": self.time_remaining / self.FPS,
            "robots_state": [r['state'] for r in self.robots],
            "enemies_count": len(self.enemies),
        }

    def _render_game(self):
        for x in range(0, self.SCREEN_WIDTH, self.GRID_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, self.GRID_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))

        self._render_pickups()
        self._render_enemies()
        self._render_robots()
        self._render_squads()
        self._render_effects()

    def _render_robots(self):
        for i, r in enumerate(self.robots):
            if r['state'] == 'combined':
                continue
            
            x, y = int(r['pos'][0]), int(r['pos'][1])
            size = self.ROBOT_SIZE
            rect = pygame.Rect(x - size / 2, y - size / 2, size, size)
            
            color = r['transformed_color'] if r['state'] == 'transformed' else r['color']
            
            if r['state'] == 'transformed':
                glow_size = size * 1.5
                glow_surf = pygame.Surface((glow_size, glow_size), pygame.SRCALPHA)
                pygame.draw.circle(glow_surf, (*color, 60), (glow_size/2, glow_size/2), glow_size/2)
                self.screen.blit(glow_surf, (x - glow_size/2, y - glow_size/2))

            pygame.draw.rect(self.screen, color, rect, border_radius=4)
            
            if r['stun_timer'] > 0:
                stun_alpha = 100 + 100 * math.sin(self.steps * 1.5)
                s = pygame.Surface((size, size), pygame.SRCALPHA)
                s.fill((255, 50, 50, stun_alpha))
                self.screen.blit(s, rect.topleft)

            if i == self.selected_robot_idx:
                pulse = (math.sin(self.steps * 0.2) + 1) / 2
                alpha = int(100 + 155 * pulse)
                pygame.gfxdraw.rectangle(self.screen, rect, (*self.COLOR_TEXT, alpha))
                pygame.gfxdraw.rectangle(self.screen, rect.inflate(2, 2), (*self.COLOR_TEXT, alpha // 2))

    def _render_enemies(self):
        for e in self.enemies:
            x, y = int(e['pos'][0]), int(e['pos'][1])
            size = self.ENEMY_SIZE
            glow_size = size * 2.5
            glow_surf = pygame.Surface((glow_size, glow_size), pygame.SRCALPHA)
            pygame.draw.circle(glow_surf, (*self.COLOR_ENEMY, 80), (glow_size/2, glow_size/2), glow_size/2)
            self.screen.blit(glow_surf, (x - glow_size/2, y - glow_size/2))
            pygame.gfxdraw.aacircle(self.screen, x, y, int(size), self.COLOR_ENEMY)
            pygame.gfxdraw.filled_circle(self.screen, x, y, int(size), self.COLOR_ENEMY)

    def _render_pickups(self):
        for p in self.energy_pickups:
            x, y = int(p['pos'][0]), int(p['pos'][1])
            size = self.GRID_SIZE * 0.4
            points = [
                (x, y - size * 0.8),
                (x - size * 0.7, y + size * 0.5),
                (x + size * 0.7, y + size * 0.5)
            ]
            glow_size = size * 3
            glow_surf = pygame.Surface((glow_size, glow_size), pygame.SRCALPHA)
            pygame.draw.circle(glow_surf, (*self.COLOR_ENERGY, 90), (glow_size/2, glow_size/2), glow_size/2)
            self.screen.blit(glow_surf, (p['pos'][0] - glow_size/2, p['pos'][1] - glow_size/2))
            pygame.gfxdraw.aatrigon(self.screen, int(points[0][0]), int(points[0][1]), int(points[1][0]), int(points[1][1]), int(points[2][0]), int(points[2][1]), self.COLOR_ENERGY)
            pygame.gfxdraw.filled_trigon(self.screen, int(points[0][0]), int(points[0][1]), int(points[1][0]), int(points[1][1]), int(points[2][0]), int(points[2][1]), self.COLOR_ENERGY)

    def _render_squads(self):
        for s in self.combined_squads:
            x, y = int(s['pos'][0]), int(s['pos'][1])
            num_robots = len(s['robots'])
            base_size = self.ROBOT_SIZE * math.sqrt(num_robots) * 0.8
            
            pulse = (math.sin(self.steps * 0.1) + 1) / 2
            glow_size = base_size * (1.8 + 0.2 * pulse)
            glow_surf = pygame.Surface((glow_size, glow_size), pygame.SRCALPHA)
            pygame.draw.circle(glow_surf, (*self.COLOR_TEXT, 40), (glow_size/2, glow_size/2), glow_size/2)
            self.screen.blit(glow_surf, (x - glow_size/2, y - glow_size/2))

            for i in range(num_robots):
                angle = 2 * math.pi * i / num_robots + self.steps * 0.02
                r_size = base_size * 0.6
                rx = x + math.cos(angle) * base_size * 0.3
                ry = y + math.sin(angle) * base_size * 0.3
                rect = pygame.Rect(rx - r_size/2, ry - r_size/2, r_size, r_size)
                pygame.draw.rect(self.screen, s['colors'][i], rect, border_radius=4)

    def _render_effects(self):
        for p in self.particles:
            color = (*p['color'], int(255 * (p['life'] / 15)))
            if p.get('is_ring', False):
                pygame.gfxdraw.aacircle(self.screen, int(p['pos'][0]), int(p['pos'][1]), int(p['radius']), color)
            else:
                pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), int(p['radius']), color)
        
        for v in self.attack_visuals:
            alpha = int(255 * (v['life'] / 5))
            color = (*v['color'], alpha)
            pygame.draw.aaline(self.screen, color, v['start'], v['end'], 2)

    def _render_ui(self):
        score_text = self.font_ui.render(f"ENEMIES: {self.score} / {self.WIN_SCORE}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        time_str = f"TIME: {max(0, self.time_remaining // self.FPS):02d}"
        time_text = self.font_ui.render(time_str, True, self.COLOR_TEXT)
        self.screen.blit(time_text, (self.SCREEN_WIDTH - time_text.get_width() - 10, 10))

        for i, r in enumerate(self.robots):
            status_y = 40 + i * 25
            pygame.draw.rect(self.screen, r['color'], (10, status_y, 20, 20), border_radius=3)
            if i == self.selected_robot_idx:
                pygame.draw.rect(self.screen, self.COLOR_TEXT, (10, status_y, 20, 20), 2, border_radius=3)
            
            energy_w = (r['energy'] / self.ROBOT_ENERGY_TO_TRANSFORM) * 100
            pygame.draw.rect(self.screen, self.COLOR_BG, (35, status_y, 102, 20))
            pygame.draw.rect(self.screen, self.COLOR_ENERGY, (36, status_y+1, energy_w, 18))
            pygame.draw.rect(self.screen, self.COLOR_TEXT, (35, status_y, 102, 20), 1)

            if r['state'] == 'transformed':
                t_text = self.font_ui.render("T", True, r['transformed_color'])
                self.screen.blit(t_text, (145, status_y - 4))
            elif r['state'] == 'combined':
                c_text = self.font_ui.render("C", True, self.COLOR_TEXT)
                self.screen.blit(c_text, (145, status_y - 4))

        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            msg = "VICTORY" if self.score >= self.WIN_SCORE else "TIME UP"
            msg_text = self.font_big.render(msg, True, self.COLOR_TEXT)
            self.screen.blit(msg_text, (self.SCREEN_WIDTH/2 - msg_text.get_width()/2, self.SCREEN_HEIGHT/2 - msg_text.get_height()/2))

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # It will not run in the evaluation environment, but is useful for development.
    # To use, you might need to remove the os.environ line and install pygame.
    
    # Re-enable display for local testing
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Mech-Combine")
    clock = pygame.time.Clock()

    terminated = False
    total_reward = 0
    
    # Use a persistent action state for better human playability
    action = np.array([0, 0, 0])
    
    while not terminated:
        # --- Human Input ---
        # Reset movement and one-shot actions
        action[0] = 0 # No movement unless a key is pressed
        
        # Check for quit event
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            # Handle key-down for one-shot actions (rising edge)
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    action[1] = 1
                if event.key in [pygame.K_LSHIFT, pygame.K_RSHIFT]:
                    action[2] = 1
        
        # Handle continuous-press actions (movement)
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: action[0] = 1
        elif keys[pygame.K_DOWN]: action[0] = 2
        elif keys[pygame.K_LEFT]: action[0] = 3
        elif keys[pygame.K_RIGHT]: action[0] = 4
        
        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Reset one-shot actions after they have been processed by the step
        action[1] = 0
        action[2] = 0

        # --- Rendering ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        # --- Clock ---
        clock.tick(GameEnv.FPS)

    print(f"Game Over. Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
    env.close()
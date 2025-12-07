import gymnasium as gym
import os
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque
import numpy as np
from gymnasium.spaces import MultiDiscrete, Box
import os
import pygame


# Ensure Pygame runs in a headless mode for server environments
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    An environment where the player navigates a bioluminescent reef.
    The goal is to teleport between coral structures to reach a safe city,
    while avoiding predators and managing a depleting oxygen supply.
    The player can manipulate time to temporarily freeze predators.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # --- METADATA ---
    game_description = (
        "Navigate a bioluminescent reef by teleporting between corals. Avoid predators and manage your oxygen to reach the safety of the abyssal city."
    )
    user_guide = (
        "Use the arrow keys (↑↓←→) to select a nearby coral. Press space to teleport. Press shift to slow down time, but watch the cooldown."
    )
    auto_advance = True

    # --- CONSTANTS ---
    # Game Feel & Sizing
    PLAYER_RADIUS = 12
    PREDATOR_RADIUS = 15
    CORAL_MIN_RADIUS = 20
    CORAL_MAX_RADIUS = 45
    CITY_RADIUS = 50
    NUM_CORALS = 15
    NUM_PREDATORS = 4
    MAX_TELEPORT_DIST = 200
    PREDATOR_PATROL_RADIUS = 100
    PREDATOR_BASE_SPEED = 0.75

    # Timing & Limits
    MAX_STEPS = 2000
    OXYGEN_MAX = 1000
    OXYGEN_DEPLETION_RATE = 0.5
    TIME_MANIP_DURATION = 90  # 3 seconds at 30fps
    TIME_MANIP_COOLDOWN = 300  # 10 seconds

    # Rewards
    REWARD_WIN = 100
    REWARD_LOSE = -100
    REWARD_STEP = 0.01
    REWARD_CLOSER_TO_CITY_MULTIPLIER = 0.2
    REWARD_UNNECESSARY_TIME_MANIP = -1
    REWARD_SUCCESSFUL_TIME_MANIP = 5
    REWARD_PREDATOR_PROXIMITY = -0.1
    PREDATOR_DANGER_ZONE = 80

    # Colors (Bioluminescent Theme)
    COLOR_BG = (5, 10, 25)
    COLOR_PLAYER = (0, 255, 255)
    COLOR_PLAYER_GLOW = (0, 150, 150)
    COLOR_PREDATOR = (255, 50, 50)
    COLOR_PREDATOR_EYE = (255, 255, 0)
    COLOR_PREDATOR_FROZEN_AURA = (100, 100, 255)
    COLOR_CORAL_PALETTE = [(20, 80, 180), (150, 40, 190), (40, 190, 150)]
    COLOR_CITY = (255, 223, 0)
    COLOR_CITY_GLOW = (180, 150, 0)
    COLOR_OXYGEN_BAR = (0, 200, 255)
    COLOR_OXYGEN_BG = (20, 50, 80)
    COLOR_TELEPORT_TARGET = (255, 255, 255)
    COLOR_TIME_MANIP_WAVE = (180, 0, 255)
    COLOR_TEXT = (220, 220, 240)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.screen_width = 640
        self.screen_height = 400

        self.observation_space = Box(
            low=0, high=255, shape=(self.screen_height, self.screen_width, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        # Use SRCALPHA to allow for surfaces with per-pixel alpha, fixing blending issues.
        self.screen = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("monospace", 18, bold=True)

        # State variables are initialized in reset()
        self.player_pos = None
        self.corals = None
        self.predators = None
        self.city = None
        self.start_pos = None
        self.steps = None
        self.score = None
        self.oxygen = None
        self.predator_speed = None
        self.time_manip_active_steps = None
        self.time_manip_cooldown_steps = None
        self.selected_coral_idx = None
        self.last_space_held = None
        self.last_shift_held = None
        self.particles = None
        self.dist_to_city = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.oxygen = self.OXYGEN_MAX
        self.predator_speed = self.PREDATOR_BASE_SPEED
        self.time_manip_active_steps = 0
        self.time_manip_cooldown_steps = 0
        self.last_space_held = False
        self.last_shift_held = False
        self.particles = []

        self._generate_level()

        start_coral_idx = -1
        for i, coral in enumerate(self.corals):
            if coral['pos'] == self.start_pos:
                start_coral_idx = i
                break
        
        self.player_pos = self.corals[start_coral_idx]['pos'].copy()
        self.selected_coral_idx = start_coral_idx

        self.dist_to_city = self.player_pos.distance_to(self.city['pos'])

        return self._get_observation(), self._get_info()

    def _generate_level(self):
        while True:
            self.corals = []
            for i in range(self.NUM_CORALS):
                pos = pygame.Vector2(
                    self.np_random.uniform(50, self.screen_width - 50),
                    self.np_random.uniform(50, self.screen_height - 50)
                )
                radius = self.np_random.uniform(self.CORAL_MIN_RADIUS, self.CORAL_MAX_RADIUS)
                color = random.choice(self.COLOR_CORAL_PALETTE)
                self.corals.append({'id': i, 'pos': pos, 'radius': radius, 'color': color})

            chosen_corals = self.np_random.choice(self.corals, size=2, replace=False)
            self.start_pos = chosen_corals[0]['pos']
            city_pos = chosen_corals[1]['pos']

            if self.start_pos.distance_to(city_pos) < self.screen_width / 2:
                continue

            self.city = {'pos': city_pos, 'radius': self.CITY_RADIUS}

            adj = [[] for _ in range(self.NUM_CORALS)]
            start_idx, city_idx = -1, -1
            for i in range(self.NUM_CORALS):
                if self.corals[i]['pos'] == self.start_pos: start_idx = i
                if self.corals[i]['pos'] == city_pos: city_idx = i
                for j in range(i + 1, self.NUM_CORALS):
                    if self.corals[i]['pos'].distance_to(self.corals[j]['pos']) < self.MAX_TELEPORT_DIST:
                        adj[i].append(j)
                        adj[j].append(i)

            q = deque([start_idx])
            visited = {start_idx}
            path_found = False
            while q:
                u = q.popleft()
                if u == city_idx:
                    path_found = True
                    break
                for v in adj[u]:
                    if v not in visited:
                        visited.add(v)
                        q.append(v)

            if path_found:
                break

        self.predators = []
        for _ in range(self.NUM_PREDATORS):
            center = pygame.Vector2(
                self.np_random.uniform(100, self.screen_width - 100),
                self.np_random.uniform(100, self.screen_height - 100)
            )
            angle1 = self.np_random.uniform(0, 2 * math.pi)
            angle2 = angle1 + self.np_random.uniform(math.pi / 2, 3 * math.pi / 2)
            p1 = center + pygame.Vector2(math.cos(angle1), math.sin(angle1)) * self.PREDATOR_PATROL_RADIUS
            p2 = center + pygame.Vector2(math.cos(angle2), math.sin(angle2)) * self.PREDATOR_PATROL_RADIUS
            self.predators.append({
                'pos': p1.copy(),
                'path': [p1, p2],
                'target_idx': 1,
            })

    def step(self, action):
        movement, space_pressed, shift_pressed = action[0], action[1] == 1, action[2] == 1
        reward = self.REWARD_STEP

        if movement != 0:
            self._update_teleport_selection(movement)

        if shift_pressed and not self.last_shift_held and self.time_manip_cooldown_steps <= 0:
            self.time_manip_active_steps = self.TIME_MANIP_DURATION
            self.time_manip_cooldown_steps = self.TIME_MANIP_COOLDOWN
            self.particles.append(self._create_particle('time_wave', self.player_pos))
            predator_was_nearby = any(p['pos'].distance_to(self.player_pos) < self.PREDATOR_DANGER_ZONE * 1.5 for p in self.predators)
            reward += self.REWARD_SUCCESSFUL_TIME_MANIP if predator_was_nearby else self.REWARD_UNNECESSARY_TIME_MANIP

        if space_pressed and not self.last_space_held:
            if self.selected_coral_idx is not None and self.corals[self.selected_coral_idx]['pos'] != self.player_pos:
                target_pos = self.corals[self.selected_coral_idx]['pos']
                if self.player_pos.distance_to(target_pos) < self.MAX_TELEPORT_DIST:
                    self.particles.append(self._create_particle('teleport_out', self.player_pos))
                    self.player_pos = target_pos.copy()
                    self.particles.append(self._create_particle('teleport_in', self.player_pos))

        self.last_space_held = space_pressed
        self.last_shift_held = shift_pressed

        self.steps += 1
        self.oxygen -= self.OXYGEN_DEPLETION_RATE
        if self.time_manip_active_steps > 0:
            self.time_manip_active_steps -= 1
        else:
            self._update_predators()
        if self.time_manip_cooldown_steps > 0:
            self.time_manip_cooldown_steps -= 1
        self._update_particles()

        if self.steps > 0 and self.steps % 200 == 0:
            self.predator_speed += 0.05

        new_dist_to_city = self.player_pos.distance_to(self.city['pos'])
        reward += (self.dist_to_city - new_dist_to_city) * self.REWARD_CLOSER_TO_CITY_MULTIPLIER
        self.dist_to_city = new_dist_to_city

        for p in self.predators:
            if p['pos'].distance_to(self.player_pos) < self.PREDATOR_DANGER_ZONE:
                reward += self.REWARD_PREDATOR_PROXIMITY

        terminated = False
        for p in self.predators:
            if p['pos'].distance_to(self.player_pos) < self.PLAYER_RADIUS + self.PREDATOR_RADIUS:
                reward = self.REWARD_LOSE
                terminated = True
                break
        if not terminated and self.player_pos.distance_to(self.city['pos']) < self.PLAYER_RADIUS + self.city['radius']:
            reward = self.REWARD_WIN
            terminated = True
        if not terminated and self.oxygen <= 0:
            reward = self.REWARD_LOSE
            terminated = True

        truncated = self.steps >= self.MAX_STEPS
        self.score += reward

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _update_teleport_selection(self, movement):
        current_coral_id = -1
        for coral in self.corals:
            if coral['pos'] == self.player_pos:
                current_coral_id = coral['id']
                break

        candidates = [(i, c['pos']) for i, c in enumerate(self.corals) if i != current_coral_id and self.player_pos.distance_to(c['pos']) < self.MAX_TELEPORT_DIST]
        if not candidates:
            return

        best_candidate_idx = self.selected_coral_idx
        min_dist = float('inf')

        for idx, pos in candidates:
            rel_pos = pos - self.player_pos
            dist = 0
            if movement == 1 and rel_pos.y < 0: dist = abs(rel_pos.y)
            elif movement == 2 and rel_pos.y > 0: dist = rel_pos.y
            elif movement == 3 and rel_pos.x < 0: dist = abs(rel_pos.x)
            elif movement == 4 and rel_pos.x > 0: dist = rel_pos.x
            
            if 0 < dist < min_dist:
                min_dist = dist
                best_candidate_idx = idx
        
        self.selected_coral_idx = best_candidate_idx

    def _update_predators(self):
        for p in self.predators:
            target_pos = p['path'][p['target_idx']]
            direction = (target_pos - p['pos'])
            if direction.length() < self.predator_speed:
                p['pos'] = target_pos.copy()
                p['target_idx'] = 1 - p['target_idx']
            else:
                p['pos'] += direction.normalize() * self.predator_speed

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['life'] -= 1
            if p['type'] in ['teleport_in', 'teleport_out', 'time_wave']:
                p['radius'] += p['vel']
            elif p['type'] == 'bubble':
                p['pos'].y -= p['vel']
                p['pos'].x += math.sin(p['life'] * 0.1) * 0.5

    def _create_particle(self, p_type, pos):
        if p_type == 'teleport_in': return {'type': p_type, 'pos': pos.copy(), 'radius': 0, 'vel': 2, 'life': 15, 'color': self.COLOR_PLAYER}
        if p_type == 'teleport_out': return {'type': p_type, 'pos': pos.copy(), 'radius': self.PLAYER_RADIUS, 'vel': 2, 'life': 15, 'color': self.COLOR_PLAYER}
        if p_type == 'time_wave': return {'type': p_type, 'pos': pos.copy(), 'radius': 0, 'vel': 10, 'life': 40, 'color': self.COLOR_TIME_MANIP_WAVE}
        if p_type == 'bubble': return {'type': p_type, 'pos': pos.copy(), 'radius': self.np_random.integers(1, 4), 'vel': self.np_random.uniform(0.5, 1.5), 'life': 60, 'color': (200, 200, 255)}
        return None

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "oxygen": self.oxygen}

    def _render_game(self):
        if self.steps % 10 == 0:
            self.particles.append(self._create_particle('bubble', self.player_pos))

        if self.selected_coral_idx is not None:
            target_pos = self.corals[self.selected_coral_idx]['pos']
            if self.player_pos.distance_to(target_pos) < self.MAX_TELEPORT_DIST:
                pygame.draw.line(self.screen, self.COLOR_TELEPORT_TARGET, self.player_pos, target_pos, 1)
                pygame.gfxdraw.aacircle(self.screen, int(target_pos.x), int(target_pos.y), int(self.corals[self.selected_coral_idx]['radius']) + 5, self.COLOR_TELEPORT_TARGET)

        self._draw_glowing_circle(self.city['pos'], self.city['radius'], self.COLOR_CITY, self.COLOR_CITY_GLOW)
        for coral in self.corals:
            if coral['pos'] != self.city['pos']:
                self._draw_glowing_circle(coral['pos'], coral['radius'], coral['color'], tuple(c * 0.5 for c in coral['color']))

        for p in self.predators:
            target_vec = p['path'][p['target_idx']] - p['pos']
            angle = math.atan2(target_vec.y, target_vec.x) if target_vec.length() > 0 else 0
            pts = [p['pos'] + pygame.Vector2(math.cos(angle), math.sin(angle)) * self.PREDATOR_RADIUS,
                   p['pos'] + pygame.Vector2(math.cos(angle + 2.2), math.sin(angle + 2.2)) * self.PREDATOR_RADIUS * 0.8,
                   p['pos'] + pygame.Vector2(math.cos(angle - 2.2), math.sin(angle - 2.2)) * self.PREDATOR_RADIUS * 0.8]
            pygame.gfxdraw.aapolygon(self.screen, pts, self.COLOR_PREDATOR)
            pygame.gfxdraw.filled_polygon(self.screen, pts, self.COLOR_PREDATOR)
            eye_pos = p['pos'] + pygame.Vector2(math.cos(angle), math.sin(angle)) * (self.PREDATOR_RADIUS * 0.6)
            pygame.gfxdraw.filled_circle(self.screen, int(eye_pos.x), int(eye_pos.y), 2, self.COLOR_PREDATOR_EYE)
            if self.time_manip_active_steps > 0:
                self._draw_glowing_circle(p['pos'], self.PREDATOR_RADIUS, (0, 0, 0, 0), self.COLOR_PREDATOR_FROZEN_AURA, rings=2, max_alpha=100)

        for p in self.particles:
            alpha = 255 * (p['life'] / (15 if 'teleport' in p['type'] else (40 if 'time' in p['type'] else 60)))
            color = (*p['color'], int(max(0, min(255, alpha))))
            if p['type'] in ['teleport_in', 'teleport_out', 'time_wave']:
                pygame.gfxdraw.aacircle(self.screen, int(p['pos'].x), int(p['pos'].y), int(p['radius']), color)
            elif p['type'] == 'bubble':
                pygame.gfxdraw.filled_circle(self.screen, int(p['pos'].x), int(p['pos'].y), p['radius'], color)
                pygame.gfxdraw.aacircle(self.screen, int(p['pos'].x), int(p['pos'].y), p['radius'], color)

        self._draw_glowing_circle(self.player_pos, self.PLAYER_RADIUS, self.COLOR_PLAYER, self.COLOR_PLAYER_GLOW)

    def _draw_glowing_circle(self, pos, radius, color_main, color_glow, rings=4, max_alpha=100):
        for i in range(rings, 0, -1):
            alpha = int(max_alpha * (1 - (i / rings)))
            glow_radius = int(radius + i * 2)
            color = (*color_glow, alpha)
            pygame.gfxdraw.filled_circle(self.screen, int(pos.x), int(pos.y), glow_radius, color)
        pygame.gfxdraw.filled_circle(self.screen, int(pos.x), int(pos.y), int(radius), color_main)
        pygame.gfxdraw.aacircle(self.screen, int(pos.x), int(pos.y), int(radius), color_main)

    def _render_ui(self):
        bar_width = 200
        bar_height = 20
        oxygen_ratio = self.oxygen / self.OXYGEN_MAX
        pygame.draw.rect(self.screen, self.COLOR_OXYGEN_BG, (10, 10, bar_width, bar_height))
        pygame.draw.rect(self.screen, self.COLOR_OXYGEN_BAR, (10, 10, bar_width * oxygen_ratio, bar_height))

        score_text = self.font.render(f"SCORE: {self.score:.0f}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.screen_width - score_text.get_width() - 10, 10))

        cooldown_ratio = self.time_manip_cooldown_steps / self.TIME_MANIP_COOLDOWN
        if cooldown_ratio > 0:
            pygame.draw.circle(self.screen, self.COLOR_TIME_MANIP_WAVE, (35, 55), 15, 2)
            arc_angle = cooldown_ratio * 2 * math.pi
            rect = pygame.Rect(20, 40, 30, 30)
            pygame.draw.arc(self.screen, self.COLOR_BG, rect, math.pi / 2, math.pi / 2 + arc_angle, 15)
        else:
            self._draw_glowing_circle(pygame.Vector2(35, 55), 12, self.COLOR_TIME_MANIP_WAVE, self.COLOR_TIME_MANIP_WAVE, rings=3, max_alpha=150)

        if self.city and self.player_pos:
            dir_to_city = self.city['pos'] - self.player_pos
            if dir_to_city.length() > 0:
                dir_to_city.normalize_ip()
                glow_intensity = max(0, 1 - (self.dist_to_city / (self.screen_width * 1.5)))
                if glow_intensity > 0.1:
                    for i in range(3):
                        alpha = int(60 * glow_intensity * ((3 - i) / 3))
                        size = 150 * ((3 - i) / 3)
                        center_x = self.screen_width / 2 - dir_to_city.x * (self.screen_width / 2 + size / 2)
                        center_y = self.screen_height / 2 - dir_to_city.y * (self.screen_height / 2 + size / 2)
                        surf = pygame.Surface((size, size), pygame.SRCALPHA)
                        pygame.draw.circle(surf, (*self.COLOR_CITY, alpha), (size / 2, size / 2), size / 2)
                        self.screen.blit(surf, (center_x - size / 2, center_y - size / 2), special_flags=pygame.BLEND_RGBA_ADD)

    def close(self):
        pygame.font.quit()
        pygame.quit()


if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()

    # Create a display window for manual testing
    pygame.display.init()
    display_screen = pygame.display.set_mode((env.screen_width, env.screen_height))
    pygame.display.set_caption("Abyssal Reef Explorer")

    terminated = False
    truncated = False
    total_reward = 0

    while not terminated and not truncated:
        movement, space, shift = 0, 0, 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True

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

        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()

        env.clock.tick(30)

    print(f"Game Over! Final Score: {info['score']:.2f}, Steps: {info['steps']}")
    env.close()
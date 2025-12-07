import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T16:16:27.665338
# Source Brief: brief_00914.md
# Brief Index: 914
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    Glacier Guardian: A Gymnasium environment where the player teleports across a
    glacier to protect ancient sculptures from melting. The player must manage a
    time-slowing ability to solidify dangerous, spreading ice cracks.
    """
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    game_description = (
        "Teleport across a glacier to protect ancient sculptures from melting. "
        "Use a time-slowing ability to solidify dangerous, spreading ice cracks."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to teleport. Hold space to slow time and solidify cracks near you."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.MAX_STEPS = 1500 # Extended for more gameplay
        self.FPS = self.metadata["render_fps"]

        # Colors
        self.COLOR_BG = (10, 20, 40)
        self.COLOR_PLAYER = (255, 255, 0)
        self.COLOR_PLAYER_GLOW = (255, 255, 150)
        self.COLOR_CRACK = (220, 220, 255)
        self.COLOR_SOLID_CRACK = (100, 150, 255)
        self.COLOR_SCULPTURE = (0, 255, 255)
        self.COLOR_SCULPTURE_GLOW_HEALTHY = (0, 255, 255)
        self.COLOR_SCULPTURE_GLOW_MELTING = (255, 100, 0)
        self.COLOR_UI_TEXT = (230, 230, 240)
        self.COLOR_UI_BAR_BG = (50, 50, 80)
        self.COLOR_TIME_SLOW_BAR = (0, 150, 255)
        self.COLOR_HEALTH_BAR = (0, 200, 100)
        self.COLOR_HEALTH_BAR_LOW = (255, 50, 50)

        # Game Parameters
        self.PLAYER_TELEPORT_DIST = 40
        self.PLAYER_RADIUS = 8
        self.SCULPTURE_RADIUS = 15
        self.SCULPTURE_COUNT = 3
        self.SCULPTURE_STABILIZE_RADIUS = 100
        self.TIME_SLOW_MAX = 200
        self.TIME_SLOW_COST = 2
        self.TIME_SLOW_REGEN = 0.5
        self.TIME_SLOW_RADIUS = 120
        self.BASE_MELT_RATE = 0.02
        self.CRACK_MELT_RATE = 0.25
        self.INITIAL_CRACK_SPAWN_PERIOD = self.FPS * 5 # Every 5 seconds
        self.CRACK_SPAWN_PERIOD_DECREASE = 0.2 # Period decreases by this each spawn
        self.MIN_CRACK_SPAWN_PERIOD = self.FPS * 1 # Max spawn rate is 1 per second

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Consolas", 16)
        self.font_large = pygame.font.SysFont("Consolas", 24, bold=True)

        # --- State Variables ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_pos = None
        self.sculptures = []
        self.cracks = []
        self.particles = []
        self.time_slow_resource = 0
        self.is_slowing_time = False
        self.crack_spawn_timer = 0
        self.current_crack_spawn_period = 0
        self.aurora_points = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False

        # Player
        self.player_pos = np.array([self.WIDTH / 2, self.HEIGHT / 2], dtype=float)

        # Sculptures
        self.sculptures = []
        padding = 80
        for i in range(self.SCULPTURE_COUNT):
            while True:
                pos = np.array([
                    self.np_random.uniform(padding, self.WIDTH - padding),
                    self.np_random.uniform(padding, self.HEIGHT - padding)
                ])
                if not any(np.linalg.norm(pos - s['pos']) < 150 for s in self.sculptures):
                    self.sculptures.append({
                        'pos': pos,
                        'health': 100.0,
                        'max_health': 100.0
                    })
                    break

        # Environment
        self.cracks = []
        self.particles = []
        self.time_slow_resource = self.TIME_SLOW_MAX
        self.is_slowing_time = False
        self.current_crack_spawn_period = self.INITIAL_CRACK_SPAWN_PERIOD
        self.crack_spawn_timer = self.current_crack_spawn_period

        # Visuals
        self._init_aurora()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0

        # --- Action Handling ---
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        self._handle_player_action(movement)
        self.is_slowing_time = space_held and self.time_slow_resource > 0

        # --- Game Logic Updates ---
        self._update_time_slow()
        reward += self._update_cracks()
        reward += self._update_sculptures()
        self._update_particles()
        self._update_aurora()

        # --- Termination Check ---
        terminated = False
        truncated = False
        if any(s['health'] <= 0 for s in self.sculptures):
            reward -= 100.0 # Penalty for losing
            terminated = True
            self.game_over = True
        elif self.steps >= self.MAX_STEPS:
            reward += 100.0 # Bonus for winning
            terminated = True # Game ends successfully
            self.game_over = True

        self.score += reward
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    # --- Game Logic Sub-routines ---
    def _handle_player_action(self, movement):
        # 0=none, 1=up, 2=down, 3=left, 4=right
        if movement > 0:
            # Sfx: Player_Teleport_Sound()
            start_pos = self.player_pos.copy()
            if movement == 1: self.player_pos[1] -= self.PLAYER_TELEPORT_DIST
            elif movement == 2: self.player_pos[1] += self.PLAYER_TELEPORT_DIST
            elif movement == 3: self.player_pos[0] -= self.PLAYER_TELEPORT_DIST
            elif movement == 4: self.player_pos[0] += self.PLAYER_TELEPORT_DIST
            
            self.player_pos[0] = np.clip(self.player_pos[0], self.PLAYER_RADIUS, self.WIDTH - self.PLAYER_RADIUS)
            self.player_pos[1] = np.clip(self.player_pos[1], self.PLAYER_RADIUS, self.HEIGHT - self.PLAYER_RADIUS)
            self._create_teleport_particles(start_pos, self.player_pos)


    def _update_time_slow(self):
        if self.is_slowing_time:
            self.time_slow_resource -= self.TIME_SLOW_COST
        else:
            self.time_slow_resource = min(self.TIME_SLOW_MAX, self.time_slow_resource + self.TIME_SLOW_REGEN)

    def _update_cracks(self):
        reward = 0
        self.crack_spawn_timer -= 1
        if self.crack_spawn_timer <= 0:
            self._spawn_crack()
            self.current_crack_spawn_period = max(self.MIN_CRACK_SPAWN_PERIOD, self.current_crack_spawn_period - self.CRACK_SPAWN_PERIOD_DECREASE)
            self.crack_spawn_timer = self.current_crack_spawn_period

        for crack in self.cracks:
            # Grow crack if it's new
            if len(crack['points']) < crack['max_len'] and crack['state'] == 'active':
                last_point = crack['points'][-1]
                angle = crack['angle'] + self.np_random.uniform(-0.5, 0.5)
                new_point = last_point + np.array([math.cos(angle), math.sin(angle)]) * crack['growth_speed']
                if 0 < new_point[0] < self.WIDTH and 0 < new_point[1] < self.HEIGHT:
                    crack['points'].append(new_point)
                    crack['angle'] = angle

            # Solidify crack
            if self.is_slowing_time and crack['state'] == 'active':
                dist_to_player = min([np.linalg.norm(p - self.player_pos) for p in crack['points']])
                if dist_to_player < self.TIME_SLOW_RADIUS:
                    crack['state'] = 'solid'
                    crack['lifetime'] = self.FPS * 10 # Solid cracks last 10 seconds
                    # Sfx: Ice_Solidify_Sound()
                    for sculpture in self.sculptures:
                        dist_to_sculpture = min([np.linalg.norm(p - sculpture['pos']) for p in crack['points']])
                        if dist_to_sculpture < self.SCULPTURE_RADIUS + 30:
                            reward += 1.0 # Reward for solidifying a threatening crack

            if crack['state'] == 'solid':
                crack['lifetime'] -= 1
        
        self.cracks = [c for c in self.cracks if c.get('lifetime', 1) > 0]
        return reward

    def _spawn_crack(self):
        side = self.np_random.integers(4)
        if side == 0: start_pos = np.array([0, self.np_random.uniform(0, self.HEIGHT)])
        elif side == 1: start_pos = np.array([self.WIDTH, self.np_random.uniform(0, self.HEIGHT)])
        elif side == 2: start_pos = np.array([self.np_random.uniform(0, self.WIDTH), 0])
        else: start_pos = np.array([self.np_random.uniform(0, self.WIDTH), self.HEIGHT])
        
        target = np.array([self.np_random.uniform(self.WIDTH*0.25, self.WIDTH*0.75), self.np_random.uniform(self.HEIGHT*0.25, self.HEIGHT*0.75)])
        angle = math.atan2(target[1] - start_pos[1], target[0] - start_pos[0])

        self.cracks.append({
            'points': [start_pos],
            'state': 'active', # 'active' or 'solid'
            'max_len': self.np_random.integers(20, 50),
            'growth_speed': self.np_random.uniform(2, 4),
            'angle': angle
        })

    def _update_sculptures(self):
        reward = 0
        for s in self.sculptures:
            if s['health'] <= 0:
                continue
            
            reward += 0.1 # Survival reward
            melt_rate = self.BASE_MELT_RATE

            # Player stabilization
            if np.linalg.norm(self.player_pos - s['pos']) < self.SCULPTURE_STABILIZE_RADIUS:
                melt_rate = 0 # Player fully stabilizes nearby sculptures

            # Crack damage
            for crack in self.cracks:
                if crack['state'] == 'active':
                    for i in range(len(crack['points']) - 1):
                        p1 = crack['points'][i]
                        p2 = crack['points'][i+1]
                        dist_sq = self._dist_point_to_segment_sq(s['pos'], p1, p2)
                        if dist_sq < self.SCULPTURE_RADIUS**2:
                            melt_rate += self.CRACK_MELT_RATE
                            break # Only take damage once per crack
            
            s['health'] -= melt_rate
            if s['health'] <= 0:
                s['health'] = 0
                # Sfx: Sculpture_Shatter_Sound()
                self._create_sculpture_destruction_particles(s['pos'])

        return reward

    def _update_particles(self):
        for p in self.particles:
            p['pos'] += p['vel']
            p['lifetime'] -= 1
            p['radius'] *= 0.98
        self.particles = [p for p in self.particles if p['lifetime'] > 0 and p['radius'] > 0.5]

    # --- Rendering ---
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_cracks()
        self._render_sculptures()
        self._render_player()
        self._render_particles()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        for band in self.aurora_points:
            points = [(p[0], p[1] + math.sin(self.steps * p[2] + p[3]) * p[4]) for p in band['points']]
            pygame.draw.polygon(self.screen, band['color'], points)

    def _render_cracks(self):
        for crack in self.cracks:
            if len(crack['points']) > 1:
                color = self.COLOR_SOLID_CRACK if crack['state'] == 'solid' else self.COLOR_CRACK
                width = 3 if crack['state'] == 'solid' else 1
                points_int = [(int(p[0]), int(p[1])) for p in crack['points']]
                pygame.draw.aalines(self.screen, color, False, points_int)

    def _render_sculptures(self):
        for s in self.sculptures:
            if s['health'] <= 0: continue
            
            # Health-based color interpolation
            health_ratio = s['health'] / s['max_health']
            glow_color = tuple(int(c1 * health_ratio + c2 * (1 - health_ratio)) for c1, c2 in zip(self.COLOR_SCULPTURE_GLOW_HEALTHY, self.COLOR_SCULPTURE_GLOW_MELTING))
            
            self._render_glow(s['pos'], self.SCULPTURE_RADIUS * 2.5, glow_color)
            
            # Main shape (diamond)
            pos = (int(s['pos'][0]), int(s['pos'][1]))
            r = self.SCULPTURE_RADIUS
            points = [(pos[0], pos[1] - r), (pos[0] + r, pos[1]), (pos[0], pos[1] + r), (pos[0] - r, pos[1])]
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_SCULPTURE)
            pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_SCULPTURE)

    def _render_player(self):
        pos = (int(self.player_pos[0]), int(self.player_pos[1]))
        self._render_glow(pos, self.PLAYER_RADIUS * 3, self.COLOR_PLAYER_GLOW)
        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.PLAYER_RADIUS, self.COLOR_PLAYER)
        pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.PLAYER_RADIUS, self.COLOR_PLAYER)

    def _render_particles(self):
        for p in self.particles:
            pos = (int(p['pos'][0]), int(p['pos'][1]))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], int(p['radius']), p['color'])

    def _render_ui(self):
        # Time Slow Bar
        bar_width = 200
        bar_height = 15
        pygame.draw.rect(self.screen, self.COLOR_UI_BAR_BG, (self.WIDTH / 2 - bar_width / 2, 10, bar_width, bar_height))
        fill_width = bar_width * (self.time_slow_resource / self.TIME_SLOW_MAX)
        pygame.draw.rect(self.screen, self.COLOR_TIME_SLOW_BAR, (self.WIDTH / 2 - bar_width / 2, 10, fill_width, bar_height))

        # Sculpture Health
        for s in self.sculptures:
            if s['health'] <= 0: continue
            health_text = f"{int(s['health'])}%"
            text_surf = self.font_small.render(health_text, True, self.COLOR_UI_TEXT)
            text_rect = text_surf.get_rect(center=(s['pos'][0], s['pos'][1] - self.SCULPTURE_RADIUS - 10))
            self.screen.blit(text_surf, text_rect)

        # Score and Steps
        score_text = f"SCORE: {int(self.score)}"
        steps_text = f"STEP: {self.steps}/{self.MAX_STEPS}"
        score_surf = self.font_large.render(score_text, True, self.COLOR_UI_TEXT)
        steps_surf = self.font_small.render(steps_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(score_surf, (10, 10))
        self.screen.blit(steps_surf, (10, 35))

    # --- Utility and Effect Methods ---
    def _init_aurora(self):
        self.aurora_points = []
        for i in range(3): # 3 bands of aurora
            points = []
            color = [
                self.np_random.integers(20, 50),
                self.np_random.integers(80, 150),
                self.np_random.integers(60, 120),
                self.np_random.integers(10, 30) # Alpha
            ]
            y_base = self.np_random.uniform(20, self.HEIGHT / 2)
            for x in range(-50, self.WIDTH + 51, 50):
                points.append((
                    x, 
                    y_base + self.np_random.uniform(-20, 20),
                    self.np_random.uniform(0.01, 0.03), # speed
                    self.np_random.uniform(0, 2 * math.pi), # phase
                    self.np_random.uniform(10, 30) # amplitude
                ))
            self.aurora_points.append({'points': points, 'color': color})

    def _update_aurora(self):
        # The calculation is done in _render_background based on self.steps
        pass

    def _render_glow(self, pos, max_radius, color):
        pos = (int(pos[0]), int(pos[1]))
        for i in range(int(max_radius), 0, -2):
            alpha = int(255 * (1 - i / max_radius)**2)
            glow_color = (*color[:3], max(0, min(255, alpha // 4)))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], i, glow_color)

    def _create_teleport_particles(self, start_pos, end_pos):
        for _ in range(30):
            self.particles.append({
                'pos': start_pos.copy() + self.np_random.uniform(-5, 5, 2),
                'vel': (end_pos - start_pos) / 15 + self.np_random.uniform(-1, 1, 2),
                'radius': self.np_random.uniform(2, 5),
                'lifetime': self.np_random.integers(15, 25),
                'color': self.COLOR_PLAYER_GLOW
            })

    def _create_sculpture_destruction_particles(self, pos):
        for _ in range(100):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 5)
            self.particles.append({
                'pos': pos.copy() + self.np_random.uniform(-5, 5, 2),
                'vel': np.array([math.cos(angle) * speed, math.sin(angle) * speed]),
                'radius': self.np_random.uniform(3, 7),
                'lifetime': self.np_random.integers(40, 80),
                'color': self.COLOR_SCULPTURE
            })

    def _dist_point_to_segment_sq(self, p, v, w):
        l2 = np.sum((v - w)**2)
        if l2 == 0: return np.sum((p - v)**2)
        t = max(0, min(1, np.dot(p - v, w - v) / l2))
        projection = v + t * (w - v)
        return np.sum((p - projection)**2)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "sculpture_healths": [s['health'] for s in self.sculptures],
            "time_slow_resource": self.time_slow_resource
        }

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # --- Manual Play Example ---
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Un-dummy the video driver for manual play
    os.environ["SDL_VIDEODRIVER"] = "x11" 
    pygame.display.init()
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Glacier Guardian")
    clock = pygame.time.Clock()
    
    terminated = False
    total_reward = 0
    
    print("\n--- Controls ---")
    print(GameEnv.user_guide)
    print("R:      Reset")
    print("Q:      Quit")
    
    while not terminated:
        # --- Action Mapping from Keyboard ---
        action = [0, 0, 0] # Default No-op
        
        # This event loop is crucial for responsive controls
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    terminated = True
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    total_reward = 0
                    print("--- Game Reset ---")

        keys = pygame.key.get_pressed()
        movement = 0 # No-op
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # --- Rendering ---
        # The observation is already a rendered frame
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(env.FPS)

    print(f"\nGame Over. Final Score: {total_reward:.2f}")
    env.close()
import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T13:50:35.204404
# Source Brief: brief_01984.md
# Brief Index: 1984
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
        "A top-down stealth game where you must collect all cargo and escape without being seen. "
        "Use portals to navigate the level and physics objects to stun guards."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move. "
        "Press space to place the blue portal and shift to place the orange portal."
    )
    auto_advance = True

    # --- CONSTANTS ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30
    
    # Colors
    COLOR_BG = (10, 20, 30)
    COLOR_WALL = (40, 50, 60)
    COLOR_PLAYER = (0, 255, 128)
    COLOR_PLAYER_GLOW = (0, 255, 128, 50)
    COLOR_GUARD = (255, 50, 50)
    COLOR_GUARD_GLOW = (255, 50, 50, 60)
    COLOR_GUARD_VISION = (255, 50, 50, 30)
    COLOR_GUARD_STUNNED = (100, 100, 200)
    COLOR_CARGO = (255, 220, 0)
    COLOR_CARGO_GLOW = (255, 220, 0, 80)
    COLOR_OBJECT = (150, 150, 150)
    COLOR_PORTAL_A = (0, 150, 255)
    COLOR_PORTAL_B = (255, 150, 0)
    COLOR_EXIT = (100, 255, 100)
    COLOR_TEXT = (220, 220, 220)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
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
        self.font = pygame.font.SysFont('Consolas', 24, bold=True)
        
        # --- Game State Variables (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.player = {}
        self.guards = []
        self.cargos = []
        self.portals = []
        self.objects = []
        self.walls = []
        self.exit = None
        self.particles = []
        
        self.previous_action = np.array([0, 0, 0])
        self.last_dist_to_cargo = float('inf')
        self.last_dist_to_guard = float('inf')
        
        self.collected_cargo_count = 0
        self.total_cargo = 0
        
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # --- Reset Core State ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.collected_cargo_count = 0
        self.previous_action = np.array([0, 0, 0])
        self.particles.clear()
        
        # --- Level Generation ---
        self._setup_level()
        
        # --- Entity Initialization ---
        self.player = {
            'pos': pygame.Vector2(self.WIDTH * 0.1, self.HEIGHT * 0.5),
            'vel': pygame.Vector2(0, 0),
            'radius': 10,
            'teleport_cooldown': 0,
            'speed': 1.5,
            'friction': 0.85
        }
        
        self.portals = [
            {'pos': None, 'radius': 15, 'color': self.COLOR_PORTAL_A, 'cooldown': 0},
            {'pos': None, 'radius': 15, 'color': self.COLOR_PORTAL_B, 'cooldown': 0}
        ]
        
        # --- Initial Reward Calculation State ---
        self.last_dist_to_cargo = self._get_dist_to_closest_uncollected_cargo()
        self.last_dist_to_guard = self._get_dist_to_closest_guard()

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        self.game_over = False

        self._handle_player_input(action)
        
        # --- Update Game Logic ---
        self._update_player()
        self._update_guards()
        self.objects = [o for o in self.objects if self._update_object(o)]
        self._update_particles()

        # --- Handle Interactions and Events ---
        teleport_reward = self._handle_portals()
        collision_rewards = self._handle_collisions()
        reward += teleport_reward + collision_rewards
        
        # --- Continuous Rewards ---
        dist_cargo = self._get_dist_to_closest_uncollected_cargo()
        if dist_cargo < self.last_dist_to_cargo:
            reward += 0.1
        self.last_dist_to_cargo = dist_cargo

        dist_guard = self._get_dist_to_closest_guard()
        if dist_guard < self.last_dist_to_guard:
            reward -= 0.1
        self.last_dist_to_guard = dist_guard
        
        # --- Termination Check ---
        self.steps += 1
        win_condition, detection_condition, timeout_condition = self._check_termination()
        
        if win_condition:
            reward += 50
            self.game_over = True
        if detection_condition:
            reward -= 100
            self.game_over = True
        if timeout_condition:
            self.game_over = True
            
        terminated = self.game_over
        truncated = False # Not using truncation based on time limits
        self.score += reward
        self.previous_action = action
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "cargo_collected": self.collected_cargo_count}

    # --- GAME LOGIC SUB-FUNCTIONS ---

    def _setup_level(self):
        self.walls = [
            pygame.Rect(0, 0, self.WIDTH, 10),
            pygame.Rect(0, self.HEIGHT - 10, self.WIDTH, 10),
            pygame.Rect(0, 0, 10, self.HEIGHT),
            pygame.Rect(self.WIDTH - 10, 0, 10, self.HEIGHT),
            pygame.Rect(self.WIDTH * 0.3, 0, 20, self.HEIGHT * 0.4),
            pygame.Rect(self.WIDTH * 0.3, self.HEIGHT * 0.6, 20, self.HEIGHT * 0.4),
            pygame.Rect(self.WIDTH * 0.65, 0, 20, self.HEIGHT * 0.7),
        ]
        
        self.cargos = [
            {'pos': pygame.Vector2(self.WIDTH * 0.5, self.HEIGHT * 0.2), 'radius': 8, 'collected': False},
            {'pos': pygame.Vector2(self.WIDTH * 0.85, self.HEIGHT * 0.85), 'radius': 8, 'collected': False},
        ]
        self.total_cargo = len(self.cargos)
        
        self.objects = [
            {'pos': pygame.Vector2(self.WIDTH*0.2, self.HEIGHT*0.2), 'vel': pygame.Vector2(0,0), 'radius': 12, 'friction': 0.96, 'mass': 2},
            {'pos': pygame.Vector2(self.WIDTH*0.5, self.HEIGHT*0.8), 'vel': pygame.Vector2(0,0), 'radius': 12, 'friction': 0.96, 'mass': 2},
        ]

        self.guards = [
            {
                'pos': pygame.Vector2(self.WIDTH * 0.4, self.HEIGHT * 0.5), 'radius': 9, 'speed': 0.8,
                'path': [pygame.Vector2(self.WIDTH*0.4, self.HEIGHT*0.2), pygame.Vector2(self.WIDTH*0.4, self.HEIGHT*0.8)],
                'path_index': 0, 'detected_player': False, 'stun_timer': 0,
                'vision_radius': 60, 'vision_angle': 45, 'direction': pygame.Vector2(0, 1)
            },
            {
                'pos': pygame.Vector2(self.WIDTH * 0.8, self.HEIGHT * 0.2), 'radius': 9, 'speed': 1.0,
                'path': [pygame.Vector2(self.WIDTH*0.8, self.HEIGHT*0.2), pygame.Vector2(self.WIDTH*0.8, self.HEIGHT*0.5)],
                'path_index': 0, 'detected_player': False, 'stun_timer': 0,
                'vision_radius': 80, 'vision_angle': 45, 'direction': pygame.Vector2(0, 1)
            }
        ]
        
        self.exit = pygame.Rect(self.WIDTH - 30, self.HEIGHT // 2 - 25, 20, 50)

    def _handle_player_input(self, action):
        movement, space_action, shift_action = action[0], action[1], action[2]
        
        accel = pygame.Vector2(0, 0)
        if movement == 1: accel.y -= self.player['speed'] # Up
        if movement == 2: accel.y += self.player['speed'] # Down
        if movement == 3: accel.x -= self.player['speed'] # Left
        if movement == 4: accel.x += self.player['speed'] # Right
        
        # Normalize diagonal movement
        if accel.length() > 0:
            accel.scale_to_length(self.player['speed'])
        self.player['vel'] += accel

        # Portal placement on button PRESS (0 -> 1 transition)
        if space_action == 1 and self.previous_action[1] == 0:
            self.portals[0]['pos'] = pygame.Vector2(self.player['pos'])
            self._spawn_particles(self.player['pos'], 20, self.portals[0]['color'], 2, 20)
            # sfx: portal_A_open.wav
        
        if shift_action == 1 and self.previous_action[2] == 0:
            self.portals[1]['pos'] = pygame.Vector2(self.player['pos'])
            self._spawn_particles(self.player['pos'], 20, self.portals[1]['color'], 2, 20)
            # sfx: portal_B_open.wav

    def _update_player(self):
        p = self.player
        p['vel'] *= p['friction']
        if p['vel'].length() < 0.1: p['vel'] = pygame.Vector2(0, 0)
        p['pos'] += p['vel']
        
        if p['teleport_cooldown'] > 0: p['teleport_cooldown'] -= 1

        # Wall collisions
        p_rect = pygame.Rect(p['pos'].x - p['radius'], p['pos'].y - p['radius'], p['radius']*2, p['radius']*2)
        for wall in self.walls:
            if p_rect.colliderect(wall):
                # Simple position correction and velocity bounce
                if abs(p_rect.centerx - wall.centerx) > abs(p_rect.centery - wall.centery):
                    p['vel'].x *= -0.5
                    p['pos'].x += p['vel'].x
                else:
                    p['vel'].y *= -0.5
                    p['pos'].y += p['vel'].y
        
        p['pos'].x = np.clip(p['pos'].x, p['radius'], self.WIDTH - p['radius'])
        p['pos'].y = np.clip(p['pos'].y, p['radius'], self.HEIGHT - p['radius'])

    def _update_guards(self):
        # Difficulty scaling
        detection_radius_increase = (self.steps // 500) * 0.5
        
        for g in self.guards:
            g['vision_radius'] = 60 + detection_radius_increase
            if g['stun_timer'] > 0:
                g['stun_timer'] -= 1
                continue

            target = g['path'][g['path_index']]
            direction_to_target = target - g['pos']
            
            if direction_to_target.length() < g['speed']:
                g['path_index'] = (g['path_index'] + 1) % len(g['path'])
            else:
                g['direction'] = direction_to_target.normalize()
                g['pos'] += g['direction'] * g['speed']

            # Check for player detection
            dist_to_player = g['pos'].distance_to(self.player['pos'])
            if dist_to_player < g['vision_radius']:
                vec_to_player = (self.player['pos'] - g['pos']).normalize()
                angle_to_player = g['direction'].angle_to(vec_to_player)
                
                if abs(angle_to_player) < g['vision_angle'] / 2:
                    # Line of sight check
                    has_los = True
                    for wall in self.walls:
                        if wall.clipline(g['pos'], self.player['pos']):
                            has_los = False
                            break
                    if has_los:
                        g['detected_player'] = True
                        # sfx: alert.wav

    def _update_object(self, obj):
        obj['vel'] *= obj['friction']
        if obj['vel'].length() < 0.1: obj['vel'] = pygame.Vector2(0, 0)
        obj['pos'] += obj['vel']
        
        if obj.get('teleport_cooldown', 0) > 0: obj['teleport_cooldown'] -= 1

        # Wall collisions
        obj_rect = pygame.Rect(obj['pos'].x - obj['radius'], obj['pos'].y - obj['radius'], obj['radius']*2, obj['radius']*2)
        for wall in self.walls:
            if obj_rect.colliderect(wall):
                if abs(obj_rect.centerx - wall.centerx) > abs(obj_rect.centery - wall.centery):
                    obj['vel'].x *= -0.5
                else:
                    obj['vel'].y *= -0.5
                # Push out of wall
                while obj_rect.colliderect(wall):
                    obj['pos'] += obj['vel'].normalize() * 0.1
                    obj_rect.center = obj['pos']

        obj['pos'].x = np.clip(obj['pos'].x, obj['radius'], self.WIDTH - obj['radius'])
        obj['pos'].y = np.clip(obj['pos'].y, obj['radius'], self.HEIGHT - obj['radius'])
        return True

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1
            p['radius'] = max(0, p['start_radius'] * (p['life'] / p['start_life']))

    def _handle_portals(self):
        p_a, p_b = self.portals[0], self.portals[1]
        if p_a['pos'] is None or p_b['pos'] is None:
            return 0
        
        entities = [self.player] + self.objects
        for entity in entities:
            if entity.get('teleport_cooldown', 0) > 0:
                continue
            
            # Check portal A
            if entity['pos'].distance_to(p_a['pos']) < p_a['radius']:
                entity['pos'] = pygame.Vector2(p_b['pos'])
                entity['teleport_cooldown'] = 30 # 1 second cooldown
                self._spawn_particles(p_b['pos'], 30, p_b['color'], 3, 25)
                # sfx: teleport.wav
                break
            
            # Check portal B
            if entity['pos'].distance_to(p_b['pos']) < p_b['radius']:
                entity['pos'] = pygame.Vector2(p_a['pos'])
                entity['teleport_cooldown'] = 30
                self._spawn_particles(p_a['pos'], 30, p_a['color'], 3, 25)
                # sfx: teleport.wav
                break
        return 0

    def _handle_collisions(self):
        reward = 0
        # Player-Cargo
        for cargo in self.cargos:
            if not cargo['collected']:
                if self.player['pos'].distance_to(cargo['pos']) < self.player['radius'] + cargo['radius']:
                    cargo['collected'] = True
                    self.collected_cargo_count += 1
                    reward += 10
                    self._spawn_particles(cargo['pos'], 50, self.COLOR_CARGO, 4, 30)
                    # sfx: cargo_collect.wav

        # Object-Guard
        for obj in self.objects:
            if obj['vel'].length() > 2.0: # Must be moving fast
                for guard in self.guards:
                    if guard['stun_timer'] == 0:
                        if obj['pos'].distance_to(guard['pos']) < obj['radius'] + guard['radius']:
                            guard['stun_timer'] = 150 # 5 seconds
                            obj['vel'] *= -0.5 # Bounce off
                            # sfx: guard_stun.wav
        
        # Object-Object
        for i in range(len(self.objects)):
            for j in range(i + 1, len(self.objects)):
                o1, o2 = self.objects[i], self.objects[j]
                dist_vec = o1['pos'] - o2['pos']
                if 0 < dist_vec.length() < o1['radius'] + o2['radius']:
                    # Resolve overlap
                    overlap = (o1['radius'] + o2['radius']) - dist_vec.length()
                    o1['pos'] += dist_vec.normalize() * overlap / 2
                    o2['pos'] -= dist_vec.normalize() * overlap / 2
                    # Elastic collision
                    v1, v2 = o1['vel'], o2['vel']
                    p1, p2 = o1['pos'], o2['pos']
                    m1, m2 = o1['mass'], o2['mass']
                    
                    dv = v1 - v2
                    dx = p1 - p2
                    
                    o1['vel'] = v1 - (2 * m2 / (m1 + m2)) * (dv.dot(dx) / dx.length_squared()) * dx
                    o2['vel'] = v2 - (2 * m1 / (m1 + m2)) * (-(dv).dot(-dx) / (-dx).length_squared()) * (-dx)

        return reward

    def _check_termination(self):
        win_condition = False
        if self.collected_cargo_count == self.total_cargo:
            player_rect = pygame.Rect(self.player['pos'].x - self.player['radius'], self.player['pos'].y - self.player['radius'], self.player['radius']*2, self.player['radius']*2)
            if player_rect.colliderect(self.exit):
                win_condition = True
        
        detection_condition = any(g['detected_player'] for g in self.guards)
        timeout_condition = self.steps >= 2000
        
        return win_condition, detection_condition, timeout_condition

    # --- HELPER & UTILITY FUNCTIONS ---

    def _get_dist_to_closest_uncollected_cargo(self):
        uncollected = [c['pos'] for c in self.cargos if not c['collected']]
        if not uncollected: return 0
        return min(self.player['pos'].distance_to(pos) for pos in uncollected)

    def _get_dist_to_closest_guard(self):
        if not self.guards: return float('inf')
        return min(self.player['pos'].distance_to(g['pos']) for g in self.guards)

    def _spawn_particles(self, pos, count, color, speed, lifespan):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * random.uniform(0.5, speed)
            self.particles.append({
                'pos': pygame.Vector2(pos),
                'vel': vel,
                'color': color,
                'life': random.randint(lifespan // 2, lifespan),
                'start_life': lifespan,
                'radius': random.uniform(2, 5),
                'start_radius': 5
            })

    # --- RENDERING ---

    def _render_game(self):
        # Walls
        for wall in self.walls:
            pygame.draw.rect(self.screen, self.COLOR_WALL, wall)
        
        # Exit
        if self.collected_cargo_count == self.total_cargo:
            self._draw_glowing_rect(self.screen, self.COLOR_EXIT, self.exit, 15)

        # Portals
        for portal in self.portals:
            if portal['pos']:
                self._draw_glowing_ellipse(self.screen, portal['color'], portal['pos'], portal['radius'] * 1.5, portal['radius'] * 0.75, 20)

        # Cargos
        for cargo in self.cargos:
            if not cargo['collected']:
                self._draw_glowing_circle(self.screen, self.COLOR_CARGO, cargo['pos'], cargo['radius'], 15, self.COLOR_CARGO_GLOW)
        
        # Physics Objects
        for obj in self.objects:
            pygame.gfxdraw.filled_circle(self.screen, int(obj['pos'].x), int(obj['pos'].y), int(obj['radius']), self.COLOR_OBJECT)
            pygame.gfxdraw.aacircle(self.screen, int(obj['pos'].x), int(obj['pos'].y), int(obj['radius']), self.COLOR_OBJECT)
            
        # Guards
        for guard in self.guards:
            self._draw_guard(guard)

        # Player
        self._draw_player()

        # Particles
        for p in self.particles:
            alpha = int(255 * (p['life'] / p['start_life']))
            color = (*p['color'], alpha)
            pygame.gfxdraw.filled_circle(self.screen, int(p['pos'].x), int(p['pos'].y), int(p['radius']), color)

    def _render_ui(self):
        score_text = self.font.render(f"SCORE: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (15, 15))
        
        steps_text = self.font.render(f"STEPS: {self.steps}", True, self.COLOR_TEXT)
        self.screen.blit(steps_text, (self.WIDTH - steps_text.get_width() - 15, 15))
        
        cargo_text = self.font.render(f"CARGO: {self.collected_cargo_count}/{self.total_cargo}", True, self.COLOR_TEXT)
        self.screen.blit(cargo_text, (self.WIDTH // 2 - cargo_text.get_width() // 2, 15))

    def _draw_player(self):
        p = self.player
        self._draw_glowing_circle(self.screen, self.COLOR_PLAYER, p['pos'], p['radius'] + 5, 20, self.COLOR_PLAYER_GLOW)
        
        # Triangle ship
        angle = p['vel'].angle_to(pygame.Vector2(1, 0)) if p['vel'].length() > 0 else 0
        points = []
        for i in range(3):
            a = math.radians(angle + i * 120 + 90)
            point = p['pos'] + pygame.Vector2(math.cos(a), -math.sin(a)) * p['radius'] * (1.5 if i == 0 else 1)
            points.append((int(point.x), int(point.y)))
        
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PLAYER)
        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_PLAYER)
        
    def _draw_guard(self, guard):
        # Vision cone
        if guard['stun_timer'] == 0:
            p1 = guard['pos']
            dir_vec = guard['direction']
            angle_rad = math.radians(guard['vision_angle'] / 2)
            
            p2_dir = dir_vec.rotate_rad(angle_rad)
            p3_dir = dir_vec.rotate_rad(-angle_rad)
            
            p2 = p1 + p2_dir * guard['vision_radius']
            p3 = p1 + p3_dir * guard['vision_radius']
            
            pygame.gfxdraw.filled_trigon(self.screen, int(p1.x), int(p1.y), int(p2.x), int(p2.y), int(p3.x), int(p3.y), self.COLOR_GUARD_VISION)
        
        # Body
        color = self.COLOR_GUARD_STUNNED if guard['stun_timer'] > 0 else self.COLOR_GUARD
        glow_color = (*color, 60)
        self._draw_glowing_circle(self.screen, color, guard['pos'], guard['radius'] + 3, 15, glow_color)
        
        angle = guard['direction'].angle_to(pygame.Vector2(1, 0))
        points = []
        for i in range(3):
            a = math.radians(angle + i * 120 + 90)
            point = guard['pos'] + pygame.Vector2(math.cos(a), -math.sin(a)) * guard['radius'] * (1.3 if i == 0 else 0.8)
            points.append((int(point.x), int(point.y)))
            
        pygame.gfxdraw.filled_polygon(self.screen, points, color)
        pygame.gfxdraw.aapolygon(self.screen, points, color)

    def _draw_glowing_circle(self, surface, color, pos, radius, glow_size, glow_color):
        for i in range(glow_size, 0, -1):
            alpha = glow_color[3] * (1 - i / glow_size)
            current_color = (*glow_color[:3], int(alpha))
            pygame.gfxdraw.aacircle(surface, int(pos.x), int(pos.y), int(radius + i), current_color)
        pygame.gfxdraw.filled_circle(surface, int(pos.x), int(pos.y), int(radius), color)
        pygame.gfxdraw.aacircle(surface, int(pos.x), int(pos.y), int(radius), color)

    def _draw_glowing_ellipse(self, surface, color, pos, rx, ry, glow_size):
        temp_surf = pygame.Surface((rx*2 + glow_size*2, ry*2 + glow_size*2), pygame.SRCALPHA)
        center_x, center_y = rx + glow_size, ry + glow_size
        
        for i in range(glow_size, 0, -2):
            alpha = int(80 * (1 - i / glow_size))
            pygame.gfxdraw.aaellipse(temp_surf, int(center_x), int(center_y), int(rx + i), int(ry + i), (*color, alpha))
        
        pygame.gfxdraw.filled_ellipse(temp_surf, int(center_x), int(center_y), int(rx), int(ry), color)
        pygame.gfxdraw.aaellipse(temp_surf, int(center_x), int(center_y), int(rx), int(ry), color)
        
        # Black inner ellipse for portal effect
        pygame.gfxdraw.filled_ellipse(temp_surf, int(center_x), int(center_y), int(rx * 0.8), int(ry * 0.8), (0,0,0))
        
        surface.blit(temp_surf, (int(pos.x - center_x), int(pos.y - center_y)))

    def _draw_glowing_rect(self, surface, color, rect, glow_size):
        temp_surf = pygame.Surface((rect.width + glow_size*2, rect.height + glow_size*2), pygame.SRCALPHA)
        glow_rect = pygame.Rect(glow_size, glow_size, rect.width, rect.height)

        for i in range(glow_size, 0, -2):
            alpha = int(100 * (1 - i / glow_size))
            pygame.draw.rect(temp_surf, (*color, alpha), glow_rect.inflate(i, i), border_radius=5)
        
        pygame.draw.rect(temp_surf, color, glow_rect, border_radius=5)
        surface.blit(temp_surf, (rect.x - glow_size, rect.y - glow_size))

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # It will not run in a headless environment
    os.environ["SDL_VIDEODRIVER"] = "x11"
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Portal Pirate")
    clock = pygame.time.Clock()
    
    terminated = False
    total_reward = 0
    
    while not terminated:
        # --- Manual Control ---
        keys = pygame.key.get_pressed()
        movement = 0 # None
        if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = np.array([movement, space_held, shift_held])
        
        # --- Gym Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # --- Pygame Rendering ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0

        clock.tick(GameEnv.FPS)

    print(f"Game Over! Final Score: {total_reward:.2f}, Steps: {info['steps']}")
    env.close()
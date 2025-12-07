import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T12:26:32.713999
# Source Brief: brief_01096.md
# Brief Index: 1096
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment where the player navigates a spaceship through a
    dynamic wormhole network. The goal is to jump between colored portals to
    score points while avoiding cosmic anomalies and a shrinking safe zone.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Navigate a spaceship through a dynamic wormhole network, jumping between portals to score points "
        "while avoiding cosmic anomalies and a shrinking safe zone."
    )
    user_guide = (
        "Controls: Use arrow keys to move. Press space to jump through a nearby wormhole."
    )
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30
    MAX_STEPS = 2000
    TARGET_SCORE = 100

    # Colors
    COLOR_BG = (15, 10, 25)
    COLOR_STAR = (200, 200, 255)
    COLOR_PLAYER = (0, 255, 255)
    COLOR_PLAYER_GLOW = (0, 150, 150, 60)
    COLOR_ANOMALY = (255, 0, 128)
    COLOR_ANOMALY_PULSE = (200, 0, 100, 100)
    COLOR_WORMHOLE_RED = (255, 50, 50)
    COLOR_WORMHOLE_GREEN = (50, 255, 50)
    COLOR_WORMHOLE_BLUE = (50, 50, 255)
    WORMHOLE_COLORS = [COLOR_WORMHOLE_RED, COLOR_WORMHOLE_GREEN, COLOR_WORMHOLE_BLUE]
    COLOR_UI_TEXT = (255, 255, 255)
    COLOR_BOUNDARY = (100, 80, 180, 50)

    # Game Parameters
    PLAYER_SPEED = 8
    PLAYER_RADIUS = 10
    ANOMALY_RADIUS = 12
    ANOMALY_BASE_SPEED = 1.0
    NUM_ANOMALIES = 5
    NUM_WORMHOLE_PAIRS = 3
    WORMHOLE_RADIUS = 20
    WORMHOLE_ACTIVATION_RANGE = 35
    GLOBAL_RADIUS_MIN = 100
    GLOBAL_RADIUS_MAX = 200
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("sans-serif", 24, bold=True)
        
        self.player_pos = None
        self.anomalies = []
        self.wormholes = []
        self.particles = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.last_space_held = False
        self.anomaly_speed = self.ANOMALY_BASE_SPEED
        self.global_radius = 0
        self.starfield = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.last_space_held = False
        self.anomaly_speed = self.ANOMALY_BASE_SPEED
        
        self.player_pos = np.array([self.WIDTH / 2, self.HEIGHT / 2], dtype=np.float32)
        
        self._spawn_starfield()
        self._spawn_wormholes()
        self._spawn_anomalies()
        
        self.particles = []
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        space_held = action[1] == 1
        
        self._update_player(movement)
        self._update_anomalies()
        self._update_particles()
        self._update_game_state()
        
        reward = 0.1 # Survival reward
        
        # Handle wormhole activation on button press (not hold)
        if space_held and not self.last_space_held:
            jump_reward = self._handle_wormhole_jump()
            reward += jump_reward
        self.last_space_held = space_held

        terminated, terminal_reward = self._check_termination()
        reward += terminal_reward
        
        self.game_over = terminated
        
        truncated = self.steps >= self.MAX_STEPS
        if truncated:
            terminated = True
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _update_player(self, movement):
        if movement == 1: # Up
            self.player_pos[1] -= self.PLAYER_SPEED
        elif movement == 2: # Down
            self.player_pos[1] += self.PLAYER_SPEED
        elif movement == 3: # Left
            self.player_pos[0] -= self.PLAYER_SPEED
        elif movement == 4: # Right
            self.player_pos[0] += self.PLAYER_SPEED
        
        # World wrapping
        self.player_pos[0] %= self.WIDTH
        self.player_pos[1] %= self.HEIGHT
        
        assert 0 <= self.player_pos[0] < self.WIDTH and 0 <= self.player_pos[1] < self.HEIGHT

    def _update_anomalies(self):
        for anomaly in self.anomalies:
            anomaly['pos'] += anomaly['vel'] * self.anomaly_speed
            anomaly['pos'][0] %= self.WIDTH
            anomaly['pos'][1] %= self.HEIGHT

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1

    def _update_game_state(self):
        self.steps += 1
        
        # Oscillating global boundary
        oscillation = math.sin(self.steps * 0.02)
        radius_range = self.GLOBAL_RADIUS_MAX - self.GLOBAL_RADIUS_MIN
        self.global_radius = self.GLOBAL_RADIUS_MIN + (radius_range / 2) * (1 + oscillation)
        assert self.GLOBAL_RADIUS_MIN <= self.global_radius <= self.GLOBAL_RADIUS_MAX

        # Increase difficulty
        if self.steps > 0 and self.steps % 500 == 0:
            self.anomaly_speed += 0.05

    def _handle_wormhole_jump(self):
        # Find nearest wormhole
        min_dist = float('inf')
        nearest_wormhole = None
        for wh in self.wormholes:
            dist = np.linalg.norm(self.player_pos - wh['pos'])
            if dist < min_dist:
                min_dist = dist
                nearest_wormhole = wh
        
        if nearest_wormhole and min_dist < self.WORMHOLE_ACTIVATION_RANGE:
            # Find linked wormhole
            linked_wormhole = next(w for w in self.wormholes if w['id'] == nearest_wormhole['linked_id'])
            
            # Teleport player
            self.player_pos = linked_wormhole['pos'].copy() + np.array([0, self.PLAYER_RADIUS + self.WORMHOLE_RADIUS + 2])
            
            self._spawn_particles(self.player_pos, linked_wormhole['color'], 50)
            
            self.score += 1
            assert self.score == self._get_info()['score']
            return 1.0 # Reward for successful jump
        return 0.0

    def _check_termination(self):
        # Anomaly collision
        for anomaly in self.anomalies:
            dist = np.linalg.norm(self.player_pos - anomaly['pos'])
            if dist < self.PLAYER_RADIUS + self.ANOMALY_RADIUS:
                return True, -100.0

        # Outside global boundary
        center = np.array([self.WIDTH / 2, self.HEIGHT / 2])
        dist_from_center = np.linalg.norm(self.player_pos - center)
        if dist_from_center > self.global_radius:
            return True, -100.0
            
        # Win condition
        if self.score >= self.TARGET_SCORE:
            return True, 100.0

        # Max steps is handled as truncation, not termination with reward
        if self.steps >= self.MAX_STEPS:
            return True, 0.0

        return False, 0.0

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_global_boundary()
        self._render_wormholes()
        self._render_anomalies()
        self._render_particles()
        self._render_player()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}
    
    # --- Spawning Methods ---
    
    def _spawn_starfield(self):
        self.starfield = []
        for _ in range(200):
            self.starfield.append({
                'pos': (random.randint(0, self.WIDTH), random.randint(0, self.HEIGHT)),
                'size': random.randint(1, 2),
                'depth': random.uniform(0.1, 0.5) # For parallax effect
            })

    def _spawn_wormholes(self):
        self.wormholes = []
        occupied_areas = []
        for i in range(self.NUM_WORMHOLE_PAIRS):
            color = self.WORMHOLE_COLORS[i % len(self.WORMHOLE_COLORS)]
            
            # Spawn first wormhole of the pair
            wh1_pos = self._get_safe_spawn_pos(occupied_areas, 50)
            occupied_areas.append((wh1_pos, 50))
            
            # Spawn second wormhole of the pair
            wh2_pos = self._get_safe_spawn_pos(occupied_areas, 50)
            occupied_areas.append((wh2_pos, 50))

            self.wormholes.append({'id': 2*i, 'linked_id': 2*i + 1, 'pos': wh1_pos, 'color': color})
            self.wormholes.append({'id': 2*i + 1, 'linked_id': 2*i, 'pos': wh2_pos, 'color': color})

    def _spawn_anomalies(self):
        self.anomalies = []
        occupied_areas = [(wh['pos'], 50) for wh in self.wormholes]
        occupied_areas.append((self.player_pos, 100)) # Don't spawn on player start
        for _ in range(self.NUM_ANOMALIES):
            pos = self._get_safe_spawn_pos(occupied_areas, 30)
            occupied_areas.append((pos, 30))
            angle = random.uniform(0, 2 * math.pi)
            vel = np.array([math.cos(angle), math.sin(angle)], dtype=np.float32)
            self.anomalies.append({'pos': pos, 'vel': vel})

    def _get_safe_spawn_pos(self, occupied_areas, clearance):
        while True:
            pos = np.array([random.uniform(50, self.WIDTH - 50), random.uniform(50, self.HEIGHT - 50)], dtype=np.float32)
            is_safe = True
            for area_pos, area_radius in occupied_areas:
                if np.linalg.norm(pos - area_pos) < area_radius + clearance:
                    is_safe = False
                    break
            if is_safe:
                return pos

    def _spawn_particles(self, pos, color, count):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            vel = np.array([math.cos(angle) * speed, math.sin(angle) * speed])
            self.particles.append({
                'pos': pos.copy(),
                'vel': vel,
                'life': random.randint(15, 30),
                'color': color
            })

    # --- Rendering Methods ---

    def _render_background(self):
        player_offset_x = (self.player_pos[0] - self.WIDTH/2) / self.WIDTH
        player_offset_y = (self.player_pos[1] - self.HEIGHT/2) / self.HEIGHT
        
        for star in self.starfield:
            x = (star['pos'][0] - player_offset_x * 20 * star['depth']) % self.WIDTH
            y = (star['pos'][1] - player_offset_y * 20 * star['depth']) % self.HEIGHT
            pygame.draw.circle(self.screen, self.COLOR_STAR, (int(x), int(y)), star['size'])

    def _render_global_boundary(self):
        pygame.gfxdraw.filled_circle(
            self.screen, self.WIDTH // 2, self.HEIGHT // 2,
            int(self.global_radius), self.COLOR_BOUNDARY
        )
        pygame.gfxdraw.aacircle(
            self.screen, self.WIDTH // 2, self.HEIGHT // 2,
            int(self.global_radius), self.COLOR_BOUNDARY
        )

    def _render_wormholes(self):
        for wh in self.wormholes:
            pos = (int(wh['pos'][0]), int(wh['pos'][1]))
            color = wh['color']
            for i in range(4):
                alpha = 150 - i * 30
                radius_offset = math.sin(self.steps * 0.05 + i * math.pi/2) * 5
                radius = int(self.WORMHOLE_RADIUS + radius_offset)
                
                c = (*color, alpha) # Add alpha to color tuple
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius + i * 4, c)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.WORMHOLE_RADIUS, (10,5,15))
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.WORMHOLE_RADIUS, color)

    def _render_anomalies(self):
        pulse = (1 + math.sin(self.steps * 0.1)) / 2 * 5
        for anomaly in self.anomalies:
            pos = (int(anomaly['pos'][0]), int(anomaly['pos'][1]))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], int(self.ANOMALY_RADIUS + pulse), self.COLOR_ANOMALY_PULSE)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.ANOMALY_RADIUS, self.COLOR_ANOMALY)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.ANOMALY_RADIUS, self.COLOR_ANOMALY)
            
    def _render_player(self):
        pos = (int(self.player_pos[0]), int(self.player_pos[1]))
        
        # Glow effect
        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.PLAYER_RADIUS + 5, self.COLOR_PLAYER_GLOW)
        
        # Player ship (triangle)
        angle = math.atan2(self.player_pos[1] - self.HEIGHT/2, self.player_pos[0] - self.WIDTH/2) # Point towards center for fun
        p1 = (
            pos[0] + self.PLAYER_RADIUS * math.cos(angle),
            pos[1] + self.PLAYER_RADIUS * math.sin(angle)
        )
        p2 = (
            pos[0] + self.PLAYER_RADIUS * 0.7 * math.cos(angle + 2.2),
            pos[1] + self.PLAYER_RADIUS * 0.7 * math.sin(angle + 2.2)
        )
        p3 = (
            pos[0] + self.PLAYER_RADIUS * 0.7 * math.cos(angle - 2.2),
            pos[1] + self.PLAYER_RADIUS * 0.7 * math.sin(angle - 2.2)
        )
        pygame.gfxdraw.aapolygon(self.screen, [p1, p2, p3], self.COLOR_PLAYER)
        pygame.gfxdraw.filled_polygon(self.screen, [p1, p2, p3], self.COLOR_PLAYER)

    def _render_particles(self):
        for p in self.particles:
            alpha = int(255 * (p['life'] / 30.0))
            color = (*p['color'], alpha)
            pos = (int(p['pos'][0]), int(p['pos'][1]))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 2, color)

    def _render_ui(self):
        score_text = self.font.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # It will not run in a headless environment.
    os.environ.pop("SDL_VIDEODRIVER", None)
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Wormhole Navigator")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    while running:
        movement = 0 # No-op
        space_held = 0
        shift_held = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
            movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]:
            movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            movement = 4
        
        if keys[pygame.K_SPACE]:
            space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift_held = 1
            
        action = [movement, space_held, shift_held]
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Render the observation from the environment to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}, Steps: {info['steps']}")
            total_reward = 0
            obs, info = env.reset()
            # Add a small delay before restarting
            pygame.time.wait(2000)

        clock.tick(GameEnv.FPS)
        
    env.close()
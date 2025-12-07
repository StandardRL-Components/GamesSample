
# Generated: 2025-08-27T21:36:28.227081
# Source Brief: brief_02842.md
# Brief Index: 2842

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    user_guide = (
        "Controls: ←→ to steer. ↑ to slow descent, ↓ to dive faster. "
        "Collect rings to score points."
    )

    game_description = (
        "Steer a skydiver through a procedurally generated sky, collecting rings "
        "while managing descent speed to reach a target score before hitting the ground."
    )

    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and world dimensions
        self.WIDTH, self.HEIGHT = 640, 400
        self.WORLD_HEIGHT = 20000

        # Gymnasium spaces
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        
        # Fonts
        try:
            self.font_small = pygame.font.SysFont("Consolas", 18)
            self.font_large = pygame.font.SysFont("Consolas", 24)
        except pygame.error:
            self.font_small = pygame.font.SysFont(None, 22)
            self.font_large = pygame.font.SysFont(None, 30)

        # Colors
        self.COLOR_SKY_TOP = (117, 166, 221)
        self.COLOR_SKY_BOTTOM = (161, 195, 232)
        self.COLOR_PLAYER = (227, 48, 48)
        self.COLOR_PLAYER_OUTLINE = (255, 255, 255)
        self.COLOR_RING = (255, 215, 0)
        self.COLOR_RING_OUTLINE = (255, 255, 255)
        self.COLOR_CLOUD = (240, 240, 240)
        self.COLOR_MOUNTAIN = (60, 70, 80)
        self.COLOR_GROUND = (76, 153, 0)
        self.COLOR_TEXT = (255, 255, 255)
        self.COLOR_SHADOW = (0, 0, 0, 128)

        # Physics constants
        self.PLAYER_X_ACCEL = 1.0
        self.PLAYER_X_DRAG = 0.85
        self.PLAYER_MAX_VX = 8.0
        self.PLAYER_Y_ACCEL = 0.4
        self.GRAVITY = 0.08
        self.MIN_SPEED = 5.0
        self.MAX_SPEED = 20.0
        self.SPEED_FAST_THRESHOLD = 15.0
        self.SPEED_SLOW_THRESHOLD = 7.0

        # Game rules
        self.MAX_STEPS = 5000
        self.RINGS_TO_WIN = 50
        self.RING_RADIUS = 20
        self.PLAYER_HITBOX_RADIUS = 10

        # State variables are initialized in reset()
        self.player_x = 0
        self.player_vx = 0
        self.player_vy = 0
        self.world_y = 0
        self.rings = []
        self.clouds = []
        self.mountains = []
        self.particles = []
        self.score = 0
        self.rings_collected = 0
        self.steps = 0
        self.game_over = False
        self.ring_spawn_prob = 0.0

        # Run validation
        # self.validate_implementation() # Commented out for submission as it prints
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.player_x = self.WIDTH / 2
        self.player_vx = 0
        self.player_vy = self.MIN_SPEED
        self.world_y = 0.0
        
        self.rings = []
        self.clouds = []
        self.mountains = []
        self.particles = []
        
        self.score = 0
        self.rings_collected = 0
        self.steps = 0
        self.game_over = False
        self.ring_spawn_prob = 0.05
        
        # Procedurally generate initial world
        for _ in range(20): self._spawn_object(self.clouds, self.HEIGHT, initial=True)
        for _ in range(5): self._spawn_object(self.mountains, self.HEIGHT, initial=True, is_mountain=True)
        for _ in range(30): self._spawn_object(self.rings, self.HEIGHT, initial=True)

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack action
        movement, _, _ = action
        
        dist_before = self._get_dist_to_nearest_ring()

        self._handle_input(movement)
        self._update_physics()
        self._update_world_positions()
        
        reward = self._handle_collisions_and_spawning()
        
        dist_after = self._get_dist_to_nearest_ring()
        
        # Continuous rewards
        if dist_before is not None and dist_after is not None:
            reward += (dist_before - dist_after) * 0.01 # Reward for getting closer
        
        if self.player_vy > self.SPEED_FAST_THRESHOLD:
            reward -= 0.1 # Small penalty for diving too fast
        elif self.player_vy < self.SPEED_SLOW_THRESHOLD:
            reward += 0.05 # Small reward for controlled descent

        self.score += reward
        self.steps += 1
        
        terminated = self._check_termination()
        if terminated:
            self.game_over = True
            if self.rings_collected >= self.RINGS_TO_WIN:
                reward += 100
                self.score += 100
            else: # Hit ground
                reward -= 100
                self.score -= 100
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, movement):
        # Horizontal movement
        if movement == 3: # Left
            self.player_vx -= self.PLAYER_X_ACCEL
        elif movement == 4: # Right
            self.player_vx += self.PLAYER_X_ACCEL
        
        # Vertical speed control
        if movement == 1: # Up
            self.player_vy -= self.PLAYER_Y_ACCEL
        elif movement == 2: # Down
            self.player_vy += self.PLAYER_Y_ACCEL

    def _update_physics(self):
        # Apply drag and clamp horizontal velocity
        self.player_vx *= self.PLAYER_X_DRAG
        self.player_vx = np.clip(self.player_vx, -self.PLAYER_MAX_VX, self.PLAYER_MAX_VX)
        
        # Update horizontal position
        self.player_x += self.player_vx
        self.player_x = np.clip(self.player_x, 0, self.WIDTH)
        
        # Apply gravity and clamp vertical speed
        self.player_vy += self.GRAVITY
        self.player_vy = np.clip(self.player_vy, self.MIN_SPEED, self.MAX_SPEED)
        
        # Update world scroll position
        self.world_y += self.player_vy

    def _update_world_positions(self):
        # Update positions of all objects based on player's vertical speed
        for obj_list in [self.rings, self.clouds, self.mountains]:
            for obj in obj_list:
                parallax = obj.get("parallax", 1.0)
                obj['y'] -= self.player_vy * parallax
        
        for p in self.particles:
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['life'] -= 1

        # Remove dead particles and off-screen objects
        self.particles = [p for p in self.particles if p['life'] > 0]
        self.rings = [r for r in self.rings if r['y'] > -self.RING_RADIUS]
        self.clouds = [c for c in self.clouds if c['y'] > -c['size']]
        self.mountains = [m for m in self.mountains if m['y'] > -m['h']]

    def _handle_collisions_and_spawning(self):
        reward = 0
        player_pos = np.array([self.player_x, self.HEIGHT * 0.3])
        
        # Ring collisions
        collected_rings = []
        for ring in self.rings:
            ring_pos = np.array([ring['x'], ring['y']])
            dist = np.linalg.norm(player_pos - ring_pos)
            if dist < self.RING_RADIUS + self.PLAYER_HITBOX_RADIUS:
                collected_rings.append(ring)
                self.rings_collected += 1
                reward += 10
                self.ring_spawn_prob = max(0.01, 0.05 - self.rings_collected * 0.001)
                # SFX: Ring collect
                self._create_particles(ring['x'], ring['y'], self.COLOR_RING)

        self.rings = [r for r in self.rings if r not in collected_rings]

        # Spawning new objects
        if self.np_random.random() < self.ring_spawn_prob:
            self._spawn_object(self.rings, self.HEIGHT)
        if self.np_random.random() < 0.1:
            self._spawn_object(self.clouds, self.HEIGHT)
        if self.np_random.random() < 0.01:
            self._spawn_object(self.mountains, self.HEIGHT, is_mountain=True)
            
        return reward

    def _check_termination(self):
        # Ground collision
        if self.world_y >= self.WORLD_HEIGHT:
            return True
        # Win condition
        if self.rings_collected >= self.RINGS_TO_WIN:
            return True
        # Step limit
        if self.steps >= self.MAX_STEPS:
            return True
        return False

    def _get_observation(self):
        self._render_background()
        self._render_parallax_objects()
        self._render_game_objects()
        self._render_player()
        self._render_particles()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        # Draw sky gradient
        for y in range(self.HEIGHT):
            ratio = y / self.HEIGHT
            color = (
                self.COLOR_SKY_TOP[0] * (1 - ratio) + self.COLOR_SKY_BOTTOM[0] * ratio,
                self.COLOR_SKY_TOP[1] * (1 - ratio) + self.COLOR_SKY_BOTTOM[1] * ratio,
                self.COLOR_SKY_TOP[2] * (1 - ratio) + self.COLOR_SKY_BOTTOM[2] * ratio,
            )
            pygame.draw.line(self.screen, color, (0, y), (self.WIDTH, y))

    def _render_parallax_objects(self):
        # Render mountains (background)
        for m in sorted(self.mountains, key=lambda m: m['parallax']):
            points = [
                (m['x'], m['y']),
                (m['x'] + m['w']/2, m['y'] - m['h']),
                (m['x'] + m['w'], m['y'])
            ]
            pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_MOUNTAIN)
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_MOUNTAIN)
            
        # Render clouds (foreground)
        for c in sorted(self.clouds, key=lambda c: c['parallax']):
            color = list(self.COLOR_CLOUD)
            color.append(int(c['alpha']))
            pygame.gfxdraw.filled_circle(self.screen, int(c['x']), int(c['y']), int(c['size']), color)
            pygame.gfxdraw.aacircle(self.screen, int(c['x']), int(c['y']), int(c['size']), color)

    def _render_game_objects(self):
        # Render rings
        for ring in self.rings:
            pygame.gfxdraw.aacircle(self.screen, int(ring['x']), int(ring['y']), self.RING_RADIUS, self.COLOR_RING_OUTLINE)
            pygame.gfxdraw.filled_circle(self.screen, int(ring['x']), int(ring['y']), self.RING_RADIUS, self.COLOR_RING_OUTLINE)
            pygame.gfxdraw.aacircle(self.screen, int(ring['x']), int(ring['y']), self.RING_RADIUS - 3, self.COLOR_RING)
            pygame.gfxdraw.filled_circle(self.screen, int(ring['x']), int(ring['y']), self.RING_RADIUS - 3, self.COLOR_RING)

        # Render ground
        ground_screen_y = self.HEIGHT + (self.WORLD_HEIGHT - self.world_y)
        if ground_screen_y < self.HEIGHT + 50:
            pygame.draw.rect(self.screen, self.COLOR_GROUND, (0, ground_screen_y, self.WIDTH, 50))

    def _render_player(self):
        y_pos = self.HEIGHT * 0.3
        
        # Dynamic shape based on speed
        speed_ratio = (self.player_vy - self.MIN_SPEED) / (self.MAX_SPEED - self.MIN_SPEED)
        width_mod = 1.0 - 0.5 * speed_ratio # Narrower at high speed
        length_mod = 1.0 + 0.5 * speed_ratio # Longer at high speed
        
        p1 = (self.player_x, y_pos - 12 * length_mod)
        p2 = (self.player_x - 8 * width_mod, y_pos + 8 * length_mod)
        p3 = (self.player_x, y_pos + 4 * length_mod)
        p4 = (self.player_x + 8 * width_mod, y_pos + 8 * length_mod)
        
        points = [p1, p2, p3, p4]
        
        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_PLAYER_OUTLINE)
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PLAYER_OUTLINE)
        
        # Inset polygon for main color
        inset_points = [
            (p1[0], p1[1]+2),
            (p2[0]+2, p2[1]-2),
            (p3[0], p3[1]-1),
            (p4[0]-2, p4[1]-2)
        ]
        pygame.gfxdraw.filled_polygon(self.screen, inset_points, self.COLOR_PLAYER)

    def _render_particles(self):
        for p in self.particles:
            alpha = max(0, 255 * (p['life'] / p['max_life']))
            color = (*p['color'], alpha)
            size = int(p['size'] * (p['life'] / p['max_life']))
            if size > 0:
                pygame.gfxdraw.filled_circle(self.screen, int(p['x']), int(p['y']), size, color)

    def _render_ui(self):
        altitude = max(0, self.WORLD_HEIGHT - self.world_y)
        
        self._draw_text_with_shadow(f"SCORE: {int(self.score)}", (10, 10), self.font_large)
        self._draw_text_with_shadow(f"RINGS: {self.rings_collected}/{self.RINGS_TO_WIN}", (self.WIDTH - 170, 10), self.font_large)
        self._draw_text_with_shadow(f"ALT: {int(altitude)} M", (10, self.HEIGHT - 30), self.font_small)
        self._draw_text_with_shadow(f"SPEED: {self.player_vy:.1f} M/S", (self.WIDTH - 150, self.HEIGHT - 30), self.font_small)

    def _draw_text_with_shadow(self, text, pos, font):
        shadow_surface = font.render(text, True, self.COLOR_SHADOW)
        text_surface = font.render(text, True, self.COLOR_TEXT)
        self.screen.blit(shadow_surface, (pos[0] + 2, pos[1] + 2))
        self.screen.blit(text_surface, pos)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "rings_collected": self.rings_collected,
            "altitude": self.WORLD_HEIGHT - self.world_y,
            "speed": self.player_vy
        }
        
    def _spawn_object(self, obj_list, spawn_range_y, initial=False, is_mountain=False):
        if initial:
            y_pos = self.np_random.uniform(0, self.WORLD_HEIGHT)
        else:
            y_pos = self.np_random.uniform(self.HEIGHT, self.HEIGHT + spawn_range_y)

        x_pos = self.np_random.uniform(0, self.WIDTH)
        
        if obj_list is self.rings:
            obj_list.append({'x': x_pos, 'y': y_pos})
        elif obj_list is self.clouds:
            obj_list.append({
                'x': x_pos,
                'y': y_pos,
                'size': self.np_random.uniform(20, 60),
                'parallax': self.np_random.uniform(0.5, 0.9),
                'alpha': self.np_random.uniform(100, 200)
            })
        elif obj_list is self.mountains:
             obj_list.append({
                'x': x_pos,
                'y': y_pos + self.np_random.uniform(100, 300),
                'w': self.np_random.uniform(200, 400),
                'h': self.np_random.uniform(150, 350),
                'parallax': self.np_random.uniform(0.1, 0.3)
            })

    def _create_particles(self, x, y, color):
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            life = self.np_random.integers(15, 30)
            self.particles.append({
                'x': x, 'y': y,
                'vx': math.cos(angle) * speed,
                'vy': math.sin(angle) * speed,
                'life': life, 'max_life': life,
                'size': self.np_random.integers(2, 5),
                'color': color
            })

    def _get_dist_to_nearest_ring(self):
        player_pos = np.array([self.player_x, self.HEIGHT * 0.3])
        min_dist = float('inf')
        
        # Consider only rings that are ahead of the player
        upcoming_rings = [r for r in self.rings if r['y'] > player_pos[1]]
        if not upcoming_rings:
            return None
            
        for ring in upcoming_rings:
            ring_pos = np.array([ring['x'], ring['y']])
            dist = np.linalg.norm(player_pos - ring_pos)
            if dist < min_dist:
                min_dist = dist
        return min_dist if min_dist != float('inf') else None

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == '__main__':
    env = GameEnv()
    obs, info = env.reset()
    
    # Pygame setup for human play
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Skydiver")
    clock = pygame.time.Clock()

    done = False
    total_reward = 0
    
    # --- Control Mapping for Human Play ---
    # action = [movement, space, shift]
    # movement: 0=none, 1=up, 2=down, 3=left, 4=right
    action = [0, 0, 0]

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        keys = pygame.key.get_pressed()
        
        # Reset movement action
        action[0] = 0 
        
        if keys[pygame.K_UP]:
            action[0] = 1
        elif keys[pygame.K_DOWN]:
            action[0] = 2
        elif keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
        
        # The brief doesn't use space/shift, but we map them anyway
        action[1] = 1 if keys[pygame.K_SPACE] else 0
        action[2] = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        # Combine actions if needed (e.g., up and left)
        # This implementation prioritizes vertical over horizontal if both are pressed
        if keys[pygame.K_LEFT]:
            action[0] = 3
        if keys[pygame.K_RIGHT]:
            action[0] = 4
        if keys[pygame.K_UP]:
            action[0] = 1
        if keys[pygame.K_DOWN]:
            action[0] = 2

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated

        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(env.metadata['render_fps'])

    print(f"Episode finished. Total Reward: {total_reward}, Info: {info}")
    pygame.quit()
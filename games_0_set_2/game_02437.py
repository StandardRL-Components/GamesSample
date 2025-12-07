
# Generated: 2025-08-28T04:49:29.075186
# Source Brief: brief_02437.md
# Brief Index: 2437

        
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
    """
    A Gymnasium environment for a fast-paced arcade game where the player pilots a ship
    through a procedurally generated asteroid field. The goal is to survive for 60 seconds
    in each of the 3 increasingly difficult stages.
    """
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use Arrow Keys (↑, ↓, ←, →) to move your ship. "
        "Survive for 60 seconds to advance to the next stage."
    )

    game_description = (
        "Pilot a ship through a dense, procedurally generated asteroid field. "
        "Dodge incoming rocks and survive for as long as you can to progress "
        "through increasingly difficult stages."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30

        # Game parameters
        self.PLAYER_SPEED = 5.0
        self.PLAYER_RADIUS = 12
        self.PLAYER_GLOW_RADIUS = 25
        self.NEAR_MISS_THRESHOLD = 30
        self.MAX_STAGES = 3
        self.STAGE_DURATION_FRAMES = 60 * self.FPS
        self.DIFFICULTY_INTERVAL_FRAMES = 15 * self.FPS
        self.SPEED_INCREMENT = 0.1
        self.STAR_COUNT = 150

        # Colors
        self.COLOR_BG = (15, 15, 25)
        self.COLOR_STAR = (100, 100, 120)
        self.COLOR_PLAYER = (230, 230, 255)
        self.COLOR_PLAYER_GLOW = (80, 80, 150)
        self.COLOR_ASTEROID = (200, 200, 200)
        self.COLOR_TEXT = (255, 255, 255)
        self.COLOR_NEAR_MISS = (255, 255, 0)
        self.COLOR_COLLISION = (255, 50, 50)
        
        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.font_large = pygame.font.SysFont('monospace', 36, bold=True)
        self.font_medium = pygame.font.SysFont('monospace', 24, bold=True)
        self.font_small = pygame.font.SysFont('monospace', 18, bold=True)

        # --- State Variables (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.stage = 1
        self.stage_timer_frames = 0
        self.difficulty_increase_timer = 0
        self.player_pos = [0, 0]
        self.asteroids = []
        self.effects = []
        self.stars = []
        
        self.reset()
        self.validate_implementation()

    def _get_stage_config(self):
        """Returns configuration for the current stage."""
        if self.stage == 1:
            return {"num_asteroids": 8, "base_speed": 2.0}
        elif self.stage == 2:
            return {"num_asteroids": 10, "base_speed": 2.5}
        else: # Stage 3 and beyond (if extended)
            return {"num_asteroids": 12, "base_speed": 3.0}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.stage = 1
        self.stage_timer_frames = 0
        self.difficulty_increase_timer = 0
        
        self.player_pos = [self.WIDTH / 2, self.HEIGHT / 2]
        
        self.asteroids = []
        self.effects = []
        self._spawn_initial_asteroids()

        if not self.stars:
            self._generate_starfield()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0.0
        terminated = False

        # --- 1. Handle Input ---
        movement = action[0]
        self._handle_player_movement(movement)

        # --- 2. Update Game State ---
        self._update_timers()
        self._update_asteroids()
        self._update_effects()
        
        # --- 3. Check Events & Calculate Reward ---
        reward += 0.1  # Survival reward

        collision_detected, near_miss_count = self._check_collisions()
        reward -= near_miss_count * 5.0
        
        if collision_detected:
            # Sound: Explosion
            reward = -100.0
            terminated = True
            self.game_over = True
            self._add_explosion_effect(self.player_pos)
        
        if self.stage_timer_frames >= self.STAGE_DURATION_FRAMES:
            self.stage += 1
            if self.stage > self.MAX_STAGES:
                # Sound: Victory
                reward += 100.0
                terminated = True
                self.game_over = True
                self._add_text_effect("VICTORY!", self.font_large)
            else:
                # Sound: Stage Clear
                reward += 10.0
                self.stage_timer_frames = 0
                self.difficulty_increase_timer = 0
                self.asteroids.clear()
                self._spawn_initial_asteroids()
                self._add_text_effect(f"STAGE {self.stage}", self.font_large)

        self.score += reward
        self.steps += 1

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_player_movement(self, movement):
        if movement == 1:  # Up
            self.player_pos[1] -= self.PLAYER_SPEED
        elif movement == 2:  # Down
            self.player_pos[1] += self.PLAYER_SPEED
        elif movement == 3:  # Left
            self.player_pos[0] -= self.PLAYER_SPEED
        elif movement == 4:  # Right
            self.player_pos[0] += self.PLAYER_SPEED
        
        # Boundary checks
        self.player_pos[0] = np.clip(self.player_pos[0], self.PLAYER_RADIUS, self.WIDTH - self.PLAYER_RADIUS)
        self.player_pos[1] = np.clip(self.player_pos[1], self.PLAYER_RADIUS, self.HEIGHT - self.PLAYER_RADIUS)

    def _update_timers(self):
        self.stage_timer_frames += 1
        self.difficulty_increase_timer += 1

        if self.difficulty_increase_timer >= self.DIFFICULTY_INTERVAL_FRAMES:
            self.difficulty_increase_timer = 0
            current_speed_mod = self.stage_timer_frames // self.DIFFICULTY_INTERVAL_FRAMES
            for a in self.asteroids:
                # Increase speed slightly
                a['speed'] += self.SPEED_INCREMENT

    def _update_asteroids(self):
        # Move existing asteroids and remove off-screen ones
        for asteroid in self.asteroids[:]:
            asteroid['pos'] += asteroid['vel']
            asteroid['rot'] = (asteroid['rot'] + asteroid['rot_speed']) % 360
            if not (-asteroid['size'] < asteroid['pos'][0] < self.WIDTH + asteroid['size'] and \
                    -asteroid['size'] < asteroid['pos'][1] < self.HEIGHT + asteroid['size']):
                self.asteroids.remove(asteroid)

        # Spawn new asteroids to maintain count
        config = self._get_stage_config()
        while len(self.asteroids) < config['num_asteroids']:
            self._spawn_asteroid()

    def _update_effects(self):
        for effect in self.effects[:]:
            effect['lifetime'] -= 1
            if 'vel' in effect:
                effect['pos'] += effect['vel']
                effect['vel'] *= 0.98 # Particle friction
            if effect['lifetime'] <= 0:
                self.effects.remove(effect)

    def _check_collisions(self):
        collision_detected = False
        near_misses = 0
        player_x, player_y = self.player_pos

        for asteroid in self.asteroids:
            dist = math.hypot(player_x - asteroid['pos'][0], player_y - asteroid['pos'][1])
            
            # Collision
            if dist < self.PLAYER_RADIUS + asteroid['size']:
                collision_detected = True
                break
            
            # Near Miss
            is_near = dist < self.PLAYER_RADIUS + asteroid['size'] + self.NEAR_MISS_THRESHOLD
            if is_near and not asteroid.get('on_cooldown', False):
                # Sound: Near miss whoosh
                near_misses += 1
                asteroid['on_cooldown'] = True
                self._add_flash_effect(self.COLOR_NEAR_MISS, 10, self.PLAYER_RADIUS + 10)
            elif not is_near:
                asteroid['on_cooldown'] = False
        
        return collision_detected, near_misses

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_starfield()
        self._render_asteroids()
        if not self.game_over:
            self._render_player()
        self._render_effects()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "stage": self.stage}

    # --- Spawning Methods ---

    def _spawn_initial_asteroids(self):
        config = self._get_stage_config()
        for _ in range(config['num_asteroids']):
            self._spawn_asteroid(at_start=True)

    def _spawn_asteroid(self, at_start=False):
        config = self._get_stage_config()
        size = self.np_random.uniform(15, 35)
        
        if at_start: # Spawn anywhere on screen for initial setup
            pos = self.np_random.uniform([0,0], [self.WIDTH, self.HEIGHT])
            # Ensure it doesn't spawn on the player
            while math.hypot(pos[0] - self.player_pos[0], pos[1] - self.player_pos[1]) < 150:
                 pos = self.np_random.uniform([0,0], [self.WIDTH, self.HEIGHT])
        else: # Spawn at edges
            edge = self.np_random.integers(4)
            if edge == 0:  # Top
                pos = np.array([self.np_random.uniform(0, self.WIDTH), -size])
            elif edge == 1:  # Bottom
                pos = np.array([self.np_random.uniform(0, self.WIDTH), self.HEIGHT + size])
            elif edge == 2:  # Left
                pos = np.array([-size, self.np_random.uniform(0, self.HEIGHT)])
            else:  # Right
                pos = np.array([self.WIDTH + size, self.np_random.uniform(0, self.HEIGHT)])

        # Velocity towards a random point near the center
        target = np.array([
            self.WIDTH / 2 + self.np_random.uniform(-self.WIDTH/4, self.WIDTH/4),
            self.HEIGHT / 2 + self.np_random.uniform(-self.HEIGHT/4, self.HEIGHT/4)
        ])
        direction = (target - pos) / np.linalg.norm(target - pos)
        
        current_speed_mod = self.difficulty_increase_timer / self.DIFFICULTY_INTERVAL_FRAMES * self.SPEED_INCREMENT
        speed = config['base_speed'] + current_speed_mod + self.np_random.uniform(-0.5, 0.5)
        vel = direction * max(1.0, speed)

        self.asteroids.append({
            'pos': pos.astype(float),
            'vel': vel.astype(float),
            'rot': self.np_random.uniform(0, 360),
            'rot_speed': self.np_random.uniform(-1.5, 1.5),
            'size': size,
            'vertices': self._generate_polygon_vertices(size, self.np_random.uniform(0.1, 0.4), self.np_random.uniform(0.3, 0.7), 12)
        })

    def _generate_polygon_vertices(self, avg_radius, irregularity, spikeyness, num_vertices):
        vertices = []
        angle_step = 2 * math.pi / num_vertices
        for i in range(num_vertices):
            angle = i * angle_step
            radius = self.np_random.normal(avg_radius, avg_radius * irregularity)
            point = (
                radius * math.cos(angle) + self.np_random.normal(0, spikeyness),
                radius * math.sin(angle) + self.np_random.normal(0, spikeyness)
            )
            vertices.append(point)
        return vertices

    def _generate_starfield(self):
        self.stars = [
            (self.np_random.integers(0, self.WIDTH), 
             self.np_random.integers(0, self.HEIGHT), 
             self.np_random.uniform(0.5, 1.5))
            for _ in range(self.STAR_COUNT)
        ]

    # --- Effect Methods ---

    def _add_flash_effect(self, color, lifetime, radius):
        self.effects.append({
            'type': 'flash', 'pos': self.player_pos.copy(), 'lifetime': lifetime,
            'color': color, 'radius': radius
        })

    def _add_explosion_effect(self, pos):
        self.effects.append({
            'type': 'flash', 'pos': pos, 'lifetime': 20,
            'color': self.COLOR_COLLISION, 'radius': 100
        })
        for _ in range(50):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 6)
            vel = np.array([math.cos(angle) * speed, math.sin(angle) * speed])
            self.effects.append({
                'type': 'particle', 'pos': np.array(pos), 'vel': vel,
                'lifetime': self.np_random.integers(20, 40),
                'color': random.choice([self.COLOR_COLLISION, self.COLOR_NEAR_MISS, (255,255,255)]),
                'size': self.np_random.uniform(1, 3)
            })

    def _add_text_effect(self, text, font):
        self.effects.append({
            'type': 'text', 'text': text, 'font': font,
            'pos': (self.WIDTH // 2, self.HEIGHT // 2),
            'lifetime': 60, 'color': self.COLOR_TEXT
        })

    # --- Rendering Methods ---

    def _render_starfield(self):
        for x, y, size in self.stars:
            pygame.draw.rect(self.screen, self.COLOR_STAR, (x, y, size, size))

    def _render_player(self):
        px, py = int(self.player_pos[0]), int(self.player_pos[1])
        
        # Glow effect
        glow_alpha = 60
        glow_surf = pygame.Surface((self.PLAYER_GLOW_RADIUS*2, self.PLAYER_GLOW_RADIUS*2), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, (*self.COLOR_PLAYER_GLOW, glow_alpha), (self.PLAYER_GLOW_RADIUS, self.PLAYER_GLOW_RADIUS), self.PLAYER_GLOW_RADIUS)
        self.screen.blit(glow_surf, (px - self.PLAYER_GLOW_RADIUS, py - self.PLAYER_GLOW_RADIUS), special_flags=pygame.BLEND_RGBA_ADD)

        # Ship body
        points = [
            (px, py - self.PLAYER_RADIUS),
            (px - self.PLAYER_RADIUS * 0.7, py + self.PLAYER_RADIUS * 0.7),
            (px + self.PLAYER_RADIUS * 0.7, py + self.PLAYER_RADIUS * 0.7)
        ]
        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_PLAYER)
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PLAYER)

    def _render_asteroids(self):
        for asteroid in self.asteroids:
            ax, ay = asteroid['pos']
            angle_rad = math.radians(asteroid['rot'])
            cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
            
            points = []
            for vx, vy in asteroid['vertices']:
                rotated_x = vx * cos_a - vy * sin_a
                rotated_y = vx * sin_a + vy * cos_a
                points.append((int(ax + rotated_x), int(ay + rotated_y)))
            
            if len(points) > 2:
                pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_ASTEROID)
                pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_ASTEROID)

    def _render_effects(self):
        for effect in self.effects:
            if effect['type'] == 'flash':
                alpha = int(255 * (effect['lifetime'] / 20)) if effect['lifetime'] < 20 else 255
                alpha = max(0, alpha)
                pos = effect['pos']
                pygame.gfxdraw.filled_circle(self.screen, int(pos[0]), int(pos[1]), int(effect['radius']), (*effect['color'], alpha))
            elif effect['type'] == 'particle':
                pygame.draw.circle(self.screen, effect['color'], effect['pos'].astype(int), effect['size'])
            elif effect['type'] == 'text':
                alpha = 255
                if effect['lifetime'] < 20:
                    alpha = int(255 * (effect['lifetime'] / 20))
                
                text_surf = effect['font'].render(effect['text'], True, effect['color'])
                text_surf.set_alpha(alpha)
                text_rect = text_surf.get_rect(center=effect['pos'])
                self.screen.blit(text_surf, text_rect)

    def _render_ui(self):
        # Timer
        time_left = max(0, (self.STAGE_DURATION_FRAMES - self.stage_timer_frames) // self.FPS)
        timer_text = self.font_medium.render(f"TIME: {time_left}", True, self.COLOR_TEXT)
        self.screen.blit(timer_text, (10, 10))

        # Stage
        stage_text = self.font_medium.render(f"STAGE: {self.stage}/{self.MAX_STAGES}", True, self.COLOR_TEXT)
        stage_rect = stage_text.get_rect(topright=(self.WIDTH - 10, 10))
        self.screen.blit(stage_text, stage_rect)

        # Score
        score_text = self.font_medium.render(f"SCORE: {int(self.score)}", True, self.COLOR_TEXT)
        score_rect = score_text.get_rect(midbottom=(self.WIDTH / 2, self.HEIGHT - 10))
        self.screen.blit(score_text, score_rect)

        if self.game_over:
            if self.stage > self.MAX_STAGES:
                pass # Victory text is handled by effects
            else:
                self._add_text_effect("GAME OVER", self.font_large)

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == self.observation_space.shape
        assert test_obs.dtype == self.observation_space.dtype
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == self.observation_space.shape
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == self.observation_space.shape
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to run the game directly to test it
    env = GameEnv(render_mode="rgb_array")
    
    # --- Manual Play ---
    # Set to False to let a random agent play
    MANUAL_PLAY = True

    obs, info = env.reset()
    done = False
    
    # Pygame window for human play
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Asteroid Dodger")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        if MANUAL_PLAY:
            keys = pygame.key.get_pressed()
            movement = 0 # No-op
            if keys[pygame.K_UP]: movement = 1
            elif keys[pygame.K_DOWN]: movement = 2
            elif keys[pygame.K_LEFT]: movement = 3
            elif keys[pygame.K_RIGHT]: movement = 4
            
            space = 1 if keys[pygame.K_SPACE] else 0
            shift = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
            
            action = [movement, space, shift]
        else:
            action = env.action_space.sample()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment.")
                obs, info = env.reset()
                total_reward = 0
                done = False

        if not done:
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

        # Render the observation from the environment to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if done:
            print(f"Episode finished. Total Reward: {total_reward}, Info: {info}")
            # In manual play, allow reset with 'R' key
            if not MANUAL_PLAY:
                obs, info = env.reset()
                total_reward = 0
                done = False

        clock.tick(env.FPS)
        
    pygame.quit()

# Generated: 2025-08-27T19:10:38.320539
# Source Brief: brief_02072.md
# Brief Index: 2072

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move. Press Space for a risky hyperspace jump."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Pilot a spaceship, collect valuable asteroids, and dodge hostile alien ships. "
        "Risky asteroid grabs near enemies yield bonus points."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400

    # Colors
    COLOR_BG = (10, 15, 30)
    COLOR_PLAYER = (0, 255, 128)
    COLOR_PLAYER_GLOW = (0, 255, 128, 50)
    COLOR_ASTEROID = (150, 150, 150)
    COLOR_ENEMY = (255, 50, 50)
    COLOR_ENEMY_GLOW = (255, 50, 50, 60)
    COLOR_PARTICLE_ASTEROID = (200, 200, 200)
    COLOR_PARTICLE_EXPLOSION = (255, 150, 50)
    COLOR_UI_TEXT = (255, 255, 255)
    
    # Game parameters
    PLAYER_ACCELERATION = 0.4
    PLAYER_FRICTION = 0.96
    PLAYER_MAX_SPEED = 6
    PLAYER_SIZE = 12
    INITIAL_LIVES = 5
    ASTEROIDS_TO_WIN = 20
    NUM_INITIAL_ASTEROIDS = 8
    NUM_INITIAL_ENEMIES = 4
    NUM_STARS = 150
    MAX_STEPS = 2000
    
    # Reward parameters
    RISK_RADIUS = 100
    SAFE_RADIUS = 250
    INVINCIBILITY_FRAMES = 60

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 28)
        self.font_game_over = pygame.font.Font(None, 64)

        # State variables are initialized in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_pos = pygame.Vector2(0, 0)
        self.player_vel = pygame.Vector2(0, 0)
        self.player_angle = 0
        self.player_lives = 0
        self.asteroids_collected = 0
        self.asteroids = []
        self.enemies = []
        self.stars = []
        self.particles = []
        self.prev_space_held = False
        self.last_dist_to_asteroid = float('inf')
        self.invincibility_timer = 0
        self.rng = None

        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        else:
            if self.rng is None:
                self.rng = np.random.default_rng()

        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.player_pos = pygame.Vector2(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2)
        self.player_vel = pygame.Vector2(0, 0)
        self.player_angle = -90
        self.player_lives = self.INITIAL_LIVES
        self.asteroids_collected = 0
        self.invincibility_timer = 0
        self.prev_space_held = False

        self.stars = [
            (self.rng.integers(0, self.SCREEN_WIDTH), self.rng.integers(0, self.SCREEN_HEIGHT), self.rng.integers(1, 3))
            for _ in range(self.NUM_STARS)
        ]
        
        self.asteroids = []
        for _ in range(self.NUM_INITIAL_ASTEROIDS):
            self._spawn_asteroid()

        self.enemies = []
        for i in range(self.NUM_INITIAL_ENEMIES):
            self._spawn_enemy(i)
            
        self.particles = []
        
        self.last_dist_to_asteroid = self._get_dist_to_nearest_asteroid()

        return self._get_observation(), self._get_info()

    def _spawn_asteroid(self):
        pos = pygame.Vector2(
            self.rng.integers(0, self.SCREEN_WIDTH), self.rng.integers(0, self.SCREEN_HEIGHT)
        )
        while pos.distance_to(self.player_pos) < 100:
            pos = pygame.Vector2(
                self.rng.integers(0, self.SCREEN_WIDTH), self.rng.integers(0, self.SCREEN_HEIGHT)
            )
        
        vel = pygame.Vector2(self.rng.uniform(-1, 1), self.rng.uniform(-1, 1))
        vel = vel.normalize() * self.rng.uniform(0.5, 1.5) if vel.length() > 0 else pygame.Vector2(0,0)
        
        self.asteroids.append({"pos": pos, "vel": vel, "size": self.rng.integers(10, 20)})

    def _spawn_enemy(self, index):
        pattern_type = self.rng.choice(['circle', 'patrol'])
        pos = pygame.Vector2(
            self.rng.integers(0, self.SCREEN_WIDTH), self.rng.integers(0, self.SCREEN_HEIGHT)
        )
        while pos.distance_to(self.player_pos) < 150:
             pos = pygame.Vector2(
                self.rng.integers(0, self.SCREEN_WIDTH), self.rng.integers(0, self.SCREEN_HEIGHT)
            )

        enemy = {"pos": pos, "size": 10, "pattern_type": pattern_type, "state": {}}
        if pattern_type == 'circle':
            enemy['state']['center'] = pos.copy()
            enemy['state']['radius'] = self.rng.uniform(50, 150)
            enemy['state']['angle'] = self.rng.uniform(0, 2 * math.pi)
            enemy['state']['speed'] = self.rng.uniform(0.02, 0.04) * self.rng.choice([-1, 1])
        elif pattern_type == 'patrol':
            enemy['state']['target'] = pygame.Vector2(
                self.rng.integers(0, self.SCREEN_WIDTH), self.rng.integers(0, self.SCREEN_HEIGHT)
            )
            enemy['state']['speed'] = self.rng.uniform(1.5, 2.5)
        self.enemies.append(enemy)

    def _get_dist_to_nearest_entity(self, entities):
        if not entities:
            return float('inf')
        
        min_dist = float('inf')
        for entity in entities:
            dx = abs(entity['pos'].x - self.player_pos.x)
            dy = abs(entity['pos'].y - self.player_pos.y)
            
            # Account for world wrap
            dx = min(dx, self.SCREEN_WIDTH - dx)
            dy = min(dy, self.SCREEN_HEIGHT - dy)

            dist = math.sqrt(dx**2 + dy**2)
            if dist < min_dist:
                min_dist = dist
        return min_dist

    def _get_dist_to_nearest_asteroid(self):
        return self._get_dist_to_nearest_entity(self.asteroids)
    
    def _get_dist_to_nearest_enemy(self):
        return self._get_dist_to_nearest_entity(self.enemies)

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        self.clock.tick(30)
        self.steps += 1
        reward = 0

        movement, space_pressed, _ = action
        space_held = space_pressed == 1
        
        # --- Player Movement ---
        acceleration = pygame.Vector2(0, 0)
        if movement == 1: # Up
            acceleration.y = -self.PLAYER_ACCELERATION
        elif movement == 2: # Down
            acceleration.y = self.PLAYER_ACCELERATION
        elif movement == 3: # Left
            acceleration.x = -self.PLAYER_ACCELERATION
        elif movement == 4: # Right
            acceleration.x = self.PLAYER_ACCELERATION
        
        self.player_vel += acceleration
        if self.player_vel.length() > self.PLAYER_MAX_SPEED:
            self.player_vel.scale_to_length(self.PLAYER_MAX_SPEED)
        
        self.player_pos += self.player_vel
        self.player_vel *= self.PLAYER_FRICTION
        
        if self.player_vel.length() > 0.1:
            self.player_angle = self.player_vel.angle_to(pygame.Vector2(1, 0))

        # World wrap
        self.player_pos.x %= self.SCREEN_WIDTH
        self.player_pos.y %= self.SCREEN_HEIGHT

        # --- Player Action: Hyperspace ---
        if space_held and not self.prev_space_held:
            # sound: hyperspace_jump.wav
            safe_spot_found = False
            for _ in range(10): # Try 10 times to find a safe spot
                self.player_pos = pygame.Vector2(
                    self.rng.integers(0, self.SCREEN_WIDTH), self.rng.integers(0, self.SCREEN_HEIGHT)
                )
                if self._get_dist_to_nearest_enemy() > 50 and self._get_dist_to_nearest_asteroid() > 50:
                    safe_spot_found = True
                    break
            self.player_vel = pygame.Vector2(0, 0)
            self._create_particles(self.player_pos, 30, self.COLOR_PLAYER, 2, 5, 20)
        self.prev_space_held = space_held
        
        # --- Update Entities ---
        self._update_asteroids()
        self._update_enemies()
        self._update_particles()
        if self.invincibility_timer > 0:
            self.invincibility_timer -= 1

        # --- Continuous Reward ---
        dist_to_asteroid = self._get_dist_to_nearest_asteroid()
        if dist_to_asteroid < self.last_dist_to_asteroid:
            reward += 0.1
        else:
            reward -= 0.1
        self.last_dist_to_asteroid = dist_to_asteroid

        # --- Collisions ---
        # Player vs Asteroids
        for asteroid in self.asteroids[:]:
            if self._check_collision(self.player_pos, self.PLAYER_SIZE, asteroid['pos'], asteroid['size']):
                # sound: collect_asteroid.wav
                self.asteroids.remove(asteroid)
                self.asteroids_collected += 1
                self.score += 10
                reward += 10
                
                dist_to_enemy = self._get_dist_to_nearest_enemy()
                if dist_to_enemy < self.RISK_RADIUS:
                    reward += 5
                    self.score += 5
                elif dist_to_enemy > self.SAFE_RADIUS:
                    reward -= 2
                    self.score = max(0, self.score - 2)
                
                self._create_particles(asteroid['pos'], 20, self.COLOR_PARTICLE_ASTEROID, 1, 3, 15)
                self._spawn_asteroid()
                break

        # Player vs Enemies
        if self.invincibility_timer == 0:
            for enemy in self.enemies:
                if self._check_collision(self.player_pos, self.PLAYER_SIZE, enemy['pos'], enemy['size']):
                    # sound: player_hit.wav
                    self.player_lives -= 1
                    reward -= 20
                    self.score = max(0, self.score - 20)
                    self.invincibility_timer = self.INVINCIBILITY_FRAMES
                    self._create_particles(self.player_pos, 40, self.COLOR_PARTICLE_EXPLOSION, 2, 6, 25)
                    break
        
        # --- Termination Check ---
        terminated = False
        if self.asteroids_collected >= self.ASTEROIDS_TO_WIN:
            terminated = True
            self.game_over = True
            reward += 100
            self.score += 100
        elif self.player_lives <= 0:
            terminated = True
            self.game_over = True
            reward -= 100
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True
            
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _update_asteroids(self):
        for asteroid in self.asteroids:
            asteroid['pos'] += asteroid['vel']
            asteroid['pos'].x %= self.SCREEN_WIDTH
            asteroid['pos'].y %= self.SCREEN_HEIGHT

    def _update_enemies(self):
        for enemy in self.enemies:
            if enemy['pattern_type'] == 'circle':
                state = enemy['state']
                state['angle'] += state['speed']
                offset = pygame.Vector2(
                    math.cos(state['angle']) * state['radius'],
                    math.sin(state['angle']) * state['radius']
                )
                enemy['pos'] = state['center'] + offset
            elif enemy['pattern_type'] == 'patrol':
                state = enemy['state']
                direction = state['target'] - enemy['pos']
                if direction.length() < 5:
                    state['target'] = pygame.Vector2(
                        self.rng.integers(0, self.SCREEN_WIDTH), self.rng.integers(0, self.SCREEN_HEIGHT)
                    )
                else:
                    enemy['pos'] += direction.normalize() * state['speed']
            
            enemy['pos'].x %= self.SCREEN_WIDTH
            enemy['pos'].y %= self.SCREEN_HEIGHT

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1

    def _create_particles(self, pos, count, color, min_speed, max_speed, life):
        for _ in range(count):
            angle = self.rng.uniform(0, 2 * math.pi)
            speed = self.rng.uniform(min_speed, max_speed)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            self.particles.append({
                "pos": pos.copy(),
                "vel": vel,
                "life": life,
                "color": color
            })
            
    def _check_collision(self, pos1, size1, pos2, size2):
        dist_sq = (pos1.x - pos2.x)**2 + (pos1.y - pos2.y)**2
        # Check wrapped distances
        dx_wrap = abs(pos1.x - pos2.x)
        dx_wrap = min(dx_wrap, self.SCREEN_WIDTH - dx_wrap)
        dy_wrap = abs(pos1.y - pos2.y)
        dy_wrap = min(dy_wrap, self.SCREEN_HEIGHT - dy_wrap)
        dist_sq_wrap = dx_wrap**2 + dy_wrap**2

        return dist_sq_wrap < (size1 + size2)**2

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Stars
        for x, y, size in self.stars:
            pygame.draw.rect(self.screen, (size * 40, size * 40, size * 50), (x, y, size, size))

        # Particles
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['life'] / 20.0))))
            color = p['color']
            if len(color) == 4: # Color with alpha
                 pygame.gfxdraw.filled_circle(self.screen, int(p['pos'].x), int(p['pos'].y), 2, (*color[:3], alpha))
            else: # Solid color
                 pygame.gfxdraw.filled_circle(self.screen, int(p['pos'].x), int(p['pos'].y), 2, (*color, alpha))


        # Asteroids
        for asteroid in self.asteroids:
            pygame.gfxdraw.filled_circle(self.screen, int(asteroid['pos'].x), int(asteroid['pos'].y), int(asteroid['size']), self.COLOR_ASTEROID)
            pygame.gfxdraw.aacircle(self.screen, int(asteroid['pos'].x), int(asteroid['pos'].y), int(asteroid['size']), self.COLOR_ASTEROID)

        # Enemies
        for enemy in self.enemies:
            pos, size = enemy['pos'], enemy['size']
            # Glow
            pygame.gfxdraw.filled_circle(self.screen, int(pos.x), int(pos.y), int(size * 2.5), self.COLOR_ENEMY_GLOW)
            # Body
            points = [
                (pos.x, pos.y - size),
                (pos.x - size * 0.8, pos.y + size * 0.8),
                (pos.x + size * 0.8, pos.y + size * 0.8)
            ]
            pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_ENEMY)
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_ENEMY)

        # Player
        if self.invincibility_timer == 0 or (self.invincibility_timer > 0 and self.steps % 4 < 2):
            pos = self.player_pos
            size = self.PLAYER_SIZE
            angle_rad = math.radians(self.player_angle)
            
            # Glow
            pygame.gfxdraw.filled_circle(self.screen, int(pos.x), int(pos.y), int(size * 2), self.COLOR_PLAYER_GLOW)

            # Ship body
            p1 = pos + pygame.Vector2(size, 0).rotate(-self.player_angle)
            p2 = pos + pygame.Vector2(-size * 0.7, size * 0.7).rotate(-self.player_angle)
            p3 = pos + pygame.Vector2(-size * 0.7, -size * 0.7).rotate(-self.player_angle)
            points = [(p1.x, p1.y), (p2.x, p2.y), (p3.x, p3.y)]
            pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_PLAYER)
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PLAYER)

    def _render_ui(self):
        score_text = self.font_ui.render(f"Score: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))

        asteroids_text = self.font_ui.render(f"Asteroids: {self.asteroids_collected} / {self.ASTEROIDS_TO_WIN}", True, self.COLOR_UI_TEXT)
        text_rect = asteroids_text.get_rect(centerx=self.SCREEN_WIDTH / 2)
        self.screen.blit(asteroids_text, (text_rect.x, 10))
        
        lives_text = self.font_ui.render(f"Lives: {self.player_lives}", True, self.COLOR_UI_TEXT)
        self.screen.blit(lives_text, (self.SCREEN_WIDTH - lives_text.get_width() - 10, 10))

        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            if self.asteroids_collected >= self.ASTEROIDS_TO_WIN:
                end_text = self.font_game_over.render("VICTORY!", True, self.COLOR_PLAYER)
            else:
                end_text = self.font_game_over.render("GAME OVER", True, self.COLOR_ENEMY)
            
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.player_lives,
            "asteroids_collected": self.asteroids_collected,
        }

    def close(self):
        pygame.quit()

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

if __name__ == '__main__':
    # This block allows you to play the game manually
    env = GameEnv()
    obs, info = env.reset()
    
    running = True
    total_reward = 0
    
    # Create a display window
    pygame.display.set_caption("Asteroid Collector")
    display_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))

    action = np.array([0, 0, 0]) # No-op, no space, no shift

    while running:
        # --- Human Input ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        
        # Movement
        action[0] = 0 # none
        if keys[pygame.K_UP]:
            action[0] = 1
        elif keys[pygame.K_DOWN]:
            action[0] = 2
        elif keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4

        # Actions
        action[1] = 1 if keys[pygame.K_SPACE] else 0
        action[2] = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        # --- Gym Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # --- Render to Display ---
        # The observation is (H, W, C), but pygame blit wants a surface.
        # So we get the surface from the env's internal screen.
        surf = pygame.transform.flip(env.screen, False, False)
        surf = pygame.transform.rotate(surf, -90)
        surf = pygame.transform.flip(surf, True, False)
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward}")
            # Wait for a moment before resetting
            pygame.time.wait(3000)
            obs, info = env.reset()
            total_reward = 0

    env.close()
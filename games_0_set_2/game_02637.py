
# Generated: 2025-08-28T05:28:48.904768
# Source Brief: brief_02637.md
# Brief Index: 2637

        
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
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Arrow keys to move. Hold Shift for a temporary shield. Press Space for a speed boost."
    )

    game_description = (
        "Pilot a lone spaceship through a dense asteroid field. Dodge the incoming "
        "rocks and use your shield and boost abilities to survive as long as possible."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.FPS = 60
        self.MAX_STEPS = 120 * self.FPS  # 120 seconds

        # Colors
        self.COLOR_BG = (10, 10, 20)
        self.COLOR_PLAYER = (255, 255, 255)
        self.COLOR_SHIELD = (100, 200, 255)
        self.COLOR_BOOST_TRAIL = (255, 165, 0)
        self.COLOR_ASTEROID = (128, 128, 128)
        self.COLOR_UI = (100, 255, 100)
        self.COLOR_EXPLOSION = (255, 100, 0)
        self.COLOR_COOLDOWN = (255, 100, 100)

        # Player settings
        self.PLAYER_SPEED = 5
        self.PLAYER_RADIUS = 12
        self.BOOST_MULTIPLIER = 2
        self.BOOST_DURATION = int(1 * self.FPS)
        self.BOOST_COOLDOWN = int(3 * self.FPS)
        self.SHIELD_DURATION = int(2 * self.FPS)
        self.SHIELD_COOLDOWN = int(5 * self.FPS)
        
        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 28)

        # --- Game State Variables ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_pos = pygame.Vector2(0, 0)
        self.player_vel = pygame.Vector2(0, 0)
        self.asteroids = []
        self.particles = []
        self.stars = []
        self.shield_timer = 0
        self.shield_cooldown = 0
        self.boost_timer = 0
        self.boost_cooldown = 0
        self.stationary_steps = 0

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False

        self.player_pos = pygame.Vector2(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT - 50)
        self.player_vel = pygame.Vector2(0, 0)

        self.asteroids = []
        self.particles = []
        self._init_stars()

        self.shield_timer = 0
        self.shield_cooldown = 0
        self.boost_timer = 0
        self.boost_cooldown = 0
        self.stationary_steps = 0

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.auto_advance:
            self.clock.tick(self.FPS)

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        reward = self._calculate_reward(movement)

        if not self.game_over:
            self.steps += 1
            self.score = self.steps
            self._handle_input(movement, space_held, shift_held)
            self._update_game_state()
            self._check_collisions()

        terminated = self.game_over or self.steps >= self.MAX_STEPS

        if terminated:
            if self.game_over:
                reward = -10.0
                if len(self.particles) == 0: # Create explosion only once
                    self._create_explosion(self.player_pos)
                    # SFX: Player explosion
            elif self.steps >= self.MAX_STEPS:
                reward = 100.0

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info(),
        )

    def _calculate_reward(self, movement):
        if self.game_over:
            return 0.0
            
        reward = 0.1  # Survival reward
        if movement == 0:
            self.stationary_steps += 1
        else:
            self.stationary_steps = 0
        
        if self.stationary_steps > 10:
            reward -= 0.2
        
        return reward

    def _handle_input(self, movement, space_held, shift_held):
        move_speed = self.PLAYER_SPEED * (self.BOOST_MULTIPLIER if self.boost_timer > 0 else 1)
        
        vel_x, vel_y = 0, 0
        if movement == 1: vel_y = -move_speed  # Up
        elif movement == 2: vel_y = move_speed   # Down
        elif movement == 3: vel_x = -move_speed  # Left
        elif movement == 4: vel_x = move_speed   # Right
        
        self.player_vel = pygame.Vector2(vel_x, vel_y)

        if space_held and self.boost_cooldown == 0 and self.boost_timer == 0:
            self.boost_timer = self.BOOST_DURATION
            self.boost_cooldown = self.BOOST_COOLDOWN
            # SFX: Boost activate

        if shift_held and self.shield_cooldown == 0 and self.shield_timer == 0:
            self.shield_timer = self.SHIELD_DURATION
            self.shield_cooldown = self.SHIELD_COOLDOWN
            # SFX: Shield activate

    def _update_game_state(self):
        # Update player position
        self.player_pos += self.player_vel
        self.player_pos.x = np.clip(self.player_pos.x, self.PLAYER_RADIUS, self.SCREEN_WIDTH - self.PLAYER_RADIUS)
        self.player_pos.y = np.clip(self.player_pos.y, self.PLAYER_RADIUS, self.SCREEN_HEIGHT - self.PLAYER_RADIUS)

        # Update ability timers
        if self.shield_timer > 0: self.shield_timer -= 1
        if self.shield_cooldown > 0: self.shield_cooldown -= 1
        if self.boost_timer > 0:
            self.boost_timer -= 1
            if self.steps % 3 == 0: self._create_boost_trail()
        if self.boost_cooldown > 0: self.boost_cooldown -= 1

        self._update_asteroids()
        self._update_particles()

    def _update_asteroids(self):
        # Spawn new asteroids
        ramp_up_duration = 100 * self.FPS
        progress = min(1.0, self.steps / ramp_up_duration)
        initial_rate_per_step = 1.0 / self.FPS
        final_rate_per_step = 5.0 / self.FPS
        current_rate_per_step = initial_rate_per_step + (final_rate_per_step - initial_rate_per_step) * progress
        
        if self.np_random.random() < current_rate_per_step:
            self._spawn_asteroid()

        # Move and remove off-screen asteroids
        for asteroid in self.asteroids[:]:
            asteroid['pos'] += asteroid['vel']
            asteroid['angle'] = (asteroid['angle'] + asteroid['rot_speed']) % 360
            if asteroid['pos'].y > self.SCREEN_HEIGHT + asteroid['radius']:
                self.asteroids.remove(asteroid)

    def _spawn_asteroid(self):
        speed_bonus = (self.steps // 100) * 0.005
        radius = self.np_random.integers(10, 41)
        asteroid = {
            'radius': radius,
            'pos': pygame.Vector2(self.np_random.integers(0, self.SCREEN_WIDTH), -radius),
            'vel': pygame.Vector2(self.np_random.uniform(-0.5, 0.5), self.np_random.uniform(1.0, 3.0) + speed_bonus),
            'angle': 0,
            'rot_speed': self.np_random.uniform(-2.0, 2.0),
            'shape': self._generate_asteroid_shape(radius),
        }
        self.asteroids.append(asteroid)

    def _generate_asteroid_shape(self, radius):
        num_vertices = self.np_random.integers(7, 13)
        points = []
        for i in range(num_vertices):
            angle = 2 * math.pi * i / num_vertices
            dist = self.np_random.uniform(0.7, 1.0) * radius
            points.append((math.cos(angle) * dist, math.sin(angle) * dist))
        return points

    def _check_collisions(self):
        if self.shield_timer > 0:
            return
        for asteroid in self.asteroids:
            dist = self.player_pos.distance_to(asteroid['pos'])
            if dist < self.PLAYER_RADIUS + asteroid['radius'] * 0.8: # Use 80% of radius for forgiving hitbox
                self.game_over = True
                break

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _create_explosion(self, pos):
        for _ in range(50):
            life = self.np_random.integers(20, 40)
            self.particles.append({
                'pos': pos.copy(),
                'vel': pygame.Vector2(self.np_random.uniform(-4, 4), self.np_random.uniform(-4, 4)),
                'life': life, 'max_life': life,
                'color': self.COLOR_EXPLOSION,
                'radius': self.np_random.integers(2, 5)
            })

    def _create_boost_trail(self):
        life = 15
        self.particles.append({
            'pos': self.player_pos.copy() + pygame.Vector2(0, self.PLAYER_RADIUS).rotate(180),
            'vel': -self.player_vel * self.np_random.uniform(0.1, 0.3),
            'life': life, 'max_life': life,
            'color': self.COLOR_BOOST_TRAIL,
            'radius': self.np_random.integers(3, 6)
        })

    def _init_stars(self):
        self.stars = []
        for _ in range(200):
            self.stars.append({
                'pos': (self.np_random.integers(0, self.SCREEN_WIDTH), self.np_random.integers(0, self.SCREEN_HEIGHT)),
                'size': self.np_random.integers(1, 3),
                'color': (c := self.np_random.integers(100, 200), c, c)
            })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_asteroids()
        self._render_particles()
        if not self.game_over:
            self._render_player()
        self._render_ui()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_background(self):
        for star in self.stars:
            pygame.draw.circle(self.screen, star['color'], star['pos'], star['size'])

    def _render_player(self):
        points = [
            (self.player_pos.x, self.player_pos.y - self.PLAYER_RADIUS),
            (self.player_pos.x - self.PLAYER_RADIUS * 0.7, self.player_pos.y + self.PLAYER_RADIUS * 0.7),
            (self.player_pos.x + self.PLAYER_RADIUS * 0.7, self.player_pos.y + self.PLAYER_RADIUS * 0.7),
        ]
        pygame.gfxdraw.aapolygon(self.screen, [(int(p[0]), int(p[1])) for p in points], self.COLOR_PLAYER)
        pygame.gfxdraw.filled_polygon(self.screen, [(int(p[0]), int(p[1])) for p in points], self.COLOR_PLAYER)

        if self.shield_timer > 0:
            alpha = 100 + 50 * math.sin(self.steps * 0.3)
            shield_color = (*self.COLOR_SHIELD, max(0, min(255, alpha)))
            temp_surf = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            pygame.gfxdraw.aacircle(temp_surf, int(self.player_pos.x), int(self.player_pos.y), int(self.PLAYER_RADIUS * 1.5), shield_color)
            pygame.gfxdraw.filled_circle(temp_surf, int(self.player_pos.x), int(self.player_pos.y), int(self.PLAYER_RADIUS * 1.5), shield_color)
            self.screen.blit(temp_surf, (0, 0))

    def _render_asteroids(self):
        for asteroid in self.asteroids:
            rotated_points = []
            for x, y in asteroid['shape']:
                rad = math.radians(asteroid['angle'])
                new_x = x * math.cos(rad) - y * math.sin(rad) + asteroid['pos'].x
                new_y = x * math.sin(rad) + y * math.cos(rad) + asteroid['pos'].y
                rotated_points.append((int(new_x), int(new_y)))
            
            if len(rotated_points) > 2:
                darker_grey = tuple(max(0, c - 20) for c in self.COLOR_ASTEROID)
                pygame.gfxdraw.filled_polygon(self.screen, rotated_points, darker_grey)
                pygame.gfxdraw.aapolygon(self.screen, rotated_points, self.COLOR_ASTEROID)

    def _render_particles(self):
        for p in self.particles:
            alpha = p['life'] / p['max_life']
            radius = p['radius'] * alpha
            if radius > 1:
                pygame.draw.circle(self.screen, p['color'], (int(p['pos'].x), int(p['pos'].y)), int(radius))

    def _render_ui(self):
        score_text = f"SCORE: {self.score}"
        score_surf = self.font.render(score_text, True, self.COLOR_UI)
        self.screen.blit(score_surf, (10, 10))

        time_left = max(0, (self.MAX_STEPS - self.steps) / self.FPS)
        minutes, seconds = divmod(int(time_left), 60)
        timer_text = f"TIME: {minutes:02}:{seconds:02}"
        timer_surf = self.font.render(timer_text, True, self.COLOR_UI)
        timer_rect = timer_surf.get_rect(topright=(self.SCREEN_WIDTH - 10, 10))
        self.screen.blit(timer_surf, timer_rect)

        shift_text = "SHIELD"
        shift_color = self.COLOR_UI if self.shield_cooldown == 0 else self.COLOR_COOLDOWN
        if self.shield_timer > 0: shift_color = self.COLOR_SHIELD
        shift_surf = self.font.render(shift_text, True, shift_color)
        shift_rect = shift_surf.get_rect(bottomleft=(10, self.SCREEN_HEIGHT - 10))
        self.screen.blit(shift_surf, shift_rect)

        space_text = "BOOST"
        space_color = self.COLOR_UI if self.boost_cooldown == 0 else self.COLOR_COOLDOWN
        if self.boost_timer > 0: space_color = self.COLOR_BOOST_TRAIL
        space_surf = self.font.render(space_text, True, space_color)
        space_rect = space_surf.get_rect(bottomright=(self.SCREEN_WIDTH - 10, self.SCREEN_HEIGHT - 10))
        self.screen.blit(space_surf, space_rect)
    
    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")
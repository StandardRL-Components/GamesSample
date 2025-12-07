
# Generated: 2025-08-28T03:55:25.389120
# Source Brief: brief_02137.md
# Brief Index: 2137

        
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
        "Controls: Arrow keys to jump. Survive the asteroid field for 60 seconds to advance."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Pilot a space hopper through an asteroid field. Each stage gets progressively harder. Survive all three to win."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30
    PLAYER_RADIUS = 12
    JUMP_DISTANCE = 80
    JUMP_DURATION_FRAMES = 8
    JUMP_COOLDOWN_FRAMES = 5
    STAGE_DURATION_SECONDS = 60

    # --- Colors ---
    COLOR_BG = (10, 15, 30)
    COLOR_PLAYER = (50, 255, 150)
    COLOR_PLAYER_GLOW = (150, 255, 200)
    COLOR_ASTEROID = (120, 130, 140)
    COLOR_ASTEROID_OUTLINE = (90, 100, 110)
    COLOR_TEXT = (240, 240, 240)
    COLOR_TIMER_WARN = (255, 100, 100)
    COLOR_PARTICLE_JUMP = (200, 255, 255)
    COLOR_PARTICLE_EXPLOSION = [ (255, 200, 50), (255, 100, 50), (255, 50, 50) ]

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 48)
        self.font_medium = pygame.font.Font(None, 32)
        
        # Initialize state variables
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.np_random = None
        self.player_pos = None
        self.player_jump_state = None
        self.player_jump_timer = 0
        self.player_jump_start_pos = None
        self.player_jump_target_pos = None
        self.player_jump_cooldown = 0
        self.asteroids = []
        self.particles = []
        self.stars = []
        self.current_stage = 1
        self.time_left_frames = 0
        self.difficulty_timer = 0
        self.stage_difficulty_multiplier = 1.0
        self.current_max_asteroid_speed = 0
        self.current_asteroid_count = 0
        self.stage_clear_message_timer = 0
        
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.particles = []

        self._generate_stars()
        self._start_stage(1)
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()

    def _start_stage(self, stage):
        self.current_stage = stage
        self.time_left_frames = self.STAGE_DURATION_SECONDS * self.FPS
        self.stage_clear_message_timer = self.FPS * 2 # Show "STAGE X" for 2 seconds

        self.player_pos = np.array([self.WIDTH / 2, self.HEIGHT / 2], dtype=float)
        self.player_jump_state = 'idle' # 'idle', 'jumping', 'landing'
        self.player_jump_timer = 0
        self.player_jump_cooldown = 0
        
        # Difficulty scaling: 1.0x, 1.5x, 2.0x for stages 1, 2, 3
        self.stage_difficulty_multiplier = 1 + (stage - 1) * 0.5
        base_speed_pps = 30.0 # pixels per second
        base_count = 5
        
        self.current_max_asteroid_speed = base_speed_pps * self.stage_difficulty_multiplier
        self.current_asteroid_count = int(base_count * self.stage_difficulty_multiplier)
        self.difficulty_timer = 0
        
        self._generate_asteroids()

    def _generate_stars(self):
        self.stars = []
        for _ in range(150):
            self.stars.append({
                'pos': (self.np_random.integers(0, self.WIDTH), self.np_random.integers(0, self.HEIGHT)),
                'size': self.np_random.choice([1, 2, 3]),
                'brightness': self.np_random.integers(50, 150)
            })

    def _generate_asteroids(self):
        self.asteroids = []
        for _ in range(self.current_asteroid_count):
            self._add_asteroid()

    def _add_asteroid(self):
        # Spawn away from the player's starting zone
        while True:
            pos = np.array([self.np_random.uniform(0, self.WIDTH), self.np_random.uniform(0, self.HEIGHT)], dtype=float)
            if np.linalg.norm(pos - np.array([self.WIDTH / 2, self.HEIGHT / 2])) > 100:
                break
        
        angle = self.np_random.uniform(0, 2 * math.pi)
        speed = self.np_random.uniform(self.current_max_asteroid_speed * 0.5, self.current_max_asteroid_speed) / self.FPS
        vel = np.array([math.cos(angle), math.sin(angle)]) * speed
        radius = self.np_random.integers(10, 30)
        
        self.asteroids.append({'pos': pos, 'vel': vel, 'radius': radius})

    def step(self, action):
        if self.auto_advance:
            self.clock.tick(self.FPS)

        reward = 0
        terminated = self.game_over

        if not terminated:
            self.steps += 1
            reward += 0.01 # Small survival reward

            movement = action[0]
            self._handle_input(movement)

            self._update_player()
            self._update_asteroids()
            self._update_particles()
            
            terminated = self._check_collisions()
            if terminated:
                self.game_over = True
                reward = -10 # Collision penalty
                self._create_explosion(self.player_pos, 30)
                # sfx: player_explosion

            self._update_timers_and_difficulty()

            if not terminated and self.time_left_frames <= 0:
                reward += 10 # RL reward for stage clear
                self.score += 100 # Stage bonus
                
                if self.current_stage == 3:
                    self.game_over = True
                    terminated = True
                    reward += 100 # RL reward for game clear
                    self.score += 1000 # Game clear bonus
                else:
                    self._start_stage(self.current_stage + 1)
        
        if not self.game_over:
             self.score += 1 # Survival score
        
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _handle_input(self, movement):
        if self.player_jump_state == 'idle' and self.player_jump_cooldown == 0 and movement != 0:
            direction_map = { 1: np.array([0, -1]), 2: np.array([0, 1]), 3: np.array([-1, 0]), 4: np.array([1, 0]) }
            direction = direction_map.get(movement)

            if direction is not None:
                self.player_jump_state = 'jumping'
                self.player_jump_timer = self.JUMP_DURATION_FRAMES
                self.player_jump_start_pos = self.player_pos.copy()
                self.player_jump_target_pos = self.player_pos + direction * self.JUMP_DISTANCE
                # sfx: player_jump

    def _update_player(self):
        if self.player_jump_cooldown > 0:
            self.player_jump_cooldown -= 1

        if self.player_jump_state == 'jumping':
            self.player_jump_timer -= 1
            progress = 1.0 - (self.player_jump_timer / self.JUMP_DURATION_FRAMES)
            eased_progress = 1 - (1 - progress) ** 2
            self.player_pos = self.player_jump_start_pos + (self.player_jump_target_pos - self.player_jump_start_pos) * eased_progress

            if self.steps % 2 == 0:
                p_pos = self.player_pos.copy()
                p_vel = self.np_random.uniform(-0.5, 0.5, 2)
                p_life = self.np_random.integers(5, 10)
                self.particles.append({'pos': p_pos, 'vel': p_vel, 'life': p_life, 'max_life': p_life, 'color': self.COLOR_PARTICLE_JUMP, 'radius': 3})

            if self.player_jump_timer <= 0:
                self.player_pos = self.player_jump_target_pos
                self.player_jump_state = 'landing'
                self.player_jump_timer = 4 # Landing squash animation duration

        elif self.player_jump_state == 'landing':
            self.player_jump_timer -= 1
            if self.player_jump_timer <= 0:
                self.player_jump_state = 'idle'
                self.player_jump_cooldown = self.JUMP_COOLDOWN_FRAMES

        self.player_pos[0] %= self.WIDTH
        self.player_pos[1] %= self.HEIGHT
    
    def _update_asteroids(self):
        for asteroid in self.asteroids:
            asteroid['pos'] += asteroid['vel']
            asteroid['pos'][0] %= self.WIDTH
            asteroid['pos'][1] %= self.HEIGHT

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['life'] -= 1
            p['pos'] += p['vel']

    def _update_timers_and_difficulty(self):
        self.time_left_frames -= 1
        self.difficulty_timer += 1
        if self.stage_clear_message_timer > 0:
            self.stage_clear_message_timer -= 1

        if self.difficulty_timer >= 10 * self.FPS:
            self.difficulty_timer = 0
            
            speed_increase_pps = 10.0
            self.current_max_asteroid_speed += speed_increase_pps * self.stage_difficulty_multiplier
            self.current_asteroid_count = int(self.current_asteroid_count * 1.05)
            
            while len(self.asteroids) < self.current_asteroid_count:
                self._add_asteroid()

    def _check_collisions(self):
        for asteroid in self.asteroids:
            dx = abs(self.player_pos[0] - asteroid['pos'][0])
            dy = abs(self.player_pos[1] - asteroid['pos'][1])
            if dx > self.WIDTH / 2: dx = self.WIDTH - dx
            if dy > self.HEIGHT / 2: dy = self.HEIGHT - dy
            
            if math.hypot(dx, dy) < self.PLAYER_RADIUS + asteroid['radius']:
                return True
        return False

    def _create_explosion(self, position, num_particles):
        for _ in range(num_particles):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 5)
            vel = np.array([math.cos(angle), math.sin(angle)]) * speed
            life = self.np_random.integers(15, 30)
            radius = self.np_random.uniform(2, 5)
            color = self.np_random.choice(self.COLOR_PARTICLE_EXPLOSION)
            self.particles.append({'pos': position.copy(), 'vel': vel, 'life': life, 'max_life': life, 'color': color, 'radius': radius})

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        for star in self.stars:
            color_val = star['brightness']
            color = (color_val, color_val, color_val)
            pygame.draw.circle(self.screen, color, star['pos'], star['size'] / 2)

    def _render_game(self):
        for p in self.particles:
            alpha = int(255 * (p['life'] / p['max_life']))
            radius = p['radius'] * (p['life'] / p['max_life'])
            pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), int(radius), (*p['color'], alpha))

        for asteroid in self.asteroids:
            x, y, r = int(asteroid['pos'][0]), int(asteroid['pos'][1]), int(asteroid['radius'])
            for dx in [-self.WIDTH, 0, self.WIDTH]:
                for dy in [-self.HEIGHT, 0, self.HEIGHT]:
                    pygame.gfxdraw.filled_circle(self.screen, x + dx, y + dy, r, self.COLOR_ASTEROID_OUTLINE)
                    pygame.gfxdraw.filled_circle(self.screen, x + dx, y + dy, max(0, r - 2), self.COLOR_ASTEROID)

        if not self.game_over:
            x, y = int(self.player_pos[0]), int(self.player_pos[1])
            scale_x, scale_y = 1.0, 1.0
            if self.player_jump_state == 'jumping':
                p = 1.0 - (self.player_jump_timer / self.JUMP_DURATION_FRAMES)
                scale_y, scale_x = 1.0 + 0.6 * math.sin(p * math.pi), 1.0 - 0.4 * math.sin(p * math.pi)
            elif self.player_jump_state == 'landing':
                p = 1.0 - (self.player_jump_timer / 4.0)
                scale_y, scale_x = 1.0 - 0.5 * math.sin(p * math.pi), 1.0 + 0.3 * math.sin(p * math.pi)

            glow_radius = int(self.PLAYER_RADIUS * 1.5)
            glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(glow_surf, (*self.COLOR_PLAYER_GLOW, 50), (glow_radius, glow_radius), glow_radius)
            self.screen.blit(glow_surf, (x - glow_radius, y - glow_radius), special_flags=pygame.BLEND_RGBA_ADD)
            
            body_surf = pygame.Surface((self.PLAYER_RADIUS * 2 * scale_x, self.PLAYER_RADIUS * 2 * scale_y), pygame.SRCALPHA)
            pygame.draw.ellipse(body_surf, self.COLOR_PLAYER, body_surf.get_rect())
            self.screen.blit(body_surf, (x - self.PLAYER_RADIUS * scale_x, y - self.PLAYER_RADIUS * scale_y))

    def _render_ui(self):
        score_text = self.font_medium.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        stage_text = self.font_medium.render(f"STAGE: {self.current_stage}/3", True, self.COLOR_TEXT)
        self.screen.blit(stage_text, (self.WIDTH // 2 - stage_text.get_width() // 2, 10))

        time_left_sec = max(0, self.time_left_frames / self.FPS)
        timer_color = self.COLOR_TEXT if time_left_sec > 10 else self.COLOR_TIMER_WARN
        timer_text = self.font_medium.render(f"TIME: {time_left_sec:.1f}", True, timer_color)
        self.screen.blit(timer_text, (self.WIDTH - timer_text.get_width() - 10, 10))

        if self.game_over:
            msg = "YOU WIN!" if self.current_stage == 3 and self.time_left_frames <= 0 else "GAME OVER"
            end_text = self.font_large.render(msg, True, self.COLOR_TEXT)
            self.screen.blit(end_text, end_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2)))
        elif self.stage_clear_message_timer > 0:
            p = self.stage_clear_message_timer / (self.FPS * 2)
            alpha = int(255 * math.sin(p * math.pi))
            start_text = self.font_large.render(f"STAGE {self.current_stage}", True, self.COLOR_TEXT)
            start_text.set_alpha(alpha)
            self.screen.blit(start_text, start_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2)))

    def _get_info(self):
        return { "score": self.score, "steps": self.steps, "stage": self.current_stage, "time_left": max(0, self.time_left_frames / self.FPS) }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
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
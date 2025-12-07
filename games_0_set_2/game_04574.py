
# Generated: 2025-08-28T02:49:35.284594
# Source Brief: brief_04574.md
# Brief Index: 4574

        
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
        "Controls: Arrow keys to move. Hold Space to fire. Press Shift to activate shield."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Survive for 60 seconds against waves of descending alien ships in this retro top-down shooter."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30
    GAME_DURATION_SECONDS = 60

    # Colors
    COLOR_BG = (10, 5, 20)
    COLOR_PLAYER = (0, 200, 255)
    COLOR_PLAYER_DAMAGED = (255, 50, 50)
    COLOR_PLAYER_PROJECTILE = (100, 255, 100)
    COLOR_ENEMY_PROJECTILE = (255, 100, 100)
    COLOR_ALIEN_1 = (200, 200, 200)
    COLOR_ALIEN_2 = (150, 150, 150)
    COLOR_SHIELD = (100, 150, 255)
    COLOR_TEXT = (255, 255, 255)
    
    # Player
    PLAYER_MAX_HEALTH = 3
    PLAYER_SPEED = 7
    PLAYER_FIRE_COOLDOWN = 5  # Allows ~6 shots/sec
    PLAYER_PROJECTILE_SPEED = 12
    SHIELD_DURATION = 15  # 0.5s at 30fps
    SHIELD_COOLDOWN = 60 # 2s at 30fps

    # Aliens
    INITIAL_ALIEN_SPAWN_RATE = 30 # 1 per second
    INITIAL_ALIEN_PROJECTILE_SPEED = 4
    ALIEN_FIRE_RATE_MIN = 45
    ALIEN_FIRE_RATE_MAX = 90
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 64)

        self.np_random = None
        
        # Initialize state variables
        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        else:
            self.np_random = np.random.default_rng()

        self.player_pos = [self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT - 50]
        self.player_health = self.PLAYER_MAX_HEALTH
        self.player_projectiles = []
        self.player_fire_cooldown = 0
        
        self.shield_active = False
        self.shield_timer = 0
        self.shield_cooldown = 0
        
        self.aliens = []
        self.enemy_projectiles = []
        self.particles = []
        
        self.stars = [
            [self.np_random.integers(0, self.SCREEN_WIDTH), self.np_random.integers(0, self.SCREEN_HEIGHT), self.np_random.uniform(0.5, 2.0)]
            for _ in range(100)
        ]

        self.timer = self.GAME_DURATION_SECONDS
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.win = False
        
        self.alien_spawn_timer = 0
        self.alien_spawn_rate = self.INITIAL_ALIEN_SPAWN_RATE
        self.alien_projectile_speed = self.INITIAL_ALIEN_PROJECTILE_SPEED
        self.next_difficulty_increase = self.GAME_DURATION_SECONDS - 10
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_held = action[1] == 1  # Boolean
        shift_held = action[2] == 1  # Boolean
        
        # Reward for surviving each step
        reward = 0.01

        # --- Update Timers ---
        self.steps += 1
        self.timer -= 1 / self.FPS
        
        if self.player_fire_cooldown > 0: self.player_fire_cooldown -= 1
        if self.shield_timer > 0: self.shield_timer -= 1
        else: self.shield_active = False
        if self.shield_cooldown > 0: self.shield_cooldown -= 1
        
        self.alien_spawn_timer -= 1

        # --- Handle Input and Update Player ---
        self._update_player(movement, space_held, shift_held)

        # --- Update Game Objects ---
        self._update_projectiles()
        reward += self._update_aliens()
        self._update_particles()
        
        # --- Handle Collisions ---
        reward += self._handle_collisions()
        
        # --- Update Difficulty ---
        self._update_difficulty()

        # --- Check Termination ---
        terminated = self.player_health <= 0 or self.timer <= 0
        if terminated:
            self.game_over = True
            if self.player_health > 0 and self.timer <= 0:
                self.win = True
                reward += 100  # Win bonus
            else:
                reward -= 10 # Loss penalty
        
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )
    
    # --- Update Logic Sub-functions ---

    def _update_player(self, movement, space_held, shift_held):
        # Movement
        if movement == 1: self.player_pos[1] -= self.PLAYER_SPEED
        if movement == 2: self.player_pos[1] += self.PLAYER_SPEED
        if movement == 3: self.player_pos[0] -= self.PLAYER_SPEED
        if movement == 4: self.player_pos[0] += self.PLAYER_SPEED

        # Screen wrapping (horizontal) and clamping (vertical)
        self.player_pos[0] %= self.SCREEN_WIDTH
        self.player_pos[1] = np.clip(self.player_pos[1], 0, self.SCREEN_HEIGHT - 20)

        # Firing
        if space_held and self.player_fire_cooldown == 0:
            # sfx: player_shoot.wav
            self.player_projectiles.append(list(self.player_pos))
            self.player_fire_cooldown = self.PLAYER_FIRE_COOLDOWN

        # Shield
        if shift_held and self.shield_cooldown == 0:
            # sfx: shield_activate.wav
            self.shield_active = True
            self.shield_timer = self.SHIELD_DURATION
            self.shield_cooldown = self.SHIELD_COOLDOWN

    def _update_projectiles(self):
        # Player projectiles
        for p in self.player_projectiles:
            p[1] -= self.PLAYER_PROJECTILE_SPEED
        self.player_projectiles = [p for p in self.player_projectiles if p[1] > 0]

        # Enemy projectiles
        for p in self.enemy_projectiles:
            p[1] += self.alien_projectile_speed
        self.enemy_projectiles = [p for p in self.enemy_projectiles if p[1] < self.SCREEN_HEIGHT]

    def _update_aliens(self):
        # Spawn new aliens
        if self.alien_spawn_timer <= 0:
            x = self.np_random.integers(20, self.SCREEN_WIDTH - 20)
            alien_type = self.np_random.choice([1, 2], p=[0.7, 0.3])
            speed = self.np_random.uniform(1.5, 3.0) if alien_type == 1 else self.np_random.uniform(1.0, 2.0)
            self.aliens.append({
                'pos': [x, -20],
                'type': alien_type,
                'fire_cooldown': self.np_random.integers(self.ALIEN_FIRE_RATE_MIN, self.ALIEN_FIRE_RATE_MAX),
                'speed': speed,
                'base_x': x,
                'angle': 0
            })
            self.alien_spawn_timer = int(self.alien_spawn_rate)

        # Move and update existing aliens
        aliens_to_keep = []
        for alien in self.aliens:
            if alien['type'] == 1: # Straight down
                alien['pos'][1] += alien['speed']
            else: # Sine wave
                alien['pos'][1] += alien['speed']
                alien['angle'] += 0.05
                alien['pos'][0] = alien['base_x'] + math.sin(alien['angle']) * 40

            # Horizontal wrap
            alien['pos'][0] %= self.SCREEN_WIDTH
            
            # Firing
            alien['fire_cooldown'] -= 1
            if alien['fire_cooldown'] <= 0:
                # sfx: enemy_shoot.wav
                self.enemy_projectiles.append(list(alien['pos']))
                alien['fire_cooldown'] = self.np_random.integers(self.ALIEN_FIRE_RATE_MIN, self.ALIEN_FIRE_RATE_MAX)
            
            if alien['pos'][1] < self.SCREEN_HEIGHT:
                aliens_to_keep.append(alien)
        
        self.aliens = aliens_to_keep
        return 0 # No reward/penalty for aliens reaching bottom in this version

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0 and p['radius'] > 0]
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            p['radius'] -= 0.2

    def _handle_collisions(self):
        reward = 0
        
        # Player projectiles vs Aliens
        for p_proj in self.player_projectiles[:]:
            for alien in self.aliens[:]:
                if math.hypot(p_proj[0] - alien['pos'][0], p_proj[1] - alien['pos'][1]) < 15:
                    # sfx: explosion.wav
                    self._create_explosion(alien['pos'])
                    self.aliens.remove(alien)
                    if p_proj in self.player_projectiles: self.player_projectiles.remove(p_proj)
                    self.score += 1
                    reward += 1
                    break 

        # Enemy projectiles vs Player
        for e_proj in self.enemy_projectiles[:]:
            if math.hypot(e_proj[0] - self.player_pos[0], e_proj[1] - self.player_pos[1]) < 15:
                if self.shield_active:
                    # sfx: shield_block.wav
                    reward += 0.5 # Successful block reward
                    self._create_explosion(e_proj, num=5, color=(150, 200, 255))
                else:
                    # sfx: player_hit.wav
                    self.player_health -= 1
                    reward -= 5 # Hit penalty
                    self._create_explosion(self.player_pos, num=10, color=self.COLOR_PLAYER_DAMAGED)
                
                self.enemy_projectiles.remove(e_proj)

        return reward

    def _update_difficulty(self):
        if self.timer > 0 and self.timer <= self.next_difficulty_increase:
            # Increase alien projectile speed
            self.alien_projectile_speed += 0.05 * self.FPS / 3.0 # Scale with FPS, slower increase
            
            # Decrease spawn rate (making them spawn faster)
            self.alien_spawn_rate = max(10, self.alien_spawn_rate - 3)
            
            self.next_difficulty_increase -= 10
            if self.next_difficulty_increase < 0:
                self.next_difficulty_increase = -1 # Prevent re-triggering

    def _create_explosion(self, pos, num=20, color=(255, 180, 50)):
        for _ in range(num):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            self.particles.append({
                'pos': list(pos),
                'vel': vel,
                'radius': self.np_random.uniform(3, 7),
                'life': self.np_random.integers(10, 20),
                'color': color
            })

    # --- Rendering ---

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_starfield()
        self._render_particles()
        self._render_aliens()
        self._render_projectiles()
        self._render_player()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_starfield(self):
        for star in self.stars:
            star[1] += star[2]
            if star[1] > self.SCREEN_HEIGHT:
                star[0] = self.np_random.integers(0, self.SCREEN_WIDTH)
                star[1] = 0
            
            brightness = int(star[2] / 2.0 * 150)
            color = (brightness, brightness, brightness)
            pygame.draw.circle(self.screen, color, (int(star[0]), int(star[1])), 1)

    def _render_particles(self):
        for p in self.particles:
            alpha = int(255 * (p['life'] / 20))
            color = (*p['color'], alpha)
            
            radius = max(0, int(p['radius']))
            if radius == 0: continue
            
            temp_surf = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (radius, radius), radius)
            self.screen.blit(temp_surf, (int(p['pos'][0] - radius), int(p['pos'][1] - radius)))

    def _render_aliens(self):
        for alien in self.aliens:
            x, y = int(alien['pos'][0]), int(alien['pos'][1])
            color = self.COLOR_ALIEN_1 if alien['type'] == 1 else self.COLOR_ALIEN_2
            if alien['type'] == 1:
                points = [(x, y-10), (x-12, y+5), (x+12, y+5)]
                pygame.gfxdraw.aapolygon(self.screen, points, color)
                pygame.gfxdraw.filled_polygon(self.screen, points, color)
                pygame.gfxdraw.filled_circle(self.screen, x, y, 5, (255, 100, 100))
            else:
                points = [(x, y+10), (x-12, y-5), (x+12, y-5)]
                pygame.gfxdraw.aapolygon(self.screen, points, color)
                pygame.gfxdraw.filled_polygon(self.screen, points, color)
                pygame.gfxdraw.filled_circle(self.screen, x, y, 3, (255, 100, 100))

    def _render_projectiles(self):
        for p in self.player_projectiles:
            pygame.draw.line(self.screen, self.COLOR_PLAYER_PROJECTILE, (p[0], p[1]), (p[0], p[1]-10), 3)
        for p in self.enemy_projectiles:
            pygame.gfxdraw.aacircle(self.screen, int(p[0]), int(p[1]), 4, self.COLOR_ENEMY_PROJECTILE)
            pygame.gfxdraw.filled_circle(self.screen, int(p[0]), int(p[1]), 4, self.COLOR_ENEMY_PROJECTILE)

    def _render_player(self):
        x, y = int(self.player_pos[0]), int(self.player_pos[1])
        
        health_ratio = max(0, self.player_health / self.PLAYER_MAX_HEALTH)
        player_color = tuple(
            int(self.COLOR_PLAYER_DAMAGED[i] * (1 - health_ratio) + self.COLOR_PLAYER[i] * health_ratio)
            for i in range(3)
        )

        points = [(x, y - 12), (x - 10, y + 8), (x + 10, y + 8)]
        pygame.gfxdraw.aapolygon(self.screen, points, player_color)
        pygame.gfxdraw.filled_polygon(self.screen, points, player_color)
        
        pygame.gfxdraw.aacircle(self.screen, x, y, 4, (200, 255, 255))
        pygame.gfxdraw.filled_circle(self.screen, x, y, 4, (200, 255, 255))

        if self.shield_active:
            pulse = abs(math.sin(self.steps * 0.5))
            radius = int(20 + pulse * 5)
            alpha = int(50 + pulse * 50)
            
            temp_surf = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
            pygame.gfxdraw.aacircle(temp_surf, radius, radius, radius-1, (*self.COLOR_SHIELD, alpha))
            pygame.gfxdraw.filled_circle(temp_surf, radius, radius, radius-1, (*self.COLOR_SHIELD, alpha//2))
            self.screen.blit(temp_surf, (x - radius, y - radius))

    def _render_ui(self):
        score_text = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        timer_text = self.font_small.render(f"TIME: {max(0, int(self.timer))}", True, self.COLOR_TEXT)
        self.screen.blit(timer_text, (self.SCREEN_WIDTH - timer_text.get_width() - 10, 10))
        
        health_text = self.font_small.render(f"HEALTH: {self.player_health}", True, self.COLOR_TEXT)
        self.screen.blit(health_text, (10, 30))

        if self.shield_cooldown > 0:
            cooldown_ratio = self.shield_cooldown / self.SHIELD_COOLDOWN
            bar_width = 100
            pygame.draw.rect(self.screen, (50, 50, 80), (self.SCREEN_WIDTH // 2 - 50, self.SCREEN_HEIGHT - 15, bar_width, 10))
            pygame.draw.rect(self.screen, self.COLOR_SHIELD, (self.SCREEN_WIDTH // 2 - 50, self.SCREEN_HEIGHT - 15, bar_width * (1-cooldown_ratio), 10))
        
        if self.game_over:
            msg, color = ("YOU WIN!", (100, 255, 100)) if self.win else ("GAME OVER", (255, 100, 100))
            
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0,0,0,128))
            self.screen.blit(overlay, (0,0))
            
            end_text = self.font_large.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "timer": round(self.timer, 2),
            "player_health": self.player_health,
            "win": self.win
        }

    def close(self):
        pygame.quit()

# Example of how to run the environment for human play
if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Retro Space Shooter")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement, space, shift = 0, 0, 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
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

        # Pygame uses a different coordinate system, so we transpose
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Info: {info}, Total Reward: {total_reward:.2f}")
            pygame.time.wait(3000)
            total_reward = 0
            obs, info = env.reset()

        clock.tick(env.FPS)

    env.close()
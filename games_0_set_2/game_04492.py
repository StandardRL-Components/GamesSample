
# Generated: 2025-08-28T02:34:13.651875
# Source Brief: brief_04492.md
# Brief Index: 4492

        
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
        "Controls: Arrow keys to move. Hold Space to fire."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Survive waves of descending aliens in a top-down retro arcade shooter. Last for 5 minutes to win."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and timing
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30

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
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 48)
        
        # Colors
        self.COLOR_BG = (15, 15, 25)
        self.COLOR_PLAYER = (50, 255, 150)
        self.COLOR_PLAYER_THRUSTER = (255, 180, 50)
        self.COLOR_ALIEN_BASIC = (255, 80, 80)
        self.COLOR_ALIEN_ADVANCED = (80, 150, 255)
        self.COLOR_PROJECTILE_PLAYER = (255, 255, 255)
        self.COLOR_PROJECTILE_ALIEN = (255, 200, 80)
        self.COLOR_EXPLOSION = (255, 150, 0)
        self.COLOR_UI_TEXT = (220, 220, 220)
        self.COLOR_HEALTH_BAR_FG = (50, 200, 50)
        self.COLOR_HEALTH_BAR_BG = (100, 40, 40)
        
        # Game constants
        self.MAX_TIME = 300  # 5 minutes in seconds
        self.MAX_STEPS = 10000
        self.PLAYER_SPEED = 6
        self.PLAYER_SHOOT_COOLDOWN = int(0.2 * self.FPS) # 6 frames
        self.PROJECTILE_SPEED = 12
        self.PLAYER_MAX_HEALTH = 100
        self.ALIEN_BASIC_DAMAGE = 10
        self.ALIEN_ADVANCED_DAMAGE = 20
        self.ALIEN_COLLISION_DAMAGE = 30

        # Initialize state variables
        self.player_pos = None
        self.player_health = None
        self.player_shoot_timer = None
        self.aliens = None
        self.player_projectiles = None
        self.alien_projectiles = None
        self.particles = None
        self.stars = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.time_elapsed = None
        self.alien_spawn_timer = None
        self.difficulty_timer = None
        self.alien_spawn_rate = None
        self.alien_base_speed = None

        self.reset()
        
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_elapsed = 0.0

        # Player state
        self.player_pos = np.array([self.WIDTH / 2, self.HEIGHT - 50], dtype=np.float32)
        self.player_health = self.PLAYER_MAX_HEALTH
        self.player_shoot_timer = 0

        # Entity lists
        self.aliens = []
        self.player_projectiles = []
        self.alien_projectiles = []
        self.particles = []
        
        # Background stars
        self.stars = [
            (self.np_random.integers(0, self.WIDTH), self.np_random.integers(0, self.HEIGHT), self.np_random.integers(1, 3))
            for _ in range(150)
        ]

        # Difficulty and spawning
        self.alien_spawn_timer = 0
        self.difficulty_timer = 0
        self.alien_spawn_rate = 3.0 * self.FPS # 3 seconds
        self.alien_base_speed = 1.0
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]
        space_held = action[1] == 1
        
        reward = 0.01  # Small survival reward per step

        # --- Update Timers ---
        self.steps += 1
        self.time_elapsed += 1.0 / self.FPS
        self.difficulty_timer += 1
        self.alien_spawn_timer += 1
        if self.player_shoot_timer > 0:
            self.player_shoot_timer -= 1

        # --- Difficulty Scaling ---
        if self.difficulty_timer >= 30 * self.FPS: # Every 30 seconds
            self.alien_base_speed = min(3.0, self.alien_base_speed + 0.05)
            self.difficulty_timer = 0
        if self.time_elapsed > 60 and self.time_elapsed % 60 < 1/self.FPS: # Every 60 seconds
             self.alien_spawn_rate = max(0.5 * self.FPS, self.alien_spawn_rate * 0.9)

        # --- Handle Player Input ---
        if movement == 1: self.player_pos[1] -= self.PLAYER_SPEED  # Up
        if movement == 2: self.player_pos[1] += self.PLAYER_SPEED  # Down
        if movement == 3: self.player_pos[0] -= self.PLAYER_SPEED  # Left
        if movement == 4: self.player_pos[0] += self.PLAYER_SPEED  # Right
        self.player_pos[0] = np.clip(self.player_pos[0], 10, self.WIDTH - 10)
        self.player_pos[1] = np.clip(self.player_pos[1], 10, self.HEIGHT - 20)

        if space_held and self.player_shoot_timer == 0:
            # SFX: Player shoot
            self.player_projectiles.append(self.player_pos.copy() - np.array([0, 15]))
            self.player_shoot_timer = self.PLAYER_SHOOT_COOLDOWN

        # --- Update Game Entities ---
        self._update_projectiles()
        self._update_aliens()
        self._update_particles()
        self._update_stars()

        # --- Spawning ---
        if self.alien_spawn_timer >= self.alien_spawn_rate:
            self._spawn_alien()
            self.alien_spawn_timer = 0
        
        # --- Handle Collisions ---
        reward += self._handle_collisions()

        # --- Check Termination ---
        terminated = False
        if self.player_health <= 0:
            reward -= 20 # Penalty for dying
            self._create_explosion(self.player_pos, 50, 2.0)
            terminated = True
        elif self.time_elapsed >= self.MAX_TIME:
            reward += 100 # Big reward for winning
            terminated = True
        elif self.steps >= self.MAX_STEPS:
            terminated = True
        
        self.game_over = terminated
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _spawn_alien(self):
        spawn_x = self.np_random.uniform(20, self.WIDTH - 20)
        speed_multiplier = self.np_random.uniform(0.9, 1.2)
        
        if self.np_random.random() < 0.7: # 70% chance for basic
            alien_type = 'basic'
            speed = self.alien_base_speed * speed_multiplier
            shoot_cooldown = 2.0 * self.FPS
        else: # 30% chance for advanced
            alien_type = 'advanced'
            speed = self.alien_base_speed * 1.2 * speed_multiplier
            shoot_cooldown = 1.0 * self.FPS
        
        self.aliens.append({
            'pos': np.array([spawn_x, -20], dtype=np.float32),
            'type': alien_type,
            'speed': speed,
            'shoot_cooldown': int(shoot_cooldown),
            'shoot_timer': self.np_random.integers(0, int(shoot_cooldown))
        })

    def _update_projectiles(self):
        self.player_projectiles = [p for p in self.player_projectiles if p[1] > -10]
        for p in self.player_projectiles:
            p[1] -= self.PROJECTILE_SPEED
            
        self.alien_projectiles = [p for p in self.alien_projectiles if p[1] < self.HEIGHT + 10]
        for p in self.alien_projectiles:
            p[1] += self.PROJECTILE_SPEED * 0.7 # Slower than player's

    def _update_aliens(self):
        aliens_to_keep = []
        for alien in self.aliens:
            alien['pos'][1] += alien['speed']
            alien['shoot_timer'] += 1
            if alien['shoot_timer'] >= alien['shoot_cooldown']:
                # SFX: Alien shoot
                self.alien_projectiles.append(alien['pos'].copy() + np.array([0, 10]))
                alien['shoot_timer'] = 0
            
            if alien['pos'][1] < self.HEIGHT + 20:
                aliens_to_keep.append(alien)
        self.aliens = aliens_to_keep
    
    def _handle_collisions(self):
        reward = 0
        
        # Player projectiles vs aliens
        projectiles_to_keep = []
        for p_idx, p in enumerate(self.player_projectiles):
            hit = False
            for a_idx, alien in reversed(list(enumerate(self.aliens))):
                if np.linalg.norm(p - alien['pos']) < 15:
                    # SFX: Explosion
                    self._create_explosion(alien['pos'], 20, 1.0)
                    if alien['type'] == 'basic':
                        self.score += 10
                        reward += 1.0
                    else:
                        self.score += 25
                        reward += 2.5
                    
                    del self.aliens[a_idx]
                    hit = True
                    break
            if not hit:
                projectiles_to_keep.append(p)
        self.player_projectiles = projectiles_to_keep

        # Alien projectiles vs player
        projectiles_to_keep = []
        for p in self.alien_projectiles:
            if np.linalg.norm(p - self.player_pos) < 15:
                # SFX: Player hit
                self.player_health -= self.ALIEN_BASIC_DAMAGE # Assume all do basic damage for now
                reward -= self.ALIEN_BASIC_DAMAGE / 10.0 # -1 reward per 10 damage
                self._create_explosion(self.player_pos, 15, 0.5)
            else:
                projectiles_to_keep.append(p)
        self.alien_projectiles = projectiles_to_keep

        # Aliens vs player
        aliens_to_keep = []
        for alien in self.aliens:
            if np.linalg.norm(alien['pos'] - self.player_pos) < 20:
                # SFX: Big explosion, player hit hard
                self.player_health -= self.ALIEN_COLLISION_DAMAGE
                reward -= self.ALIEN_COLLISION_DAMAGE / 10.0 # -3 reward
                self._create_explosion(alien['pos'], 30, 1.5)
            else:
                aliens_to_keep.append(alien)
        self.aliens = aliens_to_keep
        
        self.player_health = max(0, self.player_health)
        return reward

    def _create_explosion(self, pos, num_particles, speed_mult):
        for _ in range(num_particles):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4) * speed_mult
            velocity = np.array([math.cos(angle) * speed, math.sin(angle) * speed])
            self.particles.append({
                'pos': pos.copy(),
                'vel': velocity,
                'life': self.np_random.uniform(0.5 * self.FPS, 1.0 * self.FPS),
                'max_life': 1.0 * self.FPS,
                'size': self.np_random.uniform(2, 6)
            })

    def _update_particles(self):
        particles_to_keep = []
        for p in self.particles:
            p['pos'] += p['vel']
            p['vel'] *= 0.95 # friction
            p['life'] -= 1
            if p['life'] > 0:
                particles_to_keep.append(p)
        self.particles = particles_to_keep
        
    def _update_stars(self):
        for i in range(len(self.stars)):
            x, y, speed = self.stars[i]
            y = (y + speed * 0.2) % self.HEIGHT
            self.stars[i] = (x, y, speed)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Stars
        for x, y, speed in self.stars:
            color_val = 50 + speed * 20
            pygame.draw.circle(self.screen, (color_val, color_val, color_val), (int(x), int(y)), speed-1)

        # Particles
        for p in self.particles:
            alpha = int(255 * (p['life'] / p['max_life']))
            color = (*self.COLOR_EXPLOSION, alpha)
            temp_surf = pygame.Surface((p['size']*2, p['size']*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (p['size'], p['size']), p['size'])
            self.screen.blit(temp_surf, (int(p['pos'][0] - p['size']), int(p['pos'][1] - p['size'])))

        # Alien Projectiles
        for p in self.alien_projectiles:
            pygame.draw.circle(self.screen, self.COLOR_PROJECTILE_ALIEN, (int(p[0]), int(p[1])), 4)
            
        # Player Projectiles
        for p in self.player_projectiles:
            pygame.draw.rect(self.screen, self.COLOR_PROJECTILE_PLAYER, (int(p[0]-2), int(p[1]), 4, 10))
            
        # Aliens
        for alien in self.aliens:
            pos_int = (int(alien['pos'][0]), int(alien['pos'][1]))
            color = self.COLOR_ALIEN_BASIC if alien['type'] == 'basic' else self.COLOR_ALIEN_ADVANCED
            pygame.draw.rect(self.screen, color, (pos_int[0]-10, pos_int[1]-10, 20, 20))
            pygame.draw.rect(self.screen, (255,255,255), (pos_int[0]-10, pos_int[1]-10, 20, 20), 1)

        # Player
        if self.player_health > 0:
            px, py = int(self.player_pos[0]), int(self.player_pos[1])
            # Thruster
            thruster_height = 10 + (self.steps % 3) * 2
            thruster_points = [(px-5, py+10), (px+5, py+10), (px, py+10+thruster_height)]
            pygame.gfxdraw.aapolygon(self.screen, thruster_points, self.COLOR_PLAYER_THRUSTER)
            pygame.gfxdraw.filled_polygon(self.screen, thruster_points, self.COLOR_PLAYER_THRUSTER)
            # Ship
            ship_points = [(px, py-15), (px-10, py+10), (px+10, py+10)]
            pygame.gfxdraw.aapolygon(self.screen, ship_points, self.COLOR_PLAYER)
            pygame.gfxdraw.filled_polygon(self.screen, ship_points, self.COLOR_PLAYER)

    def _render_ui(self):
        # Health Bar
        health_ratio = self.player_health / self.PLAYER_MAX_HEALTH
        bar_width = 200
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (10, 10, bar_width, 20))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_FG, (10, 10, int(bar_width * health_ratio), 20))
        
        # Score
        score_text = self.font_large.render(f"{self.score:06d}", True, self.COLOR_UI_TEXT)
        score_rect = score_text.get_rect(center=(self.WIDTH // 2, 22))
        self.screen.blit(score_text, score_rect)
        
        # Time
        time_left = max(0, self.MAX_TIME - self.time_elapsed)
        minutes = int(time_left // 60)
        seconds = int(time_left % 60)
        time_text = self.font_large.render(f"{minutes:02}:{seconds:02}", True, self.COLOR_UI_TEXT)
        time_rect = time_text.get_rect(topright=(self.WIDTH - 10, 8))
        self.screen.blit(time_text, time_rect)
        
        if self.game_over:
            msg = "YOU WIN!" if self.time_elapsed >= self.MAX_TIME else "GAME OVER"
            end_text = self.font_large.render(msg, True, (255, 255, 50))
            end_rect = end_text.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
            self.screen.blit(end_text, end_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_elapsed": round(self.time_elapsed, 2),
            "player_health": self.player_health,
        }

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

if __name__ == "__main__":
    # This block allows you to play the game directly
    env = GameEnv()
    obs, info = env.reset()
    
    # Use a dummy window for human play
    pygame.display.set_caption(env.game_description)
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    terminated = False
    
    while not terminated:
        # --- Action mapping for human play ---
        keys = pygame.key.get_pressed()
        movement = 0 # None
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        action = [movement, space_held, shift_held]
        
        # --- Environment step ---
        obs, reward, terminated, truncated, info = env.step(action)
        
        # --- Rendering ---
        # The observation is already a rendered frame
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Event handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
        
        # --- Tick clock ---
        env.clock.tick(env.FPS)

    print(f"Game Over! Final Info: {info}")
    env.close()
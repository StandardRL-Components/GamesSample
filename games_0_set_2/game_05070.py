
# Generated: 2025-08-28T03:53:21.841249
# Source Brief: brief_05070.md
# Brief Index: 5070

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = "Controls: Use arrow keys (↑↓←→) to move. Press Space to fire your weapon."

    # Must be a short, user-facing description of the game:
    game_description = "A fast-paced top-down shooter. Destroy waves of descending aliens while dodging their projectiles to achieve a high score."

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.width, self.height = 640, 400
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.height, self.width, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.width, self.height))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 28)
        self.font_game_over = pygame.font.Font(None, 64)

        # Game constants
        self.PLAYER_SPEED = 6
        self.PLAYER_SIZE = 12
        self.PLAYER_FIRE_COOLDOWN = 5
        self.PROJECTILE_SPEED = 10
        self.PROJECTILE_SIZE = 4
        self.ALIEN_BASE_SPEED = 1.5
        self.ALIEN_BASE_FIRE_RATE = 0.005
        self.MAX_STEPS = 2500
        self.TOTAL_ALIENS = 50

        # Colors
        self.COLOR_BG = (15, 15, 30)
        self.COLOR_PLAYER = (50, 255, 50)
        self.COLOR_PLAYER_OUTLINE = (150, 255, 150)
        self.COLOR_PLAYER_PROJECTILE = (255, 255, 255)
        self.COLOR_ALIEN_PROJECTILE = (255, 100, 255)
        self.COLOR_EXPLOSION = [(255, 255, 100), (255, 150, 50), (200, 50, 0)]
        self.COLOR_UI = (200, 200, 200)
        self.ALIEN_COLORS = {
            1: (255, 80, 80),  # Red
            2: (80, 150, 255), # Blue
            3: (255, 255, 80), # Yellow
        }
        
        # Initialize state variables
        self.player_pos = None
        self.player_lives = None
        self.player_fire_cooldown_timer = None
        self.player_invulnerability_timer = None
        
        self.aliens = None
        self.player_projectiles = None
        self.alien_projectiles = None
        self.particles = None
        self.stars = None

        self.steps = None
        self.score = None
        self.game_over = None
        
        self.aliens_killed_total = None
        self.aliens_to_spawn_total = None
        self.current_aliens_on_screen = None
        self.stage = None
        self.stage_clear_bonus_given = None
        
        self.reset()
        
        # Run validation check
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.player_pos = [self.width / 2, self.height - 40]
        self.player_lives = 3
        self.player_fire_cooldown_timer = 0
        self.player_invulnerability_timer = 0
        
        self.aliens = []
        self.player_projectiles = []
        self.alien_projectiles = []
        self.particles = []

        self.steps = 0
        self.score = 0
        self.game_over = False

        self.aliens_killed_total = 0
        self.aliens_to_spawn_total = self.TOTAL_ALIENS
        self.current_aliens_on_screen = 0
        self.stage = 1
        self.stage_clear_bonus_given = {1: False, 2: False, 3: False}

        if self.stars is None:
            self._spawn_stars()
        
        self._spawn_wave()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = 0.0
        self.clock.tick(30)
        
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        
        if movement == 0 and not space_held:
            reward -= 0.001 # Small penalty for inactivity

        self._handle_input(movement, space_held)
        self._update_projectiles()
        self._update_aliens()
        collision_reward = self._handle_collisions()
        reward += collision_reward
        self._update_particles()

        if self.current_aliens_on_screen == 0 and self.aliens_to_spawn_total > 0:
            self._spawn_wave()
        
        stage_reward = self._check_stage_clear()
        reward += stage_reward

        self.steps += 1
        terminated = self._check_termination()

        if terminated and self.aliens_killed_total >= self.TOTAL_ALIENS:
            reward += 500 # Victory bonus

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_input(self, movement, space_held):
        # Decrement timers
        self.player_fire_cooldown_timer = max(0, self.player_fire_cooldown_timer - 1)
        self.player_invulnerability_timer = max(0, self.player_invulnerability_timer - 1)

        # Movement
        if movement == 1: self.player_pos[1] -= self.PLAYER_SPEED # Up
        if movement == 2: self.player_pos[1] += self.PLAYER_SPEED # Down
        if movement == 3: self.player_pos[0] -= self.PLAYER_SPEED # Left
        if movement == 4: self.player_pos[0] += self.PLAYER_SPEED # Right

        # Clamp player position to screen bounds
        self.player_pos[0] = np.clip(self.player_pos[0], self.PLAYER_SIZE, self.width - self.PLAYER_SIZE)
        self.player_pos[1] = np.clip(self.player_pos[1], self.PLAYER_SIZE, self.height - self.PLAYER_SIZE)

        # Firing
        if space_held and self.player_fire_cooldown_timer == 0:
            # sfx: player_shoot.wav
            self.player_projectiles.append(list(self.player_pos))
            self.player_fire_cooldown_timer = self.PLAYER_FIRE_COOLDOWN

    def _update_projectiles(self):
        # Player projectiles
        for proj in self.player_projectiles[:]:
            proj[1] -= self.PROJECTILE_SPEED
            if proj[1] < 0:
                self.player_projectiles.remove(proj)
        
        # Alien projectiles
        for proj in self.alien_projectiles[:]:
            proj['pos'][1] += proj['speed']
            if proj['pos'][1] > self.height:
                self.alien_projectiles.remove(proj)

    def _update_aliens(self):
        wave_number = 1 + self.aliens_killed_total // 10
        alien_speed = self.ALIEN_BASE_SPEED + (wave_number - 1) * 0.2
        alien_fire_rate = self.ALIEN_BASE_FIRE_RATE + (wave_number - 1) * 0.002

        for alien in self.aliens[:]:
            # Movement
            if alien['type'] == 1: # Straight down
                alien['pos'][1] += alien_speed
            elif alien['type'] == 2: # Zig-zag
                alien['pos'][1] += alien_speed * 0.8
                alien['pos'][0] += alien['dir'] * alien_speed * 1.2
                if not (self.PLAYER_SIZE < alien['pos'][0] < self.width - self.PLAYER_SIZE):
                    alien['dir'] *= -1
            elif alien['type'] == 3: # Sine wave
                alien['pos'][1] += alien_speed
                alien['pos'][0] = alien['center_x'] + math.sin(alien['pos'][1] * 0.05) * 50

            # Firing
            if self.np_random.random() < alien_fire_rate:
                # sfx: alien_shoot.wav
                self.alien_projectiles.append({
                    'pos': list(alien['pos']),
                    'speed': alien_speed * 2
                })

            # Removal if off-screen
            if alien['pos'][1] > self.height + alien['size']:
                self.aliens.remove(alien)
                self.current_aliens_on_screen -= 1

    def _handle_collisions(self):
        reward = 0
        
        # Player projectiles vs aliens
        for proj in self.player_projectiles[:]:
            for alien in self.aliens[:]:
                dist = math.hypot(proj[0] - alien['pos'][0], proj[1] - alien['pos'][1])
                if dist < alien['size'] + self.PROJECTILE_SIZE:
                    # sfx: explosion.wav
                    self._create_explosion(alien['pos'], 20, 15)
                    self.aliens.remove(alien)
                    self.current_aliens_on_screen -= 1
                    if proj in self.player_projectiles: self.player_projectiles.remove(proj)
                    
                    self.score += 100
                    self.aliens_killed_total += 1
                    reward += 1.1 # +1 for kill, +0.1 for hit
                    break

        if self.player_invulnerability_timer > 0:
            return reward

        # Alien projectiles vs player
        for proj in self.alien_projectiles[:]:
            dist = math.hypot(proj['pos'][0] - self.player_pos[0], proj['pos'][1] - self.player_pos[1])
            if dist < self.PLAYER_SIZE + self.PROJECTILE_SIZE:
                self.alien_projectiles.remove(proj)
                reward += self._player_hit()
                break
        
        # Aliens vs player
        for alien in self.aliens[:]:
            dist = math.hypot(alien['pos'][0] - self.player_pos[0], alien['pos'][1] - self.player_pos[1])
            if dist < self.PLAYER_SIZE + alien['size']:
                self._create_explosion(alien['pos'], 20, 15)
                self.aliens.remove(alien)
                self.current_aliens_on_screen -= 1
                self.aliens_killed_total += 1 # Counts as a kill
                reward += self._player_hit()
                break
                
        return reward

    def _player_hit(self):
        if self.player_invulnerability_timer > 0:
            return 0
        # sfx: player_hit.wav
        self.player_lives -= 1
        self.player_invulnerability_timer = 60 # 2 seconds of invulnerability
        self._create_explosion(self.player_pos, 30, 20)
        if self.player_lives <= 0:
            self.game_over = True
        return -10.0 # Penalty for getting hit

    def _spawn_stars(self):
        self.stars = []
        for _ in range(150):
            self.stars.append({
                'pos': [self.np_random.integers(0, self.width), self.np_random.integers(0, self.height)],
                'size': self.np_random.integers(1, 3),
                'color': self.np_random.integers(50, 150)
            })

    def _spawn_wave(self):
        num_to_spawn = min(5, self.aliens_to_spawn_total)
        if num_to_spawn == 0: return

        self.aliens_to_spawn_total -= num_to_spawn
        self.current_aliens_on_screen += num_to_spawn

        if self.aliens_killed_total < 15: self.stage = 1
        elif self.aliens_killed_total < 30: self.stage = 2
        else: self.stage = 3

        for i in range(num_to_spawn):
            alien_type = self.stage
            if self.stage == 2 and self.np_random.random() < 0.5: alien_type = 1
            if self.stage == 3 and self.np_random.random() < 0.6: alien_type = self.np_random.integers(1, 3)
            
            x_pos = (i + 1) * self.width / (num_to_spawn + 1)
            self.aliens.append({
                'pos': [x_pos, self.np_random.integers(-80, -20)],
                'size': 14,
                'type': alien_type,
                'dir': 1 if self.np_random.random() > 0.5 else -1,
                'center_x': x_pos
            })

    def _check_stage_clear(self):
        reward = 0
        stage_thresholds = {1: 15, 2: 30}
        for stage, threshold in stage_thresholds.items():
            if self.aliens_killed_total >= threshold and not self.stage_clear_bonus_given[stage]:
                reward += 100
                self.stage_clear_bonus_given[stage] = True
        return reward

    def _check_termination(self):
        if self.player_lives <= 0:
            self.game_over = True
            return True
        if self.aliens_killed_total >= self.TOTAL_ALIENS:
            self.game_over = True
            return True
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            return True
        return False

    def _create_explosion(self, pos, num_particles, max_life):
        for _ in range(num_particles):
            angle = self.np_random.random() * 2 * math.pi
            speed = self.np_random.random() * 4 + 1
            self.particles.append({
                'pos': list(pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'life': max_life,
                'max_life': max_life,
                'color': random.choice(self.COLOR_EXPLOSION)
            })

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][0] *= 0.95 # friction
            p['vel'][1] *= 0.95
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_stars()
        self._render_particles()
        self._render_projectiles()
        self._render_aliens()
        if not self.game_over:
            self._render_player()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "lives": self.player_lives, "aliens_left": self.TOTAL_ALIENS - self.aliens_killed_total}

    def _render_stars(self):
        for star in self.stars:
            c = star['color']
            pygame.draw.circle(self.screen, (c, c, c), star['pos'], star['size'])

    def _render_player(self):
        if self.player_invulnerability_timer > 0 and self.steps % 4 < 2:
            return # Flicker effect when invulnerable

        p = self.player_pos
        s = self.PLAYER_SIZE
        points = [(p[0], p[1] - s), (p[0] - s / 1.5, p[1] + s / 2), (p[0] + s / 1.5, p[1] + s / 2)]
        
        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_PLAYER_OUTLINE)
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PLAYER)

    def _render_aliens(self):
        for alien in self.aliens:
            x, y = int(alien['pos'][0]), int(alien['pos'][1])
            size = int(alien['size'])
            color = self.ALIEN_COLORS[alien['type']]
            
            if alien['type'] == 1: # Diamond
                points = [(x, y - size), (x + size, y), (x, y + size), (x - size, y)]
            elif alien['type'] == 2: # Square
                points = [(x - size, y - size), (x + size, y - size), (x + size, y + size), (x - size, y + size)]
            else: # Triangle
                points = [(x, y - size), (x - size, y + size), (x + size, y + size)]
            
            pygame.gfxdraw.aapolygon(self.screen, points, color)
            pygame.gfxdraw.filled_polygon(self.screen, points, color)

    def _render_projectiles(self):
        for proj in self.player_projectiles:
            pygame.draw.rect(self.screen, self.COLOR_PLAYER_PROJECTILE, (proj[0] - self.PROJECTILE_SIZE / 2, proj[1] - self.PROJECTILE_SIZE / 2, self.PROJECTILE_SIZE, self.PROJECTILE_SIZE))
        for proj in self.alien_projectiles:
            pos = proj['pos']
            pygame.draw.rect(self.screen, self.COLOR_ALIEN_PROJECTILE, (pos[0] - self.PROJECTILE_SIZE / 2, pos[1] - self.PROJECTILE_SIZE / 2, self.PROJECTILE_SIZE, self.PROJECTILE_SIZE))

    def _render_particles(self):
        for p in self.particles:
            life_ratio = p['life'] / p['max_life']
            radius = int(life_ratio * 10)
            color = tuple(int(c * life_ratio) for c in p['color'])
            if radius > 0:
                pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), radius, color)

    def _render_ui(self):
        # Score
        score_surf = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_UI)
        self.screen.blit(score_surf, (10, 10))

        # Lives
        for i in range(self.player_lives):
            p_size = self.PLAYER_SIZE * 0.8
            x_pos = self.width - 20 - (i * (p_size * 2))
            points = [(x_pos, 15), (x_pos - p_size/1.5, 15 + p_size*1.5), (x_pos + p_size/1.5, 15 + p_size*1.5)]
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PLAYER)

        # Game Over message
        if self.game_over:
            msg = "VICTORY!" if self.aliens_killed_total >= self.TOTAL_ALIENS else "GAME OVER"
            color = self.COLOR_PLAYER if msg == "VICTORY!" else self.ALIEN_COLORS[1]
            
            over_surf = self.font_game_over.render(msg, True, color)
            over_rect = over_surf.get_rect(center=(self.width / 2, self.height / 2))
            self.screen.blit(over_surf, over_rect)

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
        assert test_obs.shape == (self.height, self.width, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.height, self.width, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.height, self.width, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Setup Pygame window for human play
    pygame.display.set_caption("Galactic Annihilator")
    screen = pygame.display.set_mode((env.width, env.height))
    
    terminated = False
    total_reward = 0
    
    # Game loop
    while not terminated:
        # --- Human Input ---
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
        # The observation is the rendered frame, so we just need to show it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True

    print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
    env.close()
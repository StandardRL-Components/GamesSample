
# Generated: 2025-08-28T02:37:49.574674
# Source Brief: brief_01749.md
# Brief Index: 1749

        
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
        "Controls: ↑↓←→ to move. Hold space to fire your weapon."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A top-down arcade shooter. Eliminate waves of descending aliens while dodging their projectiles."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and game dimensions
        self.WIDTH, self.HEIGHT = 640, 400
        
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
        
        # Fonts
        self.font_ui = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_msg = pygame.font.SysFont("monospace", 40, bold=True)

        # Colors
        self.COLOR_BG = (15, 15, 25)
        self.COLOR_PLAYER = (0, 255, 128)
        self.COLOR_PLAYER_PROJECTILE = (255, 255, 0)
        self.COLOR_ALIEN_PROJECTILE = (255, 80, 80)
        self.COLOR_ALIEN_RED = (255, 50, 50)
        self.COLOR_ALIEN_BLUE = (50, 150, 255)
        self.COLOR_ALIEN_YELLOW = (255, 255, 50)
        self.COLOR_EXPLOSION = (255, 128, 0)
        self.COLOR_UI = (200, 200, 220)
        self.COLOR_STAR = (100, 100, 120)

        # Game parameters
        self.MAX_STEPS = 10000
        self.TOTAL_ALIENS = 50
        self.PLAYER_SPEED = 6
        self.PLAYER_FIRE_COOLDOWN = 6 # frames
        self.PLAYER_PROJECTILE_SPEED = 12
        self.ALIEN_PROJECTILE_SPEEDS = {'red': 4, 'blue': 6, 'yellow': 8}
        self.INITIAL_ALIEN_FIRE_RATE = 0.01
        self.ALIEN_FIRE_RATE_INCREASE = 0.001
        
        # Etc...
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_pos = None
        self.player_lives = None
        self.player_fire_cooldown_timer = 0
        self.player_hit_timer = 0
        self.aliens = []
        self.player_projectiles = []
        self.alien_projectiles = []
        self.particles = []
        self.stars = []
        self.last_kill_step = -100 # for combo reward
        self.alien_fire_rate = self.INITIAL_ALIEN_FIRE_RATE
        self.aliens_killed = 0
        
        # Initialize state variables
        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_pos = [self.WIDTH // 2, self.HEIGHT - 40]
        self.player_lives = 3
        self.player_fire_cooldown_timer = 0
        self.player_hit_timer = 0
        
        self.aliens = []
        self.player_projectiles = []
        self.alien_projectiles = []
        self.particles = []

        self.last_kill_step = -100
        self.alien_fire_rate = self.INITIAL_ALIEN_FIRE_RATE
        self.aliens_killed = 0

        # Create starfield
        self.stars = [
            (self.np_random.integers(0, self.WIDTH), self.np_random.integers(0, self.HEIGHT), self.np_random.integers(1, 3))
            for _ in range(100)
        ]
        
        # Spawn aliens
        alien_types = ['red', 'blue', 'yellow']
        for i in range(self.TOTAL_ALIENS):
            alien_type = self.np_random.choice(alien_types)
            self.aliens.append({
                'pos': [self.np_random.integers(20, self.WIDTH - 20), self.np_random.integers(-300, -20)],
                'type': alien_type,
                'initial_x': self.np_random.integers(100, self.WIDTH - 100),
                'size': 12,
                'speed_y': self.np_random.uniform(0.5, 1.5)
            })
            
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.auto_advance:
            self.clock.tick(30)

        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_held = action[1] == 1  # Boolean
        shift_held = action[2] == 1  # Boolean
        
        # Update game logic
        reward = -0.02  # Per-step penalty

        # Store state for reward calculation
        dist_before = self._get_distance_to_nearest_alien()
        
        # Handle player actions
        self._handle_actions(movement, space_held)
        
        # Update game state
        self._update_player()
        self._update_aliens()
        reward_from_projectiles = self._update_projectiles()
        reward += reward_from_projectiles
        self._update_particles()
        self._update_starfield()

        # Handle collisions and get event-based rewards
        reward += self._handle_collisions()

        # Calculate reward for moving closer to alien
        dist_after = self._get_distance_to_nearest_alien()
        if dist_after is not None and dist_before is not None:
            if dist_after < dist_before:
                reward += 0.1

        self.steps += 1
        terminated = self._check_termination()

        # Terminal rewards
        if terminated:
            if self.player_lives <= 0:
                reward -= 100 # Lost
            elif len(self.aliens) == 0:
                reward += 100 # Won
        
        self.score += reward
        
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )
    
    def _get_distance_to_nearest_alien(self):
        if not self.aliens:
            return None
        player_x, player_y = self.player_pos
        min_dist = float('inf')
        for alien in self.aliens:
            alien_x, alien_y = alien['pos']
            dist = math.sqrt((player_x - alien_x)**2 + (player_y - alien_y)**2)
            if dist < min_dist:
                min_dist = dist
        return min_dist

    def _handle_actions(self, movement, space_held):
        # Movement
        if movement == 1: self.player_pos[1] -= self.PLAYER_SPEED # Up
        if movement == 2: self.player_pos[1] += self.PLAYER_SPEED # Down
        if movement == 3: self.player_pos[0] -= self.PLAYER_SPEED # Left
        if movement == 4: self.player_pos[0] += self.PLAYER_SPEED # Right

        # Firing
        if space_held and self.player_fire_cooldown_timer == 0:
            # SFX: player_shoot
            self.player_projectiles.append({
                'pos': [self.player_pos[0], self.player_pos[1] - 20],
                'size': (4, 12),
                'hit': False
            })
            self.player_fire_cooldown_timer = self.PLAYER_FIRE_COOLDOWN
            # Muzzle flash particle
            self.particles.append({
                'pos': [self.player_pos[0], self.player_pos[1] - 20],
                'radius': 10,
                'max_life': 3,
                'life': 3,
                'color': (255, 255, 200)
            })
    
    def _update_player(self):
        # Cooldowns
        if self.player_fire_cooldown_timer > 0:
            self.player_fire_cooldown_timer -= 1
        if self.player_hit_timer > 0:
            self.player_hit_timer -= 1

        # Boundaries
        self.player_pos[0] = np.clip(self.player_pos[0], 15, self.WIDTH - 15)
        self.player_pos[1] = np.clip(self.player_pos[1], 15, self.HEIGHT - 15)

    def _update_aliens(self):
        for alien in self.aliens[:]:
            # Movement
            if alien['type'] == 'red': # Linear
                alien['pos'][1] += alien['speed_y']
            elif alien['type'] == 'blue': # Sinusoidal
                alien['pos'][1] += alien['speed_y']
                alien['pos'][0] = alien['initial_x'] + math.sin(alien['pos'][1] * 0.02) * 100
            elif alien['type'] == 'yellow': # Fast linear
                alien['pos'][1] += alien['speed_y'] * 1.5
            
            # Alien Firing
            if alien['pos'][1] > 0 and self.np_random.random() < self.alien_fire_rate:
                # SFX: alien_shoot
                self.alien_projectiles.append({
                    'pos': list(alien['pos']),
                    'speed': self.ALIEN_PROJECTILE_SPEEDS[alien['type']],
                    'size': 6
                })

            # Despawn if off-screen
            if alien['pos'][1] > self.HEIGHT + alien['size']:
                self.aliens.remove(alien)

    def _update_projectiles(self):
        reward = 0
        # Player projectiles
        for proj in self.player_projectiles[:]:
            proj['pos'][1] -= self.PLAYER_PROJECTILE_SPEED
            if proj['pos'][1] < -proj['size'][1]:
                if not proj['hit']:
                    reward -= 0.2 # Miss penalty
                self.player_projectiles.remove(proj)
        
        # Alien projectiles
        for proj in self.alien_projectiles[:]:
            proj['pos'][1] += proj['speed']
            if proj['pos'][1] > self.HEIGHT + proj['size']:
                self.alien_projectiles.remove(proj)
        return reward

    def _update_particles(self):
        for p in self.particles[:]:
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _update_starfield(self):
        for i in range(len(self.stars)):
            x, y, speed = self.stars[i]
            y = (y + speed) % self.HEIGHT
            self.stars[i] = (x, y, speed)

    def _handle_collisions(self):
        reward = 0
        player_rect = pygame.Rect(self.player_pos[0] - 12, self.player_pos[1] - 10, 24, 20)

        # Player projectiles vs Aliens
        for proj in self.player_projectiles[:]:
            proj_rect = pygame.Rect(proj['pos'][0] - proj['size'][0]/2, proj['pos'][1], proj['size'][0], proj['size'][1])
            for alien in self.aliens[:]:
                alien_rect = pygame.Rect(alien['pos'][0] - alien['size'], alien['pos'][1] - alien['size'], alien['size']*2, alien['size']*2)
                if proj_rect.colliderect(alien_rect):
                    # SFX: alien_explosion
                    self._create_explosion(alien['pos'])
                    self.aliens.remove(alien)
                    proj['hit'] = True
                    if proj in self.player_projectiles: self.player_projectiles.remove(proj)

                    reward += 1 # Hit reward
                    self.aliens_killed += 1
                    self.alien_fire_rate += self.ALIEN_FIRE_RATE_INCREASE

                    # Combo reward
                    if self.steps - self.last_kill_step <= 5:
                        reward += 2
                        # Combo text particle
                        self.particles.append({
                            'pos': [alien['pos'][0], alien['pos'][1] - 20],
                            'text': "COMBO!",
                            'life': 20,
                            'max_life': 20,
                            'color': (255, 255, 0)
                        })

                    self.last_kill_step = self.steps
                    break

        # Alien projectiles vs Player
        for proj in self.alien_projectiles[:]:
            proj_rect = pygame.Rect(proj['pos'][0] - proj['size']/2, proj['pos'][1] - proj['size']/2, proj['size'], proj['size'])
            if player_rect.colliderect(proj_rect) and self.player_hit_timer == 0:
                # SFX: player_hit
                self.alien_projectiles.remove(proj)
                self.player_lives -= 1
                self.player_hit_timer = 60 # 2 seconds of invincibility
                self._create_explosion(self.player_pos, size_mult=1.5)
                reward -= 1 # Hit penalty
                if self.player_lives <= 0:
                    self.game_over = True
                break
        
        return reward

    def _create_explosion(self, pos, size_mult=1.0):
        num_particles = int(20 * size_mult)
        for _ in range(num_particles):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 5) * size_mult
            self.particles.append({
                'pos': list(pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'radius': self.np_random.integers(2, 5),
                'life': self.np_random.integers(15, 30),
                'max_life': 30,
                'color': self.COLOR_EXPLOSION
            })

    def _check_termination(self):
        if self.game_over:
            return True
        if self.player_lives <= 0:
            self.game_over = True
            return True
        if not self.aliens:
            self.game_over = True
            return True
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            return True
        return False

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Stars
        for x, y, speed in self.stars:
            size = speed
            pygame.draw.rect(self.screen, self.COLOR_STAR, (x, y, size, size))

        # Player
        player_color = self.COLOR_PLAYER
        if self.player_hit_timer > 0 and (self.player_hit_timer // 3) % 2 == 0:
            player_color = (255, 255, 255) # Flash white when hit
        
        px, py = int(self.player_pos[0]), int(self.player_pos[1])
        player_points = [(px, py - 15), (px - 12, py + 10), (px + 12, py + 10)]
        pygame.draw.polygon(self.screen, player_color, player_points)
        pygame.gfxdraw.aapolygon(self.screen, player_points, player_color)
        
        # Player projectiles
        for proj in self.player_projectiles:
            px, py = int(proj['pos'][0]), int(proj['pos'][1])
            w, h = proj['size']
            rect = pygame.Rect(px - w/2, py - h/2, w, h)
            pygame.draw.rect(self.screen, self.COLOR_PLAYER_PROJECTILE, rect, border_radius=2)
        
        # Aliens
        for alien in self.aliens:
            ax, ay = int(alien['pos'][0]), int(alien['pos'][1])
            size = int(alien['size'])
            color = {'red': self.COLOR_ALIEN_RED, 'blue': self.COLOR_ALIEN_BLUE, 'yellow': self.COLOR_ALIEN_YELLOW}[alien['type']]
            if alien['type'] == 'red':
                pygame.draw.rect(self.screen, color, (ax - size/2, ay - size/2, size, size))
            elif alien['type'] == 'blue':
                points = [(ax, ay - size), (ax - size, ay), (ax, ay + size), (ax + size, ay)]
                pygame.draw.polygon(self.screen, color, points)
            elif alien['type'] == 'yellow':
                pygame.draw.circle(self.screen, color, (ax, ay), int(size * 0.8))

        # Alien projectiles
        for proj in self.alien_projectiles:
            px, py = int(proj['pos'][0]), int(proj['pos'][1])
            size = int(proj['size'])
            pygame.draw.rect(self.screen, self.COLOR_ALIEN_PROJECTILE, (px - size/2, py - size/2, size, size))

        # Particles (explosions, muzzle flash, text)
        for p in self.particles:
            life_ratio = p['life'] / p['max_life']
            if 'vel' in p: # Explosion particle
                p['pos'][0] += p['vel'][0]
                p['pos'][1] += p['vel'][1]
                p['vel'][0] *= 0.95 # Damping
                p['vel'][1] *= 0.95
                alpha = int(255 * life_ratio)
                color = (p['color'][0], p['color'][1], p['color'][2], alpha)
                temp_surf = pygame.Surface((p['radius']*2, p['radius']*2), pygame.SRCALPHA)
                pygame.draw.circle(temp_surf, color, (p['radius'], p['radius']), p['radius'])
                self.screen.blit(temp_surf, (int(p['pos'][0] - p['radius']), int(p['pos'][1] - p['radius'])))
            elif 'text' in p: # Text particle
                alpha = int(255 * (life_ratio ** 0.5))
                text_surf = self.font_ui.render(p['text'], True, p['color'])
                text_surf.set_alpha(alpha)
                p['pos'][1] -= 1 # Move up
                self.screen.blit(text_surf, (int(p['pos'][0] - text_surf.get_width()/2), int(p['pos'][1])))
            else: # Simple radial particle (muzzle flash)
                radius = int(p['radius'] * life_ratio)
                if radius > 0:
                    alpha = int(255 * life_ratio)
                    color = (p['color'][0], p['color'][1], p['color'][2], alpha)
                    temp_surf = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
                    pygame.draw.circle(temp_surf, color, (radius, radius), radius)
                    self.screen.blit(temp_surf, (int(p['pos'][0] - radius), int(p['pos'][1] - radius)))

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {int(self.score)}", True, self.COLOR_UI)
        self.screen.blit(score_text, (10, 10))

        # Lives
        lives_text = self.font_ui.render(f"LIVES: {self.player_lives}", True, self.COLOR_UI)
        self.screen.blit(lives_text, (self.WIDTH - lives_text.get_width() - 10, 10))

        # Aliens remaining
        aliens_text = self.font_ui.render(f"ALIENS: {len(self.aliens)}", True, self.COLOR_UI)
        self.screen.blit(aliens_text, (self.WIDTH // 2 - aliens_text.get_width() // 2, 10))

        # Game over messages
        if self.game_over:
            if self.player_lives <= 0:
                msg = "GAME OVER"
            elif len(self.aliens) == 0:
                msg = "YOU WIN!"
            else: # Max steps
                msg = "TIME UP"
            
            msg_surf = self.font_msg.render(msg, True, (255, 255, 255))
            msg_rect = msg_surf.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
            self.screen.blit(msg_surf, msg_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.player_lives,
            "aliens_remaining": len(self.aliens),
            "aliens_killed": self.aliens_killed,
        }

    def close(self):
        pygame.quit()

# Example of how to run the environment for human play
if __name__ == '__main__':
    env = GameEnv()
    env.reset()
    
    # Use a real screen for testing
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Arcade Alien Shooter")
    
    running = True
    terminated = False
    
    while running:
        action = [0, 0, 0] # Default to no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r and terminated:
                env.reset()
                terminated = False
        
        if not terminated:
            keys = pygame.key.get_pressed()
            # Movement
            if keys[pygame.K_UP]:
                action[0] = 1
            elif keys[pygame.K_DOWN]:
                action[0] = 2
            elif keys[pygame.K_LEFT]:
                action[0] = 3
            elif keys[pygame.K_RIGHT]:
                action[0] = 4
            else:
                action[0] = 0
            
            # Fire
            if keys[pygame.K_SPACE]:
                action[1] = 1
            
            # Shift (unused in this game)
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
                action[2] = 1

            obs, reward, terminated, truncated, info = env.step(action)
            
            # For human display, we need to get the surface from the env
            # and blit it. The obs is transposed, so we need to fix it.
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            
            if terminated:
                print(f"Game Over! Final Score: {info['score']}, Steps: {info['steps']}. Press 'R' to restart.")

    env.close()
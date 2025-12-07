
# Generated: 2025-08-27T13:38:22.029073
# Source Brief: brief_00432.md
# Brief Index: 432

        
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
        "Controls: ↑↓←→ to move. Hold shift for a temporary shield. Press space to fire."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Defend Earth from three waves of descending alien invaders in this retro top-down arcade shooter."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.WIDTH, self.HEIGHT = 640, 400
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 48, bold=True)
        
        # Game constants
        self.FPS = 30
        self.MAX_STEPS = 10000
        
        # Colors
        self.COLOR_BG = (10, 5, 20)
        self.COLOR_PLAYER = (0, 255, 128)
        self.COLOR_PLAYER_BULLET = (255, 255, 0)
        self.COLOR_ALIEN_SHOT = (0, 200, 255)
        self.COLOR_SHIELD = (100, 150, 255)
        self.COLOR_WAVE_1_ALIEN = (255, 80, 80)
        self.COLOR_WAVE_2_ALIEN = (220, 100, 255)
        self.COLOR_WAVE_3_ALIEN = (255, 150, 50)
        self.COLOR_TEXT = (220, 220, 220)
        
        # Player constants
        self.PLAYER_SPEED = 8
        self.PLAYER_SHOOT_COOLDOWN = 6 # frames
        self.PLAYER_BULLET_SPEED = 15
        self.SHIELD_DURATION = 3 * self.FPS # 3 seconds
        self.SHIELD_COOLDOWN = 10 * self.FPS # 10 seconds
        self.SHIELD_RADIUS = 40
        
        # Initial state variables (will be properly set in reset)
        self.player_pos = [0, 0]
        self.player_lives = 0
        self.player_shoot_timer = 0
        self.shield_active_timer = 0
        self.shield_cooldown_timer = 0
        self.aliens = []
        self.player_bullets = []
        self.alien_shots = []
        self.particles = []
        self.stars = []
        self.current_wave = 1
        self.wave_info = {}
        self.steps = 0
        self.score = 0
        self.game_over = False

        # Initialize state
        self.reset()

        # Validate implementation
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.player_pos = [self.WIDTH // 2, self.HEIGHT - 50]
        self.player_lives = 3
        self.player_shoot_timer = 0
        self.shield_active_timer = 0
        self.shield_cooldown_timer = 0
        
        self.aliens = []
        self.player_bullets = []
        self.alien_shots = []
        self.particles = []
        
        self.current_wave = 1
        self._spawn_wave(self.current_wave)
        
        if not self.stars:
            self._create_starfield()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]
        space_held = action[1] == 1
        shift_held = action[2] == 1
        
        # Update timers
        if self.player_shoot_timer > 0: self.player_shoot_timer -= 1
        if self.shield_active_timer > 0: self.shield_active_timer -= 1
        if self.shield_cooldown_timer > 0: self.shield_cooldown_timer -= 1
            
        # Handle player actions
        reward += self._handle_player_movement(movement)
        if space_held: reward += self._handle_player_shooting()
        if shift_held: reward += self._handle_player_shield()
            
        # Update game entities
        self._update_player_bullets()
        self._update_aliens()
        self._update_alien_shots()
        self._update_particles()
        self._update_starfield()
        
        # Check for collisions and calculate rewards
        reward += self._check_collisions()
        
        # Check for wave progression
        if not self.aliens and not self.game_over:
            reward += 100  # Wave clear reward
            self.current_wave += 1
            if self.current_wave > 3:
                self.game_over = True
                reward += 500  # Game win reward
            else:
                self._spawn_wave(self.current_wave)
        
        self.steps += 1
        
        # Check termination conditions
        terminated = self.game_over or self.steps >= self.MAX_STEPS
        if self.player_lives <= 0 and not self.game_over:
            self.game_over = True
            self._create_explosion(self.player_pos, 50, 100)
            # sound: player_explosion.wav
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_starfield()
        self._render_particles()
        self._render_aliens()
        self._render_projectiles()
        if self.player_lives > 0:
            self._render_player()
        self._render_ui()
        if self.game_over:
            self._render_game_over()
            
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.current_wave,
            "lives": self.player_lives,
            "shield_cooldown": self.shield_cooldown_timer > 0
        }

    # --- Player Action Handlers ---
    def _handle_player_movement(self, movement):
        if movement == 1: self.player_pos[1] -= self.PLAYER_SPEED  # Up
        if movement == 2: self.player_pos[1] += self.PLAYER_SPEED  # Down
        if movement == 3: self.player_pos[0] -= self.PLAYER_SPEED  # Left
        if movement == 4: self.player_pos[0] += self.PLAYER_SPEED  # Right
        self.player_pos[0] = np.clip(self.player_pos[0], 20, self.WIDTH - 20)
        self.player_pos[1] = np.clip(self.player_pos[1], 20, self.HEIGHT - 20)
        return 0

    def _handle_player_shooting(self):
        if self.player_shoot_timer == 0 and self.player_lives > 0:
            self.player_shoot_timer = self.PLAYER_SHOOT_COOLDOWN
            self.player_bullets.append(list(self.player_pos))
            # sound: player_shoot.wav
            return -0.01
        return 0

    def _handle_player_shield(self):
        if self.shield_cooldown_timer == 0 and self.player_lives > 0:
            self.shield_active_timer = self.SHIELD_DURATION
            self.shield_cooldown_timer = self.SHIELD_COOLDOWN
            # sound: shield_activate.wav
            
            # Check for nearby shots for reward penalty
            is_shot_nearby = any(math.dist(self.player_pos, shot) < 150 for shot in self.alien_shots)
            if not is_shot_nearby:
                return -0.1 # Penalty for unnecessary shield use
        return 0

    # --- Game State Updates ---
    def _update_player_bullets(self):
        for bullet in self.player_bullets[:]:
            bullet[1] -= self.PLAYER_BULLET_SPEED
            if bullet[1] < 0:
                self.player_bullets.remove(bullet)

    def _update_aliens(self):
        if not self.aliens: return

        # Alien Firing Logic
        fire_prob_per_frame = (0.2 + (self.current_wave - 1) * 0.05) / self.FPS
        for alien in self.aliens:
            if self.np_random.random() < fire_prob_per_frame:
                self.alien_shots.append(list(alien['pos']))
                # sound: alien_shoot.wav
        
        # Alien Movement Logic
        if self.current_wave == 1: # Horizontal sweep
            self.wave_info['y_pos'] += 0.2
            for alien in self.aliens:
                alien['pos'][0] += self.wave_info['dir'] * 2
                alien['pos'][1] = self.wave_info['y_pos'] + alien['offset_y']
            if any(a['pos'][0] < 30 or a['pos'][0] > self.WIDTH - 30 for a in self.aliens):
                self.wave_info['dir'] *= -1
        
        elif self.current_wave == 2: # Diagonal sweep
            for alien in self.aliens:
                alien['pos'][0] += alien['dir'] * 2.5
                alien['pos'][1] += 0.7
                if alien['pos'][0] < 20 or alien['pos'][0] > self.WIDTH - 20:
                    alien['dir'] *= -1
        
        elif self.current_wave == 3: # Circular motion
            self.wave_info['center'][1] += 0.3 # Formation moves down
            self.wave_info['angle'] += 0.02 # Formation rotates
            for alien in self.aliens:
                angle = self.wave_info['angle'] + alien['angle_offset']
                alien['pos'][0] = self.wave_info['center'][0] + math.cos(angle) * alien['radius']
                alien['pos'][1] = self.wave_info['center'][1] + math.sin(angle) * alien['radius']

    def _update_alien_shots(self):
        for shot in self.alien_shots[:]:
            shot[1] += 5
            if shot[1] > self.HEIGHT:
                self.alien_shots.remove(shot)

    def _check_collisions(self):
        reward = 0
        # Player bullets vs Aliens
        for bullet in self.player_bullets[:]:
            for alien in self.aliens[:]:
                if math.dist(bullet, alien['pos']) < 15: # 15px hit radius
                    self.player_bullets.remove(bullet)
                    self.aliens.remove(alien)
                    self._create_explosion(alien['pos'], 20, 40)
                    # sound: alien_explosion.wav
                    self.score += 10
                    reward += 10
                    break
        
        if self.player_lives <= 0: return reward
        
        # Alien shots vs Player
        is_shielded = self.shield_active_timer > 0
        for shot in self.alien_shots[:]:
            if math.dist(shot, self.player_pos) < (self.SHIELD_RADIUS if is_shielded else 15):
                self.alien_shots.remove(shot)
                if is_shielded:
                    self._create_explosion(shot, 5, 10, self.COLOR_SHIELD)
                    # sound: shield_block.wav
                else:
                    self.player_lives -= 1
                    self.score = max(0, self.score - 10)
                    reward -= 10
                    self._create_explosion(self.player_pos, 30, 60)
                    # sound: player_hit.wav
        return reward

    # --- Spawning and Creation ---
    def _spawn_wave(self, wave_num):
        self.aliens.clear()
        if wave_num == 1:
            self.wave_info = {'y_pos': 60, 'dir': 1}
            for i in range(8):
                self.aliens.append({
                    'pos': [self.WIDTH // 2 - 140 + i * 40, 60],
                    'offset_y': (i % 2) * 20,
                    'color': self.COLOR_WAVE_1_ALIEN
                })
        elif wave_num == 2:
            self.wave_info = {}
            for i in range(12):
                self.aliens.append({
                    'pos': [100 + (i % 6) * 80, 50 + (i // 6) * 40],
                    'dir': 1 if (i % 2 == 0) else -1,
                    'color': self.COLOR_WAVE_2_ALIEN
                })
        elif wave_num == 3:
            self.wave_info = {'center': [self.WIDTH / 2, 120], 'angle': 0}
            for i in range(16):
                self.aliens.append({
                    'pos': [0, 0],
                    'angle_offset': (i / 16) * 2 * math.pi,
                    'radius': 80 + (i % 2) * 40,
                    'color': self.COLOR_WAVE_3_ALIEN
                })

    def _create_starfield(self):
        self.stars = []
        for _ in range(150):
            self.stars.append({
                'pos': [self.np_random.integers(0, self.WIDTH), self.np_random.integers(0, self.HEIGHT)],
                'speed': self.np_random.random() * 1.5 + 0.5,
                'size': self.np_random.integers(1, 3),
                'color': random.choice([(100,100,100), (150,150,150), (200,200,200)])
            })

    def _create_explosion(self, pos, num_particles, max_life, color=None):
        for _ in range(num_particles):
            angle = self.np_random.random() * 2 * math.pi
            speed = self.np_random.random() * 4 + 1
            self.particles.append({
                'pos': list(pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'life': self.np_random.integers(max_life // 2, max_life),
                'max_life': max_life,
                'color': color or (255, self.np_random.integers(100, 255), 0)
            })

    # --- Rendering ---
    def _render_player(self):
        x, y = int(self.player_pos[0]), int(self.player_pos[1])
        # Ship body
        points = [(x, y - 15), (x - 12, y + 10), (x + 12, y + 10)]
        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_PLAYER)
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PLAYER)
        # Engine glow
        flame_y = y + 10 + (self.steps % 4)
        flame_points = [(x - 5, y + 10), (x + 5, y + 10), (x, flame_y)]
        pygame.gfxdraw.aapolygon(self.screen, flame_points, (255, 200, 0))
        pygame.gfxdraw.filled_polygon(self.screen, flame_points, (255, 200, 0))

        # Shield effect
        if self.shield_active_timer > 0:
            alpha = 100 + int(math.sin(self.steps * 0.5) * 50)
            radius = self.SHIELD_RADIUS
            if self.shield_active_timer < 20: # Fade out effect
                alpha = int(alpha * (self.shield_active_timer / 20))
            
            temp_surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
            pygame.gfxdraw.aacircle(temp_surf, radius, radius, radius - 1, self.COLOR_SHIELD + (alpha,))
            pygame.gfxdraw.filled_circle(temp_surf, radius, radius, radius - 1, self.COLOR_SHIELD + (alpha // 4,))
            self.screen.blit(temp_surf, (x - radius, y - radius))

    def _render_aliens(self):
        for alien in self.aliens:
            x, y = int(alien['pos'][0]), int(alien['pos'][1])
            pygame.gfxdraw.aacircle(self.screen, x, y, 10, alien['color'])
            pygame.gfxdraw.filled_circle(self.screen, x, y, 10, alien['color'])
            # Add a simple "eye"
            eye_x = x + int(3 * math.sin(self.steps * 0.1))
            pygame.gfxdraw.filled_circle(self.screen, eye_x, y, 3, (255,255,255))


    def _render_projectiles(self):
        for bullet in self.player_bullets:
            x, y = int(bullet[0]), int(bullet[1])
            pygame.draw.line(self.screen, self.COLOR_PLAYER_BULLET, (x, y), (x, y - 10), 3)
        for shot in self.alien_shots:
            x, y = int(shot[0]), int(shot[1])
            pygame.gfxdraw.aacircle(self.screen, x, y, 4, self.COLOR_ALIEN_SHOT)
            pygame.gfxdraw.filled_circle(self.screen, x, y, 4, self.COLOR_ALIEN_SHOT)

    def _render_particles(self):
        for p in self.particles:
            life_ratio = p['life'] / p['max_life']
            color = p['color']
            alpha = int(255 * life_ratio)
            radius = int(5 * life_ratio)
            if radius > 0:
                pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), radius, color + (alpha,))

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1 # gravity
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _render_starfield(self):
        for star in self.stars:
            pygame.draw.circle(self.screen, star['color'], star['pos'], star['size'])
    
    def _update_starfield(self):
        for star in self.stars:
            star['pos'][1] += star['speed']
            if star['pos'][1] > self.HEIGHT:
                star['pos'] = [self.np_random.integers(0, self.WIDTH), 0]

    def _render_ui(self):
        # Score
        score_text = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        # Wave
        wave_text = self.font_small.render(f"WAVE: {self.current_wave}/3", True, self.COLOR_TEXT)
        self.screen.blit(wave_text, (self.WIDTH - wave_text.get_width() - 10, 10))
        # Lives
        for i in range(self.player_lives):
            x, y = 20 + i * 25, self.HEIGHT - 25
            points = [(x, y-5), (x-8, y), (x, y+8), (x+8, y)]
            pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_PLAYER)
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PLAYER)
        # Shield Cooldown Indicator
        if self.shield_cooldown_timer > 0:
            bar_width = 100
            fill_width = int(bar_width * (self.shield_cooldown_timer / self.SHIELD_COOLDOWN))
            pygame.draw.rect(self.screen, (50,50,80), (10, self.HEIGHT - 45, bar_width, 10))
            pygame.draw.rect(self.screen, self.COLOR_SHIELD, (10, self.HEIGHT - 45, fill_width, 10))

    def _render_game_over(self):
        s = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        s.fill((0,0,0,180))
        self.screen.blit(s, (0,0))
        
        msg = "YOU WIN!" if self.current_wave > 3 else "GAME OVER"
        text = self.font_large.render(msg, True, self.COLOR_PLAYER if self.current_wave > 3 else self.COLOR_WAVE_1_ALIEN)
        text_rect = text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
        self.screen.blit(text, text_rect)

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

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Set up a window to display the game
    pygame.display.set_caption("Galactic Defender")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    while running:
        # Player controls
        keys = pygame.key.get_pressed()
        movement = 0 # none
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}")
            # Wait for a moment before restarting
            pygame.time.wait(3000)
            obs, info = env.reset()

        # Render the observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # Cap the frame rate
        clock.tick(env.FPS)
        
    env.close()
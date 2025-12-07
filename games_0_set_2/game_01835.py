
# Generated: 2025-08-28T02:51:29.150622
# Source Brief: brief_01835.md
# Brief Index: 1835

        
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
        "Controls: ←→ to move. Press space to fire your weapon."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A top-down space shooter where you must survive 10 waves of increasingly difficult alien invaders."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Screen dimensions
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
        self.font_ui = pygame.font.Font(None, 24)
        self.font_game_over = pygame.font.Font(None, 72)

        # Colors
        self.COLOR_BG = (15, 15, 30)
        self.COLOR_STAR = (100, 100, 120)
        self.COLOR_PLAYER = (0, 255, 150)
        self.COLOR_PLAYER_DMG = (255, 50, 50)
        self.COLOR_ENEMY_1 = (255, 80, 80)
        self.COLOR_ENEMY_2 = (80, 150, 255)
        self.COLOR_ENEMY_3 = (200, 80, 255)
        self.COLOR_PLAYER_PROJ = (255, 255, 255)
        self.COLOR_ENEMY_PROJ = (255, 200, 0)
        self.COLOR_EXPLOSION = [(255, 255, 100), (255, 150, 50), (255, 50, 50)]
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_HEALTH_BAR = (0, 200, 100)
        self.COLOR_HEALTH_BAR_BG = (50, 50, 50)

        # Game constants
        self.MAX_STEPS = 10000
        self.MAX_WAVES = 10
        self.PLAYER_SPEED = 6
        self.PLAYER_FIRE_COOLDOWN = 6  # frames
        self.PLAYER_MAX_HEALTH = 100
        self.PROJECTILE_SPEED = 10

        # Initialize state variables
        self.player_pos = None
        self.player_health = None
        self.player_fire_timer = None
        self.enemies = []
        self.projectiles = []
        self.particles = []
        self.stars = []
        self.wave_number = None
        self.score = None
        self.steps = None
        self.game_over = None
        self.game_won = None

        self._generate_stars()
        
        # This will be called by the environment runner, but good for standalone validation
        # self.reset()
        
        # self.validate_implementation()


    def _generate_stars(self):
        self.stars = []
        for _ in range(150):
            self.stars.append(
                (
                    random.randint(0, self.WIDTH),
                    random.randint(0, self.HEIGHT),
                    random.choice([1, 2, 2, 3]), # size
                )
            )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.player_pos = [self.WIDTH / 2, self.HEIGHT - 50]
        self.player_health = self.PLAYER_MAX_HEALTH
        self.player_fire_timer = 0

        self.enemies = []
        self.projectiles = []
        self.particles = []

        self.wave_number = 1
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.game_won = False

        self._spawn_wave()

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0.01  # Survival reward per frame
        
        if not self.game_over:
            # --- Handle Input ---
            movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
            
            # Player Movement
            if movement == 3:  # Left
                self.player_pos[0] -= self.PLAYER_SPEED
            elif movement == 4:  # Right
                self.player_pos[0] += self.PLAYER_SPEED
            self.player_pos[0] = np.clip(self.player_pos[0], 20, self.WIDTH - 20)
            
            # Player Firing
            if space_held and self.player_fire_timer <= 0:
                self._fire_player_projectile()
                self.player_fire_timer = self.PLAYER_FIRE_COOLDOWN

            # --- Update Game State ---
            self._update_timers()
            self._update_projectiles()
            self._update_enemies()
            self._update_particles()
            
            # --- Handle Collisions ---
            collision_rewards, hits_on_player = self._handle_collisions()
            reward += collision_rewards
            if hits_on_player > 0:
                self.player_health -= 25 * hits_on_player
                if self.player_health <= 0:
                    self.game_over = True
                    self.game_won = False
                    reward = -100.0 # Large penalty for dying
                    self._create_explosion(self.player_pos, 50, 5.0)

            # --- Wave Progression ---
            if not self.enemies and not self.game_over:
                if self.wave_number == self.MAX_WAVES:
                    self.game_over = True
                    self.game_won = True
                    reward += 100.0 # Large reward for winning
                    self.score += 1000
                else:
                    reward += 10.0 # Reward for clearing a wave
                    self.score += 100
                    self.wave_number += 1
                    self._spawn_wave()

        # --- Termination Check ---
        self.steps += 1
        terminated = self.game_over or self.steps >= self.MAX_STEPS
        if self.steps >= self.MAX_STEPS and not self.game_over:
            self.game_over = True
            reward = -50.0 # Penalty for timeout

        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info(),
        )

    def _update_timers(self):
        if self.player_fire_timer > 0:
            self.player_fire_timer -= 1
        for enemy in self.enemies:
            if enemy['fire_timer'] > 0:
                enemy['fire_timer'] -= 1

    def _fire_player_projectile(self):
        # sound: player_shoot.wav
        self.projectiles.append({
            'pos': list(self.player_pos),
            'vel': [0, -self.PROJECTILE_SPEED],
            'owner': 'player',
            'size': (4, 12)
        })

    def _fire_enemy_projectile(self, pos, vel):
        # sound: enemy_shoot.wav
        self.projectiles.append({
            'pos': list(pos),
            'vel': vel,
            'owner': 'enemy',
            'size': (8, 8)
        })

    def _update_projectiles(self):
        for proj in self.projectiles[:]:
            proj['pos'][0] += proj['vel'][0]
            proj['pos'][1] += proj['vel'][1]
            if not (0 < proj['pos'][1] < self.HEIGHT):
                self.projectiles.remove(proj)

    def _update_enemies(self):
        for enemy in self.enemies:
            # Movement patterns
            if enemy['type'] == 1: # Sinusoidal
                enemy['pos'][0] += enemy['vel'][0]
                enemy['pos'][1] = enemy['start_y'] + math.sin(self.steps * enemy['freq']) * enemy['amp']
                if not (20 < enemy['pos'][0] < self.WIDTH - 20):
                    enemy['vel'][0] *= -1
            elif enemy['type'] == 2: # Zig-zag
                enemy['pos'][0] += enemy['vel'][0]
                enemy['pos'][1] += enemy['vel'][1]
                if not (20 < enemy['pos'][0] < self.WIDTH - 20):
                    enemy['vel'][0] *= -1
            elif enemy['type'] == 3: # Homing
                target_x = self.player_pos[0]
                enemy['pos'][0] += np.clip(target_x - enemy['pos'][0], -abs(enemy['vel'][0]), abs(enemy['vel'][0]))
                enemy['pos'][1] += enemy['vel'][1]

            # Firing logic
            if enemy['fire_timer'] <= 0:
                if self.np_random.random() < enemy['fire_rate']:
                    direction = np.array(self.player_pos) - np.array(enemy['pos'])
                    dist = np.linalg.norm(direction)
                    if dist > 0:
                        vel = (direction / dist) * 4 # Enemy projectile speed
                        self._fire_enemy_projectile(enemy['pos'], vel.tolist())
                    enemy['fire_timer'] = enemy['fire_cooldown']

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _handle_collisions(self):
        reward = 0
        hits_on_player = 0
        player_rect = pygame.Rect(self.player_pos[0] - 12, self.player_pos[1] - 10, 24, 20)

        for proj in self.projectiles[:]:
            proj_rect = pygame.Rect(proj['pos'][0] - proj['size'][0]/2, proj['pos'][1] - proj['size'][1]/2, *proj['size'])
            
            if proj['owner'] == 'player':
                for enemy in self.enemies[:]:
                    enemy_rect = pygame.Rect(enemy['pos'][0] - 12, enemy['pos'][1] - 12, 24, 24)
                    if enemy_rect.colliderect(proj_rect):
                        # sound: enemy_explode.wav
                        self._create_explosion(enemy['pos'], 30, 3.0)
                        self.enemies.remove(enemy)
                        if proj in self.projectiles: self.projectiles.remove(proj)
                        reward += 1.0
                        self.score += 10
                        break
            
            elif proj['owner'] == 'enemy':
                if player_rect.colliderect(proj_rect):
                    # sound: player_hit.wav
                    self._create_explosion(self.player_pos, 15, 2.0, self.COLOR_PLAYER_DMG)
                    if proj in self.projectiles: self.projectiles.remove(proj)
                    hits_on_player += 1
        
        return reward, hits_on_player

    def _spawn_wave(self):
        w = self.wave_number
        num_enemies = 8 + w * 2
        
        # Difficulty scaling
        speed_mod = 1 + w * 0.1
        fire_rate_mod = 0.005 + w * 0.002
        fire_cooldown_mod = max(30, 60 - w * 3)
        
        rows = math.ceil(num_enemies / 8)
        for i in range(num_enemies):
            row = i // 8
            col = i % 8
            
            enemy_type = 1
            if w >= 4: enemy_type = self.np_random.choice([1, 2], p=[0.5, 0.5])
            if w >= 7: enemy_type = self.np_random.choice([1, 2, 3], p=[0.2, 0.4, 0.4])

            enemy = {
                'pos': [col * 70 + 80, -30 - row * 50],
                'start_y': 60 + row * 50,
                'fire_timer': self.np_random.integers(0, 60),
                'type': enemy_type
            }

            if enemy_type == 1:
                enemy.update({
                    'vel': [1.5 * speed_mod, 0],
                    'fire_rate': fire_rate_mod,
                    'fire_cooldown': fire_cooldown_mod,
                    'freq': 0.05 * self.np_random.uniform(0.8, 1.2),
                    'amp': 30
                })
            elif enemy_type == 2:
                enemy.update({
                    'vel': [2.5 * speed_mod, 0.5 * speed_mod],
                    'fire_rate': fire_rate_mod * 1.2,
                    'fire_cooldown': fire_cooldown_mod * 0.8,
                })
            elif enemy_type == 3:
                enemy.update({
                    'vel': [1.8 * speed_mod, 0.7 * speed_mod],
                    'fire_rate': fire_rate_mod * 1.5,
                    'fire_cooldown': fire_cooldown_mod * 0.7,
                })
            
            self.enemies.append(enemy)

    def _create_explosion(self, pos, num_particles, max_speed, color=None):
        for _ in range(num_particles):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(0.5, max_speed)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            self.particles.append({
                'pos': list(pos),
                'vel': vel,
                'life': self.np_random.integers(10, 25),
                'color': random.choice(self.COLOR_EXPLOSION) if color is None else color,
                'size': self.np_random.integers(2, 5)
            })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Stars
        for x, y, size in self.stars:
            pygame.draw.rect(self.screen, self.COLOR_STAR, (x, y, size, size))

        # Particles
        for p in self.particles:
            alpha = int(255 * (p['life'] / 25.0))
            color = p['color']
            s = pygame.Surface((p['size']*2, p['size']*2), pygame.SRCALPHA)
            pygame.draw.circle(s, (*color, alpha), (p['size'], p['size']), p['size'])
            self.screen.blit(s, (int(p['pos'][0] - p['size']), int(p['pos'][1] - p['size'])))

        # Projectiles
        for proj in self.projectiles:
            if proj['owner'] == 'player':
                pygame.draw.rect(self.screen, self.COLOR_PLAYER_PROJ, (int(proj['pos'][0] - proj['size'][0]/2), int(proj['pos'][1] - proj['size'][1]/2), *proj['size']))
            else:
                pygame.gfxdraw.filled_circle(self.screen, int(proj['pos'][0]), int(proj['pos'][1]), int(proj['size'][0]/2), self.COLOR_ENEMY_PROJ)

        # Enemies
        for enemy in self.enemies:
            x, y = int(enemy['pos'][0]), int(enemy['pos'][1])
            if enemy['type'] == 1:
                pygame.draw.rect(self.screen, self.COLOR_ENEMY_1, (x - 10, y - 10, 20, 20))
            elif enemy['type'] == 2:
                points = [(x, y - 12), (x + 11, y - 4), (x + 7, y + 10), (x - 7, y + 10), (x - 11, y - 4)]
                pygame.draw.polygon(self.screen, self.COLOR_ENEMY_2, points)
            elif enemy['type'] == 3:
                points = [(x + 12, y), (x + 6, y - 10), (x - 6, y - 10), (x - 12, y), (x - 6, y + 10), (x + 6, y + 10)]
                pygame.draw.polygon(self.screen, self.COLOR_ENEMY_3, points)
        
        # Player
        if self.player_health > 0:
            health_ratio = self.player_health / self.PLAYER_MAX_HEALTH
            player_color = (
                int(self.COLOR_PLAYER_DMG[0] * (1-health_ratio) + self.COLOR_PLAYER[0] * health_ratio),
                int(self.COLOR_PLAYER_DMG[1] * (1-health_ratio) + self.COLOR_PLAYER[1] * health_ratio),
                int(self.COLOR_PLAYER_DMG[2] * (1-health_ratio) + self.COLOR_PLAYER[2] * health_ratio)
            )
            px, py = int(self.player_pos[0]), int(self.player_pos[1])
            player_points = [(px, py - 15), (px - 12, py + 10), (px + 12, py + 10)]
            pygame.draw.polygon(self.screen, player_color, player_points)
            # Engine trail
            if self.np_random.random() > 0.3:
                self.particles.append({
                    'pos': [px + self.np_random.uniform(-5, 5), py + 12],
                    'vel': [0, self.np_random.uniform(2, 4)],
                    'life': self.np_random.integers(5, 10),
                    'color': random.choice([(0,150,255), (100,200,255)]),
                    'size': self.np_random.integers(2,4)
                })

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Wave
        wave_text = self.font_ui.render(f"WAVE: {self.wave_number}/{self.MAX_WAVES}", True, self.COLOR_TEXT)
        self.screen.blit(wave_text, (self.WIDTH - wave_text.get_width() - 10, 10))

        # Health Bar
        health_bar_width = 200
        health_ratio = max(0, self.player_health / self.PLAYER_MAX_HEALTH)
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, ((self.WIDTH - health_bar_width) / 2, self.HEIGHT - 25, health_bar_width, 15))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR, ((self.WIDTH - health_bar_width) / 2, self.HEIGHT - 25, health_bar_width * health_ratio, 15))

        # Game Over / Win Message
        if self.game_over:
            msg = "YOU WIN!" if self.game_won else "GAME OVER"
            color = self.COLOR_PLAYER if self.game_won else self.COLOR_ENEMY_1
            end_text = self.font_game_over.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.wave_number,
            "health": self.player_health,
        }

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
        # Need to reset first to initialize state for rendering
        self.reset()
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
    # Set `render_mode` to "human" if you want to see the pygame window
    # Note: The provided class is built for "rgb_array", so we'll simulate a "human" mode
    
    env = GameEnv()
    env.reset()
    
    # To display the game, we need a separate pygame window
    pygame.display.set_caption("Galactic Survivor")
    display_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    running = True
    total_reward = 0
    
    # Use a default action (no-op)
    action = env.action_space.sample()
    action[0] = 0
    action[1] = 0
    action[2] = 0
    
    while running:
        # --- Human Controls ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        
        # Reset action
        action[0] = 0 # Movement
        action[1] = 0 # Space
        action[2] = 0 # Shift (unused)
        
        if keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
        
        if keys[pygame.K_SPACE]:
            action[1] = 1
        
        if keys[pygame.K_r]: # Press R to reset
            env.reset()

        # --- Gym Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # --- Rendering ---
        # The observation is already a rendered frame, so we just need to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(30) # Control the frame rate

        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            # Wait a bit before auto-resetting
            pygame.time.wait(2000)
            total_reward = 0
            obs, info = env.reset()

    env.close()
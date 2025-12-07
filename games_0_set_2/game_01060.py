
# Generated: 2025-08-27T15:44:01.138890
# Source Brief: brief_01060.md
# Brief Index: 1060

        
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
        "Controls: Arrow keys to move. Press space to fire your weapon."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Survive a relentless alien invasion for 3 minutes in a top-down shooter, "
        "blasting as many aliens as possible while dodging their projectiles."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.GAME_DURATION_SECONDS = 180
        self.MAX_STEPS = self.GAME_DURATION_SECONDS * self.FPS
        
        # Colors
        self.COLOR_BG = (10, 5, 20)
        self.COLOR_PLAYER = (0, 255, 150)
        self.COLOR_PLAYER_PROJECTILE = (200, 255, 255)
        self.COLOR_ALIEN_PROJECTILE = (255, 100, 100)
        self.COLOR_EXPLOSION = [(255, 255, 0), (255, 165, 0), (255, 69, 0)]
        self.COLOR_UI_TEXT = (220, 220, 220)
        self.COLOR_HEALTH_BAR = (0, 200, 80)
        self.COLOR_HEALTH_BAR_BG = (80, 0, 0)
        
        self.ALIEN_COLORS = [
            (255, 80, 80), (200, 80, 255), (80, 150, 255), (255, 150, 80)
        ]

        # Player settings
        self.PLAYER_SPEED = 6
        self.PLAYER_FIRE_COOLDOWN = 6 # frames (5 shots/sec)
        self.PLAYER_MAX_HEALTH = 100
        self.PLAYER_SIZE = 12

        # Projectile settings
        self.PLAYER_PROJECTILE_SPEED = 12
        self.ALIEN_PROJECTILE_SPEED = 5

        # Alien settings
        self.INITIAL_SPAWN_RATE = 1.0 # aliens per second
        
        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_timer = pygame.font.SysFont("monospace", 22, bold=True)
        
        # Initialize state variables
        self.player_pos = None
        self.player_health = None
        self.player_projectiles = None
        self.player_last_shot_step = None
        
        self.aliens = None
        self.alien_projectiles = None
        self.particles = None
        
        self.steps = None
        self.score = None
        self.game_over = None
        
        self.spawn_timer = None
        self.current_spawn_rate = None
        self.difficulty_multiplier = None
        
        self.reset()
        
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.player_pos = pygame.Vector2(self.WIDTH / 2, self.HEIGHT - 40)
        self.player_health = self.PLAYER_MAX_HEALTH
        self.player_projectiles = []
        self.player_last_shot_step = -self.PLAYER_FIRE_COOLDOWN
        
        self.aliens = []
        self.alien_projectiles = []
        self.particles = []
        
        self.current_spawn_rate = self.INITIAL_SPAWN_RATE
        self.spawn_timer = 0
        self.difficulty_multiplier = 1.0
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0.01  # Small reward for surviving a step
        
        # --- Handle Input ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # Player Movement
        if movement == 1: self.player_pos.y -= self.PLAYER_SPEED
        if movement == 2: self.player_pos.y += self.PLAYER_SPEED
        if movement == 3: self.player_pos.x -= self.PLAYER_SPEED
        if movement == 4: self.player_pos.x += self.PLAYER_SPEED
        
        # Player Firing
        if space_held and (self.steps - self.player_last_shot_step) >= self.PLAYER_FIRE_COOLDOWN:
            # sfx: player_shoot.wav
            proj_pos = self.player_pos + pygame.Vector2(0, -self.PLAYER_SIZE)
            self.player_projectiles.append(pygame.Rect(proj_pos.x - 2, proj_pos.y, 4, 10))
            self.player_last_shot_step = self.steps

        # --- Update Game State ---
        self._update_player()
        self._update_projectiles()
        self._update_aliens()
        self._update_particles()
        
        # --- Spawning and Difficulty ---
        self._spawn_aliens()
        self._update_difficulty()
        
        # --- Collisions and Rewards ---
        reward += self._handle_collisions()
        
        # --- Termination Check ---
        self.steps += 1
        terminated = self.player_health <= 0 or self.steps >= self.MAX_STEPS
        
        if terminated:
            if self.player_health > 0:
                reward += 100 # Win bonus
            else:
                reward -= 100 # Loss penalty
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _update_player(self):
        self.player_pos.x = np.clip(self.player_pos.x, self.PLAYER_SIZE, self.WIDTH - self.PLAYER_SIZE)
        self.player_pos.y = np.clip(self.player_pos.y, self.PLAYER_SIZE, self.HEIGHT - self.PLAYER_SIZE)

    def _update_projectiles(self):
        # Player projectiles
        self.player_projectiles = [p for p in self.player_projectiles if p.bottom > 0]
        for p in self.player_projectiles:
            p.y -= self.PLAYER_PROJECTILE_SPEED
            
        # Alien projectiles
        self.alien_projectiles = [p for p in self.alien_projectiles if p.top < self.HEIGHT]
        for p in self.alien_projectiles:
            p.y += self.ALIEN_PROJECTILE_SPEED * self.difficulty_multiplier

    def _update_aliens(self):
        aliens_to_keep = []
        for alien in self.aliens:
            # Alien Movement
            pattern = alien['pattern']
            t = self.steps - alien['spawn_step']
            
            if pattern == 0: # Sinusoidal
                alien['pos'].x = alien['start_x'] + math.sin(t * 0.05 * self.difficulty_multiplier) * alien['amplitude']
                alien['pos'].y += alien['speed'] * self.difficulty_multiplier
            elif pattern == 1: # Diagonal Bounce
                alien['pos'] += alien['vel'] * self.difficulty_multiplier
                if alien['pos'].x <= alien['size'] or alien['pos'].x >= self.WIDTH - alien['size']:
                    alien['vel'].x *= -1
            elif pattern == 2: # Circular
                angle = alien['start_angle'] + t * 0.04 * self.difficulty_multiplier
                alien['pos'].x = alien['center_x'] + math.cos(angle) * alien['amplitude']
                alien['pos'].y = alien['center_y'] + math.sin(angle) * alien['amplitude']
            elif pattern == 3: # Homing (slowly)
                direction = (self.player_pos - alien['pos']).normalize()
                alien['pos'] += direction * alien['speed'] * self.difficulty_multiplier * 0.6

            # Alien Firing
            if (self.steps - alien['last_shot_step']) >= alien['fire_rate']:
                # sfx: alien_shoot.wav
                proj_pos = alien['pos'].copy()
                self.alien_projectiles.append(pygame.Rect(proj_pos.x - 2, proj_pos.y, 4, 8))
                alien['last_shot_step'] = self.steps + self.np_random.integers(-15, 15)

            if alien['pos'].y < self.HEIGHT + alien['size']:
                aliens_to_keep.append(alien)
        self.aliens = aliens_to_keep

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1
            p['size'] = max(0, p['size'] - 0.2)

    def _spawn_aliens(self):
        self.spawn_timer += self.current_spawn_rate / self.FPS
        if self.spawn_timer >= 1:
            self.spawn_timer -= 1
            
            size = self.np_random.integers(10, 16)
            start_x = self.np_random.uniform(size, self.WIDTH - size)
            start_y = -size
            pattern = self.np_random.integers(0, 4)
            color = self.np_random.choice(self.ALIEN_COLORS)
            
            alien = {
                'pos': pygame.Vector2(start_x, start_y),
                'size': size,
                'color': color,
                'pattern': pattern,
                'spawn_step': self.steps,
                'last_shot_step': self.steps,
                'fire_rate': self.np_random.integers(60, 120) / self.difficulty_multiplier,
            }
            
            if pattern == 0: # Sinusoidal
                alien['start_x'] = start_x
                alien['speed'] = self.np_random.uniform(1.0, 2.0)
                alien['amplitude'] = self.np_random.uniform(50, 150)
            elif pattern == 1: # Diagonal Bounce
                alien['vel'] = pygame.Vector2(self.np_random.choice([-1, 1]), 0.5) * self.np_random.uniform(2.0, 3.0)
            elif pattern == 2: # Circular
                alien['center_x'] = start_x
                alien['center_y'] = self.np_random.uniform(80, 150)
                alien['amplitude'] = self.np_random.uniform(40, 100)
                alien['start_angle'] = math.atan2(start_y - alien['center_y'], start_x - alien['center_x'])
            elif pattern == 3: # Homing
                alien['speed'] = self.np_random.uniform(1.0, 1.5)

            self.aliens.append(alien)

    def _update_difficulty(self):
        if self.steps > 0 and self.steps % (30 * self.FPS) == 0:
            self.difficulty_multiplier += 0.05
            self.current_spawn_rate += 0.1

    def _handle_collisions(self):
        reward = 0
        
        # Player projectiles vs Aliens
        aliens_to_keep = []
        for alien in self.aliens:
            hit = False
            alien_rect = pygame.Rect(alien['pos'].x - alien['size'], alien['pos'].y - alien['size'], alien['size']*2, alien['size']*2)
            for proj in self.player_projectiles:
                if alien_rect.colliderect(proj):
                    # sfx: explosion.wav
                    self._create_explosion(alien['pos'], alien['color'], 20)
                    self.player_projectiles.remove(proj)
                    self.score += 1
                    reward += 1
                    hit = True
                    break
            if not hit:
                aliens_to_keep.append(alien)
        self.aliens = aliens_to_keep
        
        # Alien projectiles vs Player
        player_rect = pygame.Rect(self.player_pos.x - self.PLAYER_SIZE, self.player_pos.y - self.PLAYER_SIZE, self.PLAYER_SIZE*2, self.PLAYER_SIZE*2)
        for proj in self.alien_projectiles:
            if player_rect.colliderect(proj):
                # sfx: player_hit.wav
                self.alien_projectiles.remove(proj)
                self.player_health -= 10
                self._create_explosion(self.player_pos, self.COLOR_PLAYER, 10)
                break
        
        self.player_health = max(0, self.player_health)
        
        return reward

    def _create_explosion(self, pos, color, num_particles):
        for _ in range(num_particles):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 5)
            self.particles.append({
                'pos': pos.copy(),
                'vel': pygame.Vector2(math.cos(angle), math.sin(angle)) * speed,
                'life': self.np_random.integers(15, 30),
                'color': self.np_random.choice(self.COLOR_EXPLOSION),
                'size': self.np_random.uniform(3, 7)
            })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Render particles
        for p in self.particles:
            pygame.draw.circle(self.screen, p['color'], p['pos'], max(0, int(p['size'] * (p['life']/30))))

        # Render alien projectiles
        for p in self.alien_projectiles:
            pygame.draw.rect(self.screen, self.COLOR_ALIEN_PROJECTILE, p)

        # Render player projectiles
        for p in self.player_projectiles:
            pygame.draw.rect(self.screen, self.COLOR_PLAYER_PROJECTILE, p)
            # Add a glow effect
            glow_rect = p.inflate(4, 4)
            s = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
            pygame.draw.rect(s, (*self.COLOR_PLAYER_PROJECTILE, 60), s.get_rect(), border_radius=2)
            self.screen.blit(s, glow_rect.topleft)

        # Render aliens
        for alien in self.aliens:
            pos_int = (int(alien['pos'].x), int(alien['pos'].y))
            size = int(alien['size'])
            if alien['pattern'] % 2 == 0: # Circle for patterns 0, 2
                pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], size, alien['color'])
                pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], size, alien['color'])
            else: # Diamond for patterns 1, 3
                points = [
                    (pos_int[0], pos_int[1] - size),
                    (pos_int[0] + size, pos_int[1]),
                    (pos_int[0], pos_int[1] + size),
                    (pos_int[0] - size, pos_int[1]),
                ]
                pygame.gfxdraw.filled_polygon(self.screen, points, alien['color'])
                pygame.gfxdraw.aapolygon(self.screen, points, alien['color'])

        # Render player
        player_points = [
            (self.player_pos.x, self.player_pos.y - self.PLAYER_SIZE),
            (self.player_pos.x - self.PLAYER_SIZE * 0.8, self.player_pos.y + self.PLAYER_SIZE * 0.8),
            (self.player_pos.x + self.PLAYER_SIZE * 0.8, self.player_pos.y + self.PLAYER_SIZE * 0.8),
        ]
        pygame.gfxdraw.filled_polygon(self.screen, player_points, self.COLOR_PLAYER)
        pygame.gfxdraw.aapolygon(self.screen, player_points, self.COLOR_PLAYER)

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (self.WIDTH - score_text.get_width() - 10, 10))
        
        # Health Bar
        health_pct = self.player_health / self.PLAYER_MAX_HEALTH
        bar_width = 150
        bar_height = 15
        fill_width = int(bar_width * health_pct)
        
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (10, 10, bar_width, bar_height))
        if fill_width > 0:
            pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR, (10, 10, fill_width, bar_height))
        
        # Timer
        time_left_secs = (self.MAX_STEPS - self.steps) / self.FPS
        minutes = int(time_left_secs // 60)
        seconds = int(time_left_secs % 60)
        timer_text = self.font_timer.render(f"{minutes:02}:{seconds:02}", True, self.COLOR_UI_TEXT)
        self.screen.blit(timer_text, (self.WIDTH / 2 - timer_text.get_width() / 2, 8))
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "health": self.player_health,
            "time_remaining_steps": self.MAX_STEPS - self.steps,
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

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv()
    obs, info = env.reset()
    
    # Setup Pygame window for human play
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Alien Invasion")
    clock = pygame.time.Clock()
    
    terminated = False
    total_reward = 0
    
    while not terminated:
        # --- Human Input ---
        movement = 0 # none
        space_held = 0
        shift_held = 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1
            
        action = [movement, space_held, shift_held]
        
        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # --- Render to screen ---
        # The observation is already a rendered frame, just need to show it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Print info ---
        if env.steps % env.FPS == 0:
            print(f"Step: {info['steps']}, Score: {info['score']}, Health: {info['health']}, Total Reward: {total_reward:.2f}")

        # Control framerate
        clock.tick(env.FPS)
        
    print("\n--- GAME OVER ---")
    print(f"Final Score: {info['score']}")
    print(f"Total Reward: {total_reward:.2f}")
    print(f"Survived for {info['steps']} steps.")
    
    env.close()
    pygame.quit()
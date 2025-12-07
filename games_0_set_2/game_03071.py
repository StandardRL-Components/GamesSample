
# Generated: 2025-08-28T06:55:10.423814
# Source Brief: brief_03071.md
# Brief Index: 3071

        
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

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ←→ to move. Press space to fire your weapon."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A minimalist, procedurally generated top-down space shooter. "
        "Destroy 5 waves of descending aliens to win."
    )

    # Frames auto-advance at 30fps.
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_STEPS = 5000
        self.NUM_WAVES = 5

        # Colors
        self.COLOR_BG = (10, 15, 30)
        self.COLOR_PLAYER = (50, 255, 150)
        self.COLOR_PLAYER_GLOW = (50, 255, 150, 50)
        self.COLOR_ALIEN_BASE = (255, 80, 80)
        self.COLOR_PLAYER_PROJ = (150, 255, 255)
        self.COLOR_ALIEN_PROJ = (255, 255, 100)
        self.COLOR_PARTICLE_1 = (255, 200, 50)
        self.COLOR_PARTICLE_2 = (255, 100, 50)
        self.COLOR_UI_TEXT = (220, 220, 240)

        # Gymnasium spaces
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_title = pygame.font.SysFont("Consolas", 48, bold=True)
        
        # Game state variables (initialized in reset)
        self.steps = None
        self.score = None
        self.game_over = None
        self.player_pos = None
        self.player_health = None
        self.player_fire_cooldown = None
        self.current_wave = None
        self.aliens = None
        self.player_projectiles = None
        self.alien_projectiles = None
        self.particles = None
        self.stars = None
        self.alien_projectile_speed = None
        self.alien_fire_rate_multiplier = None

        # Initialize state
        self.reset()

        # Run validation check
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        # Player state
        self.player_pos = [self.WIDTH // 2, self.HEIGHT - 40]
        self.player_health = 3
        self.player_fire_cooldown = 0
        
        # Game progression
        self.current_wave = 1
        
        # Entity lists
        self.aliens = []
        self.player_projectiles = []
        self.alien_projectiles = []
        self.particles = []
        
        # Generate starfield background
        self.stars = []
        for _ in range(150):
            x = self.np_random.uniform(0, self.WIDTH)
            y = self.np_random.uniform(0, self.HEIGHT)
            speed = self.np_random.uniform(0.1, 0.5)
            size = self.np_random.uniform(0.5, 1.5)
            self.stars.append([x, y, speed, size])

        # Spawn first wave
        self._spawn_wave()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = -0.01  # Small penalty for time to encourage speed

        if self.game_over:
            # If the game has ended, do not update the state.
            # Just return the final observation and a reward of 0.
            return (
                self._get_observation(), 0, True, False, self._get_info()
            )

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_held = action[1] == 1
        
        # Handle player actions
        self._handle_input(movement, space_held)

        # Update game state
        self._update_player()
        self._update_projectiles()
        self._update_aliens()
        self._update_particles()
        
        # Handle collisions and calculate rewards
        collision_reward = self._handle_collisions()
        reward += collision_reward

        # Check for game status changes (wave clear, win/loss)
        status_reward, terminated = self._check_game_status()
        reward += status_reward
        self.game_over = terminated
        
        # Update step counter
        self.steps += 1
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            reward -= 100 # Penalty for running out of time

        return (
            self._get_observation(),
            reward,
            self.game_over,
            False,  # truncated is always False
            self._get_info()
        )

    def _handle_input(self, movement, space_held):
        # Player movement
        player_speed = 8
        if movement == 3:  # Left
            self.player_pos[0] -= player_speed
        elif movement == 4:  # Right
            self.player_pos[0] += player_speed
        
        # Clamp player position to screen bounds
        self.player_pos[0] = max(20, min(self.WIDTH - 20, self.player_pos[0]))

        # Player firing
        if space_held and self.player_fire_cooldown <= 0:
            # Sfx: Player shoot
            self.player_projectiles.append(list(self.player_pos))
            self.player_fire_cooldown = 6  # 5 shots per second

    def _update_player(self):
        if self.player_fire_cooldown > 0:
            self.player_fire_cooldown -= 1

    def _update_projectiles(self):
        # Player projectiles
        for proj in self.player_projectiles[:]:
            proj[1] -= 12  # Move up
            if proj[1] < 0:
                self.player_projectiles.remove(proj)
        
        # Alien projectiles
        for proj in self.alien_projectiles[:]:
            proj[1] += self.alien_projectile_speed  # Move down
            if proj[1] > self.HEIGHT:
                self.alien_projectiles.remove(proj)

    def _update_aliens(self):
        for alien in self.aliens[:]:
            # Movement
            alien['pos'][1] += alien['v_speed']
            if alien['move_pattern'] == 'sinusoid':
                alien['pos'][0] = alien['origin_x'] + math.sin(self.steps * 0.05 + alien['phase']) * alien['amplitude']
            
            # Firing
            alien['fire_cooldown'] -= 1
            if alien['fire_cooldown'] <= 0:
                if self.np_random.random() < alien['fire_prob']:
                    # Sfx: Alien shoot
                    self.alien_projectiles.append(list(alien['pos']))
                alien['fire_cooldown'] = int(self.FPS * (1 / alien['fire_rate']))

            # Check if alien reached bottom
            if alien['pos'][1] > self.HEIGHT - 20:
                self.player_health = 0 # Instant loss

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _handle_collisions(self):
        reward = 0
        player_rect = pygame.Rect(self.player_pos[0] - 10, self.player_pos[1] - 8, 20, 16)

        # Player projectiles vs aliens
        for proj in self.player_projectiles[:]:
            proj_rect = pygame.Rect(proj[0] - 2, proj[1] - 8, 4, 16)
            for alien in self.aliens[:]:
                alien_rect = pygame.Rect(alien['pos'][0] - 12, alien['pos'][1] - 10, 24, 20)
                if proj_rect.colliderect(alien_rect):
                    # Sfx: Explosion
                    self._create_explosion(alien['pos'], 20, self.COLOR_PARTICLE_1)
                    self.aliens.remove(alien)
                    if proj in self.player_projectiles: self.player_projectiles.remove(proj)
                    self.score += 10
                    reward += 0.1
                    break
        
        # Alien projectiles vs player
        for proj in self.alien_projectiles[:]:
            proj_rect = pygame.Rect(proj[0] - 2, proj[1] - 4, 4, 8)
            if player_rect.colliderect(proj_rect):
                # Sfx: Player hit
                self.alien_projectiles.remove(proj)
                self.player_health -= 1
                self._create_explosion(self.player_pos, 30, self.COLOR_PLAYER)
                break
        
        # Aliens vs player
        for alien in self.aliens[:]:
            alien_rect = pygame.Rect(alien['pos'][0] - 12, alien['pos'][1] - 10, 24, 20)
            if player_rect.colliderect(alien_rect):
                # Sfx: Player hit & Explosion
                self.aliens.remove(alien)
                self.player_health -= 1
                self._create_explosion(self.player_pos, 30, self.COLOR_PLAYER)
                self._create_explosion(alien['pos'], 20, self.COLOR_PARTICLE_1)
                break
                
        return reward

    def _check_game_status(self):
        reward = 0
        terminated = False

        # Player loss
        if self.player_health <= 0:
            reward = -100
            terminated = True
            return reward, terminated

        # Wave completion
        if not self.aliens and not self.game_over:
            self.score += 100
            reward += 1.0
            self.current_wave += 1
            
            if self.current_wave > self.NUM_WAVES:
                # Player win
                reward += 100
                terminated = True
            else:
                # Spawn next wave
                self._spawn_wave()
        
        return reward, terminated

    def _spawn_wave(self):
        num_aliens_x = 8
        num_aliens_y = 3
        
        # Difficulty scaling
        self.alien_projectile_speed = 3.0 + (self.current_wave - 1) * 0.5
        base_fire_rate = 0.5 + (self.current_wave - 1) * 0.2
        
        wave_config = {
            1: {'pattern': 'static', 'v_speed': 0.3, 'rows': 2},
            2: {'pattern': 'static', 'v_speed': 0.4, 'rows': 3},
            3: {'pattern': 'sinusoid', 'v_speed': 0.3, 'rows': 3},
            4: {'pattern': 'sinusoid', 'v_speed': 0.4, 'rows': 4},
            5: {'pattern': 'sinusoid', 'v_speed': 0.5, 'rows': 4, 'fire_rate_mult': 1.5},
        }.get(self.current_wave, {})

        num_aliens_y = wave_config.get('rows', 3)
        fire_rate_mult = wave_config.get('fire_rate_mult', 1.0)
        
        for i in range(num_aliens_x):
            for j in range(num_aliens_y):
                x = self.WIDTH * 0.2 + i * (self.WIDTH * 0.6 / (num_aliens_x -1))
                y = 60 + j * 40
                
                alien = {
                    'pos': [x, y],
                    'origin_x': x,
                    'v_speed': wave_config.get('v_speed', 0.3),
                    'move_pattern': wave_config.get('pattern', 'static'),
                    'phase': self.np_random.uniform(0, 2 * math.pi),
                    'amplitude': 40,
                    'fire_rate': base_fire_rate * fire_rate_mult,
                    'fire_prob': 0.01,
                    'fire_cooldown': self.np_random.integers(0, self.FPS * 2),
                }
                self.aliens.append(alien)
                
    def _create_explosion(self, pos, num_particles, color):
        for _ in range(num_particles):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            life = self.np_random.integers(15, 30)
            self.particles.append({'pos': list(pos), 'vel': vel, 'life': life, 'color': color})

    def _get_observation(self):
        # Clear screen
        self.screen.fill(self.COLOR_BG)
        
        # Render game elements
        self._render_starfield()
        self._render_particles()
        self._render_projectiles()
        self._render_aliens()
        self._render_player()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_starfield(self):
        for star in self.stars:
            star[1] = (star[1] + star[2]) % self.HEIGHT
            color_val = int(100 + star[2] * 100)
            color = (color_val, color_val, color_val)
            pygame.draw.circle(self.screen, color, (int(star[0]), int(star[1])), star[3])

    def _render_player(self):
        if self.player_health > 0:
            x, y = int(self.player_pos[0]), int(self.player_pos[1])
            # Ship body
            points = [(x, y - 10), (x - 12, y + 8), (x + 12, y + 8)]
            # Glow effect
            glow_points = [(x, y - 15), (x - 18, y + 12), (x + 18, y + 12)]
            pygame.gfxdraw.aapolygon(self.screen, glow_points, self.COLOR_PLAYER_GLOW)
            pygame.gfxdraw.filled_polygon(self.screen, glow_points, self.COLOR_PLAYER_GLOW)
            # Main ship
            pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_PLAYER)
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PLAYER)
            # Cockpit
            pygame.gfxdraw.aacircle(self.screen, x, y, 3, (200, 255, 255))
            pygame.gfxdraw.filled_circle(self.screen, x, y, 3, (200, 255, 255))

    def _render_aliens(self):
        for alien in self.aliens:
            x, y = int(alien['pos'][0]), int(alien['pos'][1])
            # U-shape alien
            points = [(x-10, y-8), (x-10, y+8), (x-5, y+8), (x-5, y-2), (x+5, y-2), (x+5, y+8), (x+10, y+8), (x+10, y-8)]
            wave_color_shift = min(100, (self.current_wave - 1) * 20)
            color = (
                self.COLOR_ALIEN_BASE[0], 
                max(0, self.COLOR_ALIEN_BASE[1] - wave_color_shift), 
                max(0, self.COLOR_ALIEN_BASE[2] - wave_color_shift)
            )
            pygame.gfxdraw.aapolygon(self.screen, points, color)
            pygame.gfxdraw.filled_polygon(self.screen, points, color)
    
    def _render_projectiles(self):
        for proj in self.player_projectiles:
            start = (int(proj[0]), int(proj[1]))
            end = (int(proj[0]), int(proj[1]) - 10)
            pygame.draw.line(self.screen, self.COLOR_PLAYER_PROJ, start, end, 3)
        for proj in self.alien_projectiles:
            start = (int(proj[0]), int(proj[1]))
            end = (int(proj[0]), int(proj[1]) + 6)
            pygame.draw.line(self.screen, self.COLOR_ALIEN_PROJ, start, end, 3)

    def _render_particles(self):
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['life'] / 30.0))))
            color = p['color'] + (alpha,)
            size = int(p['life'] * 0.3)
            if size > 0:
                pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), size, color)

    def _render_ui(self):
        # Score
        score_text = self.font_main.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Wave
        wave_text = self.font_main.render(f"WAVE: {self.current_wave}/{self.NUM_WAVES}", True, self.COLOR_UI_TEXT)
        self.screen.blit(wave_text, (10, 35))
        
        # Health
        for i in range(self.player_health):
            x, y = self.WIDTH - 30 - (i * 30), 25
            points = [(x, y - 6), (x - 7, y + 5), (x + 7, y + 5)]
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PLAYER)

        # Game Over / Win message
        if self.game_over:
            if self.player_health <= 0 or self.steps >= self.MAX_STEPS:
                msg = "GAME OVER"
            else:
                msg = "YOU WIN!"
            
            title_text = self.font_title.render(msg, True, (255, 255, 255))
            text_rect = title_text.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
            self.screen.blit(title_text, text_rect)
            
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.current_wave,
            "player_health": self.player_health
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """
        Call this at the end of __init__ to verify implementation.
        """
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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    running = True
    terminated = False
    
    # Create a display for human play
    pygame.display.set_caption("Space Shooter")
    display_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))

    while running:
        # Action defaults
        movement = 0  # No-op
        space = 0     # Released
        shift = 0     # Released (unused)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                terminated = False

        if not terminated:
            keys = pygame.key.get_pressed()
            if keys[pygame.K_LEFT]:
                movement = 3
            elif keys[pygame.K_RIGHT]:
                movement = 4
            
            if keys[pygame.K_SPACE]:
                space = 1

            action = [movement, space, shift]
            obs, reward, terminated, truncated, info = env.step(action)

        # Display the observation from the environment
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        display_screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        env.clock.tick(env.FPS)

    env.close()
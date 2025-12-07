
# Generated: 2025-08-27T12:42:22.297612
# Source Brief: brief_00136.md
# Brief Index: 136

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ←→ to move. Hold space to fire your weapon."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A minimalist top-down space shooter. Destroy all invading aliens before they overwhelm you."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    
    COLOR_BG = (10, 10, 20)
    COLOR_PLAYER = (255, 255, 255)
    COLOR_PROJECTILE = (255, 255, 255)
    COLOR_ALIEN_S1 = (255, 80, 80)
    COLOR_ALIEN_S2 = (80, 255, 80)
    COLOR_ALIEN_S3 = (80, 150, 255)
    COLOR_TEXT = (240, 240, 240)
    COLOR_GAMEOVER = (255, 50, 50)
    COLOR_WIN = (50, 255, 50)

    PLAYER_WIDTH = 30
    PLAYER_HEIGHT = 15
    PLAYER_SPEED = 8
    PLAYER_Y_POS = 370
    
    ALIEN_SIZE = 18
    TOTAL_STAGES = 3
    ALIENS_PER_STAGE = 50
    
    PROJECTILE_SPEED = 12
    PROJECTILE_RADIUS = 3
    FIRE_COOLDOWN_FRAMES = 6
    
    MAX_EPISODE_STEPS = 5000
    INITIAL_LIVES = 3

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
        self.font_main = pygame.font.SysFont("monospace", 24, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 48, bold=True)
        
        self.render_mode = render_mode
        self.np_random = None

        # Initialize state variables
        self.reset()
        
        # Run validation check
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.np_random is None:
            self.np_random, seed = gym.utils.seeding.np_random(seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.lives = self.INITIAL_LIVES
        self.stage = 1
        self.game_over = False
        self.win = False
        
        self.player_pos = [self.SCREEN_WIDTH / 2, self.PLAYER_Y_POS]
        self.projectiles = []
        self.aliens = []
        self.particles = []
        
        self.fire_cooldown = 0
        
        self._spawn_aliens()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0
        
        if self.game_over or self.win:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_held = action[1] == 1  # Boolean
        
        # --- Update game logic ---
        self._handle_input(movement, space_held)
        
        self._update_player()
        reward += self._update_projectiles()
        reward += self._update_aliens()
        self._update_particles()
        
        collision_reward = self._handle_collisions()
        reward += collision_reward
        
        # --- Check for stage/game transitions ---
        if not self.aliens and not self.win:
            if self.stage < self.TOTAL_STAGES:
                self.stage += 1
                reward += 100  # Stage clear bonus
                self._spawn_aliens()
            else:
                self.win = True
                reward += 300  # Game win bonus
        
        if self.lives <= 0 and not self.game_over:
            self.game_over = True
            if collision_reward == 0: # Only apply penalty if not already penalized by collision
                 reward -= 100 # Penalty for losing a life (alien reached bottom)

        self.steps += 1
        terminated = self.game_over or self.win or self.steps >= self.MAX_EPISODE_STEPS
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _handle_input(self, movement, space_held):
        # Movement
        if movement == 3:  # Left
            self.player_pos[0] -= self.PLAYER_SPEED
        elif movement == 4:  # Right
            self.player_pos[0] += self.PLAYER_SPEED
        
        # Firing
        if space_held and self.fire_cooldown <= 0:
            # SFX: Player shoot
            self.projectiles.append(list(self.player_pos))
            self.fire_cooldown = self.FIRE_COOLDOWN_FRAMES

    def _update_player(self):
        self.player_pos[0] = np.clip(self.player_pos[0], self.PLAYER_WIDTH / 2, self.SCREEN_WIDTH - self.PLAYER_WIDTH / 2)
        if self.fire_cooldown > 0:
            self.fire_cooldown -= 1

    def _update_projectiles(self):
        reward = 0
        for p in self.projectiles[:]:
            p[1] -= self.PROJECTILE_SPEED
            if p[1] < 0:
                self.projectiles.remove(p)
                reward -= 0.02 # Penalty for missing
        return reward

    def _update_aliens(self):
        reward = 0
        stage_speed_mod = 0.5 + (self.stage - 1) * 0.2
        
        for alien in self.aliens[:]:
            alien['pos'][1] += stage_speed_mod
            
            # Horizontal movement patterns
            if alien['pattern'] == 'sine':
                alien['wave_angle'] += alien['frequency']
                alien['pos'][0] = alien['initial_x'] + math.sin(alien['wave_angle']) * alien['amplitude']
            elif alien['pattern'] == 'diag':
                alien['pos'][0] += alien['vx'] * stage_speed_mod
                if not (0 < alien['pos'][0] < self.SCREEN_WIDTH):
                    alien['vx'] *= -1

            # Check if alien reached bottom
            if alien['pos'][1] > self.SCREEN_HEIGHT:
                self.aliens.remove(alien)
                self.lives -= 1
                reward -= 100 # Penalty for losing a life
                # SFX: Life lost
        return reward

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['lifespan'] -= 1
            p['vel'][1] += 0.05 # Gravity
            if p['lifespan'] <= 0:
                self.particles.remove(p)

    def _handle_collisions(self):
        reward = 0
        
        # Projectile-Alien collisions
        for proj in self.projectiles[:]:
            for alien in self.aliens[:]:
                dist = math.hypot(proj[0] - alien['pos'][0], proj[1] - alien['pos'][1])
                if dist < self.ALIEN_SIZE / 2 + self.PROJECTILE_RADIUS:
                    self.aliens.remove(alien)
                    if proj in self.projectiles: self.projectiles.remove(proj)
                    self.score += 10
                    reward += 1 # Reward for destroying an alien
                    self._create_explosion(alien['pos'], alien['color'])
                    # SFX: Explosion
                    break

        # Alien-Player collisions
        player_rect = pygame.Rect(self.player_pos[0] - self.PLAYER_WIDTH / 2, self.player_pos[1] - self.PLAYER_HEIGHT / 2, self.PLAYER_WIDTH, self.PLAYER_HEIGHT)
        for alien in self.aliens[:]:
            alien_rect = pygame.Rect(alien['pos'][0] - self.ALIEN_SIZE / 2, alien['pos'][1] - self.ALIEN_SIZE / 2, self.ALIEN_SIZE, self.ALIEN_SIZE)
            if player_rect.colliderect(alien_rect):
                self.aliens.remove(alien)
                self.lives -= 1
                reward -= 100 # Penalty for losing a life
                self._create_explosion(self.player_pos, self.COLOR_PLAYER)
                # SFX: Player hit/explosion
        
        return reward
        
    def _spawn_aliens(self):
        self.aliens.clear()
        colors = [self.COLOR_ALIEN_S1, self.COLOR_ALIEN_S2, self.COLOR_ALIEN_S3]
        alien_color = colors[self.stage - 1]
        
        for i in range(self.ALIENS_PER_STAGE):
            pattern_choice = self.np_random.choice(['sine', 'diag'])
            start_x = self.np_random.uniform(self.ALIEN_SIZE, self.SCREEN_WIDTH - self.ALIEN_SIZE)
            start_y = self.np_random.uniform(-500, -50)
            
            alien = {'pos': [start_x, start_y], 'color': alien_color}
            
            if pattern_choice == 'sine':
                alien.update({
                    'pattern': 'sine',
                    'initial_x': start_x,
                    'amplitude': self.np_random.uniform(50, 200),
                    'frequency': self.np_random.uniform(0.02, 0.05),
                    'wave_angle': self.np_random.uniform(0, 2 * math.pi)
                })
            else: # 'diag'
                alien.update({
                    'pattern': 'diag',
                    'vx': self.np_random.choice([-1.5, 1.5])
                })
            self.aliens.append(alien)
            
    def _create_explosion(self, pos, color):
        num_particles = 20
        for _ in range(num_particles):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifespan = self.np_random.integers(15, 30)
            self.particles.append({
                'pos': list(pos),
                'vel': vel,
                'lifespan': lifespan,
                'max_lifespan': lifespan,
                'color': color
            })
    
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
        # Render aliens
        for alien in self.aliens:
            pos = (int(alien['pos'][0]), int(alien['pos'][1]))
            pygame.gfxdraw.box(self.screen, (pos[0] - self.ALIEN_SIZE//2, pos[1] - self.ALIEN_SIZE//2, self.ALIEN_SIZE, self.ALIEN_SIZE), alien['color'])
            
        # Render projectiles
        for p in self.projectiles:
            pos = (int(p[0]), int(p[1]))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.PROJECTILE_RADIUS, self.COLOR_PROJECTILE)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.PROJECTILE_RADIUS, self.COLOR_PROJECTILE)
    
        # Render player
        player_rect = (self.player_pos[0] - self.PLAYER_WIDTH / 2, self.player_pos[1] - self.PLAYER_HEIGHT / 2, self.PLAYER_WIDTH, self.PLAYER_HEIGHT)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect)
        
        # Render particles
        for p in self.particles:
            life_ratio = p['lifespan'] / p['max_lifespan']
            radius = int(life_ratio * 5)
            alpha = int(life_ratio * 255)
            color = (*p['color'], alpha)
            pos = (int(p['pos'][0]), int(p['pos'][1]))
            if radius > 0:
                # Use a temporary surface for alpha blending
                temp_surf = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
                pygame.draw.circle(temp_surf, color, (radius, radius), radius)
                self.screen.blit(temp_surf, (pos[0] - radius, pos[1] - radius))

    def _render_ui(self):
        # Score
        score_text = self.font_main.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        # Lives
        lives_text = self.font_main.render(f"LIVES: {self.lives}", True, self.COLOR_TEXT)
        self.screen.blit(lives_text, (self.SCREEN_WIDTH - lives_text.get_width() - 10, 10))

        # Stage
        stage_text = self.font_main.render(f"STAGE: {self.stage}/{self.TOTAL_STAGES}", True, self.COLOR_TEXT)
        self.screen.blit(stage_text, (self.SCREEN_WIDTH//2 - stage_text.get_width()//2, 10))
        
        # Game Over / Win message
        if self.game_over:
            msg = self.font_large.render("GAME OVER", True, self.COLOR_GAMEOVER)
            self.screen.blit(msg, (self.SCREEN_WIDTH // 2 - msg.get_width() // 2, self.SCREEN_HEIGHT // 2 - msg.get_height() // 2))
        elif self.win:
            msg = self.font_large.render("YOU WIN!", True, self.COLOR_WIN)
            self.screen.blit(msg, (self.SCREEN_WIDTH // 2 - msg.get_width() // 2, self.SCREEN_HEIGHT // 2 - msg.get_height() // 2))
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "stage": self.stage,
            "win": self.win,
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
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    # Set `render_mode` to "human" if you want a visible window
    import os
    os.environ['SDL_VIDEODRIVER'] = 'x11' # Use 'dummy' for headless, 'x11', 'wayland', 'windows', or 'quartz' for visible window
    
    env = GameEnv(render_mode="rgb_array")
    
    # For human play
    pygame.display.set_caption("Space Invaders Gym")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    
    obs, info = env.reset()
    terminated = False
    
    while not terminated:
        # --- Human Controls ---
        keys = pygame.key.get_pressed()
        movement = 0 # none
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        # --- End Human Controls ---
        
        # --- Agent Controls (uncomment for random agent) ---
        # action = env.action_space.sample()
        # --- End Agent Controls ---

        obs, reward, terminated, truncated, info = env.step(action)
        
        # --- Rendering for Human ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        # --- End Rendering ---
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
        
        env.clock.tick(30) # Limit to 30 FPS for smooth play

    print(f"Game Over! Final Info: {info}")
    env.close()
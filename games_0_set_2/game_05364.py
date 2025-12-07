
# Generated: 2025-08-28T04:47:17.655249
# Source Brief: brief_05364.md
# Brief Index: 5364

        
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
        "Defend Earth from a descending alien horde in this fast-paced, top-down arcade shooter."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and world dimensions
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
        try:
            self.font_ui = pygame.font.Font(pygame.font.get_default_font(), 18)
            self.font_game_over = pygame.font.Font(pygame.font.get_default_font(), 48)
        except:
            self.font_ui = pygame.font.SysFont("monospace", 18)
            self.font_game_over = pygame.font.SysFont("monospace", 48)

        # Colors
        self.COLOR_BG = (10, 5, 20)
        self.COLOR_PLAYER = (50, 255, 50)
        self.COLOR_ALIEN = (255, 50, 50)
        self.COLOR_PROJECTILE = (255, 255, 100)
        self.COLOR_EXPLOSION = (255, 150, 50)
        self.COLOR_UI = (200, 200, 255)
        
        # Game constants
        self.MAX_STEPS = 10000
        self.INITIAL_LIVES = 3
        self.TOTAL_ALIENS = 50
        self.PLAYER_SPEED = 8
        self.PROJECTILE_SPEED = 12
        self.FIRE_COOLDOWN = 6 # frames
        self.PLAYER_WIDTH = 30
        self.PLAYER_HEIGHT = 15
        
        # Initialize state variables
        self.stars = []
        self.player_rect = None
        self.player_lives = 0
        self.projectiles = []
        self.aliens = []
        self.explosions = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        self.last_shot_time = 0
        self.aliens_destroyed_count = 0
        self.initial_alien_descent_speed = 1.0
        self.initial_alien_lateral_amplitude = 2.0
        self.alien_descent_speed = 1.0
        self.alien_lateral_amplitude = 2.0
        
        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        self.player_lives = self.INITIAL_LIVES
        
        self.player_rect = pygame.Rect(
            self.WIDTH // 2 - self.PLAYER_WIDTH // 2, 
            self.HEIGHT - self.PLAYER_HEIGHT - 10,
            self.PLAYER_WIDTH,
            self.PLAYER_HEIGHT
        )
        
        self.projectiles = []
        self.aliens = []
        self.explosions = []
        
        self.last_shot_time = 0
        self.aliens_destroyed_count = 0
        self.alien_descent_speed = self.initial_alien_descent_speed
        self.alien_lateral_amplitude = self.initial_alien_lateral_amplitude
        
        self._spawn_stars()
        self._spawn_aliens()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0
        terminated = False

        if not self.game_over:
            # Unpack factorized action
            movement = action[0]
            space_held = action[1] == 1
            
            # Update game logic
            self._handle_input(movement, space_held)
            self._update_projectiles()
            self._update_aliens()
            reward += self._handle_collisions()
            self._update_explosions()
            self._update_stars()

            # Continuous survival reward
            reward += 0.01

            # Check for termination conditions
            if self.player_lives <= 0:
                self.game_over = True
                self.game_won = False
                reward = -100.0 # Large penalty for losing
            elif len(self.aliens) == 0:
                self.game_over = True
                self.game_won = True
                reward = 100.0 # Large reward for winning
        
        self.steps += 1
        if self.steps >= self.MAX_STEPS:
            terminated = True

        terminated = terminated or self.game_over
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, movement, space_held):
        # Movement: 3=left, 4=right
        if movement == 3:
            self.player_rect.x -= self.PLAYER_SPEED
        elif movement == 4:
            self.player_rect.x += self.PLAYER_SPEED
        
        # Clamp player position to screen bounds
        self.player_rect.x = max(0, min(self.WIDTH - self.player_rect.width, self.player_rect.x))

        # Firing
        if space_held and (self.steps - self.last_shot_time) > self.FIRE_COOLDOWN:
            # sfx: Player shoot sound
            self.last_shot_time = self.steps
            proj_rect = pygame.Rect(self.player_rect.centerx - 2, self.player_rect.top - 10, 4, 10)
            self.projectiles.append(proj_rect)

    def _update_projectiles(self):
        for proj in self.projectiles[:]:
            proj.y -= self.PROJECTILE_SPEED
            if proj.bottom < 0:
                self.projectiles.remove(proj)

    def _update_aliens(self):
        for alien in self.aliens[:]:
            alien['y'] += self.alien_descent_speed
            offset = self.alien_lateral_amplitude * math.sin(alien['initial_x'] + self.steps * 0.05)
            alien['rect'].x = int(alien['initial_x'] + offset)
            alien['rect'].y = int(alien['y'])

            if alien['rect'].top > self.HEIGHT:
                self.aliens.remove(alien)
                self.player_lives -= 1
                # sfx: Life lost sound

    def _handle_collisions(self):
        reward = 0
        for proj in self.projectiles[:]:
            for alien in self.aliens[:]:
                if proj.colliderect(alien['rect']):
                    # sfx: Alien explosion sound
                    self.projectiles.remove(proj)
                    self.aliens.remove(alien)
                    self._create_explosion(alien['rect'].center)
                    self.score += 10
                    reward += 1.0

                    self.aliens_destroyed_count += 1
                    if self.aliens_destroyed_count % 10 == 0:
                        self.alien_descent_speed += 0.05
                    if self.aliens_destroyed_count % 20 == 0:
                        self.alien_lateral_amplitude += 0.2
                    
                    break # Projectile can only hit one alien
        return reward

    def _spawn_aliens(self):
        rows = 5
        cols = self.TOTAL_ALIENS // rows
        h_spacing = self.WIDTH // (cols + 1)
        v_spacing = 50
        alien_size = 20
        
        for row in range(rows):
            for col in range(cols):
                initial_x = (col + 1) * h_spacing
                initial_y = -((row + 1) * v_spacing)
                rect = pygame.Rect(initial_x - alien_size//2, initial_y, alien_size, alien_size)
                self.aliens.append({'rect': rect, 'initial_x': initial_x, 'y': float(initial_y)})

    def _spawn_stars(self):
        self.stars = []
        for _ in range(150):
            x = self.np_random.integers(0, self.WIDTH)
            y = self.np_random.integers(0, self.HEIGHT)
            speed = self.np_random.uniform(0.2, 1.0)
            size = int(speed * 2)
            self.stars.append([x, y, speed, size])

    def _update_stars(self):
        for star in self.stars:
            star[1] += star[2]
            if star[1] > self.HEIGHT:
                star[0] = self.np_random.integers(0, self.WIDTH)
                star[1] = 0

    def _create_explosion(self, pos):
        particles = []
        for _ in range(20):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifespan = self.np_random.integers(15, 30)
            particles.append({'pos': list(pos), 'vel': vel, 'life': lifespan})
        self.explosions.append(particles)

    def _update_explosions(self):
        for particle_group in self.explosions[:]:
            for particle in particle_group[:]:
                particle['pos'][0] += particle['vel'][0]
                particle['pos'][1] += particle['vel'][1]
                particle['life'] -= 1
                if particle['life'] <= 0:
                    particle_group.remove(particle)
            if not particle_group:
                self.explosions.remove(particle_group)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_aliens()
        self._render_projectiles()
        self._render_player()
        self._render_explosions()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.player_lives,
            "aliens_remaining": len(self.aliens),
        }

    def _render_background(self):
        for x, y, speed, size in self.stars:
            color_val = int(speed * 100)
            color = (color_val, color_val, color_val + 50)
            pygame.draw.rect(self.screen, color, (int(x), int(y), size, size))

    def _render_player(self):
        # Ship body
        p1 = (self.player_rect.centerx, self.player_rect.top)
        p2 = (self.player_rect.left, self.player_rect.bottom)
        p3 = (self.player_rect.right, self.player_rect.bottom)
        pygame.gfxdraw.aapolygon(self.screen, (p1, p2, p3), self.COLOR_PLAYER)
        pygame.gfxdraw.filled_polygon(self.screen, (p1, p2, p3), self.COLOR_PLAYER)

        # Engine glow
        engine_y = self.player_rect.bottom + 2
        engine_x1 = self.player_rect.left + self.player_rect.width * 0.3
        engine_x2 = self.player_rect.right - self.player_rect.width * 0.3
        glow_size = self.np_random.integers(3, 6)
        pygame.draw.circle(self.screen, (255,100,0), (int(engine_x1), engine_y), glow_size)
        pygame.draw.circle(self.screen, (255,200,0), (int(engine_x1), engine_y), glow_size // 2)
        pygame.draw.circle(self.screen, (255,100,0), (int(engine_x2), engine_y), glow_size)
        pygame.draw.circle(self.screen, (255,200,0), (int(engine_x2), engine_y), glow_size // 2)

    def _render_aliens(self):
        for alien in self.aliens:
            pygame.gfxdraw.aacircle(self.screen, alien['rect'].centerx, alien['rect'].centery, alien['rect'].width // 2, self.COLOR_ALIEN)
            pygame.gfxdraw.filled_circle(self.screen, alien['rect'].centerx, alien['rect'].centery, alien['rect'].width // 2, self.COLOR_ALIEN)
            # "Eyes"
            eye_y = alien['rect'].centery - 2
            eye_x1 = alien['rect'].centerx - 4
            eye_x2 = alien['rect'].centerx + 4
            pygame.draw.circle(self.screen, self.COLOR_BG, (eye_x1, eye_y), 2)
            pygame.draw.circle(self.screen, self.COLOR_BG, (eye_x2, eye_y), 2)

    def _render_projectiles(self):
        for proj in self.projectiles:
            pygame.draw.rect(self.screen, self.COLOR_PROJECTILE, proj, border_radius=2)

    def _render_explosions(self):
        for particle_group in self.explosions:
            for particle in particle_group:
                size = int(max(1, particle['life'] / 6))
                pygame.draw.circle(self.screen, self.COLOR_EXPLOSION, (int(particle['pos'][0]), int(particle['pos'][1])), size)

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_UI)
        self.screen.blit(score_text, (10, 10))

        # Aliens remaining
        aliens_text = self.font_ui.render(f"ALIENS: {len(self.aliens)}/{self.TOTAL_ALIENS}", True, self.COLOR_UI)
        self.screen.blit(aliens_text, (self.WIDTH - aliens_text.get_width() - 10, 10))

        # Lives
        for i in range(self.player_lives):
            ship_icon_rect = pygame.Rect(10 + i * (self.PLAYER_WIDTH * 0.7 + 5), 35, self.PLAYER_WIDTH * 0.7, self.PLAYER_HEIGHT * 0.7)
            p1 = (ship_icon_rect.centerx, ship_icon_rect.top)
            p2 = (ship_icon_rect.left, ship_icon_rect.bottom)
            p3 = (ship_icon_rect.right, ship_icon_rect.bottom)
            pygame.gfxdraw.filled_polygon(self.screen, (p1, p2, p3), self.COLOR_PLAYER)
        
        # Game Over message
        if self.game_over:
            msg = "MISSION COMPLETE" if self.game_won else "GAME OVER"
            color = (100, 255, 100) if self.game_won else (255, 100, 100)
            end_text = self.font_game_over.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

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
    # Set SDL_VIDEODRIVER to "dummy" if you're running headless
    # import os
    # os.environ["SDL_VIDEODRIVER"] = "dummy"

    env = GameEnv()
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Alien Horde Defender")
    clock = pygame.time.Clock()
    
    terminated = False
    
    while not terminated:
        # Map keyboard keys to the MultiDiscrete action space
        keys = pygame.key.get_pressed()
        
        movement = 0 # no-op
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            movement = 4
            
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 0 # Not used in this game
        
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation from the environment to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting game...")
                obs, info = env.reset()
                terminated = False

        clock.tick(30) # Run at 30 FPS

    print(f"Game Over. Final Score: {info['score']}")
    env.close()
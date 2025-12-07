
# Generated: 2025-08-27T15:19:11.177276
# Source Brief: brief_00958.md
# Brief Index: 958

        
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
        "Controls: ←→ to move the ship. Hold Space to fire your weapon."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A retro-inspired arcade shooter. Dodge alien projectiles and blast them out of the sky across five increasingly difficult waves to win."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Screen and game constants
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.MAX_STEPS = 10000
        self.WAVE_COUNT = 5

        # EXACT spaces:
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.SysFont("monospace", 36, bold=True)
        self.font_small = pygame.font.SysFont("monospace", 18, bold=True)
        
        # Colors
        self.COLOR_BG = (15, 15, 30)
        self.COLOR_PLAYER = (50, 150, 255)
        self.COLOR_PLAYER_EXHAUST = (255, 200, 50)
        self.COLOR_PLAYER_PROJECTILE = (100, 255, 100)
        self.COLOR_ALIEN_PROJECTILE = (255, 100, 100)
        self.ALIEN_COLORS = [(200, 50, 200), (255, 150, 0), (255, 255, 0)]
        self.COLOR_WHITE = (240, 240, 240)
        self.COLOR_HEALTH_FG = (50, 220, 50)
        self.COLOR_HEALTH_BG = (180, 50, 50)

        # Game entity constants
        self.PLAYER_SPEED = 8
        self.PLAYER_FIRE_COOLDOWN = 6 # frames
        self.PROJECTILE_SPEED = 12
        
        # State variables are initialized in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.np_random = None
        self.player_rect = None
        self.player_health = 0
        self.player_fire_timer = 0
        self.current_wave = 0
        self.aliens = []
        self.projectiles = []
        self.explosions = []
        self.stars = []

        # Initialize state for the first time
        self.reset()
        
        # self.validate_implementation() # Optional: Call to test during development

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize RNG
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)
        else:
            # Fallback if seed is None
            if self.np_random is None:
                self.np_random, _ = gym.utils.seeding.np_random(random.randint(0, 1e9))

        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        
        self.player_rect = pygame.Rect(
            self.SCREEN_WIDTH / 2 - 15, self.SCREEN_HEIGHT - 60, 30, 20
        )
        self.player_health = 100
        self.player_fire_timer = 0

        self.current_wave = 1
        self.aliens = []
        self.projectiles = []
        self.explosions = []
        
        self._spawn_stars()
        self._spawn_wave()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.auto_advance:
            self.clock.tick(30)

        reward = 0.0

        if not self.game_over:
            # Survival reward
            reward += 0.01

            # Unpack factorized action
            movement = action[0]
            space_held = action[1] == 1

            # Update game logic
            self._handle_input(movement, space_held)
            self._update_aliens()
            self._update_projectiles()
            self._update_particles()
            
            collision_rewards = self._handle_collisions()
            reward += collision_rewards

            # Check for wave completion
            if not self.aliens and not self.win:
                if self.current_wave == self.WAVE_COUNT:
                    self.win = True
                else:
                    reward += 100
                    self.current_wave += 1
                    self._spawn_wave()

        # Update step counter
        self.steps += 1

        # Check for termination
        terminated = self.player_health <= 0 or self.win or self.steps >= self.MAX_STEPS
        if terminated and not self.game_over:
            self.game_over = True
            if self.win:
                reward += 500  # Win bonus
            elif self.player_health <= 0:
                reward -= 100 # Lose penalty

        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _spawn_stars(self):
        self.stars = []
        for _ in range(150):
            self.stars.append({
                "pos": [self.np_random.integers(0, self.SCREEN_WIDTH), self.np_random.integers(0, self.SCREEN_HEIGHT)],
                "speed": self.np_random.uniform(0.5, 2.0),
                "size": self.np_random.integers(1, 3),
                "color": random.choice([(100, 100, 100), (150, 150, 150), (200, 200, 200)])
            })

    def _spawn_wave(self):
        num_aliens = 3 + 2 * self.current_wave
        rows = math.ceil(num_aliens / 8)
        cols = min(num_aliens, 8)
        
        for i in range(num_aliens):
            row = i // cols
            col = i % cols
            
            x = (self.SCREEN_WIDTH / (cols + 1)) * (col + 1)
            y = 60 + row * 50
            
            self.aliens.append({
                "rect": pygame.Rect(x - 15, y - 10, 30, 20),
                "color": random.choice(self.ALIEN_COLORS),
                "fire_cooldown": self.np_random.integers(60, 120),
                "move_phase": self.np_random.uniform(0, 2 * math.pi),
                "base_x": x
            })

    def _handle_input(self, movement, space_held):
        # Player movement
        if movement == 3: # Left
            self.player_rect.x -= self.PLAYER_SPEED
        elif movement == 4: # Right
            self.player_rect.x += self.PLAYER_SPEED
        
        self.player_rect.left = max(0, self.player_rect.left)
        self.player_rect.right = min(self.SCREEN_WIDTH, self.player_rect.right)

        # Player firing
        if self.player_fire_timer > 0:
            self.player_fire_timer -= 1
        
        if space_held and self.player_fire_timer == 0:
            # sfx: player_shoot.wav
            self.projectiles.append({
                "rect": pygame.Rect(self.player_rect.centerx - 2, self.player_rect.top, 4, 10),
                "type": "player"
            })
            self.player_fire_timer = self.PLAYER_FIRE_COOLDOWN

    def _update_aliens(self):
        alien_projectile_speed = 4 + 0.2 * self.current_wave
        
        for alien in self.aliens:
            # Movement
            alien['move_phase'] += 0.03
            offset = 40 * math.sin(alien['move_phase'])
            alien['rect'].centerx = alien['base_x'] + offset

            # Firing
            alien['fire_cooldown'] -= 1
            if alien['fire_cooldown'] <= 0:
                # sfx: alien_shoot.wav
                self.projectiles.append({
                    "rect": pygame.Rect(alien['rect'].centerx - 3, alien['rect'].bottom, 6, 12),
                    "type": "alien",
                    "speed": alien_projectile_speed
                })
                alien['fire_cooldown'] = self.np_random.integers(90, 150) - (self.current_wave * 10)

    def _update_projectiles(self):
        for proj in self.projectiles[:]:
            if proj['type'] == 'player':
                proj['rect'].y -= self.PROJECTILE_SPEED
            else: # alien
                proj['rect'].y += proj['speed']
            
            if proj['rect'].bottom < 0 or proj['rect'].top > self.SCREEN_HEIGHT:
                self.projectiles.remove(proj)

    def _update_particles(self):
        # Explosions
        for exp in self.explosions[:]:
            exp['radius'] += exp['speed']
            if exp['radius'] > exp['max_radius']:
                self.explosions.remove(exp)
        
        # Stars
        for star in self.stars:
            star['pos'][1] += star['speed']
            if star['pos'][1] > self.SCREEN_HEIGHT:
                star['pos'][1] = 0
                star['pos'][0] = self.np_random.integers(0, self.SCREEN_WIDTH)

    def _handle_collisions(self):
        reward = 0
        
        # Player projectiles vs Aliens
        for proj in self.projectiles[:]:
            if proj['type'] != 'player':
                continue
            
            for alien in self.aliens[:]:
                if proj['rect'].colliderect(alien['rect']):
                    # sfx: explosion.wav
                    self.aliens.remove(alien)
                    if proj in self.projectiles: self.projectiles.remove(proj)
                    self.score += 10
                    reward += 10
                    self.explosions.append({
                        "pos": alien['rect'].center,
                        "radius": 5, "max_radius": 40, "speed": 2, "color": (255, 255, 150)
                    })
                    break

        # Alien projectiles vs Player
        for proj in self.projectiles[:]:
            if proj['type'] != 'alien':
                continue
                
            if self.player_rect.colliderect(proj['rect']):
                # sfx: player_hit.wav
                if proj in self.projectiles: self.projectiles.remove(proj)
                self.player_health -= 10
                reward -= 0.2
                self.explosions.append({
                    "pos": self.player_rect.center,
                    "radius": 3, "max_radius": 25, "speed": 1.5, "color": (255, 100, 50)
                })
                if self.player_health <= 0:
                    self.player_health = 0
                    self._create_player_explosion()
                break
        return reward
    
    def _create_player_explosion(self):
        self.explosions.append({
            "pos": self.player_rect.center,
            "radius": 10, "max_radius": 80, "speed": 2, "color": (255, 255, 255)
        })

    def _get_observation(self):
        # Clear screen with background color
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_background()
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        for star in self.stars:
            pygame.draw.circle(self.screen, star['color'], (int(star['pos'][0]), int(star['pos'][1])), star['size'])

    def _render_game(self):
        # Render player
        if self.player_health > 0:
            # Ship body
            p = self.player_rect
            ship_points = [(p.centerx, p.top), (p.left, p.bottom), (p.right, p.bottom)]
            pygame.draw.polygon(self.screen, self.COLOR_PLAYER, ship_points)
            pygame.gfxdraw.aapolygon(self.screen, ship_points, self.COLOR_WHITE)
            # Exhaust flame
            flame_height = 5 + (self.steps % 3) * 2
            flame_points = [(p.centerx - 5, p.bottom), (p.centerx + 5, p.bottom), (p.centerx, p.bottom + flame_height)]
            pygame.draw.polygon(self.screen, self.COLOR_PLAYER_EXHAUST, flame_points)

        # Render aliens
        for alien in self.aliens:
            r = alien['rect']
            alien_points = [(r.left, r.centery), (r.centerx, r.top), (r.right, r.centery), (r.centerx, r.bottom)]
            pygame.draw.polygon(self.screen, alien['color'], alien_points)
            pygame.gfxdraw.aapolygon(self.screen, alien_points, self.COLOR_WHITE)

        # Render projectiles
        for proj in self.projectiles:
            color = self.COLOR_PLAYER_PROJECTILE if proj['type'] == 'player' else self.COLOR_ALIEN_PROJECTILE
            pygame.draw.rect(self.screen, color, proj['rect'], border_radius=3)

        # Render explosions
        for exp in self.explosions:
            pos = (int(exp['pos'][0]), int(exp['pos'][1]))
            # Draw multiple circles for a glow effect
            for i in range(4):
                alpha = 255 * (1 - (exp['radius'] / exp['max_radius']))**2
                radius = exp['radius'] * (1 - i * 0.2)
                if radius > 0 and alpha > 0:
                    color = (*exp['color'], int(alpha))
                    # Pygame doesn't directly support drawing transparent shapes on a non-alpha surface
                    # We create a temporary surface to achieve this effect
                    temp_surf = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
                    pygame.draw.circle(temp_surf, color, (radius, radius), radius)
                    self.screen.blit(temp_surf, (pos[0] - radius, pos[1] - radius), special_flags=pygame.BLEND_RGBA_ADD)

    def _render_ui(self):
        # Score
        score_text = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_WHITE)
        self.screen.blit(score_text, (self.SCREEN_WIDTH - score_text.get_width() - 10, 10))

        # Wave
        wave_text = self.font_small.render(f"WAVE: {self.current_wave}/{self.WAVE_COUNT}", True, self.COLOR_WHITE)
        self.screen.blit(wave_text, (10, 10))

        # Health Bar
        health_bar_width = 200
        health_bar_height = 20
        health_bar_x = (self.SCREEN_WIDTH - health_bar_width) / 2
        health_bar_y = self.SCREEN_HEIGHT - health_bar_height - 10
        
        current_health_width = health_bar_width * (max(0, self.player_health) / 100)
        
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BG, (health_bar_x, health_bar_y, health_bar_width, health_bar_height), border_radius=5)
        if current_health_width > 0:
            pygame.draw.rect(self.screen, self.COLOR_HEALTH_FG, (health_bar_x, health_bar_y, current_health_width, health_bar_height), border_radius=5)

        # Game Over / Win Text
        if self.game_over:
            message = "YOU WIN!" if self.win else "GAME OVER"
            color = self.COLOR_HEALTH_FG if self.win else self.COLOR_ALIEN_PROJECTILE
            end_text = self.font_large.render(message, True, color)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "player_health": self.player_health,
            "current_wave": self.current_wave,
            "aliens_remaining": len(self.aliens)
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation.
        '''
        print("Validating implementation...")
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
        assert "score" in info
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc is False
        assert isinstance(info, dict)
        assert "steps" in info

        # Test specific game mechanics
        self.reset()
        initial_score = self.score
        self.aliens = [{"rect": pygame.Rect(300, 100, 30, 20)}]
        self.projectiles = [{"rect": pygame.Rect(300, 110, 4, 10), "type": "player"}]
        self._handle_collisions()
        assert self.score == initial_score + 10, "Score did not increase correctly on alien hit"
        assert len(self.aliens) == 0, "Alien was not destroyed on hit"
        
        print("✓ Implementation validated successfully")


if __name__ == "__main__":
    # This block allows you to run the game directly to test it
    env = GameEnv(render_mode="rgb_array")
    
    # To test the implementation (optional)
    # env.validate_implementation()
    
    obs, info = env.reset()
    done = False
    
    # Pygame window for human play
    pygame.display.set_caption("Arcade Shooter")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    
    running = True
    total_reward = 0
    
    while running:
        # Action defaults to no-op
        action = [0, 0, 0] # [movement, space, shift]
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0
                done = False

        if not done:
            keys = pygame.key.get_pressed()
            
            movement_action = 0 # None
            if keys[pygame.K_LEFT] or keys[pygame.K_a]:
                movement_action = 3 # Left
            elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
                movement_action = 4 # Right
            
            space_action = 1 if keys[pygame.K_SPACE] else 0
            shift_action = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

            action = [movement_action, space_action, shift_action]
        
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

    env.close()
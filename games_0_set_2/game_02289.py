
# Generated: 2025-08-28T04:20:36.245491
# Source Brief: brief_02289.md
# Brief Index: 2289

        
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
        "Controls: Use ← and → to move the ship. Press Space to fire your weapon."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A retro arcade shooter. Survive 5 waves of descending aliens by shooting them down and dodging their projectiles. The difficulty increases with each wave."
    )

    # Frames auto-advance at 30fps for smooth arcade gameplay.
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
        
        # Pygame setup for headless rendering
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        
        # Colors
        self.COLOR_BG = (16, 16, 32)         # Dark blue/purple
        self.COLOR_PLAYER = (0, 255, 128)    # Bright cyan/green
        self.COLOR_ALIEN = (255, 32, 64)     # Bright red
        self.COLOR_BULLET = (255, 255, 255)  # White
        self.COLOR_EXPLOSION = (255, 255, 0) # Yellow
        self.COLOR_TEXT = (224, 224, 240)    # Light gray/lavender
        self.COLOR_UI_ACCENT = (0, 192, 255) # UI Accent
        
        # Fonts
        self.font_large = pygame.font.Font(None, 50)
        self.font_medium = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)

        # Game constants
        self.MAX_STEPS = 10000
        self.TOTAL_WAVES = 5
        self.PLAYER_SPEED = 8
        self.PLAYER_FIRE_COOLDOWN = 6 # frames
        self.PLAYER_BULLET_SPEED = 12
        self.WAVE_CLEAR_MESSAGE_DURATION = 90 # frames (3 seconds at 30fps)
        
        # Initialize state variables
        self.player_pos = None
        self.player_fire_cooldown_timer = None
        self.aliens = None
        self.player_bullets = None
        self.alien_bullets = None
        self.particles = None
        self.current_wave = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.victory = None
        self.wave_clear_timer = None
        
        # Call reset to populate the state
        self.reset()

        # Validate implementation after initialization
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.player_pos = pygame.Vector2(self.WIDTH / 2, self.HEIGHT - 40)
        self.player_fire_cooldown_timer = 0
        
        self.aliens = []
        self.player_bullets = []
        self.alien_bullets = []
        self.particles = []
        
        self.current_wave = 1
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.victory = False
        self.wave_clear_timer = 0
        
        self._spawn_wave()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = -0.01  # Small penalty for each step to encourage efficiency
        self.steps += 1
        
        if not self.game_over:
            self._handle_input(action)
            self._update_game_state()
            reward += self._handle_collisions()
            reward += self._check_wave_status()

        terminated = self.game_over or self.victory or self.steps >= self.MAX_STEPS
        if self.game_over:
             reward = -10.0 # Final penalty for losing
        elif self.victory:
             reward = 100.0 # Final large bonus for winning the game
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        movement = action[0]
        space_held = action[1] == 1

        # Player Movement
        if movement == 3:  # Left
            self.player_pos.x -= self.PLAYER_SPEED
        elif movement == 4:  # Right
            self.player_pos.x += self.PLAYER_SPEED
        self.player_pos.x = np.clip(self.player_pos.x, 20, self.WIDTH - 20)
        
        # Player Firing
        if space_held and self.player_fire_cooldown_timer == 0:
            # sfx: player_shoot.wav
            bullet_pos = self.player_pos + pygame.Vector2(0, -20)
            self.player_bullets.append(bullet_pos)
            self.player_fire_cooldown_timer = self.PLAYER_FIRE_COOLDOWN
            # Muzzle flash particle
            for _ in range(5):
                self.particles.append({
                    "pos": bullet_pos.copy(),
                    "vel": pygame.Vector2(random.uniform(-2, 2), random.uniform(-4, -1)),
                    "radius": random.randint(2, 4),
                    "lifetime": 5,
                    "color": self.COLOR_BULLET
                })

    def _update_game_state(self):
        # Cooldowns
        if self.player_fire_cooldown_timer > 0:
            self.player_fire_cooldown_timer -= 1
        if self.wave_clear_timer > 0:
            self.wave_clear_timer -= 1

        # Move player bullets
        self.player_bullets = [b + pygame.Vector2(0, -self.PLAYER_BULLET_SPEED) for b in self.player_bullets if b.y > 0]

        # Update aliens
        for alien in self.aliens:
            alien["pos"] += alien["vel"]
            # Horizontal boundary check
            if alien["pos"].x < alien["size"] or alien["pos"].x > self.WIDTH - alien["size"]:
                alien["vel"].x *= -1
            
            # Alien Firing
            if random.random() < alien["fire_rate"]:
                # sfx: alien_shoot.wav
                self.alien_bullets.append(alien["pos"].copy())
        
        # Move alien bullets
        alien_bullet_speed = 1 + self.current_wave
        self.alien_bullets = [b + pygame.Vector2(0, alien_bullet_speed) for b in self.alien_bullets if b.y < self.HEIGHT]

        # Update particles
        for p in self.particles:
            p["pos"] += p["vel"]
            p["lifetime"] -= 1
        self.particles = [p for p in self.particles if p["lifetime"] > 0]
        
    def _handle_collisions(self):
        reward = 0
        
        # Player bullets hitting aliens
        for bullet in self.player_bullets[:]:
            for alien in self.aliens[:]:
                if bullet.distance_to(alien["pos"]) < alien["size"]:
                    # sfx: explosion.wav
                    self._create_explosion(alien["pos"])
                    self.aliens.remove(alien)
                    if bullet in self.player_bullets: self.player_bullets.remove(bullet)
                    self.score += 10
                    reward += 1.0
                    break
        
        # Alien bullets hitting player
        player_rect = pygame.Rect(self.player_pos.x - 15, self.player_pos.y - 10, 30, 20)
        for bullet in self.alien_bullets[:]:
            if player_rect.collidepoint(bullet.x, bullet.y):
                # sfx: player_hit.wav
                self._create_explosion(self.player_pos)
                self.game_over = True
                self.alien_bullets.remove(bullet)
                break
                
        # Aliens hitting player (or reaching bottom)
        for alien in self.aliens[:]:
            if alien["pos"].y > self.HEIGHT - 50:
                 self.game_over = True
                 self._create_explosion(alien["pos"])
                 self.aliens.remove(alien)
                 break
        
        return reward

    def _check_wave_status(self):
        reward = 0
        if not self.aliens and not self.game_over:
            if self.current_wave == self.TOTAL_WAVES:
                self.victory = True
            else:
                if self.wave_clear_timer == 0: # Check if we just cleared it
                    # sfx: wave_clear.wav
                    self.current_wave += 1
                    self.wave_clear_timer = self.WAVE_CLEAR_MESSAGE_DURATION
                    reward += 100.0 # Big reward for clearing a wave
                    self._spawn_wave()
        return reward

    def _spawn_wave(self):
        self.player_bullets.clear()
        self.alien_bullets.clear()
        
        rows, cols = 2 + self.current_wave, 5 + self.current_wave
        x_spacing = self.WIDTH * 0.8 / (cols -1)
        y_spacing = 50
        
        descent_speed = 0.2 + self.current_wave * 0.1
        fire_prob = 0.001 + self.current_wave * 0.002

        for r in range(rows):
            for c in range(cols):
                alien_type = random.choice(["normal", "sine", "fast"])
                
                vel = pygame.Vector2(0, descent_speed)
                if alien_type == "sine":
                    vel.x = random.choice([-1, 1]) * (1 + self.current_wave * 0.2)
                elif alien_type == "fast":
                    vel.y *= 1.5

                self.aliens.append({
                    "pos": pygame.Vector2(self.WIDTH * 0.1 + c * x_spacing, 60 + r * y_spacing),
                    "vel": vel,
                    "size": 12,
                    "fire_rate": fire_prob * (1.5 if alien_type == "fast" else 1),
                    "type": alien_type,
                })

    def _create_explosion(self, pos):
        num_particles = 30
        for _ in range(num_particles):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 5)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            self.particles.append({
                "pos": pos.copy(),
                "vel": vel,
                "radius": random.randint(5, 10),
                "lifetime": random.randint(15, 30),
                "color": self.COLOR_EXPLOSION
            })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw particles
        for p in self.particles:
            p_pos = (int(p["pos"].x), int(p["pos"].y))
            alpha = int(255 * (p["lifetime"] / 30))
            pygame.gfxdraw.filled_circle(self.screen, p_pos[0], p_pos[1], int(p["radius"]), (*p["color"], alpha))

        # Draw alien bullets
        for b in self.alien_bullets:
            pygame.draw.rect(self.screen, self.COLOR_BULLET, (int(b.x)-2, int(b.y)-4, 4, 8))
            
        # Draw player bullets
        for b in self.player_bullets:
            pygame.draw.rect(self.screen, self.COLOR_PLAYER, (int(b.x)-2, int(b.y)-8, 4, 16))
            pygame.gfxdraw.box(self.screen, (int(b.x)-2, int(b.y)-8, 4, 16), (*self.COLOR_PLAYER, 128))


        # Draw aliens
        for a in self.aliens:
            pos = (int(a["pos"].x), int(a["pos"].y))
            size = int(a["size"])
            p1 = (pos[0], pos[1] - size)
            p2 = (pos[0] - size, pos[1] + size)
            p3 = (pos[0] + size, pos[1] + size)
            pygame.gfxdraw.aapolygon(self.screen, (p1, p2, p3), self.COLOR_ALIEN)
            pygame.gfxdraw.filled_polygon(self.screen, (p1, p2, p3), self.COLOR_ALIEN)

        # Draw player
        if not self.game_over:
            p_pos = (int(self.player_pos.x), int(self.player_pos.y))
            ship_points = [
                (p_pos[0], p_pos[1] - 15),
                (p_pos[0] - 18, p_pos[1] + 10),
                (p_pos[0] + 18, p_pos[1] + 10)
            ]
            pygame.gfxdraw.aapolygon(self.screen, ship_points, self.COLOR_PLAYER)
            pygame.gfxdraw.filled_polygon(self.screen, ship_points, self.COLOR_PLAYER)
            # Engine glow
            engine_y = p_pos[1] + 12
            for i in range(5):
                alpha = 200 - i * 40
                radius = 8 - i
                pygame.gfxdraw.filled_circle(self.screen, p_pos[0], engine_y, radius, (*self.COLOR_UI_ACCENT, alpha))

    def _render_ui(self):
        # Draw Score
        score_text = self.font_medium.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, 10))

        # Draw Wave
        wave_text = self.font_medium.render(f"WAVE: {self.current_wave}/{self.TOTAL_WAVES}", True, self.COLOR_TEXT)
        self.screen.blit(wave_text, (self.WIDTH - wave_text.get_width() - 20, 10))
        
        # Game Over / Victory / Wave Clear Messages
        if self.game_over:
            self._render_centered_text("GAME OVER", self.font_large, self.COLOR_ALIEN)
        elif self.victory:
            self._render_centered_text("VICTORY", self.font_large, self.COLOR_PLAYER)
        elif self.wave_clear_timer > 0:
            alpha = 255
            if self.wave_clear_timer < 30: # Fade out
                alpha = int(255 * (self.wave_clear_timer / 30))
            self._render_centered_text(f"WAVE {self.current_wave - 1} CLEARED", self.font_large, (*self.COLOR_UI_ACCENT, alpha))

    def _render_centered_text(self, text, font, color):
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
        self.screen.blit(text_surface, text_rect)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.current_wave,
            "victory": self.victory,
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

# Example of how to run the environment
if __name__ == '__main__':
    # Set Pygame to use a visible display
    import os
    os.environ.pop('SDL_VIDEODRIVER', None)

    env = GameEnv(render_mode="rgb_array")
    
    # --- Manual Play ---
    obs, info = env.reset()
    terminated = False
    
    # Pygame setup for display
    display_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Alien Invaders")
    clock = pygame.time.Clock()

    while not terminated:
        # Action defaults
        movement = 0 # no-op
        space_held = 0
        shift_held = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        
        if keys[pygame.K_SPACE]:
            space_held = 1

        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Display the observation from the environment
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        display_screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(30) # Match the auto-advance rate

    print(f"Game Over! Final Info: {info}")
    env.close()